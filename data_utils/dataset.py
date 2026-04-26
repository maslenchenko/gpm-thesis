import glob
import json
import os
import sys
import re

from natsort import natsorted
import tensorflow as tf

# TFRecord constants
TFR_INPUT = "sequence"
TFR_OUTPUT = "target"

# Prevent Tensorflow from grabbing GPU memory
try:
    tf.config.experimental.set_visible_devices([], "GPU")
except Exception:
    pass


def file_to_records(filename):
    return tf.data.TFRecordDataset(filename, compression_type="ZLIB")


def get_target_type(description):
    match = re.search(r"^(\w+):", description)
    return match.group(1) if match else None


def get_orientation_type(identifier):
    c = identifier[-1]
    if c == "+":
        return 1
    if c == "-":
        return -1
    return 0


class SeqDataset:
    def __init__(
        self,
        data_dir="data",
        split_label="train",
        batch_size=2,
        shuffle_buffer=4,
        prefetch=1,
        seq_length_crop=None,
        mode="eval",
        drop_remainder=False,
    ):
        self.data_dir = data_dir
        self.split_label = split_label
        self.batch_size = batch_size
        self.shuffle_buffer = shuffle_buffer
        self.prefetch = prefetch
        self.seq_length_crop = seq_length_crop
        self.mode = mode
        self.drop_remainder = drop_remainder

        data_stats_file = os.path.join(self.data_dir, "statistics.json")
        with open(data_stats_file) as data_stats_open:
            data_stats = json.load(data_stats_open)

        self.seq_length = data_stats["seq_length"]
        self.seq_depth = data_stats.get("seq_depth", 4)
        self.seq_1hot = data_stats.get("seq_1hot", False)
        self.target_length = data_stats["target_length"]
        self.num_targets = data_stats["num_targets"]
        self.pool_width = data_stats["pool_width"]

        if self.seq_length_crop is not None:
            if self.seq_length_crop <= 0:
                raise ValueError(f"seq_length_crop must be positive, got {self.seq_length_crop}")
            if self.seq_length_crop > self.seq_length:
                raise ValueError(
                    f"seq_length_crop ({self.seq_length_crop}) exceeds seq_length ({self.seq_length})"
                )
            diff = self.seq_length - self.seq_length_crop
            if diff % 2 != 0:
                raise ValueError(
                    "seq_length - seq_length_crop must be even for centered crop "
                    f"(diff={diff})"
                )
        if self.shuffle_buffer < 0:
            raise ValueError(f"shuffle_buffer must be >= 0, got {self.shuffle_buffer}")
        if self.prefetch < -1:
            raise ValueError(f"prefetch must be -1 or >= 0, got {self.prefetch}")

        self.tfr_path = os.path.join(self.data_dir, "tfrecords", f"{self.split_label}-*.tfr")
        self.num_seqs = data_stats[f"{self.split_label}_seqs"]

        data_targets_file = os.path.join(self.data_dir, "targets.txt")
        with open(data_targets_file) as data_targets_open:
            targets_header, *targets = data_targets_open.read().splitlines()
            header_cols = targets_header.split("\t")
            id_index = header_cols.index("identifier")
            strand_pair_index = header_cols.index("strand_pair")
            description_index = header_cols.index("description")
            self.strand_pair = [int(line.split("\t")[strand_pair_index]) for line in targets]
            if len(self.strand_pair) != self.num_targets:
                raise ValueError("strand_pair length does not match num_targets")
            target_type_str = [get_target_type(line.split("\t")[description_index]) for line in targets]
            self.target_type_name = list(set(target_type_str))
            self.target_type = [self.target_type_name.index(t) for t in target_type_str]
            self.orientation_type = [get_orientation_type(line.split("\t")[id_index]) for line in targets]

        self.make_dataset()

    def generate_parser(self, raw=False):
        def parse_proto(example_protos):
            features = {
                TFR_INPUT: tf.io.FixedLenFeature([], tf.string),
                TFR_OUTPUT: tf.io.FixedLenFeature([], tf.string),
            }

            parsed_features = tf.io.parse_single_example(example_protos, features=features)

            sequence = tf.io.decode_raw(parsed_features[TFR_INPUT], tf.uint8)
            if not raw:
                if self.seq_1hot:
                    sequence = tf.reshape(sequence, [self.seq_length])
                    sequence = tf.one_hot(sequence, 1 + self.seq_depth, dtype=tf.uint8)
                    sequence = sequence[:, :-1]  # drop N
                else:
                    sequence = tf.reshape(sequence, [self.seq_length, self.seq_depth])
                if self.seq_length_crop is not None:
                    crop_len = (self.seq_length - self.seq_length_crop) // 2
                    if crop_len > 0:
                        sequence = sequence[crop_len:-crop_len, :]
                sequence = tf.cast(sequence, tf.float32)

            targets = tf.io.decode_raw(parsed_features[TFR_OUTPUT], tf.float16)
            if not raw:
                targets = tf.reshape(targets, [self.target_length, self.num_targets])
                targets = tf.cast(targets, tf.float32)

            return sequence, targets

        return parse_proto

    def make_dataset(self, cycle_length=4):
        tfr_files = natsorted(glob.glob(self.tfr_path))
        if tfr_files:
            dataset = tf.data.Dataset.from_tensor_slices(tfr_files)
        else:
            print(f"Cannot order TFRecords {self.tfr_path}", file=sys.stderr)
            dataset = tf.data.Dataset.list_files(self.tfr_path)

        if self.mode == "train":
            dataset = dataset.repeat()
            dataset = dataset.interleave(
                map_func=file_to_records,
                cycle_length=cycle_length,
                num_parallel_calls=tf.data.experimental.AUTOTUNE,
            )
            if self.shuffle_buffer > 0:
                dataset = dataset.shuffle(buffer_size=self.shuffle_buffer, reshuffle_each_iteration=True)
        else:
            dataset = dataset.flat_map(file_to_records)

        dataset = dataset.map(self.generate_parser())
        dataset = dataset.batch(self.batch_size, drop_remainder=self.drop_remainder)
        if self.prefetch < 0:
            dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
        elif self.prefetch > 0:
            dataset = dataset.prefetch(self.prefetch)
        self.dataset = dataset

    @property
    def effective_seq_length(self):
        return self.seq_length_crop if self.seq_length_crop is not None else self.seq_length
