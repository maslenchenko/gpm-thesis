import csv
import math
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Sequence, Tuple

import numpy as np
from torch.utils.data import Dataset


EcoliSplit = Literal["train", "valid", "test"]
ContigMode = Literal["concat", "shuffle"]

_LABEL_MAP: Dict[str, int] = {"S": 0, "R": 1}

_ONE_HOT_LOOKUP = np.zeros((256, 4), dtype=np.float32)
_ONE_HOT_LOOKUP[ord("A"), 0] = 1.0
_ONE_HOT_LOOKUP[ord("C"), 1] = 1.0
_ONE_HOT_LOOKUP[ord("G"), 2] = 1.0
_ONE_HOT_LOOKUP[ord("T"), 3] = 1.0
_ONE_HOT_LOOKUP[ord("a"), 0] = 1.0
_ONE_HOT_LOOKUP[ord("c"), 1] = 1.0
_ONE_HOT_LOOKUP[ord("g"), 2] = 1.0
_ONE_HOT_LOOKUP[ord("t"), 3] = 1.0


@dataclass(frozen=True)
class EcoliRecord:
    isolate: str
    contig_path: str
    label: int
    year: Optional[int]


def parse_gff_fasta_contigs(gff_path: str) -> List[str]:
    """Extract contig sequences from the ##FASTA section of a GFF file."""
    in_fasta = False
    contigs: List[str] = []
    current_seq: List[str] = []

    with open(gff_path, "r", encoding="utf-8", errors="ignore") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not in_fasta:
                if line == "##FASTA":
                    in_fasta = True
                continue

            if not line:
                continue
            if line.startswith(">"):
                if current_seq:
                    contigs.append("".join(current_seq))
                    current_seq = []
                continue
            current_seq.append(line.upper())

    if current_seq:
        contigs.append("".join(current_seq))

    if not in_fasta:
        raise ValueError(f"GFF file has no ##FASTA section: {gff_path}")
    if not contigs:
        raise ValueError(f"GFF FASTA section contains no contigs: {gff_path}")

    return contigs


def parse_gff_fasta_contig_lengths(gff_path: str) -> List[int]:
    """Extract contig lengths from the ##FASTA section of a GFF file."""
    in_fasta = False
    contig_lengths: List[int] = []
    current_len = 0

    with open(gff_path, "r", encoding="utf-8", errors="ignore") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not in_fasta:
                if line == "##FASTA":
                    in_fasta = True
                continue

            if not line:
                continue
            if line.startswith(">"):
                if current_len > 0:
                    contig_lengths.append(current_len)
                    current_len = 0
                continue
            current_len += len(line)

    if current_len > 0:
        contig_lengths.append(current_len)

    if not in_fasta:
        raise ValueError(f"GFF file has no ##FASTA section: {gff_path}")
    if not contig_lengths:
        raise ValueError(f"GFF FASTA section contains no contigs: {gff_path}")

    return contig_lengths


def dna_to_one_hot(sequence: str) -> np.ndarray:
    if not sequence:
        return np.zeros((0, 4), dtype=np.float32)
    encoded = np.frombuffer(sequence.encode("ascii", errors="replace"), dtype=np.uint8)
    return _ONE_HOT_LOOKUP[encoded]


def compose_pseudogenome(
    contigs: Sequence[str],
    mode: ContigMode,
    separator_length: int = 50,
    rng: Optional[random.Random] = None,
) -> Tuple[str, List[str]]:
    if mode not in ("concat", "shuffle"):
        raise ValueError(f"Unknown contig composition mode: {mode}")
    ordered = list(contigs)
    if mode == "shuffle":
        if rng is None:
            rng = random.Random()
        rng.shuffle(ordered)
    separator = "N" * max(int(separator_length), 0)
    return separator.join(ordered), ordered


def load_ecoli_records(
    metadata_csv: str,
    contigs_dir: str,
    antibiotic: str = "CIP",
    label_map: Optional[Dict[str, int]] = None,
) -> List[EcoliRecord]:
    if label_map is None:
        label_map = _LABEL_MAP
    antibiotic = str(antibiotic)

    records: List[EcoliRecord] = []
    with open(metadata_csv, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if "Isolate" not in (reader.fieldnames or []):
            raise ValueError("Metadata.csv must contain 'Isolate' column.")
        if antibiotic not in (reader.fieldnames or []):
            raise ValueError(f"Metadata.csv must contain '{antibiotic}' column.")

        for row in reader:
            isolate = (row.get("Isolate") or "").strip()
            if not isolate:
                continue

            label_raw = (row.get(antibiotic) or "").strip().upper()
            if label_raw not in label_map:
                continue

            gff_path = os.path.join(contigs_dir, f"{isolate}.velvet.gff")
            if not os.path.isfile(gff_path):
                continue

            year_raw = (row.get("Year") or "").strip()
            year = int(year_raw) if year_raw.isdigit() else None
            records.append(
                EcoliRecord(
                    isolate=isolate,
                    contig_path=gff_path,
                    label=int(label_map[label_raw]),
                    year=year,
                )
            )
    return records


def _split_counts(n: int, train_fraction: float, valid_fraction: float) -> Tuple[int, int, int]:
    n_train = int(n * train_fraction)
    n_valid = int(n * valid_fraction)
    if n_train + n_valid > n:
        n_valid = max(0, n - n_train)
    n_test = n - n_train - n_valid
    return n_train, n_valid, n_test


def stratified_split_records(
    records: Sequence[EcoliRecord],
    split: EcoliSplit,
    train_fraction: float = 0.8,
    valid_fraction: float = 0.1,
    seed: int = 42,
) -> List[EcoliRecord]:
    if split not in ("train", "valid", "test"):
        raise ValueError(f"Unknown split: {split}")
    if not (0.0 < train_fraction < 1.0):
        raise ValueError(f"train_fraction must be in (0,1), got {train_fraction}")
    if not (0.0 <= valid_fraction < 1.0):
        raise ValueError(f"valid_fraction must be in [0,1), got {valid_fraction}")
    if train_fraction + valid_fraction >= 1.0:
        raise ValueError("train_fraction + valid_fraction must be < 1.0")

    by_label: Dict[int, List[EcoliRecord]] = {}
    for rec in records:
        by_label.setdefault(rec.label, []).append(rec)

    rng = random.Random(seed)
    out: List[EcoliRecord] = []
    for label_records in by_label.values():
        label_records = list(label_records)
        rng.shuffle(label_records)
        n_train, n_valid, _n_test = _split_counts(len(label_records), train_fraction, valid_fraction)
        if split == "train":
            chosen = label_records[:n_train]
        elif split == "valid":
            chosen = label_records[n_train : n_train + n_valid]
        else:
            chosen = label_records[n_train + n_valid :]
        out.extend(chosen)

    rng.shuffle(out)
    return out


class EcoliIsolateDataset(Dataset):
    """PyTorch-style dataset for E. coli isolate-level CIP classification."""

    def __init__(
        self,
        metadata_csv: Optional[str] = None,
        contigs_dir: Optional[str] = None,
        split: EcoliSplit = "train",
        antibiotic: str = "CIP",
        train_fraction: float = 0.8,
        valid_fraction: float = 0.1,
        split_seed: int = 42,
        contig_mode: ContigMode = "concat",
        dynamic_shuffle: bool = False,
        shuffle_seed: int = 0,
        separator_length: int = 50,
        max_genome_length: Optional[int] = None,
        pad_to_multiple: Optional[int] = None,
        return_metadata: bool = False,
        cache_contigs: bool = False,
        records: Optional[Sequence[EcoliRecord]] = None,
    ):
        if contig_mode not in ("concat", "shuffle"):
            raise ValueError(f"Unknown contig_mode: {contig_mode}")
        if separator_length < 0:
            raise ValueError(f"separator_length must be >= 0, got {separator_length}")
        if max_genome_length is not None and max_genome_length <= 0:
            raise ValueError(f"max_genome_length must be > 0, got {max_genome_length}")
        if pad_to_multiple is not None and pad_to_multiple <= 0:
            raise ValueError(f"pad_to_multiple must be > 0, got {pad_to_multiple}")

        self.split = split
        self.antibiotic = antibiotic
        self.contig_mode = contig_mode
        self.dynamic_shuffle = bool(dynamic_shuffle)
        self.shuffle_seed = int(shuffle_seed)
        self.separator_length = int(separator_length)
        self.max_genome_length = max_genome_length
        self.pad_to_multiple = pad_to_multiple
        self.return_metadata = bool(return_metadata)
        self._epoch = 0
        self._dynamic_draw_counts: Dict[int, int] = {}
        self._contig_cache = {} if cache_contigs else None
        self._contig_lengths_cache: Dict[str, Tuple[int, ...]] = {}
        self._sequence_length_cache: Dict[int, int] = {}

        if records is None:
            if metadata_csv is None or contigs_dir is None:
                raise ValueError("Provide either `records` or both (`metadata_csv`, `contigs_dir`).")
            all_records = load_ecoli_records(
                metadata_csv=metadata_csv,
                contigs_dir=contigs_dir,
                antibiotic=antibiotic,
            )
            self.records = stratified_split_records(
                records=all_records,
                split=split,
                train_fraction=train_fraction,
                valid_fraction=valid_fraction,
                seed=split_seed,
            )
        else:
            if metadata_csv is not None or contigs_dir is not None:
                raise ValueError("Cannot provide `records` together with `metadata_csv`/`contigs_dir`.")
            self.records = list(records)

    def __len__(self) -> int:
        return len(self.records)

    def set_epoch(self, epoch: int):
        """Set epoch for deterministic dynamic shuffle scheduling."""
        self._epoch = int(epoch)
        self._dynamic_draw_counts.clear()

    def _load_contigs(self, path: str) -> List[str]:
        if self._contig_cache is not None and path in self._contig_cache:
            return list(self._contig_cache[path])
        contigs = parse_gff_fasta_contigs(path)
        if self._contig_cache is not None:
            self._contig_cache[path] = tuple(contigs)
        return contigs

    def _load_contig_lengths(self, path: str) -> List[int]:
        cached = self._contig_lengths_cache.get(path)
        if cached is not None:
            return list(cached)
        lengths = parse_gff_fasta_contig_lengths(path)
        self._contig_lengths_cache[path] = tuple(int(x) for x in lengths)
        return lengths

    def _compose_sequence(self, index: int, contigs: Sequence[str]) -> Tuple[str, List[str]]:
        if self.contig_mode == "concat":
            return compose_pseudogenome(contigs, mode="concat", separator_length=self.separator_length)

        if self.dynamic_shuffle:
            draw_count = self._dynamic_draw_counts.get(index, 0)
            self._dynamic_draw_counts[index] = draw_count + 1
            seed = (
                self.shuffle_seed
                + self._epoch * 1_000_003
                + index * 10_007
                + draw_count
            )
            rng = random.Random(seed)
        else:
            rng = random.Random(self.shuffle_seed + index)
        return compose_pseudogenome(
            contigs,
            mode="shuffle",
            separator_length=self.separator_length,
            rng=rng,
        )

    def get_sequence_length(self, index: int) -> int:
        """Return final sequence length after truncation/padding without one-hot allocation."""
        cached = self._sequence_length_cache.get(index)
        if cached is not None:
            return int(cached)

        record = self.records[index]
        contig_lengths = self._load_contig_lengths(record.contig_path)
        total_length = int(sum(contig_lengths))
        if len(contig_lengths) > 1 and self.separator_length > 0:
            total_length += int(self.separator_length) * (len(contig_lengths) - 1)
        if self.max_genome_length is not None and total_length > self.max_genome_length:
            total_length = int(self.max_genome_length)
        if self.pad_to_multiple is not None and total_length % self.pad_to_multiple != 0:
            total_length += self.pad_to_multiple - (total_length % self.pad_to_multiple)

        self._sequence_length_cache[index] = int(total_length)
        return int(total_length)

    def __getitem__(self, index: int):
        record = self.records[index]
        contigs = self._load_contigs(record.contig_path)
        sequence, order = self._compose_sequence(index, contigs)

        if self.max_genome_length is not None and len(sequence) > self.max_genome_length:
            sequence = sequence[: self.max_genome_length]
        if self.pad_to_multiple is not None and len(sequence) % self.pad_to_multiple != 0:
            pad_len = self.pad_to_multiple - (len(sequence) % self.pad_to_multiple)
            sequence = sequence + ("N" * pad_len)

        x = dna_to_one_hot(sequence)
        y = np.array([[float(record.label)]], dtype=np.float32)

        if self.return_metadata:
            return x, y, {"isolate": record.isolate, "year": record.year, "num_contigs": len(order), "length": len(sequence)}
        return x, y


def ecoli_outer_iter(
    dataset: EcoliIsolateDataset,
    batch_size: int = 1,
    shuffle: bool = False,
    drop_remainder: bool = False,
    seed: int = 0,
    bucket_by_length: bool = False,
    bucket_size: int = 64,
    pad_to_multiple: Optional[int] = None,
):
    """Yield epoch iterators of numpy batches to match gpm training-loop iterator contract."""
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")

    if bucket_size < 1:
        raise ValueError(f"bucket_size must be >= 1, got {bucket_size}")
    if pad_to_multiple is not None and pad_to_multiple < 1:
        raise ValueError(f"pad_to_multiple must be >= 1 if set, got {pad_to_multiple}")

    def _sample_to_xy(sample):
        if not isinstance(sample, tuple):
            raise ValueError("Dataset sample must be a tuple.")
        if len(sample) < 2:
            raise ValueError("Dataset sample tuple must contain at least (x, y).")
        return sample[0], sample[1]

    def _pad_x_batch(xs: List[np.ndarray]) -> np.ndarray:
        if not xs:
            return np.zeros((0, 0, 0), dtype=np.float32)
        max_len = max(int(x.shape[0]) for x in xs)
        if pad_to_multiple is not None and max_len % pad_to_multiple != 0:
            max_len += pad_to_multiple - (max_len % pad_to_multiple)
        channels = int(xs[0].shape[1])
        out = np.zeros((len(xs), max_len, channels), dtype=np.float32)
        for i, x in enumerate(xs):
            out[i, : x.shape[0], :] = x
        return out

    def _sequence_length_for_index(index: int) -> int:
        if hasattr(dataset, "get_sequence_length"):
            return int(dataset.get_sequence_length(int(index)))
        x, _ = _sample_to_xy(dataset[int(index)])
        return int(x.shape[0])

    epoch = 0
    while True:
        if hasattr(dataset, "set_epoch"):
            dataset.set_epoch(epoch)

        indices = np.arange(len(dataset))
        rng = np.random.default_rng(seed + epoch)
        if shuffle:
            rng.shuffle(indices)

        if bucket_by_length and batch_size > 1 and len(indices) > 0:
            bucket_span = max(int(bucket_size), int(batch_size))
            chunked_batches = []
            for start in range(0, len(indices), bucket_span):
                chunk = [int(x) for x in indices[start : start + bucket_span]]
                chunk.sort(key=_sequence_length_for_index)
                batches = [chunk[i : i + batch_size] for i in range(0, len(chunk), batch_size)]
                if shuffle and len(batches) > 1:
                    rng.shuffle(batches)
                chunked_batches.extend(batches)
            indices = np.array([idx for batch in chunked_batches for idx in batch], dtype=np.int64)

        def epoch_iter():
            for start in range(0, len(indices), batch_size):
                batch_indices = indices[start : start + batch_size]
                if len(batch_indices) < batch_size and drop_remainder:
                    continue
                xs = []
                ys = []
                for idx in batch_indices:
                    x, y = _sample_to_xy(dataset[int(idx)])
                    xs.append(x)
                    ys.append(y)
                yield (
                    _pad_x_batch(xs),
                    np.stack(ys, axis=0).astype(np.float32, copy=False),
                )

        yield epoch_iter()
        epoch += 1


def ecoli_count_batches(dataset: EcoliIsolateDataset, batch_size: int, limit=None, first=None):
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    n = math.ceil(len(dataset) / batch_size)
    if first is not None:
        n = max(n - first, 0)
    if limit is not None:
        n = min(n, limit)
    return n
