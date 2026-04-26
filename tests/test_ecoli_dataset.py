import csv
import os
import tempfile
import unittest

import numpy as np

from gpm.data_utils.ecoli_dataset import (
    EcoliIsolateDataset,
    EcoliRecord,
    ecoli_outer_iter,
    load_ecoli_records,
    parse_gff_fasta_contigs,
    stratified_split_records,
)


def _write_gff_with_fasta(path, contigs):
    with open(path, "w", encoding="utf-8") as f:
        f.write("##gff-version 3\n")
        f.write("##FASTA\n")
        for idx, seq in enumerate(contigs, start=1):
            f.write(f">contig{idx}\n")
            f.write(f"{seq}\n")


class EcoliDatasetTests(unittest.TestCase):
    def test_parse_gff_fasta_contigs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "iso.velvet.gff")
            _write_gff_with_fasta(path, ["ACGT", "TTAA", "GGCC"])
            contigs = parse_gff_fasta_contigs(path)
        self.assertEqual(contigs, ["ACGT", "TTAA", "GGCC"])

    def test_load_records_filters_missing_labels_and_missing_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            contig_dir = os.path.join(tmpdir, "contigs")
            os.makedirs(contig_dir, exist_ok=True)

            _write_gff_with_fasta(os.path.join(contig_dir, "iso1.velvet.gff"), ["AAAA"])
            _write_gff_with_fasta(os.path.join(contig_dir, "iso2.velvet.gff"), ["CCCC"])

            metadata = os.path.join(tmpdir, "Metadata.csv")
            with open(metadata, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["Isolate", "Year", "CIP"])
                writer.writeheader()
                writer.writerow({"Isolate": "iso1", "Year": "2010", "CIP": "S"})
                writer.writerow({"Isolate": "iso2", "Year": "2011", "CIP": "R"})
                writer.writerow({"Isolate": "iso3", "Year": "2012", "CIP": ""})  # missing label
                writer.writerow({"Isolate": "iso4", "Year": "2013", "CIP": "S"})  # missing file

            records = load_ecoli_records(metadata, contig_dir, antibiotic="CIP")

        self.assertEqual(len(records), 2)
        labels = sorted(r.label for r in records)
        self.assertEqual(labels, [0, 1])

    def test_stratified_split_preserves_label_balance(self):
        records = []
        for i in range(10):
            records.append(EcoliRecord(isolate=f"s{i}", contig_path=f"/tmp/s{i}.gff", label=0, year=2010))
            records.append(EcoliRecord(isolate=f"r{i}", contig_path=f"/tmp/r{i}.gff", label=1, year=2010))

        train = stratified_split_records(records, split="train", train_fraction=0.6, valid_fraction=0.2, seed=7)
        valid = stratified_split_records(records, split="valid", train_fraction=0.6, valid_fraction=0.2, seed=7)
        test = stratified_split_records(records, split="test", train_fraction=0.6, valid_fraction=0.2, seed=7)

        self.assertEqual(len(train), 12)
        self.assertEqual(len(valid), 4)
        self.assertEqual(len(test), 4)
        self.assertEqual(sum(r.label == 0 for r in train), 6)
        self.assertEqual(sum(r.label == 1 for r in train), 6)

    def test_concat_mode_is_deterministic(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            gff_path = os.path.join(tmpdir, "iso.velvet.gff")
            _write_gff_with_fasta(gff_path, ["AAAA", "CCCC", "GGGG"])
            dataset = EcoliIsolateDataset(
                records=[EcoliRecord(isolate="iso", contig_path=gff_path, label=1, year=2010)],
                contig_mode="concat",
                separator_length=1,
            )
            x1, y1 = dataset[0]
            x2, y2 = dataset[0]

        self.assertTrue(np.array_equal(x1, x2))
        self.assertTrue(np.array_equal(y1, y2))
        self.assertEqual(y1.shape, (1, 1))
        self.assertEqual(x1.shape[1], 4)

    def test_shuffle_mode_dynamic_changes_order_over_time(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            gff_path = os.path.join(tmpdir, "iso.velvet.gff")
            _write_gff_with_fasta(gff_path, ["AAAA", "CCCC", "GGGG"])
            dataset = EcoliIsolateDataset(
                records=[EcoliRecord(isolate="iso", contig_path=gff_path, label=1, year=2010)],
                contig_mode="shuffle",
                dynamic_shuffle=True,
                shuffle_seed=123,
                separator_length=1,
            )

            first_base_indices = set()
            for _ in range(12):
                x, _y = dataset[0]
                first_base_indices.add(int(np.argmax(x[0])))

        self.assertGreater(len(first_base_indices), 1)

    def test_return_metadata_includes_expected_fields(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            gff_path = os.path.join(tmpdir, "iso.velvet.gff")
            _write_gff_with_fasta(gff_path, ["AAAA", "CCCC"])
            dataset = EcoliIsolateDataset(
                records=[EcoliRecord(isolate="iso", contig_path=gff_path, label=0, year=2015)],
                contig_mode="concat",
                separator_length=2,
                return_metadata=True,
            )
            x, y, meta = dataset[0]

        self.assertEqual(x.shape[1], 4)
        self.assertEqual(y.shape, (1, 1))
        self.assertEqual(meta["isolate"], "iso")
        self.assertEqual(meta["year"], 2015)
        self.assertIn("num_contigs", meta)
        self.assertIn("length", meta)

    def test_records_and_paths_are_mutually_exclusive(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            gff_path = os.path.join(tmpdir, "iso.velvet.gff")
            _write_gff_with_fasta(gff_path, ["AAAA"])
            rec = EcoliRecord(isolate="iso", contig_path=gff_path, label=0, year=None)
            with self.assertRaises(ValueError):
                EcoliIsolateDataset(
                    records=[rec],
                    metadata_csv=os.path.join(tmpdir, "Metadata.csv"),
                    contigs_dir=tmpdir,
                )

    def test_ecoli_outer_iter_returns_batched_numpy(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            gff_path = os.path.join(tmpdir, "iso.velvet.gff")
            _write_gff_with_fasta(gff_path, ["AAAA", "CCCC"])
            dataset = EcoliIsolateDataset(
                records=[EcoliRecord(isolate="iso", contig_path=gff_path, label=1, year=2010)],
                contig_mode="concat",
                separator_length=1,
            )
            outer = ecoli_outer_iter(dataset, batch_size=1, shuffle=False, seed=0)
            x_batch, y_batch = next(next(outer))

        self.assertEqual(x_batch.ndim, 3)
        self.assertEqual(y_batch.ndim, 3)
        self.assertEqual(x_batch.shape[0], 1)
        self.assertEqual(y_batch.shape, (1, 1, 1))

    def test_ecoli_outer_iter_pads_variable_length_batch(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            gff_path1 = os.path.join(tmpdir, "iso1.velvet.gff")
            gff_path2 = os.path.join(tmpdir, "iso2.velvet.gff")
            _write_gff_with_fasta(gff_path1, ["A" * 12])
            _write_gff_with_fasta(gff_path2, ["C" * 20])
            dataset = EcoliIsolateDataset(
                records=[
                    EcoliRecord(isolate="iso1", contig_path=gff_path1, label=0, year=2010),
                    EcoliRecord(isolate="iso2", contig_path=gff_path2, label=1, year=2010),
                ],
                contig_mode="concat",
                separator_length=0,
                pad_to_multiple=None,
            )
            outer = ecoli_outer_iter(dataset, batch_size=2, shuffle=False, seed=0)
            x_batch, y_batch = next(next(outer))

        self.assertEqual(x_batch.shape, (2, 20, 4))
        self.assertEqual(y_batch.shape, (2, 1, 1))
        self.assertTrue(np.allclose(x_batch[0, 12:, :], 0.0))

    def test_ecoli_outer_iter_length_bucketing_groups_similar_lengths(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            gff_paths = []
            lengths = [10, 30, 11, 31]
            for i, n in enumerate(lengths, start=1):
                path = os.path.join(tmpdir, f"iso{i}.velvet.gff")
                _write_gff_with_fasta(path, ["A" * n])
                gff_paths.append(path)
            records = [
                EcoliRecord(isolate=f"iso{i+1}", contig_path=path, label=i % 2, year=2010)
                for i, path in enumerate(gff_paths)
            ]
            dataset = EcoliIsolateDataset(
                records=records,
                contig_mode="concat",
                separator_length=0,
                pad_to_multiple=None,
            )
            outer = ecoli_outer_iter(
                dataset,
                batch_size=2,
                shuffle=False,
                seed=0,
                bucket_by_length=True,
                bucket_size=4,
            )
            epoch = next(outer)
            first_x, _first_y = next(epoch)
            second_x, _second_y = next(epoch)

        first_lengths = sorted(int((sample.sum(axis=1) > 0).sum()) for sample in first_x)
        second_lengths = sorted(int((sample.sum(axis=1) > 0).sum()) for sample in second_x)
        self.assertEqual(first_lengths, [10, 11])
        self.assertEqual(second_lengths, [30, 31])


if __name__ == "__main__":
    unittest.main()
