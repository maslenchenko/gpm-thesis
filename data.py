"""Shim re-exporting data utilities from gpm.data_utils.

This enables imports like:
    from gpm.data import SeqDataset

Keep __all__ in sync with gpm/data_utils/__init__.py.
"""

from .data_utils import (
    SeqDataset,
    EcoliIsolateDataset,
    EcoliRecord,
    load_ecoli_records,
    parse_gff_fasta_contigs,
    ecoli_outer_iter,
    ecoli_count_batches,
    round_robin_iter,
    real_data_iter,
    fake_data_iter,
    batch_limiter,
    count_batches,
)

__all__ = [
    "SeqDataset",
    "EcoliIsolateDataset",
    "EcoliRecord",
    "load_ecoli_records",
    "parse_gff_fasta_contigs",
    "ecoli_outer_iter",
    "ecoli_count_batches",
    "round_robin_iter",
    "real_data_iter",
    "fake_data_iter",
    "batch_limiter",
    "count_batches",
]
