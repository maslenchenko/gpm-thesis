try:
    from .dataset import SeqDataset
except Exception as _seqdataset_import_exc:  # pragma: no cover - optional for ecoli-only usage
    class SeqDataset:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "SeqDataset requires optional borzoi dependencies. "
                f"Original import error: {_seqdataset_import_exc}"
            )

from .ecoli_dataset import (
    EcoliIsolateDataset,
    EcoliRecord,
    load_ecoli_records,
    parse_gff_fasta_contigs,
    ecoli_outer_iter,
    ecoli_count_batches,
)
from .iterators import round_robin_iter, real_data_iter, fake_data_iter, batch_limiter, count_batches

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
