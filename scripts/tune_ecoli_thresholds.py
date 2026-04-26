import argparse
import json
import logging
import os
import pickle
import sys
from typing import Dict, Iterable, Tuple

import numpy as np
import torch

from gpm.data import EcoliIsolateDataset, ecoli_count_batches, ecoli_outer_iter
from gpm.models import models
from gpm.utils.input_interface import inject_input_interface_model_args
from gpm.utils.metrics import binary_classification_metrics_from_logits

THRESHOLD_OBJECTIVES = ("accuracy", "balanced_accuracy", "precision", "recall", "f1")


def bool_arg(val: str):
    return str(val).lower() in ("1", "true", "yes", "y")


def _normalize_metric_value(value):
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value)
    return value


def _extract_model_state_dict(payload) -> Tuple[Dict[str, torch.Tensor], Dict[str, int]]:
    if isinstance(payload, dict) and "model" in payload and isinstance(payload["model"], dict):
        metadata = {
            "epoch": int(payload.get("epoch", 0) or 0),
            "step": int(payload.get("step", 0) or 0),
        }
        return payload["model"], metadata
    if isinstance(payload, dict):
        return payload, {}
    raise ValueError("Checkpoint payload must be a dict or a dict containing a `model` state_dict.")


def build_threshold_grid(threshold_min: float, threshold_max: float, threshold_points: int) -> np.ndarray:
    if not (0.0 <= float(threshold_min) <= 1.0):
        raise ValueError(f"threshold_min must be in [0, 1], got {threshold_min}")
    if not (0.0 <= float(threshold_max) <= 1.0):
        raise ValueError(f"threshold_max must be in [0, 1], got {threshold_max}")
    if float(threshold_min) >= float(threshold_max):
        raise ValueError(
            f"threshold_min must be < threshold_max, got min={threshold_min}, max={threshold_max}"
        )
    if int(threshold_points) < 2:
        raise ValueError(f"threshold_points must be >= 2, got {threshold_points}")
    return np.linspace(float(threshold_min), float(threshold_max), int(threshold_points), dtype=np.float64)


def _is_better_candidate(
    candidate: Dict[str, float],
    best: Dict[str, float] | None,
    objective: str,
    baseline_threshold: float,
) -> bool:
    if best is None:
        return True
    eps = 1e-12
    c_score = float(candidate[objective])
    b_score = float(best[objective])
    if c_score > b_score + eps:
        return True
    if c_score < b_score - eps:
        return False

    c_bal = float(candidate["balanced_accuracy"])
    b_bal = float(best["balanced_accuracy"])
    if c_bal > b_bal + eps:
        return True
    if c_bal < b_bal - eps:
        return False

    c_dist = abs(float(candidate["threshold"]) - float(baseline_threshold))
    b_dist = abs(float(best["threshold"]) - float(baseline_threshold))
    if c_dist < b_dist - eps:
        return True
    if c_dist > b_dist + eps:
        return False

    return float(candidate["threshold"]) < float(best["threshold"])


def find_best_threshold(
    logits: torch.Tensor,
    targets: torch.Tensor,
    thresholds: Iterable[float],
    objective: str = "f1",
    baseline_threshold: float = 0.5,
) -> Dict[str, object]:
    if objective not in THRESHOLD_OBJECTIVES:
        raise ValueError(f"Unknown objective '{objective}'. Available: {', '.join(THRESHOLD_OBJECTIVES)}")

    best = None
    for th in thresholds:
        metrics = binary_classification_metrics_from_logits(logits, targets, threshold=float(th))
        candidate = {
            "threshold": float(th),
            "objective": objective,
            "objective_value": float(metrics[objective]),
            **{k: _normalize_metric_value(v) for k, v in metrics.items()},
        }
        if _is_better_candidate(candidate, best, objective=objective, baseline_threshold=baseline_threshold):
            best = candidate
    if best is None:
        raise ValueError("Empty threshold grid.")
    return best


def collect_eval_logits_targets(
    model: torch.nn.Module,
    iterator,
    n_batches: int,
    device: str,
    amp_enabled: bool = False,
    amp_dtype: str = "float16",
) -> Tuple[torch.Tensor, torch.Tensor]:
    device_type = torch.device(device).type
    amp_enabled = amp_enabled and device_type == "cuda"
    dtype = torch.float16 if amp_dtype == "float16" else torch.bfloat16
    autocast_ctx = torch.autocast(device_type=device_type, dtype=dtype, enabled=amp_enabled)

    all_logits = []
    all_targets = []
    model.eval()
    processed = 0
    with torch.no_grad():
        epoch_iter = next(iterator)
        for i, (x, y_true) in enumerate(epoch_iter):
            if i >= n_batches:
                break
            x_t = torch.tensor(x, device=device)
            y_t = torch.tensor(y_true, device=device)
            with autocast_ctx:
                y_pred = model(x_t)
            if y_pred.shape != y_t.shape:
                raise ValueError(f"predicted shape {y_pred.shape} != required shape {y_t.shape}")
            all_logits.append(y_pred.detach().float().cpu())
            all_targets.append(y_t.detach().float().cpu())
            processed += 1
            if (i + 1) % 25 == 0 or (i + 1) == n_batches:
                logging.warning(f"Processed evaluation batch {i+1}/{n_batches}")
    if processed != n_batches:
        logging.warning(f"Evaluation iterator produced {processed} batches, expected {n_batches}.")

    if all_logits:
        return torch.cat(all_logits, dim=0), torch.cat(all_targets, dim=0)
    return torch.empty(0, dtype=torch.float32), torch.empty(0, dtype=torch.float32)


def _classification_metrics_payload(logits: torch.Tensor, targets: torch.Tensor, threshold: float) -> Dict[str, object]:
    metrics = binary_classification_metrics_from_logits(logits, targets, threshold=float(threshold))
    metrics = {k: _normalize_metric_value(v) for k, v in metrics.items()}
    targets_for_bce = (targets.reshape(-1).float() >= 0.5).float()
    metrics["bce_loss"] = float(
        torch.nn.functional.binary_cross_entropy_with_logits(
            logits.reshape(-1).float(),
            targets_for_bce,
        ).item()
    )
    return metrics


def build_report(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold_grid: np.ndarray,
    selected_objective: str,
    baseline_threshold: float = 0.5,
    test_logits: torch.Tensor | None = None,
    test_targets: torch.Tensor | None = None,
) -> Dict[str, object]:
    if (test_logits is None) != (test_targets is None):
        raise ValueError("test_logits and test_targets must be provided together.")

    baseline_metrics = _classification_metrics_payload(logits, targets, threshold=baseline_threshold)

    best_by_objective = {}
    for objective in THRESHOLD_OBJECTIVES:
        best_by_objective[objective] = find_best_threshold(
            logits=logits,
            targets=targets,
            thresholds=threshold_grid,
            objective=objective,
            baseline_threshold=baseline_threshold,
        )

    selected_best = best_by_objective[selected_objective]
    best_metrics = {
        key: selected_best[key]
        for key in (
            "accuracy",
            "balanced_accuracy",
            "precision",
            "recall",
            "f1",
            "auroc",
            "auprc",
            "n_samples",
            "n_positive",
            "n_negative",
        )
    }
    best_metrics["bce_loss"] = baseline_metrics["bce_loss"]

    report = {
        "baseline_threshold": float(baseline_threshold),
        "baseline_metrics": baseline_metrics,
        "selected_objective": selected_objective,
        "best_threshold": float(selected_best["threshold"]),
        "best_objective_value": float(selected_best["objective_value"]),
        "best_metrics": best_metrics,
        "best_by_objective": best_by_objective,
        "threshold_grid": {
            "min": float(threshold_grid[0]),
            "max": float(threshold_grid[-1]),
            "points": int(len(threshold_grid)),
        },
        "n_validation_samples": int(baseline_metrics["n_samples"]),
    }
    if test_logits is not None and test_targets is not None:
        selected_threshold = float(selected_best["threshold"])
        report["test"] = {
            "baseline_threshold": float(baseline_threshold),
            "selected_threshold": selected_threshold,
            "baseline_metrics": _classification_metrics_payload(
                test_logits,
                test_targets,
                threshold=baseline_threshold,
            ),
            "selected_threshold_metrics": _classification_metrics_payload(
                test_logits,
                test_targets,
                threshold=selected_threshold,
            ),
        }
    return report


def parse_args():
    parser = argparse.ArgumentParser(description="Tune decision threshold for E. coli classification checkpoints.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint file (.pt or .pt.best).")
    parser.add_argument("--output_json", type=str, default=None, help="Optional path to write JSON report.")
    parser.add_argument("--model_name", type=str, default="stripedmamba_isolate")
    parser.add_argument("--model_args", type=str, default="{}")
    parser.add_argument("--use_input_interface", action="store_true")
    parser.add_argument(
        "--input_interface_preset",
        type=str,
        choices=["none", "borzoi", "ecoli"],
        default="none",
    )
    parser.add_argument("--input_interface_args", type=str, default="{}")
    parser.add_argument("--device", type=str, choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--amp_dtype", type=str, choices=["float16", "bfloat16"], default="float16")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument(
        "--ecoli_metadata_csv",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "../../ecoli-data/Metadata.csv"),
    )
    parser.add_argument(
        "--ecoli_contigs_dir",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "../../ecoli-data/contigs-ecoli"),
    )
    parser.add_argument("--ecoli_antibiotic", type=str, default="CIP")
    parser.add_argument("--ecoli_train_fraction", type=float, default=0.8)
    parser.add_argument("--ecoli_valid_fraction", type=float, default=0.1)
    parser.add_argument("--ecoli_split_seed", type=int, default=42)
    parser.add_argument("--ecoli_separator_length", type=int, default=50)
    parser.add_argument("--ecoli_max_genome_length", type=int, default=None)
    parser.add_argument("--ecoli_pad_to_multiple", type=int, default=128)
    parser.add_argument("--ecoli_cache_contigs", action="store_true")
    parser.add_argument("--ecoli_length_bucketing", type=str, default="true")
    parser.add_argument("--ecoli_bucket_size", type=int, default=64)
    parser.add_argument(
        "--ecoli_contig_mode",
        type=str,
        choices=["concat", "shuffle"],
        default="concat",
        help="How to combine contigs: 'concat' (separator-joined) or 'shuffle'.",
    )
    parser.add_argument(
        "--ecoli_dynamic_shuffle",
        type=str,
        default="false",
        help="Enable dynamic per-draw contig shuffling when contig_mode=shuffle.",
    )
    parser.add_argument("--baseline_threshold", type=float, default=0.5)
    parser.add_argument("--threshold_min", type=float, default=0.0)
    parser.add_argument("--threshold_max", type=float, default=1.0)
    parser.add_argument("--threshold_points", type=int, default=1001)
    parser.add_argument("--objective", type=str, choices=THRESHOLD_OBJECTIVES, default="f1")
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(message)s")

    if args.model_name not in models:
        sys.exit(f"Model '{args.model_name}' not known. Available models: {' '.join(models.keys())}")
    if args.model_name != "stripedmamba_isolate":
        sys.exit("Threshold tuning script currently supports only `--model_name stripedmamba_isolate`.")
    if args.batch_size < 1:
        sys.exit(f"--batch_size must be >= 1, got {args.batch_size}")
    if args.ecoli_bucket_size < 1:
        sys.exit(f"--ecoli_bucket_size must be >= 1, got {args.ecoli_bucket_size}")
    if not os.path.isfile(args.checkpoint):
        sys.exit(f"Checkpoint file not found: {args.checkpoint}")
    if not (0.0 <= args.baseline_threshold <= 1.0):
        sys.exit(f"--baseline_threshold must be in [0, 1], got {args.baseline_threshold}")

    try:
        threshold_grid = build_threshold_grid(args.threshold_min, args.threshold_max, args.threshold_points)
    except ValueError as exc:
        sys.exit(str(exc))

    try:
        model_args = json.loads(args.model_args)
    except json.JSONDecodeError as exc:
        sys.exit(f"Invalid JSON in --model_args: {exc}")
    if not isinstance(model_args, dict):
        sys.exit("--model_args must be a JSON object")
    try:
        model_args = inject_input_interface_model_args(
            model_name=args.model_name,
            model_args=model_args,
            use_input_interface=args.use_input_interface,
            preset=args.input_interface_preset,
            overrides_json=args.input_interface_args,
        )
    except ValueError as exc:
        sys.exit(str(exc))

    valid_dataset = EcoliIsolateDataset(
        metadata_csv=args.ecoli_metadata_csv,
        contigs_dir=args.ecoli_contigs_dir,
        split="valid",
        antibiotic=args.ecoli_antibiotic,
        train_fraction=args.ecoli_train_fraction,
        valid_fraction=args.ecoli_valid_fraction,
        split_seed=args.ecoli_split_seed,
        contig_mode=args.ecoli_contig_mode,
        dynamic_shuffle=bool_arg(args.ecoli_dynamic_shuffle),
        shuffle_seed=args.ecoli_split_seed,
        separator_length=args.ecoli_separator_length,
        max_genome_length=args.ecoli_max_genome_length,
        pad_to_multiple=args.ecoli_pad_to_multiple,
        cache_contigs=args.ecoli_cache_contigs,
    )
    test_dataset = EcoliIsolateDataset(
        metadata_csv=args.ecoli_metadata_csv,
        contigs_dir=args.ecoli_contigs_dir,
        split="test",
        antibiotic=args.ecoli_antibiotic,
        train_fraction=args.ecoli_train_fraction,
        valid_fraction=args.ecoli_valid_fraction,
        split_seed=args.ecoli_split_seed,
        contig_mode=args.ecoli_contig_mode,
        dynamic_shuffle=bool_arg(args.ecoli_dynamic_shuffle),
        shuffle_seed=args.ecoli_split_seed,
        separator_length=args.ecoli_separator_length,
        max_genome_length=args.ecoli_max_genome_length,
        pad_to_multiple=args.ecoli_pad_to_multiple,
        cache_contigs=args.ecoli_cache_contigs,
    )
    if len(valid_dataset) == 0:
        sys.exit("Validation split is empty. Check metadata paths and split parameters.")
    if len(test_dataset) == 0:
        sys.exit("Test split is empty. Check metadata paths and split parameters.")
    x0, _y0 = valid_dataset[0]
    seq_depth = int(x0.shape[1])

    valid_iter = ecoli_outer_iter(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        seed=args.ecoli_split_seed,
        bucket_by_length=bool_arg(args.ecoli_length_bucketing) and args.batch_size > 1,
        bucket_size=args.ecoli_bucket_size,
        pad_to_multiple=args.ecoli_pad_to_multiple,
    )
    test_iter = ecoli_outer_iter(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        seed=args.ecoli_split_seed,
        bucket_by_length=bool_arg(args.ecoli_length_bucketing) and args.batch_size > 1,
        bucket_size=args.ecoli_bucket_size,
        pad_to_multiple=args.ecoli_pad_to_multiple,
    )
    n_valid_batches = ecoli_count_batches(valid_dataset, args.batch_size, None, None)
    n_test_batches = ecoli_count_batches(test_dataset, args.batch_size, None, None)

    model_cls = models[args.model_name]["new_model"]
    model = model_cls(seq_depth=seq_depth, features=1, **model_args)
    with open(args.checkpoint, "rb") as f:
        payload = pickle.load(f)
    try:
        state_dict, checkpoint_meta = _extract_model_state_dict(payload)
    except ValueError as exc:
        sys.exit(str(exc))
    try:
        model.load_state_dict(state_dict, strict=False)
    except RuntimeError as exc:
        sys.exit(
            "Failed to load checkpoint into model. This usually means `--model_args` / "
            "`--input_interface_args` do not match training.\n"
            f"PyTorch error: {exc}"
        )

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    elif args.device == "cuda":
        if not torch.cuda.is_available():
            sys.exit("CUDA requested but no GPU is available.")
        device = "cuda"
    else:
        device = "cpu"
    model.to(device)

    logits, targets = collect_eval_logits_targets(
        model=model,
        iterator=valid_iter,
        n_batches=n_valid_batches,
        device=device,
        amp_enabled=args.amp,
        amp_dtype=args.amp_dtype,
    )
    if logits.numel() == 0:
        sys.exit("No validation samples were produced by iterator.")
    test_logits, test_targets = collect_eval_logits_targets(
        model=model,
        iterator=test_iter,
        n_batches=n_test_batches,
        device=device,
        amp_enabled=args.amp,
        amp_dtype=args.amp_dtype,
    )
    if test_logits.numel() == 0:
        sys.exit("No test samples were produced by iterator.")

    report = build_report(
        logits=logits,
        targets=targets,
        threshold_grid=threshold_grid,
        selected_objective=args.objective,
        baseline_threshold=args.baseline_threshold,
        test_logits=test_logits,
        test_targets=test_targets,
    )
    report["checkpoint"] = {
        "path": args.checkpoint,
        "epoch": int(checkpoint_meta.get("epoch", 0)),
        "step": int(checkpoint_meta.get("step", 0)),
    }
    report["validation"] = {
        "num_isolates": int(len(valid_dataset)),
        "num_batches": int(n_valid_batches),
        "batch_size": int(args.batch_size),
        "antibiotic": args.ecoli_antibiotic,
        "split_seed": int(args.ecoli_split_seed),
    }
    report["test"].update(
        {
            "num_isolates": int(len(test_dataset)),
            "num_batches": int(n_test_batches),
            "batch_size": int(args.batch_size),
            "antibiotic": args.ecoli_antibiotic,
            "split_seed": int(args.ecoli_split_seed),
        }
    )
    report["model"] = {
        "model_name": args.model_name,
        "model_args": model_args,
    }

    out_json = json.dumps(report, indent=2)
    print(out_json)
    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            f.write(out_json + "\n")
        logging.warning(f"Wrote threshold tuning report to {args.output_json}")


if __name__ == "__main__":
    main()
