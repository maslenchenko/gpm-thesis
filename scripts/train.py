import argparse
import json
import logging
import os
import socket
import sys
from datetime import datetime

import numpy as np
import torch

from gpm.data import (
    SeqDataset,
    EcoliIsolateDataset,
    round_robin_iter,
    ecoli_outer_iter,
    batch_limiter,
    count_batches,
    ecoli_count_batches,
    fake_data_iter,
)
from gpm.models import models
from gpm.training.state import TrainConfig, run_training_loop
from gpm.utils.losses import poisson_loss, poisson_multinomial_loss
from gpm.utils.dna import ensemble_fwd_rev, ensemble_shift
from gpm.utils.input_interface import inject_input_interface_model_args
from gpm.utils.metrics import (
    binary_classification_metrics_from_logits,
    compute_xy_moments,
    pearson_r,
    r_squared,
)

WANDB_PROJECT = "gpm"
WANDB_ENTITY = "maslenchenko"
WANDB_RESUME = "allow"
WANDB_LOG_EVERY = 10
WANDB_LOG_ARTIFACTS = False
WANDB_ARTIFACT_NAME = "gpm-checkpoints"
WANDB_CHECKPOINT_EVERY_N_EPOCHS = 1
EVAL_EVERY_STEPS = 100
SAVE_BEST_ONLY = False


def parse_args():
    parser = argparse.ArgumentParser(description="Train gpm models")
    parser.add_argument(
        "--data_backend",
        type=str,
        choices=["borzoi", "ecoli"],
        default="borzoi",
        help="Dataset backend: TFRecord borzoi pipeline or standalone E. coli isolate loader.",
    )
    parser.add_argument("--data_dir", type=str, default=os.path.join(os.path.dirname(__file__), "../../data"))
    parser.add_argument("--train", nargs="+", default=["train"])
    parser.add_argument("--valid", nargs="+", default=["valid"])
    parser.add_argument("--test", nargs="+", default=["test"])
    parser.add_argument("--model_name", type=str, default="stripedmamba")
    parser.add_argument("--model_args", type=str, default="{}")
    parser.add_argument(
        "--use_input_interface",
        action="store_true",
        help="Enable the vendored input_interface_split trunk integration.",
    )
    parser.add_argument(
        "--input_interface_preset",
        type=str,
        choices=["none", "borzoi", "ecoli"],
        default="none",
        help="Preset for input-interface configuration.",
    )
    parser.add_argument(
        "--input_interface_args",
        type=str,
        default="{}",
        help="JSON object with input-interface config overrides (applied after preset).",
    )
    parser.add_argument("--poisson", action="store_true")
    parser.add_argument("--poisson_weight", type=float, default=0.2)
    parser.add_argument(
        "--loss_type",
        type=str,
        choices=["auto", "poisson_multinomial", "poisson", "bce"],
        default="auto",
        help="Loss function selection. 'auto' => BCE for ecoli backend, Poisson variant otherwise.",
    )
    parser.add_argument(
        "--bce_pos_weight",
        type=float,
        default=None,
        help="Positive class weight for BCEWithLogitsLoss. If unset in ecoli mode, derived from train split.",
    )
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--amp_dtype", type=str, choices=["float16", "bfloat16"], default="float16")
    parser.add_argument("--load", type=str, default=None)
    parser.add_argument("--save", type=str, default=None)
    parser.add_argument("--tabulate", action="store_true")
    parser.add_argument("--eval", type=str, default=None)
    parser.add_argument("--rc_ensemble_eval", type=str, default="true")
    parser.add_argument("--shift_ensemble_eval", type=str, default="true")
    parser.add_argument("--dummy", action="store_true")
    parser.add_argument("--logdir", type=str, default=None)
    parser.add_argument("--logfile", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument(
        "--ecoli_metadata_csv",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "../../ecoli-data/Metadata.csv"),
        help="Path to E. coli metadata CSV with CIP labels.",
    )
    parser.add_argument(
        "--ecoli_contigs_dir",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "../../ecoli-data/contigs-ecoli"),
        help="Path to E. coli contig GFF files.",
    )
    parser.add_argument(
        "--ecoli_antibiotic",
        type=str,
        default="CIP",
        help="Antibiotic column name in metadata CSV (default CIP).",
    )
    parser.add_argument(
        "--ecoli_contig_mode",
        type=str,
        choices=["concat", "shuffle"],
        default="concat",
        help="How to compose isolate contigs into pseudo-genome for training.",
    )
    parser.add_argument(
        "--ecoli_dynamic_shuffle",
        type=str,
        default="true",
        help="Enable dynamic per-epoch/per-draw contig shuffling when contig_mode=shuffle.",
    )
    parser.add_argument(
        "--ecoli_train_fraction",
        type=float,
        default=0.8,
        help="Train split fraction for ecoli backend.",
    )
    parser.add_argument(
        "--ecoli_valid_fraction",
        type=float,
        default=0.1,
        help="Validation split fraction for ecoli backend.",
    )
    parser.add_argument(
        "--ecoli_split_seed",
        type=int,
        default=42,
        help="Seed for stratified ecoli split generation.",
    )
    parser.add_argument(
        "--ecoli_separator_length",
        type=int,
        default=50,
        help="Number of Ns inserted between concatenated contigs.",
    )
    parser.add_argument(
        "--ecoli_max_genome_length",
        type=int,
        default=None,
        help="Optional hard truncation length for pseudo-genome.",
    )
    parser.add_argument(
        "--ecoli_pad_to_multiple",
        type=int,
        default=128,
        help="Pad pseudo-genome length to this multiple (helps pooling divisibility).",
    )
    parser.add_argument(
        "--ecoli_cache_contigs",
        action="store_true",
        help="Cache parsed contig FASTA strings in memory.",
    )
    parser.add_argument(
        "--ecoli_length_bucketing",
        type=str,
        default="true",
        help="Enable length bucketing before batching for ecoli backend.",
    )
    parser.add_argument(
        "--ecoli_bucket_size",
        type=int,
        default=64,
        help="Number of samples per length-sorted bucket when ecoli_length_bucketing is enabled.",
    )
    parser.add_argument(
        "--shuffle_buffer",
        type=int,
        default=4,
        help="Shuffle buffer size (0 disables shuffling; applied only in train mode).",
    )
    parser.add_argument(
        "--prefetch",
        type=int,
        default=1,
        help="Prefetch buffer size (-1 uses AUTOTUNE, 0 disables prefetch).",
    )
    parser.add_argument(
        "--seq_length_crop",
        type=int,
        default=None,
        help="Crop input sequences to this centered length (seq_length - seq_length_crop must be even).",
    )
    parser.add_argument(
        "--shuffle",
        type=str,
        default="false",
        help="Enable aligned sequence-target chunk shuffle augmentation for training batches.",
    )
    parser.add_argument(
        "--p_shuffle",
        type=float,
        default=0.0,
        help="Per-sample probability to apply chunk shuffle when --shuffle is enabled.",
    )
    parser.add_argument(
        "--shuffle_min_chunks",
        type=int,
        default=3,
        help="Minimum number of chunks for sequence-target shuffle augmentation.",
    )
    parser.add_argument(
        "--shuffle_max_chunks",
        type=int,
        default=6,
        help="Maximum number of chunks for sequence-target shuffle augmentation.",
    )
    parser.add_argument(
        "--shuffle_log_per_example",
        action="store_true",
        help="Log per-sample shuffle decisions (batch/sample/applied/chunks) during training.",
    )
    parser.add_argument("--checkpoint_blocks", action="store_true")
    parser.add_argument("--first_batch", type=int, default=None)
    parser.add_argument("--batch_limit", type=int, default=None)
    parser.add_argument("--max_epochs", type=int, default=None)
    parser.add_argument("--max_seconds", type=float, default=None)
    parser.add_argument("--prevalidate", action="store_true")
    parser.add_argument(
        "--patience",
        type=int,
        default=25,
        help="Early-stopping patience in number of validation events. Set -1 to disable early stopping.",
    )
    parser.add_argument("--recompute_train_loss", action="store_true")
    parser.add_argument("--max_shift", type=int, default=3)
    parser.add_argument("--rng_key", type=int, default=42)
    parser.add_argument("--rnd_valid", action="store_true")
    parser.add_argument("--learn_rate", type=float, default=1e-4)
    parser.add_argument("--warmup_steps", type=int, default=10000)
    parser.add_argument(
        "--lr_schedule",
        type=str,
        choices=["constant", "cosine"],
        default="constant",
        help="Learning-rate schedule after warmup.",
    )
    parser.add_argument(
        "--cosine_min_lr_ratio",
        type=float,
        default=0.0,
        help="Final LR ratio for cosine schedule (final_lr = learn_rate * cosine_min_lr_ratio).",
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument(
        "--optimizer",
        type=str,
        choices=["adam", "sgd"],
        default="adam",
        help="Optimizer type.",
    )
    parser.add_argument(
        "--sgd_momentum",
        type=float,
        default=0.9,
        help="Momentum for SGD optimizer.",
    )
    parser.add_argument(
        "--sgd_nesterov",
        type=str,
        default="true",
        help="Use Nesterov momentum when optimizer=sgd.",
    )
    parser.add_argument("--block_clip", type=float, default=5.0)
    parser.add_argument("--global_clip", type=float, default=10.0)
    parser.add_argument(
        "--grad_accum_steps",
        type=int,
        default=1,
        help="Number of micro-batches to accumulate before each optimizer step.",
    )
    parser.add_argument("--experiment_name", type=str, default=None, help="W&B run name for this experiment.")
    return parser.parse_args()


def bool_arg(val: str):
    return str(val).lower() in ("1", "true", "yes", "y")


def main():
    args = parse_args()

    if not (args.save or args.dummy or args.tabulate or args.eval):
        sys.exit("You must specify --save, --eval, --tabulate, or --dummy")

    if args.model_name not in models:
        sys.exit(f"Model '{args.model_name}' not known. Available models: {' '.join(models.keys())}")
    if args.data_backend == "ecoli" and args.model_name != "stripedmamba_isolate":
        sys.exit("E. coli backend currently requires `--model_name stripedmamba_isolate`.")
    if args.loss_type == "bce" and args.model_name != "stripedmamba_isolate":
        sys.exit("`--loss_type bce` is supported only with `--model_name stripedmamba_isolate`.")
    if args.bce_pos_weight is not None and args.loss_type not in ("auto", "bce"):
        logging.warning("`--bce_pos_weight` is ignored unless loss_type is `auto` or `bce`.")
    if not (args.eval or args.tabulate) and not args.experiment_name:
        sys.exit("Please provide `--experiment_name` for W&B logging.")
    if not (0.0 <= args.cosine_min_lr_ratio <= 1.0):
        sys.exit(f"--cosine_min_lr_ratio must be in [0, 1], got {args.cosine_min_lr_ratio}")
    if args.grad_accum_steps < 1:
        sys.exit(f"--grad_accum_steps must be >= 1, got {args.grad_accum_steps}")
    if args.patience < -1:
        sys.exit(f"--patience must be >= -1, got {args.patience}")
    if not (0.0 <= args.sgd_momentum < 1.0):
        sys.exit(f"--sgd_momentum must be in [0, 1), got {args.sgd_momentum}")
    if args.optimizer == "sgd" and bool_arg(args.sgd_nesterov) and args.sgd_momentum <= 0.0:
        sys.exit("--sgd_nesterov requires --sgd_momentum > 0.")

    if args.logdir or args.logfile:
        logdir = args.logdir or "."
        logfile = args.logfile or "log"
        os.makedirs(logdir, exist_ok=True)
        logging.basicConfig(filename=f"{logdir}/{logfile}", level=logging.WARNING, format="%(asctime)s %(message)s")
        logging.getLogger().addHandler(logging.StreamHandler())
    else:
        logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(message)s")

    logging.warning("Args: " + " ".join(sys.argv))
    logging.warning("Date: " + datetime.now().strftime("%Y/%m/%d %H:%M:%S"))
    logging.warning("Host: " + socket.gethostname())

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
    if args.checkpoint_blocks:
        model_args["checkpoint_blocks"] = True

    train_dataset_ecoli = None
    if args.data_backend == "borzoi":
        train_sets = [
            SeqDataset(
                data_dir=args.data_dir,
                split_label=label,
                batch_size=1,
                mode="train",
                seq_length_crop=args.seq_length_crop,
                shuffle_buffer=args.shuffle_buffer,
                prefetch=args.prefetch,
            )
            for label in args.train
        ]
        valid_sets = [
            SeqDataset(
                data_dir=args.data_dir,
                split_label=label,
                batch_size=1,
                mode="eval",
                seq_length_crop=args.seq_length_crop,
                shuffle_buffer=args.shuffle_buffer,
                prefetch=args.prefetch,
            )
            for label in args.valid
        ]
        test_sets = [
            SeqDataset(
                data_dir=args.data_dir,
                split_label=label,
                batch_size=1,
                mode="eval",
                seq_length_crop=args.seq_length_crop,
                shuffle_buffer=args.shuffle_buffer,
                prefetch=args.prefetch,
            )
            for label in args.test
        ]

        representative = (train_sets + valid_sets + test_sets)[0]
        seq_length = representative.seq_length
        effective_seq_length = representative.effective_seq_length
        seq_depth = representative.seq_depth
        target_length = representative.target_length
        n_targets = representative.num_targets

        if effective_seq_length != seq_length:
            logging.warning(
                f"seq_length_crop={args.seq_length_crop}: seq_length={seq_length} -> {effective_seq_length} (cropped), "
                f"seq_depth={seq_depth}, target_length={target_length}, n_targets={n_targets}"
            )
        else:
            logging.warning(f"seq_length={seq_length}, seq_depth={seq_depth}, target_length={target_length}, n_targets={n_targets}")

        shuffle_enabled = bool_arg(args.shuffle)
        if shuffle_enabled:
            logging.warning(
                "Enabled shuffle augmentation: "
                f"p_shuffle={args.p_shuffle}, "
                f"chunks=[{args.shuffle_min_chunks}, {args.shuffle_max_chunks}], "
                f"pool_width={representative.pool_width}"
            )
        if args.shuffle_log_per_example and not shuffle_enabled:
            logging.warning("`--shuffle_log_per_example` is set but shuffle is disabled; no per-sample shuffle events will be emitted.")

        valid_iter = batch_limiter(
            fake_data_iter(valid_sets, effective_seq_length, seq_depth, target_length, n_targets, seed=args.rng_key)
            if args.rnd_valid
            else round_robin_iter(valid_sets, args.batch_size),
            args.batch_limit,
            args.first_batch,
        )
        train_iter = batch_limiter(
            round_robin_iter(
                train_sets,
                args.batch_size,
                shuffle=shuffle_enabled,
                p_shuffle=args.p_shuffle,
                min_chunks=args.shuffle_min_chunks,
                max_chunks=args.shuffle_max_chunks,
                pool_width=representative.pool_width,
                rng=np.random.default_rng(args.rng_key),
                shuffle_log_per_example=args.shuffle_log_per_example,
            ),
            args.batch_limit,
            args.first_batch,
        )
        test_iter = batch_limiter(round_robin_iter(test_sets, args.batch_size), args.batch_limit, args.first_batch)

        n_valid_batches = count_batches(valid_sets, args.batch_size, args.batch_limit, args.first_batch)
        n_train_batches = count_batches(train_sets, args.batch_size, args.batch_limit, args.first_batch)
        n_test_batches = count_batches(test_sets, args.batch_size, args.batch_limit, args.first_batch)
    else:
        if args.batch_size < 1:
            sys.exit(f"E. coli backend requires --batch_size >= 1, got {args.batch_size}.")

        dynamic_shuffle_train = bool_arg(args.ecoli_dynamic_shuffle) and args.ecoli_contig_mode == "shuffle"
        length_bucketing = bool_arg(args.ecoli_length_bucketing)
        train_dataset_ecoli = EcoliIsolateDataset(
            metadata_csv=args.ecoli_metadata_csv,
            contigs_dir=args.ecoli_contigs_dir,
            split="train",
            antibiotic=args.ecoli_antibiotic,
            train_fraction=args.ecoli_train_fraction,
            valid_fraction=args.ecoli_valid_fraction,
            split_seed=args.ecoli_split_seed,
            contig_mode=args.ecoli_contig_mode,
            dynamic_shuffle=dynamic_shuffle_train,
            shuffle_seed=args.rng_key,
            separator_length=args.ecoli_separator_length,
            max_genome_length=args.ecoli_max_genome_length,
            pad_to_multiple=args.ecoli_pad_to_multiple,
            cache_contigs=args.ecoli_cache_contigs,
        )
        valid_dataset_ecoli = EcoliIsolateDataset(
            metadata_csv=args.ecoli_metadata_csv,
            contigs_dir=args.ecoli_contigs_dir,
            split="valid",
            antibiotic=args.ecoli_antibiotic,
            train_fraction=args.ecoli_train_fraction,
            valid_fraction=args.ecoli_valid_fraction,
            split_seed=args.ecoli_split_seed,
            contig_mode="concat",
            dynamic_shuffle=False,
            shuffle_seed=args.rng_key,
            separator_length=args.ecoli_separator_length,
            max_genome_length=args.ecoli_max_genome_length,
            pad_to_multiple=args.ecoli_pad_to_multiple,
            cache_contigs=args.ecoli_cache_contigs,
        )
        test_dataset_ecoli = EcoliIsolateDataset(
            metadata_csv=args.ecoli_metadata_csv,
            contigs_dir=args.ecoli_contigs_dir,
            split="test",
            antibiotic=args.ecoli_antibiotic,
            train_fraction=args.ecoli_train_fraction,
            valid_fraction=args.ecoli_valid_fraction,
            split_seed=args.ecoli_split_seed,
            contig_mode="concat",
            dynamic_shuffle=False,
            shuffle_seed=args.rng_key,
            separator_length=args.ecoli_separator_length,
            max_genome_length=args.ecoli_max_genome_length,
            pad_to_multiple=args.ecoli_pad_to_multiple,
            cache_contigs=args.ecoli_cache_contigs,
        )
        if len(train_dataset_ecoli) == 0 or len(valid_dataset_ecoli) == 0 or len(test_dataset_ecoli) == 0:
            sys.exit(
                "E. coli split produced an empty subset. "
                "Check metadata paths, filters, and split fractions."
            )

        x0, _y0 = train_dataset_ecoli[0]
        seq_length = int(x0.shape[0])
        effective_seq_length = seq_length
        seq_depth = int(x0.shape[1])
        target_length = 1
        n_targets = 1
        logging.warning(
            "E. coli backend: "
            f"train={len(train_dataset_ecoli)}, valid={len(valid_dataset_ecoli)}, test={len(test_dataset_ecoli)}, "
            f"seq_length(first)={seq_length}, seq_depth={seq_depth}, targets={n_targets}, "
            f"contig_mode_train={args.ecoli_contig_mode}, dynamic_shuffle_train={dynamic_shuffle_train}, "
            f"batch_size={args.batch_size}, length_bucketing={length_bucketing}, bucket_size={args.ecoli_bucket_size}"
        )
        if args.rnd_valid:
            logging.warning("`--rnd_valid` is ignored for ecoli backend.")
        if bool_arg(args.shuffle):
            logging.warning("Sequence-target chunk shuffle args are ignored for ecoli backend.")

        train_iter = batch_limiter(
            ecoli_outer_iter(
                train_dataset_ecoli,
                batch_size=args.batch_size,
                shuffle=True,
                seed=args.rng_key,
                bucket_by_length=length_bucketing and args.batch_size > 1,
                bucket_size=args.ecoli_bucket_size,
                pad_to_multiple=args.ecoli_pad_to_multiple,
            ),
            args.batch_limit,
            args.first_batch,
        )
        valid_iter = batch_limiter(
            ecoli_outer_iter(
                valid_dataset_ecoli,
                batch_size=args.batch_size,
                shuffle=False,
                seed=args.rng_key,
                bucket_by_length=length_bucketing and args.batch_size > 1,
                bucket_size=args.ecoli_bucket_size,
                pad_to_multiple=args.ecoli_pad_to_multiple,
            ),
            args.batch_limit,
            args.first_batch,
        )
        test_iter = batch_limiter(
            ecoli_outer_iter(
                test_dataset_ecoli,
                batch_size=args.batch_size,
                shuffle=False,
                seed=args.rng_key,
                bucket_by_length=length_bucketing and args.batch_size > 1,
                bucket_size=args.ecoli_bucket_size,
                pad_to_multiple=args.ecoli_pad_to_multiple,
            ),
            args.batch_limit,
            args.first_batch,
        )
        n_train_batches = ecoli_count_batches(train_dataset_ecoli, args.batch_size, args.batch_limit, args.first_batch)
        n_valid_batches = ecoli_count_batches(valid_dataset_ecoli, args.batch_size, args.batch_limit, args.first_batch)
        n_test_batches = ecoli_count_batches(test_dataset_ecoli, args.batch_size, args.batch_limit, args.first_batch)

    logging.warning(f"counted {n_valid_batches} validation and {n_train_batches} training batches")

    model_cls = models[args.model_name]["new_model"]
    model = model_cls(seq_depth=seq_depth, features=n_targets, **model_args)

    if args.tabulate:
        logging.warning(model)
        sys.exit()

    selected_loss = args.loss_type
    if selected_loss == "auto":
        if args.data_backend == "ecoli":
            selected_loss = "bce"
        elif args.poisson:
            selected_loss = "poisson"
        else:
            selected_loss = "poisson_multinomial"

    if selected_loss == "poisson":
        loss_fn = poisson_loss
    elif selected_loss == "poisson_multinomial":
        loss_fn = lambda y_pred, y_true: poisson_multinomial_loss(y_pred, y_true, total_weight=args.poisson_weight)
    elif selected_loss == "bce":
        if args.bce_pos_weight is not None:
            pos_weight_value = float(args.bce_pos_weight)
        elif args.data_backend == "ecoli":
            positives = sum(int(r.label == 1) for r in train_dataset_ecoli.records)
            negatives = sum(int(r.label == 0) for r in train_dataset_ecoli.records)
            if positives <= 0:
                pos_weight_value = 1.0
            else:
                pos_weight_value = float(negatives) / float(positives)
            pos_weight_value = min(max(pos_weight_value, 0.01), 100.0)
            logging.warning(f"Auto-derived BCE pos_weight={pos_weight_value:.6f} from ecoli train split.")
        else:
            pos_weight_value = 1.0
        pos_weight_tensor = torch.tensor(pos_weight_value, dtype=torch.float32)

        def loss_fn(y_pred, y_true):
            return torch.nn.functional.binary_cross_entropy_with_logits(
                y_pred,
                y_true,
                pos_weight=pos_weight_tensor.to(y_pred.device),
            )
    else:
        sys.exit(f"Unsupported loss_type: {selected_loss}")

    metric_mode = "classification" if selected_loss == "bce" else "regression"

    if args.eval:
        model.eval()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        device_type = "cuda" if device.startswith("cuda") else "cpu"
        amp_enabled = args.amp and device_type == "cuda"
        if args.amp and not amp_enabled:
            logging.warning("AMP requested but CUDA is not available; disabling AMP.")
        amp_dtype = torch.float16 if args.amp_dtype == "float16" else torch.bfloat16
        autocast_ctx = torch.autocast(device_type=device_type, dtype=amp_dtype, enabled=amp_enabled)
        model.to(device)
        if args.data_backend == "ecoli":
            strand_pair = torch.tensor([0], device=device)
        else:
            strand_pair = torch.tensor(test_sets[0].strand_pair, device=device)

        def predict_fn(m, x):
            return m(x)

        if bool_arg(args.rc_ensemble_eval):
            predict_fn = ensemble_fwd_rev(predict_fn, strand_pair)
        if bool_arg(args.shift_ensemble_eval):
            predict_fn = ensemble_shift(predict_fn, args.max_shift)

        eta = 0
        loss = 0.0
        n_test = 0
        moments = None
        all_logits = []
        all_targets = []
        with torch.no_grad():
            for i, (x, y_true) in enumerate(next(test_iter)):
                x = torch.tensor(x, device=device)
                y_true = torch.tensor(y_true, device=device)
                with autocast_ctx:
                    y_pred = predict_fn(model, x)
                    l = loss_fn(y_pred, y_true)
                if metric_mode == "classification":
                    all_logits.append(y_pred.detach().float().cpu())
                    all_targets.append(y_true.detach().float().cpu())
                else:
                    m = compute_xy_moments(y_pred, y_true)
                    moments = m if moments is None else moments + m
                loss += l.item() * x.shape[0]
                n_test += x.shape[0]
        loss = loss / max(n_test, 1)
        if metric_mode == "classification":
            logits = torch.cat(all_logits, dim=0) if all_logits else torch.empty(0, dtype=torch.float32)
            targets = torch.cat(all_targets, dim=0) if all_targets else torch.empty(0, dtype=torch.float32)
            out = {"loss": float(loss)}
            out.update(binary_classification_metrics_from_logits(logits, targets, threshold=0.5))
        else:
            R = pearson_r(moments)
            R2 = r_squared(moments)
            n_features = moments.shape[0]
            by_feature = {
                "pearson_r": [float(pearson_r(moments[n, :], keep_features=True)) for n in range(n_features)],
                "r_squared": [float(r_squared(moments[n, :], keep_features=True)) for n in range(n_features)],
            }
            out = {"loss": float(loss), "pearson_r": float(R), "r_squared": float(R2), "by_feature": by_feature}
        with open(args.eval, "w") as f:
            f.write(json.dumps(out))
        sys.exit()

    if args.dummy and not args.save:
        args.save = os.path.join(os.getcwd(), "dummy.pt")

    resume_epoch = 0
    resume_step = 0
    resume_optimizer_state = None
    resume_scaler_state = None

    if args.load and not os.path.isfile(args.load):
        sys.exit(f"Checkpoint file not found: {args.load}")

    if args.load or (args.save and os.path.isfile(args.save)):
        load_path = args.load or args.save
        if os.path.isfile(load_path):
            logging.warning(f"loading parameters from {load_path}")
            import pickle
            with open(load_path, "rb") as f:
                payload = pickle.load(f)
            if isinstance(payload, dict) and "model" in payload:
                model.load_state_dict(payload["model"], strict=False)
                resume_epoch = int(payload.get("epoch", 0) or 0)
                resume_step = int(payload.get("step", 0) or 0)
                resume_optimizer_state = payload.get("optimizer")
                resume_scaler_state = payload.get("scaler")
                logging.warning(
                    "Loaded checkpoint payload with "
                    f"epoch={resume_epoch}, step={resume_step}, "
                    f"optimizer_state={'present' if resume_optimizer_state is not None else 'missing'}, "
                    f"scaler_state={'present' if resume_scaler_state is not None else 'missing'}."
                )
            elif isinstance(payload, dict):
                model.load_state_dict(payload, strict=False)
                logging.warning("Loaded model-only raw state_dict checkpoint (no optimizer/epoch/step metadata).")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    if args.optimizer == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.learn_rate,
            betas=(args.adam_beta1, args.adam_beta2),
        )
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.learn_rate,
            momentum=args.sgd_momentum,
            nesterov=bool_arg(args.sgd_nesterov),
        )
    else:  # pragma: no cover - guarded by argparse choices
        sys.exit(f"Unsupported optimizer: {args.optimizer}")
    if resume_optimizer_state is not None:
        try:
            optimizer.load_state_dict(resume_optimizer_state)
            logging.warning("Restored optimizer state from checkpoint.")
        except Exception as exc:
            logging.warning(f"Failed to restore optimizer state; continuing with a fresh optimizer: {exc}")

    config = TrainConfig(
        learn_rate=args.learn_rate,
        warmup_steps=args.warmup_steps,
        lr_schedule=args.lr_schedule,
        cosine_min_lr_ratio=args.cosine_min_lr_ratio,
        adam_beta1=args.adam_beta1,
        adam_beta2=args.adam_beta2,
        block_clip=args.block_clip,
        global_clip=args.global_clip,
        max_shift=args.max_shift,
        max_epochs=args.max_epochs,
        max_seconds=args.max_seconds,
        prevalidate=args.prevalidate,
        patience=None if args.patience < 0 else args.patience,
        recompute_train_metrics=args.recompute_train_loss,
        grad_accum_steps=args.grad_accum_steps,
        metric_mode=metric_mode,
        classification_threshold=0.5,
        amp_enabled=args.amp,
        amp_dtype=args.amp_dtype,
        device=device,
    )

    if args.data_backend == "ecoli":
        strand_pair = torch.tensor([0], device=device)
    else:
        strand_pair = torch.tensor(train_sets[0].strand_pair, device=device)
    try:
        import wandb as wandb_module
    except ImportError:
        sys.exit("`wandb` is not installed. Install with `pip install wandb`.")
    wandb = wandb_module
    try:
        wandb_run = wandb.init(
            project=WANDB_PROJECT,
            entity=WANDB_ENTITY,
            name=args.experiment_name,
            job_type="train",
            resume=WANDB_RESUME,
            config={
                **vars(args),
                "model_args_parsed": model_args,
                "resolved_loss_type": selected_loss,
                "metric_mode": metric_mode,
                "device": device,
                "n_train_batches": n_train_batches,
                "n_valid_batches": n_valid_batches,
                "n_test_batches": n_test_batches,
            },
        )
    except Exception as exc:
        sys.exit(f"Failed to initialize W&B run: {exc}")
    logging.warning(f"Initialized W&B run: {wandb_run.url}")

    try:
        run_training_loop(
            model=model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            train_iter=train_iter,
            valid_iter=valid_iter,
            n_train_batches=n_train_batches,
            n_valid_batches=n_valid_batches,
            strand_pair=strand_pair,
            config=config,
            save_filename=args.save,
            wandb_run=wandb_run,
            wandb_module=wandb,
            wandb_log_every=WANDB_LOG_EVERY,
            wandb_log_artifacts=WANDB_LOG_ARTIFACTS,
            wandb_artifact_name=WANDB_ARTIFACT_NAME,
            wandb_checkpoint_every_n_epochs=WANDB_CHECKPOINT_EVERY_N_EPOCHS,
            eval_every_steps=EVAL_EVERY_STEPS,
            save_best_only=SAVE_BEST_ONLY,
            initial_epoch=resume_epoch,
            initial_step=resume_step,
            initial_scaler_state=resume_scaler_state,
        )
    finally:
        wandb_run.finish()


if __name__ == "__main__":
    main()
