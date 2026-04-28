# GPM

PyTorch codebase for the bachelor thesis:
**"Adapting Hybrid State-Space–Attention Architectures for Organism-Level Genotype-to-Phenotype Mapping"**.

This repository contains a full training/evaluation pipeline for two settings:

- Regulatory prediction on Borzoi hg38-style data (sequence-to-coverage regression).
- Whole-genome *E. coli* CIP resistance prediction (isolate-level classification).

## 1) What This Project Does

The project studies whether a hybrid SSM+attention architecture can be adapted from regulatory genomics tasks to organism-level phenotype prediction from raw DNA.

Main components implemented here:

- PyTorch reimplementation of Striped Mamba.
- Configurable input interface for aggressive sequence compression.
- Sequence shuffling regularization:
  - aligned chunk shuffling for Borzoi-style sequence/target pairs,
  - dynamic contig-order shuffling for *E. coli* isolate classification.
- Threshold tuning script for classification checkpoints with validation calibration and held-out test reporting.

## 2) Provenance and Attribution

The Striped Mamba architecture in this repository is a PyTorch reimplementation of the Striped Mamba model from the Bilby ecosystem:

- Bilby repository: https://github.com/ihh/bilby
- Striped Mamba paper/preprint:
  - "Selective State Space Models Outperform Transformers at Predicting RNA-Seq Read Coverage"
  - https://doi.org/10.1101/2025.02.13.638190

This repository reimplements the model/training stack in PyTorch and extends it with the configurable input interface and organism-level phenotype workflow.

## 3) Repository Layout

- `scripts/train.py` - main training entrypoint (both Borzoi and *E. coli* backends).
- `scripts/infer.py` - single-batch shape/inference smoke check for Borzoi backend.
- `scripts/tune_ecoli_thresholds.py` - threshold calibration and report generation.
- `requirements.txt` - pinned environment from a working setup.
- `setup_gpm_env.sh` - automated environment bootstrap with ABI-aware `mamba-ssm` wheel selection.
- `models/striped_mamba.py` - model definitions.
- `models/input_interface.py` - configurable input interface modules.
- `data_utils/dataset.py` - Borzoi TFRecord dataset loader.
- `data_utils/ecoli_dataset.py` - *E. coli* isolate dataset and iterators.
- `training/state.py` - training loop, checkpointing, validation/test evaluation.
- `utils/` - losses, metrics, and input-interface arg injection helpers.

## 4) Important Package Naming Note

Code imports use the package name `gpm` (for example `python -m gpm.scripts.train`).

Your local directory must therefore be named `gpm`, or exposed as `gpm` on `PYTHONPATH`.

Recommended clone pattern:

```bash
git clone https://github.com/maslenchenko/gpm.git
cd gpm
```

If you run commands from inside `gpm`, use:

```bash
PYTHONPATH="$(pwd)/.." python -m gpm.scripts.train ...
```

because `python -m gpm.*` resolves package `gpm` from the parent directory.

## 5) Environment Requirements

Tested target environment:

- Linux x86_64
- Python 3.12
- NVIDIA GPU (for practical training speed)
- CUDA-compatible PyTorch build

Core dependency constraints in this repo:

- `torch==2.6.0+cu118`
- `mamba-ssm` wheel tied to torch/CUDA/Python/ABI compatibility
- `tensorflow-cpu`, `tensorflow-datasets`, `natsort` for Borzoi TFRecord pipeline
- `wandb` for training logging (training path initializes W&B)

## 6) Setup

### Option A (recommended): use provided setup script

```bash
cd gpm
chmod +x setup_gpm_env.sh
./setup_gpm_env.sh
```

What it does:

- creates/uses `.venv`,
- installs torch with CUDA 11.8 index,
- auto-selects matching `mamba-ssm` wheel by C++ ABI and Python tag,
- installs runtime deps,
- runs a quick import verification.

### Option B (manual install)

```bash
cd gpm
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip setuptools wheel
pip install -r requirements.txt
```

If `mamba_ssm` import fails with undefined symbols, reinstall with a wheel matching your torch ABI (`cxx11abiTRUE` or `cxx11abiFALSE`).

### Path setup (recommended for commands below)

Set these once per shell session:

```bash
export GPM_ROOT=/path/to/gpm
export BORZOI_DATA=/path/to/borzoi_data/hg38
export ECOLI_DATA=/path/to/ecoli-data
export CKPT_DIR="$GPM_ROOT/ckpts"
export LOG_DIR="$GPM_ROOT/logs"
mkdir -p "$CKPT_DIR" "$LOG_DIR"
```

Then use `$BORZOI_DATA`, `$ECOLI_DATA`, `$CKPT_DIR`, and `$LOG_DIR` in commands instead of machine-specific absolute paths.

## 7) Data Requirements

### 7.1 Borzoi backend (`--data_backend borzoi`)

Expected structure under `--data_dir`:

- `statistics.json`
- `targets.txt`
- `tfrecords/<split>-*.tfr` (for example `fold0-*.tfr`, `fold1-*.tfr`, ...)

The code expects split sequence counts in `statistics.json` with keys like `<split>_seqs`.

### 7.2 *E. coli* backend (`--data_backend ecoli`)

Required inputs:

- `Metadata.csv` containing columns:
  - `Isolate`
  - antibiotic column (default `CIP`)
  - `Year` (optional)
- contig directory containing files named:
  - `<Isolate>.velvet.gff`

The loader reads contigs from the `##FASTA` section of each GFF file.

## 8) Training

All commands assume:

- you run from the parent directory of `gpm`,
- virtualenv is active,
- invocation uses module mode.

### 8.1 Borzoi baseline (Experiment 1a style)

```bash
python -m gpm.scripts.train \
  --data_dir "$BORZOI_DATA" \
  --train fold0 fold1 fold2 fold3 \
  --valid fold4 \
  --test fold5 \
  --seq_length_crop 393216 \
  --model_name stripedmamba \
  --model_args '{"crop":3072}' \
  --batch_size 1 \
  --amp --amp_dtype float16 \
  --checkpoint_blocks \
  --learn_rate 1e-4 \
  --warmup_steps 10000 \
  --lr_schedule cosine \
  --global_clip 10 \
  --shuffle_buffer 4 \
  --max_epochs 10 \
  --patience 5 \
  --save "$CKPT_DIR/stripedmamba_fold0-3.pt" \
  --experiment_name stripedmamba_fold0-3
```

### 8.2 Borzoi with shuffling + configurable input interface (Experiment 1b/1c style)

```bash
python -m gpm.scripts.train \
  --data_dir "$BORZOI_DATA" \
  --train fold0 fold1 fold2 fold3 \
  --valid fold4 \
  --test fold5 \
  --seq_length_crop 393216 \
  --model_name stripedmamba \
  --model_args '{"crop":3072,"bn_momentum":0.1}' \
  --use_input_interface \
  --input_interface_preset borzoi \
  --input_interface_args '{}' \
  --shuffle true \
  --p_shuffle 0.5 \
  --shuffle_min_chunks 3 \
  --shuffle_max_chunks 6 \
  --batch_size 1 \
  --amp --amp_dtype bfloat16 \
  --checkpoint_blocks \
  --learn_rate 1e-4 \
  --warmup_steps 10000 \
  --lr_schedule cosine \
  --global_clip 10 \
  --max_epochs 10 \
  --patience 5 \
  --save "$CKPT_DIR/stripedmamba_shuffle_input_interface.pt" \
  --experiment_name stripedmamba_shuffle_input_interface
```

### 8.3 *E. coli* isolate classification (Experiment 2a/2b style)

```bash
python -m gpm.scripts.train \
  --data_backend ecoli \
  --model_name stripedmamba_isolate \
  --model_args '{"crop":0,"bn_momentum":0.1,"classifier_pool":"mean","classifier_dropout_rate":0.0,"positional_encoding":"rope","trans_pool_size":16}' \
  --use_input_interface \
  --input_interface_preset ecoli \
  --input_interface_args '{"num_layers":11,"num_channels_initial":128,"channels_increase_rate":1.16,"maxpooling":2,"kernel_sizes":1,"dilation":1,"norm_type":"batch","context_separate":false,"average_interfaces":false,"concat":false}' \
  --ecoli_metadata_csv "$ECOLI_DATA/Metadata.csv" \
  --ecoli_contigs_dir "$ECOLI_DATA/contigs-ecoli" \
  --ecoli_antibiotic CIP \
  --ecoli_contig_mode shuffle \
  --ecoli_dynamic_shuffle true \
  --ecoli_train_fraction 0.8 \
  --ecoli_valid_fraction 0.1 \
  --ecoli_split_seed 42 \
  --ecoli_separator_length 50 \
  --ecoli_pad_to_multiple 32768 \
  --ecoli_length_bucketing true \
  --ecoli_bucket_size 128 \
  --batch_size 2 \
  --grad_accum_steps 4 \
  --loss_type bce \
  --amp --amp_dtype bfloat16 \
  --checkpoint_blocks \
  --max_shift 0 \
  --learn_rate 1e-4 \
  --warmup_steps 500 \
  --lr_schedule cosine \
  --global_clip 10 \
  --max_epochs 200 \
  --patience -1 \
  --save "$CKPT_DIR/stripedmamba_isolate_ecoli_cip_shuffle.pt" \
  --experiment_name stripedmamba-isolate-ecoli-cip-shuffle
```

Notes:

- *E. coli* backend currently requires `--model_name stripedmamba_isolate`.
- `--loss_type auto` selects BCE for *E. coli*.
- If `--bce_pos_weight` is not provided in *E. coli* mode, it is auto-derived from train label counts.

## 9) Evaluation and Inference

### 9.1 Evaluate checkpoint on test split and write JSON

```bash
python -m gpm.scripts.train \
  --data_dir "$BORZOI_DATA" \
  --test fold5 \
  --seq_length_crop 393216 \
  --model_name stripedmamba \
  --model_args '{"crop":3072}' \
  --batch_size 1 \
  --load "$CKPT_DIR/model.pt.best" \
  --eval "$LOG_DIR/metrics.json"
```

### 9.2 Single-batch inference smoke test

```bash
python -m gpm.scripts.infer \
  --data_dir "$BORZOI_DATA" \
  --split fold5 \
  --model_name stripedmamba \
  --model_args '{"crop":3072}' \
  --load "$CKPT_DIR/model.pt.best"
```

## 10) Threshold Tuning for *E. coli*

`scripts/tune_ecoli_thresholds.py`:

- computes metrics at baseline threshold (`--baseline_threshold`, default 0.5),
- sweeps a threshold grid (`--threshold_min`, `--threshold_max`, `--threshold_points`),
- finds the best threshold for **every objective**:
  - `accuracy`
  - `balanced_accuracy`
  - `precision`
  - `recall`
  - `f1`
- uses `--objective` to choose the threshold that is then reported as `best_threshold` and used for selected-threshold test metrics.

Example:

```bash
python -m gpm.scripts.tune_ecoli_thresholds \
  --checkpoint "$CKPT_DIR/stripedmamba_isolate_ecoli_cip_shuffle.pt.best" \
  --model_name stripedmamba_isolate \
  --model_args '{"crop":0,"bn_momentum":0.1,"classifier_pool":"mean","classifier_dropout_rate":0.0,"positional_encoding":"rope","trans_pool_size":16}' \
  --use_input_interface \
  --input_interface_preset ecoli \
  --input_interface_args '{"num_layers":11,"num_channels_initial":128,"channels_increase_rate":1.16,"maxpooling":2,"kernel_sizes":1,"dilation":1,"norm_type":"batch","context_separate":false,"average_interfaces":false,"concat":false}' \
  --ecoli_metadata_csv "$ECOLI_DATA/Metadata.csv" \
  --ecoli_contigs_dir "$ECOLI_DATA/contigs-ecoli" \
  --ecoli_antibiotic CIP \
  --ecoli_train_fraction 0.8 \
  --ecoli_valid_fraction 0.1 \
  --ecoli_split_seed 42 \
  --ecoli_contig_mode shuffle \
  --ecoli_separator_length 50 \
  --ecoli_pad_to_multiple 32768 \
  --ecoli_length_bucketing true \
  --ecoli_bucket_size 128 \
  --batch_size 2 \
  --amp --amp_dtype bfloat16 \
  --objective balanced_accuracy \
  --threshold_min 0.0 \
  --threshold_max 1.0 \
  --threshold_points 1001 \
  --output_json "$LOG_DIR/ecoli_threshold_tuning.json"
```

## 11) Pretrained Checkpoints (*E. coli* + Borzoi)

Google Drive folder (contains *E. coli* and Borzoi experiment checkpoints, plus threshold-tuning reports):

- https://drive.google.com/drive/folders/1MyxjEZWoBqLZMqB0qVSZ90-5vLcpBojc?usp=sharing

Checkpoints  and experiments:

- Experiment 1a (Borzoi baseline, no shuffle): `stripedmamba_fold0-3_v2.pt`
- Experiment 1b (Borzoi + chunk shuffling): `stripedmamba_fold0-3__shuffle_v1.pt`
- Experiment 1c (Borzoi + chunk shuffling + input interface): `stripedmamba_fold0-3_shuffle_input_interface_v3.pt`
- Experiment 2a (*E. coli* CIP, concat/no contig shuffling): `stripedmamba_isolate_ecoli_cip_concat_v1.pt`
- Experiment 2b (*E. coli* CIP, dynamic contig-order shuffling): `stripedmamba_isolate_ecoli_cip_shuffle_v1.pt`

Threshold tuning reports:

- Experiment 2a report: `ecoli_cip_concat_v1_threshold_tuning.json`
- Experiment 2b report: `ecoli_cip_shuffle_v1_threshold_tuning.json`

## 12) W&B Logging

Training path initializes Weights & Biases runs.

- Ensure `wandb` is installed.
- Authenticate (`wandb login`) or use offline mode:

```bash
export WANDB_MODE=offline
```

## 13) Troubleshooting

- `ImportError: mamba-ssm is required for BidirectionalMamba`:
  - `mamba_ssm` import failed (often ABI/CUDA mismatch), reinstall matching wheel.
- `undefined symbol ... selective_scan_cuda`:
  - C++ ABI mismatch between installed torch and `mamba-ssm` wheel.
- CUDA driver warning:
  - installed torch CUDA runtime is newer than system NVIDIA driver; install compatible torch build or update driver.
- `E. coli backend currently requires --model_name stripedmamba_isolate`:
  - use isolate model for `--data_backend ecoli`.
- `SeqDataset requires optional borzoi dependencies`:
  - install `tensorflow-cpu`, `tensorflow-datasets`, and `natsort` for Borzoi backend.
