import logging
import math
import os
import time
import pickle
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import timedelta
from itertools import islice

import psutil
import torch

from gpm.utils.dna import stochastic_revcomp_batch
from gpm.utils.metrics import (
    binary_classification_metrics_from_logits,
    compute_xy_moments,
    pearson_r,
    r_squared,
)


@dataclass
class TrainConfig:
    learn_rate: float = 1e-4
    warmup_steps: int = 10000
    lr_schedule: str = "constant"
    cosine_min_lr_ratio: float = 0.0
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    block_clip: float = 5.0
    global_clip: float = 10.0
    max_shift: int = 3
    max_epochs: int = None
    max_seconds: float = None
    prevalidate: bool = False
    patience: int | None = 25
    recompute_train_metrics: bool = False
    grad_accum_steps: int = 1
    amp_enabled: bool = False
    amp_dtype: str = "float16"
    force_fp32_validation: bool = True
    fail_on_non_finite: bool = True
    check_batchnorm_buffers: bool = True
    use_threads: bool = False
    use_tracemalloc: bool = False
    metric_mode: str = "regression"
    classification_threshold: float = 0.5
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class ETA:
    def __init__(self, n=None, limit=None):
        self.start_time = time.time()
        self.limit = limit
        self.n = n

    def __call__(self, i):
        lapsed_secs = self.lapsed_secs()
        eta = lapsed_secs * (self.n - 1 - i) / (i + 1)
        if self.limit is not None:
            eta = min(eta, self.limit - lapsed_secs)
        return timedelta(seconds=eta)

    def lapsed_secs(self):
        return time.time() - self.start_time

    def lapsed(self):
        return timedelta(seconds=self.lapsed_secs())

    def past_limit(self):
        return self.limit is not None and self.lapsed_secs() > self.limit


def _clip_by_block_rms(params, clip):
    if clip is None:
        return
    for p in params:
        if p.grad is None:
            continue
        rms = torch.sqrt(torch.mean(p.grad.detach() ** 2))
        if rms > clip:
            p.grad.mul_(clip / (rms + 1e-12))


def _clip_by_global_norm(params, clip):
    if clip is None:
        return
    total = 0.0
    for p in params:
        if p.grad is None:
            continue
        total += torch.sum(p.grad.detach() ** 2).item()
    total = math.sqrt(total)
    if total > clip:
        scale = clip / (total + 1e-12)
        for p in params:
            if p.grad is None:
                continue
            p.grad.mul_(scale)


def _amp_dtype_from_str(name: str):
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"Unknown amp_dtype: {name}")


def _autocast_context(device_type: str, amp_enabled: bool, amp_dtype: str):
    if not amp_enabled:
        return nullcontext()
    dtype = _amp_dtype_from_str(amp_dtype)
    return torch.autocast(device_type=device_type, dtype=dtype, enabled=True)


def _make_grad_scaler(amp_enabled: bool, device_type: str):
    if not amp_enabled or device_type != "cuda":
        return None
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            return torch.amp.GradScaler(device_type=device_type)
        except TypeError:
            return torch.amp.GradScaler()
    return torch.cuda.amp.GradScaler()


def _is_finite_tensor(t: torch.Tensor) -> bool:
    return bool(torch.isfinite(t).all().item())


def _assert_finite_tensor(name: str, t: torch.Tensor, step: int):
    if _is_finite_tensor(t):
        return
    nan_count = int(torch.isnan(t).sum().item())
    inf_count = int(torch.isinf(t).sum().item())
    raise FloatingPointError(f"Non-finite values detected in {name} at step {step}: nan={nan_count}, inf={inf_count}")


def _assert_finite_gradients(params, step: int):
    for idx, p in enumerate(params):
        if p.grad is None:
            continue
        if not _is_finite_tensor(p.grad):
            shape = tuple(p.grad.shape)
            raise FloatingPointError(f"Non-finite gradients detected at step {step} for parameter index {idx}, grad_shape={shape}")


def _assert_finite_batchnorm_buffers(model, step: int):
    for module_name, module in model.named_modules():
        if not isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            continue
        if module.running_mean is not None and not _is_finite_tensor(module.running_mean):
            raise FloatingPointError(f"Non-finite BatchNorm running_mean at step {step} in module '{module_name}'")
        if module.running_var is not None and not _is_finite_tensor(module.running_var):
            raise FloatingPointError(f"Non-finite BatchNorm running_var at step {step} in module '{module_name}'")


def _optimizer_state_to_device(optimizer, device):
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)


def _compute_learning_rate(step: int, config: TrainConfig, total_train_steps=None, schedule_override=None) -> float:
    if int(step) < 0:
        raise ValueError(f"step must be non-negative, got {step}")
    if not (0.0 <= float(config.cosine_min_lr_ratio) <= 1.0):
        raise ValueError(f"cosine_min_lr_ratio must be in [0, 1], got {config.cosine_min_lr_ratio}")

    schedule = str(schedule_override or config.lr_schedule or "constant").lower()
    if schedule not in ("constant", "cosine"):
        raise ValueError(f"Unknown lr_schedule: {config.lr_schedule}")

    base_lr = float(config.learn_rate)
    warmup_steps = max(int(config.warmup_steps or 0), 0)
    if warmup_steps > 0 and step < warmup_steps:
        return base_lr * float(step + 1) / float(warmup_steps)

    if schedule == "cosine":
        if total_train_steps is None:
            return base_lr
        if int(total_train_steps) <= 0:
            raise ValueError(f"total_train_steps must be positive for cosine schedule, got {total_train_steps}")
        decay_steps = max(int(total_train_steps) - warmup_steps, 1)
        decay_step = min(max(int(step) - warmup_steps, 0), decay_steps)
        min_lr = base_lr * float(config.cosine_min_lr_ratio)
        cosine = 0.5 * (1.0 + math.cos(math.pi * float(decay_step) / float(decay_steps)))
        return min_lr + (base_lr - min_lr) * cosine

    return base_lr


def compute_metrics(
    model,
    loss_fn,
    iterator,
    n_batches,
    device,
    return_per_feature_metrics=False,
    amp_enabled=False,
    amp_dtype="float16",
    force_fp32=False,
    fail_on_non_finite=True,
    metric_mode="regression",
    classification_threshold=0.5,
):
    model.eval()
    eta = ETA(n=n_batches)
    loss = 0.0
    n_train_seqs = 0
    metric_mode = str(metric_mode or "regression").lower()
    if metric_mode not in ("regression", "classification"):
        raise ValueError(f"Unknown metric_mode: {metric_mode}")

    moments = None
    all_logits = []
    all_targets = []
    device_type = "cuda" if str(device).startswith("cuda") else "cpu"
    amp_enabled = amp_enabled and device_type == "cuda" and not force_fp32
    autocast_ctx = _autocast_context(device_type, amp_enabled, amp_dtype)

    with torch.no_grad():
        for i, (x, y_true) in enumerate(islice(next(iterator), max(n_batches, 0))):
            x = torch.tensor(x, device=device)
            y_true = torch.tensor(y_true, device=device)
            with autocast_ctx:
                y_pred = model(x)
                if y_pred.shape != y_true.shape:
                    raise ValueError(f"predicted shape {y_pred.shape} != required shape {y_true.shape}")
                l = loss_fn(y_pred, y_true)
            if fail_on_non_finite:
                _assert_finite_tensor("validation/y_pred", y_pred, i)
                _assert_finite_tensor("validation/loss", l, i)
            if metric_mode == "regression":
                m = compute_xy_moments(y_pred, y_true)
                moments = m if moments is None else moments + m
                batch_metric_msg = f"r: {pearson_r(m):.4f}"
            else:
                all_logits.append(y_pred.detach().float().cpu())
                all_targets.append(y_true.detach().float().cpu())
                batch_cls = binary_classification_metrics_from_logits(
                    y_pred,
                    y_true,
                    threshold=classification_threshold,
                )
                batch_metric_msg = (
                    f"acc: {batch_cls['accuracy']:.4f}, "
                    f"f1: {batch_cls['f1']:.4f}"
                )
            loss += l.item() * x.shape[0]
            n_train_seqs += x.shape[0]
            used_gb = psutil.virtual_memory().used / 1024 / 1024 / 1024
            logging.warning(
                f"computed predictions for batch {i+1}/{n_batches}, {batch_metric_msg}, input {tuple(x.shape)}, output {tuple(y_pred.shape)}, used {used_gb:.2f} Gb, ETA {eta(i)}"
            )

    loss = loss / max(n_train_seqs, 1)
    if metric_mode == "classification":
        if all_logits:
            logits = torch.cat(all_logits, dim=0)
            targets = torch.cat(all_targets, dim=0)
        else:
            logits = torch.empty(0, dtype=torch.float32)
            targets = torch.empty(0, dtype=torch.float32)
        result = {"loss": float(loss)}
        result.update(
            binary_classification_metrics_from_logits(
                logits,
                targets,
                threshold=classification_threshold,
            )
        )
        return result

    R = pearson_r(moments)
    R2 = r_squared(moments)
    result = {"loss": float(loss), "pearson_r": float(R), "r_squared": float(R2)}

    if return_per_feature_metrics:
        n_features = moments.shape[0]
        result["by_feature"] = {
            "pearson_r": [float(pearson_r(moments[n, :], keep_features=True)) for n in range(n_features)],
            "r_squared": [float(r_squared(moments[n, :], keep_features=True)) for n in range(n_features)],
        }

    return result


def run_training_loop(
    model,
    loss_fn,
    optimizer,
    train_iter,
    valid_iter,
    n_train_batches,
    n_valid_batches,
    strand_pair,
    config: TrainConfig,
    save_filename: str = None,
    wandb_run=None,
    wandb_module=None,
    wandb_log_every: int = 1,
    wandb_log_artifacts: bool = False,
    wandb_artifact_name: str = "gpm-checkpoints",
    wandb_checkpoint_every_n_epochs: int = 1,
    eval_every_steps: int = None,
    save_best_only: bool = False,
    initial_epoch: int = 0,
    initial_step: int = 0,
    initial_scaler_state=None,
):
    device = config.device
    model.to(device)
    _optimizer_state_to_device(optimizer, device)
    step = int(initial_step or 0)
    epoch = int(initial_epoch or 0)
    device_type = "cuda" if str(device).startswith("cuda") else "cpu"
    amp_enabled = config.amp_enabled and device_type == "cuda"
    if config.amp_enabled and not amp_enabled:
        logging.warning("AMP requested but CUDA is not available; disabling AMP.")
    autocast_ctx = _autocast_context(device_type, amp_enabled, config.amp_dtype)
    scaler = _make_grad_scaler(amp_enabled, device_type)
    if scaler is not None and initial_scaler_state is not None:
        try:
            scaler.load_state_dict(initial_scaler_state)
            logging.warning("Restored AMP GradScaler state from checkpoint.")
        except Exception as exc:
            logging.warning(f"Failed to restore AMP GradScaler state; using fresh scaler: {exc}")
    wandb_log_every = max(int(wandb_log_every or 1), 1)
    wandb_checkpoint_every_n_epochs = max(int(wandb_checkpoint_every_n_epochs or 1), 1)
    if eval_every_steps is not None:
        eval_every_steps = max(int(eval_every_steps), 1)
    grad_accum_steps = max(int(config.grad_accum_steps or 1), 1)
    if epoch > 0 or step > 0:
        logging.warning(f"Resuming training loop from epoch={epoch}, step={step}")
    effective_lr_schedule = str(config.lr_schedule or "constant").lower()
    total_train_steps = None
    if effective_lr_schedule == "cosine":
        if config.max_epochs is None:
            logging.warning("Cosine LR schedule requested but max_epochs is not set; falling back to constant LR after warmup.")
            effective_lr_schedule = "constant"
        else:
            updates_per_epoch = max(math.ceil(max(int(n_train_batches), 0) / float(grad_accum_steps)), 1)
            total_train_steps = max(int(config.max_epochs) * updates_per_epoch, 1)
            if int(config.warmup_steps or 0) >= total_train_steps:
                logging.warning(
                    f"warmup_steps ({config.warmup_steps}) >= total_train_steps ({total_train_steps}); "
                    "cosine decay will have minimal effect."
                )
    logging.warning(
        "LR schedule config: "
        f"schedule={effective_lr_schedule}, "
        f"base_lr={config.learn_rate}, "
        f"warmup_steps={config.warmup_steps}, "
        f"grad_accum_steps={grad_accum_steps}, "
        f"cosine_min_lr_ratio={config.cosine_min_lr_ratio}, "
        f"total_train_steps={total_train_steps}"
    )
    metric_mode = str(config.metric_mode or "regression").lower()
    if metric_mode not in ("regression", "classification"):
        raise ValueError(f"Unknown metric_mode: {metric_mode}")

    def _validation_score(metrics):
        if metric_mode == "classification":
            for key in ("auprc", "auroc", "f1", "balanced_accuracy", "accuracy"):
                value = float(metrics.get(key, float("nan")))
                if math.isfinite(value):
                    return value
            return -float(metrics["loss"])
        return float(metrics["pearson_r"]) + float(metrics["r_squared"]) / 4.0

    if step > 0 and grad_accum_steps > 1:
        resume_skip_batches = 0
        logging.warning(
            "Resuming with grad_accum_steps > 1: micro-batch skip alignment is disabled; "
            "the current epoch may replay some batches."
        )
    else:
        resume_skip_batches = step % max(n_train_batches, 1) if step > 0 else 0
    wandb_step_offset = 0
    if wandb_run is not None:
        try:
            wandb_current_step = int(getattr(wandb_run, "step", 0) or 0)
        except Exception:
            wandb_current_step = 0
        should_offset_wandb_step = (step < wandb_current_step) or (step == wandb_current_step and step > 0)
        if should_offset_wandb_step:
            wandb_step_offset = (wandb_current_step - step) + 1
            logging.warning(
                "W&B resume step alignment active: "
                f"internal_step={step}, wandb_current_step={wandb_current_step}, "
                f"wandb_step_offset={wandb_step_offset}"
            )

    def _safe_wandb_log(metrics, step_idx):
        if wandb_run is None:
            return
        try:
            wandb_step = int(step_idx) + wandb_step_offset
            wandb_run.log(metrics, step=wandb_step)
        except Exception as exc:
            logging.warning(f"W&B metric logging failed at step {step_idx}: {exc}")

    def _safe_wandb_artifact(path, aliases, metadata):
        if wandb_run is None or not wandb_log_artifacts or wandb_module is None:
            return
        try:
            artifact = wandb_module.Artifact(name=wandb_artifact_name, type="model", metadata=metadata)
            artifact.add_file(path, name=os.path.basename(path))
            wandb_run.log_artifact(artifact, aliases=aliases)
        except Exception as exc:
            logging.warning(f"W&B artifact logging failed for {path}: {exc}")

    latest_step_checkpoint_path = None
    best_step_checkpoint_path = None

    def save_state(best=False):
        nonlocal latest_step_checkpoint_path, best_step_checkpoint_path
        if save_filename is None:
            return
        path_base = save_filename + (".best" if best else "")
        path_step = f"{path_base}.step{step}"
        payload = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "step": step,
            "scaler": scaler.state_dict() if scaler is not None else None,
        }
        with open(path_base, "wb") as f:
            pickle.dump(payload, f)
        previous_step_path = best_step_checkpoint_path if best else latest_step_checkpoint_path
        if previous_step_path and previous_step_path != path_step and os.path.exists(previous_step_path):
            try:
                os.remove(previous_step_path)
            except OSError as exc:
                logging.warning(f"Failed to remove previous step checkpoint {previous_step_path}: {exc}")
        with open(path_step, "wb") as f:
            pickle.dump(payload, f)
        if best:
            best_step_checkpoint_path = path_step
        else:
            latest_step_checkpoint_path = path_step
        should_log = best or (epoch % wandb_checkpoint_every_n_epochs == 0)
        if should_log:
            aliases = ["best"] if best else ["latest", f"epoch-{epoch}"]
            _safe_wandb_artifact(
                path=path_step,
                aliases=aliases,
                metadata={"epoch": epoch, "step": step, "best": bool(best)},
            )

    def _emergency_save_on_error():
        if save_filename is None:
            return
        try:
            save_state(best=False)
            logging.error(f"Saved emergency checkpoint at step {step} after non-finite detection.")
        except Exception as exc:
            logging.error(f"Failed to save emergency checkpoint at step {step}: {exc}")

    if config.prevalidate:
        vmetrics = compute_metrics(
            model,
            loss_fn,
            valid_iter,
            n_valid_batches,
            device,
            amp_enabled=amp_enabled,
            amp_dtype=config.amp_dtype,
            force_fp32=config.force_fp32_validation,
            fail_on_non_finite=config.fail_on_non_finite,
            metric_mode=metric_mode,
            classification_threshold=config.classification_threshold,
        )
        if metric_mode == "classification":
            logging.warning(
                "Validation metrics before training: "
                f"loss={vmetrics['loss']:.6f} "
                f"auprc={vmetrics['auprc']:.4f} "
                f"auroc={vmetrics['auroc']:.4f} "
                f"f1={vmetrics['f1']:.4f} "
                f"acc={vmetrics['accuracy']:.4f}"
            )
            _safe_wandb_log(
                {
                    "valid/pre/loss": float(vmetrics["loss"]),
                    "valid/pre/auprc": float(vmetrics["auprc"]),
                    "valid/pre/auroc": float(vmetrics["auroc"]),
                    "valid/pre/f1": float(vmetrics["f1"]),
                    "valid/pre/accuracy": float(vmetrics["accuracy"]),
                    "valid/pre/balanced_accuracy": float(vmetrics["balanced_accuracy"]),
                    "epoch": 0,
                },
                step,
            )
        else:
            logging.warning(
                f"Validation metrics before training: loss={vmetrics['loss']:.6f} r={vmetrics['pearson_r']:.4f} r2={vmetrics['r_squared']:.4f}"
            )
            _safe_wandb_log(
                {
                    "valid/pre/loss": float(vmetrics["loss"]),
                    "valid/pre/pearson_r": float(vmetrics["pearson_r"]),
                    "valid/pre/r_squared": float(vmetrics["r_squared"]),
                    "epoch": 0,
                },
                step,
            )

    vmetrics_history = []
    best_idx = -1
    n_evals = 0
    stop_requested = False
    last_vmetrics = None
    train_eta = ETA(n=config.max_epochs, limit=config.max_seconds)
    micro_step = int(step * grad_accum_steps)

    def run_validation():
        nonlocal best_idx, n_evals, stop_requested, last_vmetrics

        try:
            vmetrics = compute_metrics(
                model,
                loss_fn,
                valid_iter,
                n_valid_batches,
                device,
                amp_enabled=amp_enabled,
                amp_dtype=config.amp_dtype,
                force_fp32=config.force_fp32_validation,
                fail_on_non_finite=config.fail_on_non_finite,
                metric_mode=metric_mode,
                classification_threshold=config.classification_threshold,
            )
        except FloatingPointError:
            _emergency_save_on_error()
            raise
        last_vmetrics = vmetrics
        vmetrics_history.append(vmetrics)
        n_evals += 1
        score = _validation_score(vmetrics)

        if best_idx < 0 or score >= _validation_score(vmetrics_history[best_idx]):
            best_idx = len(vmetrics_history) - 1
            save_state(best=True)
        if not save_best_only:
            save_state(best=False)

        if metric_mode == "classification":
            _safe_wandb_log(
                {
                    "valid/loss": float(vmetrics["loss"]),
                    "valid/auprc": float(vmetrics["auprc"]),
                    "valid/auroc": float(vmetrics["auroc"]),
                    "valid/f1": float(vmetrics["f1"]),
                    "valid/accuracy": float(vmetrics["accuracy"]),
                    "valid/balanced_accuracy": float(vmetrics["balanced_accuracy"]),
                    "valid/precision": float(vmetrics["precision"]),
                    "valid/recall": float(vmetrics["recall"]),
                    "epoch": epoch,
                },
                step,
            )
            logging.warning(
                f"Validation {n_evals} at step {step}: "
                f"loss={vmetrics['loss']:.6f} "
                f"auprc={vmetrics['auprc']:.4f} "
                f"auroc={vmetrics['auroc']:.4f} "
                f"f1={vmetrics['f1']:.4f} "
                f"acc={vmetrics['accuracy']:.4f}"
                + (" ...nice" if best_idx == len(vmetrics_history) - 1 else f" ...best was at validation {best_idx+1}")
            )
        else:
            _safe_wandb_log(
                {
                    "valid/loss": float(vmetrics["loss"]),
                    "valid/pearson_r": float(vmetrics["pearson_r"]),
                    "valid/r_squared": float(vmetrics["r_squared"]),
                    "epoch": epoch,
                },
                step,
            )
            logging.warning(
                f"Validation {n_evals} at step {step}: loss={vmetrics['loss']:.6f} r={vmetrics['pearson_r']:.4f} r2={vmetrics['r_squared']:.4f}"
                + (" ...nice" if best_idx == len(vmetrics_history) - 1 else f" ...best was at validation {best_idx+1}")
            )

        if config.patience is not None and len(vmetrics_history) - best_idx > config.patience + 1:
            logging.warning("Patience exceeded")
            stop_requested = True

    while True:
        if config.max_epochs and epoch >= config.max_epochs:
            logging.warning("Max epochs reached")
            break

        epoch_eta = ETA(n=n_train_batches)
        train_loss = 0.0
        n_train_seqs = 0
        epoch_data_iter = next(train_iter)
        if resume_skip_batches > 0:
            logging.warning(f"Skipping {resume_skip_batches} already-processed batches to align resumed step.")
            epoch_data_iter = islice(epoch_data_iter, resume_skip_batches, None)
            resume_skip_batches = 0
        batch_iter = enumerate(islice(epoch_data_iter, max(n_train_batches, 0)))
        optimizer.zero_grad(set_to_none=True)
        accum_counter = 0

        for i, (x, y) in batch_iter:
            model.train()
            x = torch.tensor(x, device=device)
            y = torch.tensor(y, device=device)

            gen = torch.Generator(device=device)
            gen.manual_seed(micro_step)
            x, y, _rev, _shift = stochastic_revcomp_batch(x, y, strand_pair, max_shift=config.max_shift, generator=gen)

            current_lr = _compute_learning_rate(
                step=step,
                config=config,
                total_train_steps=total_train_steps,
                schedule_override=effective_lr_schedule,
            )
            for pg in optimizer.param_groups:
                pg["lr"] = current_lr

            with autocast_ctx:
                y_pred = model(x)
                loss = loss_fn(y_pred, y)
                loss_for_backward = loss / float(grad_accum_steps)
            if config.fail_on_non_finite:
                try:
                    _assert_finite_tensor("train/y_pred", y_pred, step)
                    _assert_finite_tensor("train/loss", loss, step)
                except FloatingPointError:
                    _emergency_save_on_error()
                    raise
            update_applied = False
            if scaler is not None:
                scaler.scale(loss_for_backward).backward()
            else:
                loss_for_backward.backward()

            accum_counter += 1
            micro_step += 1
            is_last_microbatch = (i + 1) >= max(n_train_batches, 0)
            should_step = (accum_counter % grad_accum_steps == 0) or is_last_microbatch
            if should_step:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                    if config.fail_on_non_finite:
                        try:
                            _assert_finite_gradients(model.parameters(), step)
                        except FloatingPointError:
                            _emergency_save_on_error()
                            raise
                    _clip_by_block_rms(model.parameters(), config.block_clip)
                    _clip_by_global_norm(model.parameters(), config.global_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    if config.fail_on_non_finite:
                        try:
                            _assert_finite_gradients(model.parameters(), step)
                        except FloatingPointError:
                            _emergency_save_on_error()
                            raise
                    _clip_by_block_rms(model.parameters(), config.block_clip)
                    _clip_by_global_norm(model.parameters(), config.global_clip)
                    optimizer.step()
                if config.fail_on_non_finite and config.check_batchnorm_buffers:
                    try:
                        _assert_finite_batchnorm_buffers(model, step)
                    except FloatingPointError:
                        _emergency_save_on_error()
                        raise
                optimizer.zero_grad(set_to_none=True)
                step += 1
                update_applied = True

            batch_size = x.shape[0]
            n_train_seqs += batch_size
            train_loss += loss.item() * batch_size

            used_gb = psutil.virtual_memory().used / 1024 / 1024 / 1024
            if metric_mode == "classification":
                batch_cls = binary_classification_metrics_from_logits(
                    y_pred,
                    y,
                    threshold=config.classification_threshold,
                )
                logging.warning(
                    f"Epoch {epoch+1} batch {i+1}/{n_train_batches} (size {batch_size}) loss: {loss.item():.6f}, "
                    f"acc: {batch_cls['accuracy']:.4f}, f1: {batch_cls['f1']:.4f}, "
                    f"used {used_gb:.2f} Gb, ETA {epoch_eta(i)}"
                )
                if update_applied and step % wandb_log_every == 0:
                    _safe_wandb_log(
                        {
                            "train/loss": float(loss.item()),
                            "train/accuracy": float(batch_cls["accuracy"]),
                            "train/f1": float(batch_cls["f1"]),
                            "train/precision": float(batch_cls["precision"]),
                            "train/recall": float(batch_cls["recall"]),
                            "train/lr": float(optimizer.param_groups[0]["lr"]),
                            "train/batch_size": int(batch_size),
                            "epoch": epoch + 1,
                        },
                        step,
                    )
            else:
                batch_moments = compute_xy_moments(y_pred, y)
                R = pearson_r(batch_moments)
                R2 = r_squared(batch_moments)
                logging.warning(
                    f"Epoch {epoch+1} batch {i+1}/{n_train_batches} (size {batch_size}) loss: {loss.item():.6f}, r: {R:.4f}, r2: {R2:.4f}, used {used_gb:.2f} Gb, ETA {epoch_eta(i)}"
                )
                if update_applied and step % wandb_log_every == 0:
                    _safe_wandb_log(
                        {
                            "train/loss": float(loss.item()),
                            "train/pearson_r": float(R),
                            "train/r_squared": float(R2),
                            "train/lr": float(optimizer.param_groups[0]["lr"]),
                            "train/batch_size": int(batch_size),
                            "epoch": epoch + 1,
                        },
                        step,
                    )
            if update_applied and eval_every_steps is not None and step % eval_every_steps == 0:
                run_validation()
                if stop_requested:
                    break

        epoch += 1
        train_loss = train_loss / max(n_train_seqs, 1)
        _safe_wandb_log({"train/epoch_loss": float(train_loss), "epoch": epoch}, step)
        if eval_every_steps is None:
            run_validation()

        if last_vmetrics is not None:
            if metric_mode == "classification":
                logging.warning(
                    f"Epoch {epoch}: time {epoch_eta.lapsed()} (total {train_eta.lapsed()}), running-total training loss={train_loss:.6f}, "
                    f"latest validation loss={last_vmetrics['loss']:.6f} "
                    f"auprc={last_vmetrics['auprc']:.4f} auroc={last_vmetrics['auroc']:.4f} "
                    f"f1={last_vmetrics['f1']:.4f} acc={last_vmetrics['accuracy']:.4f}"
                )
            else:
                logging.warning(
                    f"Epoch {epoch}: time {epoch_eta.lapsed()} (total {train_eta.lapsed()}), running-total training loss={train_loss:.6f}, "
                    f"latest validation loss={last_vmetrics['loss']:.6f} r={last_vmetrics['pearson_r']:.4f} r2={last_vmetrics['r_squared']:.4f}"
                )
        else:
            logging.warning(
                f"Epoch {epoch}: time {epoch_eta.lapsed()} (total {train_eta.lapsed()}), running-total training loss={train_loss:.6f}, no validation yet"
            )

        if stop_requested:
            break

        if train_eta.past_limit():
            logging.warning("Max wall clock time exceeded")
            break

    if n_evals == 0:
        logging.warning("No periodic validation was triggered; running a final validation pass.")
        run_validation()

    logging.warning(
        f"Training finished after {epoch} epochs, {step} batches, {train_eta.lapsed()} elapsed time"
    )
    if wandb_run is not None:
        try:
            wandb_run.summary["train/epochs_completed"] = int(epoch)
            wandb_run.summary["train/steps_completed"] = int(step)
        except Exception as exc:
            logging.warning(f"W&B summary update failed: {exc}")
    return model
