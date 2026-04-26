import torch

EPSILON = 1e-5


def compute_xy_moments(x, y, weights=None):
    if x.shape != y.shape:
        raise ValueError(f"shape of predicted values {x.shape} != shape of true values {y.shape}")

    axis = tuple(range(x.ndim - 1))
    if weights is not None:
        x = x * weights
        y = y * weights
        n = torch.ones(x.shape[-1], device=x.device) * torch.sum(weights, dim=axis)
    elif x.ndim > 1:
        n = torch.ones(x.shape[-1], device=x.device) * torch.prod(torch.tensor(x.shape[:-1], device=x.device))
    else:
        n = torch.tensor(x.shape[0], device=x.device)

    ex = torch.sum(x, dim=axis)
    ey = torch.sum(y, dim=axis)
    exx = torch.sum(x * x, dim=axis)
    eyy = torch.sum(y * y, dim=axis)
    exy = torch.sum(x * y, dim=axis)

    result = torch.stack((n, ex, ey, exx, eyy, exy), dim=-1)
    return result


def _validate_xy_moments_shape(xy_moments):
    if xy_moments.shape[-1] != 6:
        raise ValueError("expected final dimension of inputs to be 6 (n, xSum, ySum, xxSum, yySum, xySum)")


def _sum_xy_moments(xy_moments):
    _validate_xy_moments_shape(xy_moments)
    return torch.sum(xy_moments, dim=tuple(range(xy_moments.ndim - 1)))


def _epsilon_where_zero(x):
    return torch.where(x == 0, torch.tensor(EPSILON, device=x.device), torch.tensor(0.0, device=x.device))


def _safe_divide(x, y):
    pc = _epsilon_where_zero(y)
    return x / (y + pc)


def pearson_r(xy_moments, keep_features=False):
    _validate_xy_moments_shape(xy_moments)
    if not keep_features:
        xy_moments = _sum_xy_moments(xy_moments)
    n, sx, sy, sxx, syy, sxy = xy_moments.transpose(0, -1)
    ex = sx / n
    ey = sy / n
    exx = sxx / n
    eyy = syy / n
    exy = sxy / n
    return _safe_divide(exy - ex * ey, torch.sqrt((exx - ex ** 2) * (eyy - ey ** 2)))


def r_squared(xy_moments, keep_features=False):
    _validate_xy_moments_shape(xy_moments)
    if not keep_features:
        xy_moments = _sum_xy_moments(xy_moments)
    n, _sx, sy, sxx, syy, sxy = xy_moments.transpose(0, -1)
    ey = sy / n
    ss_res = sxx + syy - 2 * sxy
    ss_tot = syy - ey * sy
    return 1 - _safe_divide(ss_res, ss_tot)


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator <= 0.0:
        return 0.0
    return float(numerator) / float(denominator)


def binary_classification_metrics_from_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
):
    """Compute binary classification metrics from raw logits and 0/1 targets."""
    probs = torch.sigmoid(logits.detach().float().reshape(-1)).cpu()
    labels = (targets.detach().float().reshape(-1) >= 0.5).to(torch.int64).cpu()
    n = int(labels.numel())
    if n == 0:
        return {
            "accuracy": float("nan"),
            "balanced_accuracy": float("nan"),
            "precision": float("nan"),
            "recall": float("nan"),
            "f1": float("nan"),
            "auroc": float("nan"),
            "auprc": float("nan"),
            "n_samples": 0,
            "n_positive": 0,
            "n_negative": 0,
        }

    preds = (probs >= float(threshold)).to(torch.int64)
    tp = int(((preds == 1) & (labels == 1)).sum().item())
    tn = int(((preds == 0) & (labels == 0)).sum().item())
    fp = int(((preds == 1) & (labels == 0)).sum().item())
    fn = int(((preds == 0) & (labels == 1)).sum().item())

    n_pos = int((labels == 1).sum().item())
    n_neg = n - n_pos

    precision = _safe_ratio(tp, tp + fp)
    recall = _safe_ratio(tp, tp + fn)
    specificity = _safe_ratio(tn, tn + fp)
    accuracy = _safe_ratio(tp + tn, n)
    balanced_accuracy = 0.5 * (recall + specificity)
    f1 = _safe_ratio(2.0 * precision * recall, precision + recall)

    # Ranking metrics (threshold-independent), undefined for degenerate label sets.
    auroc = float("nan")
    auprc = float("nan")
    if n_pos > 0:
        order = torch.argsort(probs, descending=True)
        y_sorted = labels[order].to(torch.float32)
        tp_cum = torch.cumsum(y_sorted, dim=0)
        fp_cum = torch.cumsum(1.0 - y_sorted, dim=0)

        precision_curve = tp_cum / torch.clamp(tp_cum + fp_cum, min=1e-12)
        recall_curve = tp_cum / float(n_pos)

        recall_ext = torch.cat([torch.zeros(1, dtype=recall_curve.dtype), recall_curve], dim=0)
        precision_ext = torch.cat([torch.ones(1, dtype=precision_curve.dtype), precision_curve], dim=0)
        auprc = float(torch.trapz(precision_ext, recall_ext).item())

        if n_neg > 0:
            tpr = tp_cum / float(n_pos)
            fpr = fp_cum / float(n_neg)
            tpr_ext = torch.cat([torch.zeros(1, dtype=tpr.dtype), tpr], dim=0)
            fpr_ext = torch.cat([torch.zeros(1, dtype=fpr.dtype), fpr], dim=0)
            auroc = float(torch.trapz(tpr_ext, fpr_ext).item())

    return {
        "accuracy": float(accuracy),
        "balanced_accuracy": float(balanced_accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "auroc": float(auroc),
        "auprc": float(auprc),
        "n_samples": int(n),
        "n_positive": int(n_pos),
        "n_negative": int(n_neg),
    }
