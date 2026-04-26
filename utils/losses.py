import torch

EPSILON = 1e-5


def poisson_loss(y_pred, y_true, epsilon=EPSILON):
    if y_pred.shape != y_true.shape:
        raise ValueError(f"predicted shape {y_pred.shape} != true shape {y_true.shape}")
    y_pred = y_pred + epsilon
    y_true = y_true + epsilon
    return torch.mean(y_pred - y_true * torch.log(y_pred))


def poisson_multinomial_loss(y_pred, y_true, epsilon=EPSILON, total_weight=1.0, rescale=False):
    seq_len = y_pred.shape[-2]
    y_pred = y_pred + epsilon
    y_true = y_true + epsilon
    s_pred = torch.sum(y_pred, dim=-2, keepdim=True)
    s_true = torch.sum(y_true, dim=-2, keepdim=True)
    p_loss = poisson_loss(s_pred, s_true, epsilon=0.0) / seq_len
    m_loss = -torch.mean(y_true * torch.log(y_pred / s_pred))
    if rescale:
        return (m_loss + total_weight * p_loss) * (2 / (1 + total_weight))
    return m_loss + total_weight * p_loss
