"""
Loss functions for boundary predictor.
"""
import torch
import torch.nn.functional as F


def soft_label_transform(
    r: torch.Tensor,
    alpha: float = 4.0,
    beta: float = 0.0
):
    """
    Soft label transformation.

    Args:
        - r: Input ratios.
        - alpha: Transformation parameter.
        - beta: Transformation parameter.

    Returns:
        - Transformed probabilities.
    """
    log_r = torch.log(r + 1e-6)
    return torch.sigmoid(alpha * (log_r - beta))


def focal_bce_loss(
    logits: torch.Tensor,         # (B, L)
    targets: torch.Tensor,        # same shape, float in [0,1]
    mask: torch.Tensor,           # bool or 0/1, same shape
    pos_weight: torch.Tensor = None,  # (B,1)  or scalar tensor
    gamma: float = 2.0,               # focusing parameter
    reduction: str = "mean"           # "mean", "sum", or "none"
):
    """
    Focal Binary Cross‑Entropy with optional pos_weight and mask.

    • If targets are hard 0/1, acts like standard focal loss.
    • If targets are soft probabilities, acts like focal-BCE for soft labels.

    Args:
        - logits: Logits from the model.
        - targets: Ground truth labels.
        - mask: Padding mask.
        - pos_weight: Positive weight.
        - gamma: Focusing parameter.
        - reduction: Reduction method.

    Returns:
        - Loss tensor.
    """
    # Standard BCE‑with‑logits (per element, no reduction yet)
    bce = F.binary_cross_entropy_with_logits(
        logits,
        targets,
        pos_weight=pos_weight,
        reduction="none"
    )   # shape (B, L)

    # Convert logits to probabilities *once* for the focal factor
    prob = torch.sigmoid(logits)
    # p_t = p when y=1, 1-p when y=0   (handles soft targets too)
    p_t = prob * targets + (1 - prob) * (1 - targets)

    focal_factor = (1.0 - p_t).pow(gamma)        # (B, L)
    loss = focal_factor * bce

    # Apply padding / validity mask
    loss = loss * mask.float()

    if reduction == "mean":
        return loss.sum() / mask.float().sum().clamp_min(1.0)
    elif reduction == "sum":
        return loss.sum()
    else:
        return loss