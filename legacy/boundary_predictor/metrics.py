"""
Metrics for boundary predictor.
"""
import torch


def precision_recall_f1(
    probs: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor | None = None
):
    """
    Precision, recall, and F1 score for binary classification.

    Args:
    - probs: Model probabilities.
    - targets: Ground-truth labels.
    - mask: Optional mask (bool, same shape) to filter positions.

    Returns:
    - precision: float
    - recall: float
    - f1: float
    """
    pred_pos = (probs >= 0.5) & mask        # predicted 1s inside mask
    true_pos = targets & mask      # ground truth 1s inside mask

    tp = (pred_pos & true_pos).sum()
    fp = (pred_pos & ~true_pos).sum()
    fn = (~pred_pos & true_pos).sum()

    precision = tp.float() / (tp + fp).clamp_min(1)
    recall = tp.float() / (tp + fn).clamp_min(1)
    f1_pos = 2 * precision * recall / (precision + recall + 1e-8)

    return precision, recall, f1_pos


def topk_overlap(
    ratios: torch.Tensor,
    logits: torch.Tensor,
    k: int,
    mask: torch.Tensor | None = None
):
    """
    Overlap@k between ground-truth `ratios` and model `logits`.

    • Both tensors may be 1-D (L,) or 2-D (B,L).
    • If `mask` (bool, same shape) is provided, only `True`
      positions are considered for ranking.

    Args:
    - ratios: Ground-truth ratios.
    - logits: Model logits.
    - k: Top-k to consider.
    - mask: Optional mask (bool, same shape) to filter positions.

    Returns:
    - overlap_count : int
    - overlap_rate  : float   # overlap_count / k (adaptive)
    """
    # Flatten batch if present
    ratios = ratios.clone().view(-1)
    logits = logits.clone().view(-1)
    if mask is not None:
        mask = mask.view(-1)
        ratios = ratios[mask]
        logits = logits[mask]

    n = ratios.numel()
    if n == 0:
        return 0, 0.0  # no valid elements

    k = min(k, n)  # make k adaptive

    topk_ratios_idx = torch.topk(ratios, k=k, largest=True).indices
    topk_logits_idx = torch.topk(logits, k=k, largest=True).indices

    # Convert to Python sets for fast intersection
    overlap = len(set(topk_ratios_idx.tolist())
                  .intersection(topk_logits_idx.tolist()))
    return overlap, overlap / float(k)