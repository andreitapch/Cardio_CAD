"""Losses for fine-tuning LoFTR on exact correspondence.

Coarse is a focal loss on the dual-softmax assignment matrix -- LoFTR's own
formulation -- restricted to the cells that carry ground truth. The restriction
is the whole point: see ``supervision.py`` for why unsupervised cells must not
be trained as negatives.

Fine is an L2 on the sub-cell offset, weighted by the inverse of the model's
own predicted spread, so confidently wrong refinements cost more than uncertain
ones. That is LoFTR's weighting, kept rather than replaced.
"""
from __future__ import annotations

import torch

__all__ = ["coarse_focal_loss", "fine_l2_loss"]


def coarse_focal_loss(
    conf_matrix: torch.Tensor,
    conf_matrix_gt: torch.Tensor,
    cell_mask0: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
    negative_weight: float = 1.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Focal loss over supervised rows of the assignment matrix.

    *cell_mask0* is ``(batch, L)``: rows with no scatterer support are dropped
    entirely, positives and negatives alike.
    """
    conf = conf_matrix.clamp(eps, 1 - eps)
    rows = cell_mask0[..., None].expand_as(conf)

    positive = (conf_matrix_gt > 0) & rows
    negative = (conf_matrix_gt == 0) & rows

    if not bool(positive.any()):
        return conf.sum() * 0.0  # keeps the graph alive on a degenerate batch

    loss_positive = -alpha * (1 - conf[positive]) ** gamma * conf[positive].log()
    loss_negative = (
        -(1 - alpha) * conf[negative] ** gamma * (1 - conf[negative]).log()
    )
    return loss_positive.mean() + negative_weight * loss_negative.mean()


def fine_l2_loss(
    expec_f: torch.Tensor, expec_f_gt: torch.Tensor, use_std: bool = True
) -> torch.Tensor:
    """L2 on the fine offset, inverse-variance weighted.

    ``expec_f`` is ``(M, 3)`` -- predicted ``(x, y)`` plus the standard
    deviation of the heatmap it came from. Targets that are NaN (no scatterer
    support for that match) are dropped rather than counted as zero error.
    """
    if expec_f.numel() == 0 or expec_f_gt.numel() == 0:
        return expec_f.sum() * 0.0

    finite = torch.isfinite(expec_f_gt).all(dim=-1)
    # Only supervise inside the window the fine head can actually reach.
    finite &= (expec_f_gt.abs() <= 1.0).all(dim=-1)
    if not bool(finite.any()):
        return expec_f.sum() * 0.0

    predicted = expec_f[finite, :2]
    target = expec_f_gt[finite]
    error = ((predicted - target) ** 2).sum(dim=-1)

    if use_std:
        std = expec_f[finite, 2]
        weight = (1.0 / std.clamp(min=1e-10)).detach()
        weight = weight / weight.mean().clamp(min=1e-10)
        return (error * weight).mean()
    return error.mean()
