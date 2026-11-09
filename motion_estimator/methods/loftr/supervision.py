"""Scatterer tracks to LoFTR's training targets.

LoFTR is supervised on its coarse assignment matrix: for each cell of the 1/8
grid in image 0, which cell of image 1 it corresponds to. The simulated dataset
gives exact positions for ~20 k myocardial scatterers per frame, so the
correspondence is known -- but only *where the scatterers are*. That is the
whole design constraint here:

* **Supervision is myocardium-only.** A coarse cell with no scatterer inside it
  gets no target and is excluded from the loss, rather than being labelled
  "no match". The blood pool and far field genuinely have no ground truth, and
  training them as negatives would teach the matcher that untextured regions
  must not match -- which is not what the data says.
* **A cell's displacement is the mean of the scatterers in it**, not a fitted
  model. Over one 8 px cell the deformation is close to constant, and the
  scatterers are dense enough (~20 k over the wall) that the mean is well
  determined where it is determined at all.

``kornia``'s ``CoarseMatching`` in training mode also reads ``spv_b_ids`` /
``spv_i_ids`` / ``spv_j_ids`` and will raise without them, so those are built
here too rather than left to the training loop.
"""
from __future__ import annotations

import torch

__all__ = ["coarse_supervision", "fine_supervision"]


def coarse_supervision(
    points0: torch.Tensor,
    points1: torch.Tensor,
    valid: torch.Tensor,
    hw0_c: tuple[int, int],
    hw1_c: tuple[int, int],
    scale: int = 8,
) -> dict:
    """Ground-truth coarse assignment from tracked point pairs.

    Parameters
    ----------
    points0, points1:
        ``(batch, n_points, 2)`` of **0-based** ``(x, y)`` pixel positions of
        the same scatterers in the two frames, in crop coordinates.
    valid:
        ``(batch, n_points)`` bool -- points actually inside both crops.
    hw0_c, hw1_c:
        Coarse grid sizes, i.e. the image size divided by *scale*.

    Returns
    -------
    A dict with ``conf_matrix_gt`` ``(batch, L, S)``, the ``spv_*_ids`` kornia
    requires, and ``cell_mask0`` ``(batch, L)`` marking which cells carry
    supervision at all.
    """
    batch = points0.shape[0]
    h0, w0 = hw0_c
    h1, w1 = hw1_c
    length, span = h0 * w0, h1 * w1
    device = points0.device

    # Which coarse cell each point falls in, at either end.
    cell0_x = (points0[..., 0] / scale).floor().long()
    cell0_y = (points0[..., 1] / scale).floor().long()
    cell1_x = (points1[..., 0] / scale).floor().long()
    cell1_y = (points1[..., 1] / scale).floor().long()

    inside = (
        valid
        & (cell0_x >= 0) & (cell0_x < w0) & (cell0_y >= 0) & (cell0_y < h0)
        & (cell1_x >= 0) & (cell1_x < w1) & (cell1_y >= 0) & (cell1_y < h1)
    )

    index0 = cell0_y * w0 + cell0_x
    index1 = cell1_y * w1 + cell1_x

    # A cell can hold several scatterers moving to different cells. Take the
    # most popular target per cell -- a majority vote, not an average of cell
    # indices, which would land on a cell nothing actually moved to.
    conf_matrix_gt = torch.zeros(batch, length, span, device=device)
    votes = torch.zeros(batch, length, span, device=device)
    batch_index = (
        torch.arange(batch, device=device)[:, None].expand_as(index0)[inside]
    )
    votes.index_put_(
        (batch_index, index0[inside], index1[inside]),
        torch.ones(int(inside.sum()), device=device),
        accumulate=True,
    )

    support = votes.sum(dim=2)                    # (batch, L)
    cell_mask0 = support > 0
    best = votes.argmax(dim=2)                    # (batch, L)

    b_ids, i_ids = torch.where(cell_mask0)
    j_ids = best[b_ids, i_ids]
    conf_matrix_gt[b_ids, i_ids, j_ids] = 1.0

    return {
        "conf_matrix_gt": conf_matrix_gt,
        "cell_mask0": cell_mask0,
        "spv_b_ids": b_ids,
        "spv_i_ids": i_ids,
        "spv_j_ids": j_ids,
    }


def fine_supervision(
    points0: torch.Tensor,
    points1: torch.Tensor,
    valid: torch.Tensor,
    b_ids: torch.Tensor,
    i_ids: torch.Tensor,
    hw0_c: tuple[int, int],
    scale: int = 8,
    window: int = 5,
) -> torch.Tensor:
    """Sub-cell offsets for the fine level, normalised to ``[-1, 1]``.

    LoFTR's fine head regresses a position inside a *window*-sized patch around
    the coarse match, expressed in normalised coordinates. The target is the
    mean position of the scatterers belonging to each matched cell, measured
    relative to the cell the coarse level chose.

    Returns ``(len(b_ids), 2)`` aligned with the coarse matches kornia kept, and
    ``nan`` for matches with no scatterer support -- the loss must drop those.
    """
    h0, w0 = hw0_c
    device = points0.device

    cell0_x = (points0[..., 0] / scale).floor().long()
    cell0_y = (points0[..., 1] / scale).floor().long()
    inside = (
        valid
        & (cell0_x >= 0) & (cell0_x < w0) & (cell0_y >= 0) & (cell0_y < h0)
    )
    index0 = cell0_y * w0 + cell0_x

    batch, _ = index0.shape
    length = h0 * w0
    total_x = torch.zeros(batch, length, device=device)
    total_y = torch.zeros(batch, length, device=device)
    count = torch.zeros(batch, length, device=device)

    batch_index = (
        torch.arange(batch, device=device)[:, None].expand_as(index0)[inside]
    )
    flat = index0[inside]
    total_x.index_put_((batch_index, flat), points1[..., 0][inside], accumulate=True)
    total_y.index_put_((batch_index, flat), points1[..., 1][inside], accumulate=True)
    count.index_put_(
        (batch_index, flat), torch.ones(int(inside.sum()), device=device), accumulate=True
    )

    mean_x = total_x[b_ids, i_ids] / count[b_ids, i_ids].clamp(min=1)
    mean_y = total_y[b_ids, i_ids] / count[b_ids, i_ids].clamp(min=1)
    has_support = count[b_ids, i_ids] > 0

    # Offset from the centre of the matched cell, in units of half the window.
    cell_x = (mean_x / scale).floor() * scale + scale / 2
    cell_y = (mean_y / scale).floor() * scale + scale / 2
    half = window * scale / 2 / scale  # window is in fine-level pixels
    offset = torch.stack(
        [(mean_x - cell_x) / (half * scale / 2), (mean_y - cell_y) / (half * scale / 2)],
        dim=-1,
    )
    return torch.where(has_support[:, None], offset, torch.full_like(offset, float("nan")))
