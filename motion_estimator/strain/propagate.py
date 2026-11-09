"""Move contour points along a flow field -- ``interpPoints`` in
``loadPredictedfloInvivo.m``.

The MATLAB samples the displacement field at the current point positions with
``interp2(..., 'linear')`` and adds it::

    [X, Y] = meshgrid(1:size(predOfX, 2), 1:size(predOfX, 1));
    dispX = interp2(X, Y, predOfX, xPoints, yPoints, 'linear');
    nextPointX = xPoints + dispX;

Two conventions carried over deliberately. Coordinates are **1-based** pixel
indices, with ``x`` indexing columns and ``y`` rows. And points that leave the
image return NaN rather than a clamped edge value, which is what ``interp2``
does -- a point that has left the field of view should be visibly missing, not
silently pinned to the border.

This module is only needed for trackers that emit dense flow. A point tracker
returns trajectories directly and skips it entirely, which is the argument for
preferring one.
"""
from __future__ import annotations

import torch

__all__ = ["sample_bilinear", "advect_points", "propagate_points"]


def sample_bilinear(field, x, y) -> torch.Tensor:
    """Bilinear sample of *field* at 1-based ``(x=column, y=row)`` positions.

    *field* is ``(height, width)``; *x* and *y* are any matching shape. Points
    outside the image come back NaN.
    """
    field = torch.as_tensor(field, dtype=torch.float64)
    x = torch.as_tensor(x, dtype=torch.float64)
    y = torch.as_tensor(y, dtype=torch.float64)
    if field.ndim != 2:
        raise ValueError(f"expected a 2-D field, got {tuple(field.shape)}")
    if x.shape != y.shape:
        raise ValueError(f"shape mismatch: x {tuple(x.shape)} vs y {tuple(y.shape)}")

    height, width = field.shape
    # MATLAB's meshgrid starts at 1; drop to 0-based for indexing.
    col = x - 1.0
    row = y - 1.0
    inside = (col >= 0) & (col <= width - 1) & (row >= 0) & (row <= height - 1)

    col_clamped = col.clamp(0, width - 1)
    row_clamped = row.clamp(0, height - 1)
    col0 = col_clamped.floor().long().clamp(0, max(width - 2, 0))
    row0 = row_clamped.floor().long().clamp(0, max(height - 2, 0))
    col1 = (col0 + 1).clamp(max=width - 1)
    row1 = (row0 + 1).clamp(max=height - 1)

    weight_col = col_clamped - col0.to(col_clamped.dtype)
    weight_row = row_clamped - row0.to(row_clamped.dtype)

    top = field[row0, col0] * (1 - weight_col) + field[row0, col1] * weight_col
    bottom = field[row1, col0] * (1 - weight_col) + field[row1, col1] * weight_col
    values = top * (1 - weight_row) + bottom * weight_row

    return torch.where(inside, values, torch.full_like(values, float("nan")))


def advect_points(flow, x, y) -> tuple[torch.Tensor, torch.Tensor]:
    """One step: sample *flow* at ``(x, y)`` and add the displacement.

    *flow* is ``(height, width, 2)`` with the x-displacement first.
    """
    flow = torch.as_tensor(flow, dtype=torch.float64)
    if flow.ndim != 3 or flow.shape[-1] != 2:
        raise ValueError(f"expected (height, width, 2), got {tuple(flow.shape)}")
    return (
        x + sample_bilinear(flow[..., 0], x, y),
        y + sample_bilinear(flow[..., 1], x, y),
    )


def propagate_points(
    flows,
    start_x,
    start_y,
    propagate_error: bool = True,
    reference_x=None,
    reference_y=None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Track a contour across a clip, one frame pair at a time.

    *flows* is ``(n_frames - 1, height, width, 2)``; the returned tracks are
    ``(n_frames, n_points)`` with frame 0 holding the given annotation.

    ``propagate_error`` reproduces the MATLAB's ``opts.propagated`` switch.
    When true, each step starts from the previous *prediction*, so tracking
    error compounds across the cycle -- the realistic setting, and the one the
    in-vivo pipeline uses. When false, each step restarts from the reference
    positions, which isolates single-step accuracy but needs ground truth for
    every frame and so cannot be run at inference.
    """
    flows = torch.as_tensor(flows, dtype=torch.float64)
    if flows.ndim != 4 or flows.shape[-1] != 2:
        raise ValueError(
            f"expected (n_frames-1, height, width, 2), got {tuple(flows.shape)}"
        )

    x = torch.as_tensor(start_x, dtype=torch.float64).clone()
    y = torch.as_tensor(start_y, dtype=torch.float64).clone()

    if not propagate_error:
        if reference_x is None or reference_y is None:
            raise ValueError(
                "propagate_error=False restarts from the reference contour each "
                "frame, so reference_x and reference_y are required"
            )
        reference_x = torch.as_tensor(reference_x, dtype=torch.float64)
        reference_y = torch.as_tensor(reference_y, dtype=torch.float64)

    tracks_x = [x]
    tracks_y = [y]
    for step in range(flows.shape[0]):
        source_x, source_y = (x, y) if propagate_error else (
            reference_x[step],
            reference_y[step],
        )
        x, y = advect_points(flows[step], source_x, source_y)
        tracks_x.append(x)
        tracks_y.append(y)

    return torch.stack(tracks_x), torch.stack(tracks_y)
