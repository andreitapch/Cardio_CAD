"""Contour arc length per frame -- ``computeArclength.m`` and ``arclength.m``.

``arclength(x, y, 'linear')`` reduces to ``sum(sqrt(sum(diff(data).^2, 2)))``:
the sum of straight-line distances between consecutive contour points. The
MATLAB offers spline and pchip variants; the pipeline only ever asks for
``'linear'``, and that is what is ported here.
"""
from __future__ import annotations

import torch

__all__ = ["LAYERS", "layer_indices", "arc_length", "contour_arc_length"]

LAYERS = ("endo", "mid", "epi", "all")


def layer_indices(layer: str, n_layers: int) -> list[int]:
    """Which myocardial layers a strain type averages over.

    Mirrors the switch in ``computeArclength.m``: ``endo`` is the first layer,
    ``epi`` the last, ``mid`` the middle one -- or the middle *two*, averaged,
    when the layer count is even.
    """
    if layer == "endo":
        return [0]
    if layer == "epi":
        return [n_layers - 1]
    if layer == "mid":
        if n_layers % 2 == 0:
            return [n_layers // 2 - 1, n_layers // 2]
        return [(n_layers - 1) // 2]
    if layer == "all":
        return list(range(n_layers))
    raise ValueError(f"layer must be one of {list(LAYERS)}, got {layer!r}")


def arc_length(points_x, points_y) -> torch.Tensor:
    """Summed chord length along the point axis.

    *points_x* / *points_y* are ``(..., n_points)``; the result drops that axis.
    """
    x = torch.as_tensor(points_x, dtype=torch.float64)
    y = torch.as_tensor(points_y, dtype=torch.float64)
    if x.shape != y.shape:
        raise ValueError(f"shape mismatch: x {tuple(x.shape)} vs y {tuple(y.shape)}")
    if x.shape[-1] < 2:
        raise ValueError(f"need at least 2 points, got {x.shape[-1]}")

    segments = torch.sqrt(
        torch.diff(x, dim=-1) ** 2 + torch.diff(y, dim=-1) ** 2
    )
    return segments.sum(dim=-1)


def contour_arc_length(
    points_x,
    points_y,
    open_contour: bool = True,
    layer: str = "endo",
) -> torch.Tensor:
    """Arc length per frame, for one strain layer.

    *points_x* / *points_y* are ``(n_frames, n_points)`` or
    ``(n_frames, n_points, n_layers)``. Returns ``(n_frames,)``.

    ``open_contour=False`` closes the ring by repeating the first point, which
    is what ``computeArclength.m`` does for circumferential contours. The
    longitudinal LV contour runs base-to-apex-to-base and is open, so the
    pipeline passes ``contour_type = true`` and no point is appended.
    """
    x = torch.as_tensor(points_x, dtype=torch.float64)
    y = torch.as_tensor(points_y, dtype=torch.float64)
    if x.shape != y.shape:
        raise ValueError(f"shape mismatch: x {tuple(x.shape)} vs y {tuple(y.shape)}")
    if x.ndim == 2:
        x, y = x.unsqueeze(-1), y.unsqueeze(-1)
    if x.ndim != 3:
        raise ValueError(
            f"expected (n_frames, n_points[, n_layers]), got {tuple(x.shape)}"
        )

    if not open_contour:
        x = torch.cat([x, x[:, :1]], dim=1)
        y = torch.cat([y, y[:, :1]], dim=1)

    indices = layer_indices(layer, x.shape[2])
    # (n_frames, n_points, n_selected) -> (n_frames, n_selected) -> (n_frames,)
    lengths = arc_length(
        x[:, :, indices].permute(0, 2, 1), y[:, :, indices].permute(0, 2, 1)
    )
    return lengths.mean(dim=-1)
