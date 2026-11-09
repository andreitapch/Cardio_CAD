"""Sampling a dense displacement field at the points we actually track.

Any dense method needs this, and the two conventions it enforces are the ones
that have caused real bugs here: coordinates are **1-based** ``(x=column,
y=row)`` as everywhere else in this package, and a point outside the field --
or outside the sector mask -- comes back NaN rather than clamped to an edge.
"""
from __future__ import annotations

import numpy as np

__all__ = ["sample_field"]


def sample_field(flow, x, y, mask):
    """Bilinear sample of a dense field at 1-based ``(x, y)``.

    Returns ``(dx, dy, inside)``. 1-based in, because that is the convention
    the rest of the package uses; the -1 happens here, once.
    """
    from scipy import ndimage

    rows, cols = flow.shape[:2]
    col = np.asarray(x, dtype=float) - 1.0
    row = np.asarray(y, dtype=float) - 1.0
    inside = (
        np.isfinite(col) & np.isfinite(row)
        & (col >= 0) & (col <= cols - 1) & (row >= 0) & (row <= rows - 1)
    )
    coords = np.stack([np.where(inside, row, 0.0), np.where(inside, col, 0.0)])
    dx = ndimage.map_coordinates(flow[..., 0], coords, order=1, mode="nearest")
    dy = ndimage.map_coordinates(flow[..., 1], coords, order=1, mode="nearest")

    if mask is not None:
        near_row = np.rint(np.where(inside, row, 0.0)).astype(int).clip(0, rows - 1)
        near_col = np.rint(np.where(inside, col, 0.0)).astype(int).clip(0, cols - 1)
        inside &= np.asarray(mask, dtype=bool)[near_row, near_col]

    return np.where(inside, dx, np.nan), np.where(inside, dy, np.nan), inside
