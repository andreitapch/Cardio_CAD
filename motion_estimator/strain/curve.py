"""Arc length to strain curve -- ``computeStrain.m`` / ``driftCompensation.m``.

Longitudinal strain is the relative change in contour length against the
reference frame::

    strainCurve = 100 * (dl - dl(1)) ./ dl(1)

Note it is ``dl(1)``, the first *valid* frame, not the frame indexed by
``refframe``. ``computeStrain.m`` takes a ``refframe`` argument and the active
body of ``driftCompensation.m`` never reads it -- the commented-out version did.
That is preserved here rather than quietly corrected, because the stored MITT
curves were produced by the code as it actually runs.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from .arclength import contour_arc_length

__all__ = ["StrainResult", "drift_compensation", "strain_from_arc_length", "compute_strain"]


@dataclass
class StrainResult:
    """A strain curve and the two scalars read off it.

    ``value_es`` samples the curve at end-systole; ``value_peak_systolic`` is
    its minimum up to and including that frame. They differ whenever a segment
    keeps shortening after end-systole -- post-systolic shortening -- which is
    why the GE ground truth carries both as separate columns.
    """

    curve: torch.Tensor
    arc_length: torch.Tensor
    value_es: float
    value_peak_systolic: float
    es_frame: int

    @property
    def n_frames(self) -> int:
        return int(self.curve.shape[0])


def drift_compensation(curve: torch.Tensor) -> torch.Tensor:
    """Remove a linear ramp so the curve returns to zero at the last frame.

    ``y = x - x(:,end) * ((1:nframes) - 1) ./ (nframes-1)``.

    Tracking drifts over a cycle, so the contour does not close exactly. This
    distributes the closing error linearly across the cycle. It is a correction
    applied after the fact -- constraining the tracker to close the cycle in the
    first place is the better fix, and is on the plan as a training loss.
    """
    n_frames = curve.shape[-1]
    if n_frames < 2:
        return curve
    ramp = torch.arange(n_frames, dtype=curve.dtype, device=curve.device) / (
        n_frames - 1
    )
    return curve - curve[..., -1:] * ramp


def strain_from_arc_length(
    lengths: torch.Tensor, drift_comp: bool = True
) -> torch.Tensor:
    """``100 * (dl - dl[0]) / dl[0]``, optionally drift-compensated.

    Frames whose arc length is not finite are dropped before the computation
    and restored as NaN afterwards, matching ``mainComputeStrain.m``'s
    ``valid_frames`` handling. Because the reference is the first *valid*
    frame, a leading NaN shifts what the curve is relative to.
    """
    lengths = torch.as_tensor(lengths, dtype=torch.float64)
    valid = torch.isfinite(lengths)
    if not bool(valid.any()):
        return torch.full_like(lengths, float("nan"))

    usable = lengths[valid]
    curve = 100.0 * (usable - usable[0]) / usable[0]
    if drift_comp:
        curve = drift_compensation(curve)

    out = torch.full_like(lengths, float("nan"))
    out[valid] = curve
    return out


def compute_strain(
    points_x,
    points_y,
    es_frame: int,
    drift_comp: bool = True,
    open_contour: bool = True,
    layer: str = "endo",
) -> StrainResult:
    """Tracked contour points to a strain curve -- ``mainComputeStrain.m``.

    *points_x* / *points_y* are ``(n_frames, n_points[, n_layers])`` in
    millimetres. *es_frame* is 1-based, as MATLAB and the MITT files store it.
    """
    lengths = contour_arc_length(points_x, points_y, open_contour, layer)
    curve = strain_from_arc_length(lengths, drift_comp)

    n_frames = int(curve.shape[0])
    if not 1 <= es_frame <= n_frames:
        raise ValueError(
            f"es_frame {es_frame} outside 1..{n_frames} (it is 1-based)"
        )

    value_es = curve[es_frame - 1]
    systole = curve[:es_frame]
    finite = systole[torch.isfinite(systole)]
    # min(strainCurve(1:ED)) -- the most negative value reached by end-systole.
    peak = finite.min() if finite.numel() else torch.tensor(float("nan"))

    return StrainResult(
        curve=curve,
        arc_length=lengths,
        value_es=float(value_es),
        value_peak_systolic=float(peak),
        es_frame=es_frame,
    )
