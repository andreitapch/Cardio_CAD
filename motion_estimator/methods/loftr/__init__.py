"""LoFTR: semi-dense learned matching, interpolated at the tracked points.

**Kept as the frozen baseline.** It is the only method measured end to end in
this project, so it is what a new method has to beat: 1.89 px endpoint error on
held-out simulated clips, 3.22 px against MITT in vivo, and the gain curve
below. It is not the recommended method -- see the coverage figures -- and
``offset_contour`` / ``track_via_midwall`` have been removed as a measured dead
end (coverage 32.1% -> 34.0% for a 5.2 mm offset, with accuracy monotonically
worse).

LoFTR is **detector-free**, so it will not track a point you name -- it chooses
its own correspondence locations on a coarse 1/8-resolution grid. Hence two
stages rather than one:

* :mod:`matcher`     -- run LoFTR on a frame pair -> (xy0, xy1, confidence)
* :mod:`tracking`    -- interpolate that field at the given points, and chain
* :mod:`supervision` -- exact scatterer tracks -> LoFTR's coarse/fine targets
* :mod:`losses`      -- focal coarse loss, inverse-variance fine loss

Two measured properties of this method that the rest of the package is built
around, both documented in ``TRACKING_EXPERIMENTS.md``:

* **It shrinks small displacements.** The fine level regresses its subpixel
  offset as the spatial expectation of a softmax heatmap, which is biased
  toward the window centre. Gain is 0.798 at a 1 px displacement and 0.987 at
  12 px. Clips here move ~1 px per frame, so consecutive chaining reads strain
  about 20% low.
* **It is sparse.** 600-800 matches over a 500x500 frame, so mean spacing is
  ~14 px and a radius-16 neighbourhood holds about 4 of them. That is what
  starves the interpolation, and it is why
  :mod:`~motion_estimator.methods.learned_flow` exists as an alternative --
  a dense field has no coverage problem by construction.
"""
from ...tracking.result import TrackResult
from .matcher import LoFTRMatcher, Matches, value_range
from .tracking import (
    displacement_at,
    to_dense_flow,
    track_bidirectional,
    track_points,
    track_strided,
)

__all__ = [
    "LoFTRMatcher",
    "Matches",
    "TrackResult",
    "displacement_at",
    "to_dense_flow",
    "track_bidirectional",
    "track_points",
    "track_strided",
    "value_range",
]
