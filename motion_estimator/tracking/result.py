"""The shape every tracker returns.

Shared rather than per method, because strain, the metrics and the notebooks
all consume it and none of them should care which estimator produced it.

``support`` is part of the type for a reason learned the hard way: a tracker
that cannot answer for a point must say so. An error averaged over only the
points that survived flatters whichever method drops the hard ones -- that is
how ``strided`` briefly appeared to beat ``reference`` in vivo.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = ["TrackResult"]


@dataclass
class TrackResult:
    """Trajectories plus the diagnostics needed to judge them.

    ``x`` / ``y`` are ``(n_frames, n_points)`` in **pixels**, 1-based, frame 0
    holding the points as given. ``support`` counts the matches that survived
    weighting and rejection for each point at each step, and is the first thing
    to look at when a track goes wrong: a column of zeros means the matcher
    found nothing near that point, which is a different failure from a
    confidently wrong displacement.
    """

    x: np.ndarray
    y: np.ndarray
    support: np.ndarray
    n_matches: np.ndarray
    pairing: str
    meta: dict = field(default_factory=dict)

    @property
    def n_frames(self) -> int:
        return int(self.x.shape[0])

    @property
    def n_points(self) -> int:
        return int(self.x.shape[1])

    @property
    def lost(self) -> np.ndarray:
        """Per point, the first frame at which it went NaN (-1 if never)."""
        bad = ~np.isfinite(self.x) | ~np.isfinite(self.y)
        any_bad = bad.any(axis=0)
        first = np.where(any_bad, bad.argmax(axis=0), -1)
        return first

    def to_mm(self, pixel_spacing) -> tuple[np.ndarray, np.ndarray]:
        """Scale to millimetres for :mod:`motion_estimator.strain`.

        ``strain.curve.compute_strain`` wants millimetres. *pixel_spacing* is
        ``(x, y)`` -- column spacing first, matching this module's coordinate
        order.

        **Not** ``PreparedClip.pixel_spacing_mm`` as it stands:
        ``image_prep.pixel_spacing_mm`` is ``voxelsize[[1, 0, 2]]``, ordered
        ``(row, col)`` to match the axes of the clinical-view image, so it is
        ``(y, x)`` here. Pass it reversed. Row and column spacing differ across
        this cohort (~0.26-0.37 mm), so confusing the two rescales strain
        rather than merely rounding it.

        The 1-based origin is dropped first: MATLAB's round trip is
        ``(value / spacing) + 1`` on the way in, so ``(index - 1) * spacing``
        on the way out.
        """
        spacing = np.asarray(pixel_spacing, dtype=float).ravel()
        if spacing.size < 2:
            raise ValueError(f"expected (x, y) spacing, got {spacing!r}")
        return (self.x - 1.0) * spacing[0], (self.y - 1.0) * spacing[1]
