"""LoFTR behind the common interface, so the baseline is comparable.

``LoFTRMatcher`` produces matches and ``displacement_at`` interpolates them;
neither satisfies :class:`~motion_estimator.methods.base.MotionEstimator` on
its own. This joins them, so the baseline runs through exactly the same
chaining and metrics as the dense candidates and a comparison is like for like.

It keeps a small per-clip cache of matches. Chaining revisits the same frame
pair -- ``reference`` matches frame 0 against every frame, ``strided`` reuses
its anchors -- and a LoFTR match is the expensive part, unlike a flow field.
"""
from __future__ import annotations

import numpy as np

from .matcher import LoFTRMatcher, value_range
from .tracking import displacement_at

__all__ = ["LoFTREstimator"]


class LoFTREstimator:
    """Sparse matching plus local-affine interpolation, as one estimator.

    ``radius`` defaults to 16 rather than 32 because that was measured better
    in vivo (5.06 -> 3.78 px for ``consecutive``), and it is the single largest
    improvement found for this method. It is still a compromise: at 16 px the
    neighbourhood holds about 4 of LoFTR's ~640 matches, below the neighbour
    floor, which is why coverage is this method's limitation rather than its
    accuracy.
    """

    def __init__(
        self,
        sector=None,
        checkpoint=None,
        weights: str = "outdoor",
        radius: float = 16.0,
        min_confidence: float = 0.2,
        scale: float = 1.0,
        **interp,
    ) -> None:
        self.matcher = LoFTRMatcher(
            weights=weights,
            min_confidence=min_confidence,
            scale=scale,
            mask=sector,
            checkpoint=checkpoint,
        )
        self.radius = radius
        self.interp = interp
        self.checkpoint = checkpoint
        self.name = "loftr" + ("_finetuned" if checkpoint else "")
        self._cache: dict[tuple[int, int], object] = {}
        self._cache_owner: int | None = None

    def displacement(self, frames, source, target, x, y, vrange=None):
        frames = np.asarray(frames)
        # Keyed on the clip's identity: a new clip must not reuse matches from
        # the previous one, and comparing arrays every call would be worse than
        # the cache is worth.
        if self._cache_owner != id(frames):
            self._cache.clear()
            self._cache_owner = id(frames)
        if vrange is None:
            vrange = value_range(frames)

        key = (int(source), int(target))
        if key not in self._cache:
            self._cache[key] = self.matcher.match(
                frames[source], frames[target], vrange
            )
        matches = self._cache[key]

        return displacement_at(matches, x, y, radius=self.radius, **self.interp)
