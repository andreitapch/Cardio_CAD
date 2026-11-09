"""The one thing every motion estimator must provide.

Everything downstream -- chaining across a cycle, strain, evaluation -- is
written against this and does not know which method produced the numbers.

The signature takes **the whole clip and two frame indices**, not two images.
That is deliberate and was forced by MEMFOF, which is multi-frame: it reads
three frames to predict one flow, so a two-image signature cannot express it.
Two-frame methods simply ignore the rest of the clip. Handing over the clip
also lets an estimator cache per-clip work -- a feature pyramid, a normalisation
range -- which a pairwise signature hides.

What is returned is the *displacement at named points*, not a match list and
not a dense field, because that is the only thing the methods have in common:

* a sparse matcher decides its own correspondence locations and must
  interpolate to reach ours;
* a dense method has a value at every pixel and must sample.

``support`` is part of the contract rather than an extra. A method that cannot
answer for a point must say so: an error averaged over only the points that
survived flatters whichever method drops the hard ones, which is exactly how
``strided`` briefly appeared to beat ``reference`` in vivo. A dense method has
no natural way to decline, so it has to manufacture one -- a sector mask, a
forward-backward consistency check, or a learned per-pixel uncertainty where
the architecture provides it.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

__all__ = ["MotionEstimator"]


@runtime_checkable
class MotionEstimator(Protocol):
    """Displacement of named points between two frames of a clip."""

    name: str

    def displacement(
        self,
        frames: np.ndarray,
        source: int,
        target: int,
        x: np.ndarray,
        y: np.ndarray,
        vrange: tuple[float, float] | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``(dx, dy, support)`` at 1-based ``(x, y)`` on frame *source*.

        *frames* is ``(n_frames, rows, cols)``; *source* and *target* index it.
        A multi-frame model may read any frames it likes; a two-frame one reads
        only these.

        *dx* / *dy* are NaN where the method has no answer. *support* is an
        integer whose meaning is method-specific -- matches used, pixels
        sampled, consistency tier -- but whose zero always means "nothing
        backed this".
        """
        ...
