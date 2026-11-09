"""Name to estimator, so switching methods is one flag.

Adding a method means one entry here and one subpackage implementing
:class:`~motion_estimator.methods.base.MotionEstimator`. Nothing else changes:
the chaining, the strain code, the metrics and every script take the estimator
through the interface and never name a method.

Factories are imported lazily, inside the builder, because the dependencies
differ -- LoFTR needs kornia, the flow methods need vendored third-party code
-- and a missing one should break only the method that needs it rather than
the whole registry.

``sector`` is passed to every factory because the dense methods genuinely need
it: they produce a value at every pixel including outside the imaged sector,
where the value is meaningless, and the mask is what lets them say "no answer"
instead. A sparse matcher uses it to discard matches between two regions of
constant zero, which come back confident and wrong.
"""
from __future__ import annotations

__all__ = ["METHODS", "available", "build", "describe"]

METHODS = ("loftr", "searaft", "memfof", "pwcnet")

_DESCRIPTIONS = {
    "loftr": (
        "Semi-dense learned matching plus local-affine interpolation. The "
        "frozen baseline: 1.89 px simulated, 3.22 px in vivo. Sparse (about "
        "640 matches per 500x500 frame), so coverage is its weakness -- 32% of "
        "tracked positions in vivo, every failure at zero support."
    ),
    "searaft": (
        "SEA-RAFT: dense flow, and the only candidate with a learned per-pixel "
        "uncertainty (mixture-of-Laplace head), which replaces the hand-rolled "
        "mask plus consistency-check heuristics. Best published 1-pixel "
        "outlier rate, which is the metric nearest this project's failure."
    ),
    "memfof": (
        "MEMFOF: multi-frame dense flow, memory-efficient by design. Reads "
        "three frames per prediction, so it can exploit the sequence rather "
        "than treating each pair independently."
    ),
    "pwcnet": (
        "PWC-Net (PyTorch rewrite of the old TF1 project): coarse-to-fine dense "
        "flow, greyscale input, trained on the simulated cohort by patient "
        "fold and fine-tuned per in-vivo fold. Pass its best.pt as checkpoint."
    ),
}


def available() -> dict:
    """Which methods can actually be built here, and why not if they cannot."""
    status = {}
    for name in METHODS:
        try:
            _factory(name)
            status[name] = "ready"
        except Exception as error:
            status[name] = f"unavailable: {type(error).__name__}: {error}"
    return status


def describe(name: str) -> str:
    return _DESCRIPTIONS.get(name, "no description")


def _factory(name: str):
    """Import and return the estimator class, raising if it is not installed."""
    if name == "loftr":
        from .loftr.estimator import LoFTREstimator

        return LoFTREstimator
    if name == "searaft":
        from .searaft.net import SeaRaftEstimator

        return SeaRaftEstimator
    if name == "memfof":
        from .memfof.net import MemfofEstimator

        return MemfofEstimator
    if name == "pwcnet":
        from .pwcnet.estimator import PWCNetEstimator

        return PWCNetEstimator
    raise KeyError(f"unknown method {name!r}; known: {', '.join(METHODS)}")


def build(name: str, sector=None, checkpoint=None, **options):
    """An estimator ready to hand to :func:`motion_estimator.tracking.chain`."""
    if name not in METHODS:
        raise KeyError(f"unknown method {name!r}; known: {', '.join(METHODS)}")
    return _factory(name)(sector=sector, checkpoint=checkpoint, **options)
