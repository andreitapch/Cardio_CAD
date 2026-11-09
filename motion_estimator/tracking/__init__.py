"""Method-agnostic tracking: the result type, the chaining, the sampling.

Everything here is written against
:class:`~motion_estimator.methods.base.MotionEstimator` and knows nothing about
any particular estimator. A new method implements ``displacement`` and gets all
four chaining strategies for free.
"""
from .chaining import chain
from .result import TrackResult
from .sampling import sample_field

__all__ = ["TrackResult", "chain", "sample_field"]
