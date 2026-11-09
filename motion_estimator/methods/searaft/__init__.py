"""SEA-RAFT: dense flow with a learned per-pixel uncertainty.

Chosen for three measured reasons, set out in :mod:`net`: dense output fixes
the coverage constraint that limits LoFTR, the mixture-of-Laplace head replaces
our hand-rolled confidence heuristics with one the model was trained to
predict, and its 1-pixel outlier rate is the published metric nearest this
project's actual failure regime.

Upstream code is vendored verbatim in ``third_party/sea_raft`` (BSD-3-Clause);
this package is only the adaptation.
"""
from .net import CONFIGS, DEFAULT_WEIGHTS, SeaRaftEstimator, confidence_from_info

__all__ = ["CONFIGS", "DEFAULT_WEIGHTS", "SeaRaftEstimator", "confidence_from_info"]
