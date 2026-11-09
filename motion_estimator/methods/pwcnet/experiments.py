"""The paper's training setups, one entry per test.

Pulido et al., *Impact of training data composition on deep learning-based
cardiac motion estimation* (OJ-UFFC), Table 1. To add a test, add a line to
``TESTS``; ``build_splits`` turns every entry into exact pair lists.

**Counts are consecutive frame pairs**, as in Table 1, even though the paper
calls them frames. This was checked against the paper's own training tree:
236 in vivo recordings hold 14,169 frames, and 14,169 - 236 = 13,933 = IV100.

``ms`` and ``iv`` are the paper's numbers, kept as printed so the table can be
checked against the manuscript. How they become pairs today:

* **In vivo**: the same pool as the paper (Retest + Test, 118 patients, 236
  recordings, 13,933 pairs), so ``iv / PAPER_IV_POOL`` of it lands on the
  paper's count.
* **Synthetic**: the same *fraction* of the new simulated pool,
  ``ms / PAPER_MS_POOL``, not the same count -- the new pool is ~1.8x larger.
  That drops two equalities the paper designed in: MS30_IV50 is no longer
  balanced, and MS100_IV0 no longer matches MS47_IV85 in total size. Give a
  test ``ms_pairs`` to pin an absolute count instead.
"""
from __future__ import annotations

from dataclasses import dataclass

__all__ = ["Test", "TESTS", "BY_NAME", "CV", "IN_VIVO", "SUS",
           "PAPER_MS_POOL", "PAPER_IV_POOL", "N_FOLDS", "VAL_FRACTION", "SEED"]

PAPER_MS_POOL = 22_405  # MS100_IV0: what the paper called 100% synthetic
PAPER_IV_POOL = 13_933  # MS0_IV100: the whole in vivo pool

# What a test is evaluated on, which decides how its in vivo data is split.
CV = "cv"            # N_FOLDS folds by in vivo patient, tested on each fold's held-out patients
IN_VIVO = "in_vivo"  # one model, no in vivo training, tested on every in vivo patient
SUS = "sus"          # one model, trained on all in vivo, tested on the external SUS dataset

N_FOLDS = 6
VAL_FRACTION = 0.1   # of training patients, on each side, used only to pick best.pt
SEED = 0


@dataclass(frozen=True)
class Test:
    name: str
    ms: int                  # synthetic pairs in the paper
    iv: int                  # in vivo pairs in the paper
    protocol: str
    description: str = ""
    ms_pairs: int | None = None  # absolute synthetic count, overriding the pool fraction

    @property
    def ms_fraction(self) -> float:
        return self.ms / PAPER_MS_POOL

    @property
    def iv_fraction(self) -> float:
        return self.iv / PAPER_IV_POOL


TESTS = [
    Test("MS0_IV85",       0, 11_785, CV,      "in vivo only"),
    Test("MS12_IV85",  2_645, 11_785, CV,      "mixed"),
    Test("MS23_IV85",  5_292, 11_785, CV,      "mixed"),
    Test("MS35_IV85",  7_937, 11_785, CV,      "mixed"),
    Test("MS47_IV85", 10_620, 11_785, CV,      "mixed, best in the paper"),
    Test("MS30_IV50",  6_612,  6_612, CV,      "balanced MS/IV"),
    Test("MS47_IV0",  10_620,      0, IN_VIVO, "synthetic only"),
    Test("MS100_IV0", 22_405,      0, IN_VIVO, "synthetic only"),
    Test("MS0_IV100",      0, 13_933, SUS,     "generalization"),
    Test("MS60_IV100", 13_225, 13_933, SUS,    "generalization"),
]

BY_NAME = {test.name: test for test in TESTS}
if len(BY_NAME) != len(TESTS):
    raise ValueError("test names must be unique")
