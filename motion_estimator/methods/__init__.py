"""Motion estimators, one subpackage each.

Each method owns its matching, its tracking and its training code, because
those are the parts that differ. What they share is the interface in
:mod:`base`: given two frames and a set of points, return a displacement per
point. Everything downstream -- chaining across a cycle, strain, evaluation --
is written against that interface and does not know which method produced it.

* :mod:`loftr`   -- semi-dense learned matching, interpolated at the points.
  Kept as the **frozen baseline**: it is the only method measured end to end
  here (1.89 px simulated, 3.22 px in vivo, and the gain curve in
  ``TRACKING_EXPERIMENTS.md``), so a new method has something to beat.
* :mod:`searaft` -- SEA-RAFT, dense learned flow with a learned per-pixel
  uncertainty. Planned.
* :mod:`memfof`  -- MEMFOF, multi-frame dense flow, memory-efficient by
  design. Planned.

Plain RAFT was wired in, measured, and removed. It was picked for availability
rather than merit, and measurement did not support the reason it was picked:
its |gain| at 1 px was 0.792 against LoFTR's 0.798, so it shares the
sub-pixel shrinkage exactly. Two architectures with different refinement
mechanisms agreeing to three decimals says the cause is the training
distribution -- natural video, where sub-pixel motion is rare -- not the
mechanism. That is what makes fine-tuning on this project's simulated cohort a
targeted fix rather than a hope.
"""
