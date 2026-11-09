"""Measuring a method, and where the numbers go.

* :mod:`metrics` -- the measurements, kept apart from the running of them so a
  metric can be reused without a script.
* ``results/``   -- every evaluation writes here, or into a run directory.

One rule the metrics encode, learned the hard way: **report end-systole
separately from the cycle median.** A median over all frames is dominated by
frames near end-diastole, where motion is small and matching easy; it read
3.65 px on IV02 while the same clip was 7.48 px at end-systole with 8 of 72
points surviving. End-systole is the frame strain is read at, so a median that
hides it is worse than no number.
"""
