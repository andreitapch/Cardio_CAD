"""Which frozen features match echo frames best, measured before any training.

A backbone is probed by matching its features between two frames of a clip
and scoring the match against known motion: exact scatterers on the simulated
cohort, MITT's contour on the in vivo one. See ``scripts/probe_backbones.py``.
"""
