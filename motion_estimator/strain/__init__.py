"""Strain reconstruction from tracked contour points.

A transcription of the MATLAB in ``cardiacMotionEstimation/matlab/gls``, in
torch so the whole chain stays differentiable:

* :mod:`arclength` -- ``computeArclength.m`` / ``arclength.m``
* :mod:`curve`     -- ``computeStrain.m``, ``driftCompensation.m``,
  ``mainComputeStrain.m``
* :mod:`propagate` -- ``interpPoints`` from ``loadPredictedfloInvivo.m``

Nothing here is learned. The point is that a tracker only has to answer *where
did these points go*; everything from there to a strain curve is arithmetic
that can be checked against MITT's own stored output before any network exists.
"""
