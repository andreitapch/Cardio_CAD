"""The paper's in vivo EPE (Table 2): contour points against MITT, in mm.

A Python port of ``cardiacMotionEstimation/matlab``: ``main.m`` with
``opts.opt = 'calculateEPE'`` -> ``loadData/loadPredPoints.m`` ->
``calculateEPE.m``, for ``opts.datasetName = 'inVivo'``.

* The endocardial contour is propagated through the predicted flow,
  unidirectionally from frame 1, each step starting from the previous
  prediction (``loadPredictedfloInvivo.m``, ``opts.propagated = 1``). That is
  :func:`motion_estimator.strain.propagate.propagate_points`.
* Each frame's points are compared with MITT's tracked points, both converted
  to mm as ``(p - 1) * spacing``; the frame's value is the mean over points.
* A sequence's value is the mean over frames 2 .. n-1 (1-based; MATLAB loops
  ``frIdx = 2 : nFrames-1``), each frame rounded to 4 decimals first.
* The table value is the mean over sequences, **+- the SD across
  sequences** (verified below). The MATLAB's printed ``STD`` pools every point
  error instead; it is returned too, but it is not what Table 2 shows.
* The paper's in vivo evaluation used the **2ch view only**
  (``loadOptions.m``: ``opts.view = '2ch'``). This replication scores 2ch
  and 4ch, reported per view and together.

**Verified** against the paper's own predicted contours
(``scripts/verify_paper_epe.py``, 2ch, ``paper_exact=True``):

=============  ===========  =================
old run        Table 2      this port
=============  ===========  =================
MS12_IV85      1.30 +-0.44  1.297 +-0.465
MS23_IV85      1.17 +-0.40  1.174 +-0.401
MS35_IV85      1.19 +-0.36  1.197 +-0.390
=============  ===========  =================

(Test_1, Test_2 and Test_6 on F: give 1.54, 2.72 and 0.90 against 1.39,
4.71 and 1.10 -- those folders are not the models Table 2 reports.)

Two MATLAB bugs, fixed here by default (``paper_exact=False``) and in
``loadPredPoints.m`` (2026-10-27; original kept as ``.before-fix``):

* **Swapped pixel spacing.** ``calculateEPE`` multiplies x (columns) by
  ``pix_spacing(1)`` and y by ``pix_spacing(2)``, but ``loadOriginalUSImage``
  builds ``pix_spacing = voxelsize([2 1 3])``, i.e. (row, column). On
  Retest IV02 2ch that is 0.301 mm applied to columns whose spacing is
  0.366 mm, so x errors come out ~18% small and y errors ~22% large.
  ``paper_exact=False`` uses each axis's own spacing.
* **The folder mean.** ``loadPredPoints.m`` overwrites ``dataFolderError(i)``
  inside the patient loop, so what it prints is the last patient's value;
  Table 2 cannot have come from that line. The mean over sequences is what the
  paper describes ("averaged per sequence"), and is what is computed here.

No scipy here, so this can run inside a training process.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["SequenceEPE", "contour_epe", "summarise"]


@dataclass
class SequenceEPE:
    key: str
    epe_mm: float                 # mean over frames 2 .. n-1
    per_frame_mm: np.ndarray      # (n_frames,), NaN where not scored
    point_errors_mm: np.ndarray   # every scored point error, for the pooled SD
    lost_points: int              # points that left the image (NaN) at some frame


def contour_epe(pred_x, pred_y, ref_x, ref_y, spacing_xy, key: str = "",
                paper_exact: bool = False) -> SequenceEPE:
    """One sequence. All coordinates ``(n_frames, n_points)``, 1-based pixels.

    *spacing_xy* is ``(column spacing, row spacing)`` in mm -- the
    ``EvalClip.spacing_xy`` order.
    """
    pred_x, pred_y = np.asarray(pred_x, float), np.asarray(pred_y, float)
    ref_x, ref_y = np.asarray(ref_x, float), np.asarray(ref_y, float)
    sx, sy = float(spacing_xy[0]), float(spacing_xy[1])
    if paper_exact:
        sx, sy = sy, sx  # MATLAB: x * pix_spacing(1) = row spacing; see module docstring
    dx = (pred_x - 1) * sx - (ref_x - 1) * sx
    dy = (pred_y - 1) * sy - (ref_y - 1) * sy
    errors = np.sqrt(dx ** 2 + dy ** 2)                         # (n_frames, n_points)

    n_frames = errors.shape[0]
    per_frame = np.full(n_frames, np.nan)
    scored = range(1, n_frames - 1)                             # frames 2 .. n-1, 1-based
    for frame in scored:
        per_frame[frame] = np.round(np.mean(errors[frame]), 4)  # NaN if any point was lost, as mean() in MATLAB
    points = errors[1:n_frames - 1].ravel()
    return SequenceEPE(
        key=key,
        epe_mm=float(np.nanmean(per_frame)),
        per_frame_mm=per_frame,
        point_errors_mm=points[np.isfinite(points)],
        lost_points=int(np.isnan(errors).any(axis=0).sum()),
    )


def summarise(sequences: list[SequenceEPE]) -> dict:
    """Table-2 style numbers over a set of sequences."""
    values = np.array([s.epe_mm for s in sequences], float)
    pooled = np.concatenate([s.point_errors_mm for s in sequences]) if sequences else np.array([])
    return {
        "n_sequences": int(np.isfinite(values).sum()),
        "epe_mm_mean": float(np.nanmean(values)) if len(values) else float("nan"),
        "epe_mm_sd": float(np.nanstd(values, ddof=1)) if len(values) > 1 else float("nan"),  # Table 2's +-
        "epe_mm_sd_pooled_points": float(np.std(pooled, ddof=1)) if pooled.size > 1 else float("nan"),
        "lost_points": int(sum(s.lost_points for s in sequences)),
    }
