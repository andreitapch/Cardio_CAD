"""Whole clips with their reference contours, for evaluation.

The counterpart to :mod:`train_loader`, which yields frame *pairs* for
training. Evaluation needs whole clips, because the quantities that matter --
cycle closure, error at end-systole, a strain curve -- are properties of a
cycle and cannot be computed from a pair.

Two sources, and they answer different questions:

* :func:`load_in_vivo` -- the GE cohort, with MITT's tracked contour as the
  reference. **MITT is not ground truth**; it is a speckle tracker with its own
  error, so a disagreement is not necessarily our error. It is the only
  per-point reference the real data has.
* :func:`load_simulated` -- held-out simulated clips, where the scatterer
  positions are exact. This is the only place a tracking error can be called
  an error.

Both return :class:`EvalClip`, so the evaluation code does not branch on which
cohort it is measuring.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

__all__ = ["EvalClip", "load_in_vivo", "load_simulated"]


@dataclass
class EvalClip:
    """One clip, its frames, and a reference to compare against.

    ``frames`` is ``(n_frames, rows, cols)`` in the clinical view.
    ``reference_x`` / ``reference_y`` are ``(n_frames, n_points)`` 1-based
    pixel coordinates -- the contour to start from and to score against.
    ``spacing_xy`` is ordered for ``TrackResult.to_mm``, i.e. column spacing
    first, which is the *reverse* of ``PreparedClip.pixel_spacing_mm``.

    ``truth`` says whether ``reference_*`` is exact. It is False for the
    in-vivo cohort, and evaluation must not describe a disagreement there as an
    error.
    """

    key: str
    frames: np.ndarray
    reference_x: np.ndarray
    reference_y: np.ndarray
    spacing_xy: np.ndarray
    start_frame: int
    es_frame: int
    truth: bool
    sector: np.ndarray | None = None
    group_id: str | None = None

    @property
    def n_frames(self) -> int:
        return int(self.frames.shape[0])

    @property
    def n_points(self) -> int:
        return int(self.reference_x.shape[1])

    def start(self):
        """The contour to seed a tracker with, at ``start_frame``."""
        return self.reference_x[self.start_frame], self.reference_y[self.start_frame]


def sector_mask(frames, threshold=0.01):
    """True inside the imaged sector.

    Derived from the clip rather than from ``<view>/track/*_mask.raw``, so it
    works for the simulated cohort too, where outside the sector is NaN and
    here comes back as zero after the cache's quantisation.
    """
    from scipy import ndimage

    peak = float(np.nanmax(frames))
    mask = np.nanmax(frames, axis=0) > peak * threshold
    mask = ndimage.binary_closing(mask, np.ones((5, 5)))
    return ndimage.binary_fill_holes(mask)


def load_in_vivo(
    dataset: str = "Retest_dataset",
    view: str | None = "4ch",
    mitt_variant: str = "uni",
    limit: int | None = None,
    root=None,
    mitt_root=None,
):
    """The GE cohort, seeded and scored against MITT's contour.

    Clips with no MITT result are skipped rather than returned without a
    reference -- there would be nothing to compare against.
    """
    from .annotations.mitt import mitt_path, read_mitt
    from .annotations.scar_struct import read_view_analysis
    from .dataset_index import index_dataset
    from .images.image_prep import frames_first, prepare_clip
    from .images.srd_reader import read_clip

    refs = [r for r in index_dataset(root, (dataset,)) if r.is_complete]
    if view:
        refs = [r for r in refs if r.key.view == view]

    clips = []
    for ref in refs:
        if limit is not None and len(clips) >= limit:
            break
        arguments = (ref.key.patient, ref.key.view, ref.key.dataset, mitt_variant)
        path = (
            mitt_path(*arguments, mitt_root) if mitt_root else mitt_path(*arguments)
        )
        if not Path(path).is_file():
            continue
        mitt = read_mitt(path)
        analysis = read_view_analysis(ref.scar_struct_path, ref.key.view)
        prepared = prepare_clip(
            read_clip(ref.clip_path), cycle_frame_limits=analysis.cycle_frame_limits
        )
        frames = frames_first(prepared.images).astype(np.float32)
        spacing_rc = np.asarray(prepared.pixel_spacing_mm, dtype=float).ravel()[:2]

        # MITT's axes are transposed relative to the clinical image, and
        # pixel_spacing_mm is ordered (row, col). Getting this wrong puts the
        # contour 90 degrees off the myocardium -- see the package README.
        clips.append(
            EvalClip(
                key=str(ref.key),
                frames=frames,
                reference_x=mitt.points_y / spacing_rc[1] + 1.0,
                reference_y=mitt.points_x / spacing_rc[0] + 1.0,
                spacing_xy=spacing_rc[::-1].copy(),
                start_frame=mitt.ref_frame - 1,
                es_frame=mitt.es_frame,
                truth=False,
                sector=sector_mask(frames),
                group_id=ref.group_id,
            )
        )
    return clips


def load_simulated(
    cache_dir=None,
    val_fraction: float = 0.2,
    seed: int = 0,
    limit: int | None = None,
    n_points: int = 200,
):
    """Held-out simulated clips, with exact scatterer correspondence.

    The split is by patient, taken from :mod:`train_loader`, so a clip returned
    here was never trained on. *n_points* subsamples the ~20 k scatterers
    evenly; the full set is far more than any interpolation needs.
    """
    from .train_loader import list_cached, split_by_patient

    if cache_dir is None:
        cache_dir = Path(__file__).resolve().parents[1] / "artifacts" / "simulated-cache"

    _, held_out = split_by_patient(list_cached(cache_dir), val_fraction, seed)
    if limit is not None:
        held_out = held_out[:limit]

    clips = []
    for clip in held_out:
        with np.load(clip.path, allow_pickle=False) as data:
            frames = data["frames"].astype(np.float32)
            scat_x, scat_y = data["scat_x"], data["scat_y"]
            pixel_mm = np.asarray(data["pixel_mm"], dtype=float).ravel()[:2]
            es_frame = int(data["es_frame"])
        columns = np.linspace(0, scat_x.shape[1] - 1, n_points).astype(int)
        clips.append(
            EvalClip(
                key=clip.key,
                frames=frames,
                reference_x=scat_x[:, columns].astype(float),
                reference_y=scat_y[:, columns].astype(float),
                # the cache resamples to an isotropic size, so the order is moot
                spacing_xy=pixel_mm[::-1].copy(),
                start_frame=0,
                es_frame=es_frame,
                truth=True,
                sector=sector_mask(frames),
                group_id=clip.group_id,
            )
        )
    return clips
