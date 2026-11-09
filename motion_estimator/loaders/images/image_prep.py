"""The clinical-view transform, ported from ``loadIntervendorData.m``.

Lines 32-40 of that file are the whole contract, and everything downstream --
the MITT contours, the pixel spacings, the published EPE and strain numbers --
is expressed in the frame they define::

    ImageData = USData.tissue3(:,:,cycle_frame_limits(1):cycle_frame_limits(2));
    ImageData = permute(ImageData(end:-1:1,:,:),[2 1 3]);
    pix_spacing = USInfo.voxelsize([2 1 3]) .* 1000;
    USInfo.gridsize = USInfo.gridsize([2 1 3]);

So this module is a transcription rather than a reimplementation: if a number
here disagrees with MATLAB, this module is wrong.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .srd_reader import UltrasoundClip

__all__ = [
    "PreparedClip",
    "crop_to_cycle",
    "to_clinical_view",
    "pixel_spacing_mm",
    "grid_size_clinical",
    "prepare_clip",
    "frames_first",
]


@dataclass
class PreparedClip:
    """A clip cropped to one cardiac cycle and rotated into the clinical view.

    ``images`` is ``(n_rows, n_cols, n_frames)``, matching MATLAB's
    ``data.ImageData``. Use :func:`frames_first` for the ``(n_frames, n_rows,
    n_cols)`` view that per-frame model code wants.
    """

    images: np.ndarray
    pixel_spacing_mm: np.ndarray
    cycle_frame_limits: tuple[int, int] | None
    source_shape: tuple[int, ...]

    @property
    def n_frames(self) -> int:
        return int(self.images.shape[2])


def crop_to_cycle(tissue: np.ndarray, cycle_frame_limits) -> np.ndarray:
    """Keep frames ``limits[0]..limits[1]`` inclusive, counting from 1.

    ``scar_struct`` stores the limits the way MATLAB indexes -- 1-based and
    inclusive at both ends -- so IV02's 2ch ``[51 94]`` means 44 frames, not
    43. Stored as ``uint8``, hence the widening to ``int`` before arithmetic.
    """
    limits = np.asarray(cycle_frame_limits).ravel()
    if limits.size != 2:
        raise ValueError(f"cycle_frame_limits must hold 2 values, got {limits.size}")

    first, last = int(limits[0]), int(limits[1])
    n_frames = tissue.shape[-1]
    if first < 1 or last > n_frames or first > last:
        raise ValueError(
            f"cycle_frame_limits [{first} {last}] outside the clip's "
            f"1..{n_frames} frames"
        )
    return tissue[..., first - 1:last]


def to_clinical_view(tissue: np.ndarray) -> np.ndarray:
    """Flip along the first axis, then swap the first two -- ``permute([2 1 3])``.

    Equivalent to ``out[i, j, k] == tissue[n_rows - 1 - j, i, k]``, which is
    what ``test_image_prep`` asserts directly.
    """
    if tissue.ndim != 3:
        raise ValueError(
            f"expected a 3-D (rows, cols, frames) array, got shape {tissue.shape}. "
            "Volumetric 3D+t clips are not part of the 2D strain pipeline."
        )
    return np.transpose(tissue[::-1, :, :], (1, 0, 2))


def pixel_spacing_mm(voxelsize) -> np.ndarray:
    """``voxelsize([2 1 3]) .* 1000`` -- metres to millimetres, reordered.

    Only the first two entries are spatial for a 2D+t clip; the third mirrors
    the frame axis and carries no length. MATLAB only ever reads
    ``pix_spacing(1)`` and ``pix_spacing(2)``, and neither should this.
    """
    values = np.asarray(voxelsize, dtype=float).ravel()
    if values.size != 3:
        raise ValueError(f"voxelsize must hold 3 values, got {values.size}")
    return values[[1, 0, 2]] * 1000.0


def grid_size_clinical(gridsize) -> np.ndarray:
    """``gridsize([2 1 3])``, to match the reordered image axes."""
    values = np.asarray(gridsize).ravel()
    if values.size != 3:
        raise ValueError(f"gridsize must hold 3 values, got {values.size}")
    return values[[1, 0, 2]]


def prepare_clip(clip: UltrasoundClip, cycle_frame_limits=None) -> PreparedClip:
    """Crop *clip* to one cycle and rotate it into the clinical view.

    Passing ``cycle_frame_limits=None`` keeps every frame, which is what an
    ad-hoc ``.srd`` without a sibling ``scar_struct.mat`` has to fall back on.
    """
    if clip.layout != "tissue3":
        raise NotImplementedError(
            f"{clip.source.name if clip.source else 'clip'}: the "
            f"{clip.layout!r} layout is still in polar coordinates "
            "(samples/beams), and scan conversion is not implemented here. "
            "Export it through Speqle first."
        )
    if clip.is_volumetric:
        raise NotImplementedError(
            f"{clip.source.name if clip.source else 'clip'}: 3D+t clip with "
            f"shape {clip.tissue.shape}; the 2D strain pipeline needs a 2D+t clip."
        )
    if clip.voxelsize is None:
        raise ValueError("clip has no USInfo.voxelsize, cannot derive pixel spacing")

    tissue = clip.tissue
    limits = None
    if cycle_frame_limits is not None:
        tissue = crop_to_cycle(tissue, cycle_frame_limits)
        pair = np.asarray(cycle_frame_limits).ravel()
        limits = (int(pair[0]), int(pair[1]))

    return PreparedClip(
        images=to_clinical_view(tissue),
        pixel_spacing_mm=pixel_spacing_mm(clip.voxelsize),
        cycle_frame_limits=limits,
        source_shape=tuple(clip.tissue.shape),
    )


def frames_first(images: np.ndarray) -> np.ndarray:
    """``(rows, cols, frames)`` -> ``(frames, rows, cols)`` for per-frame models."""
    if images.ndim != 3:
        raise ValueError(f"expected a 3-D array, got shape {images.shape}")
    return np.transpose(images, (2, 0, 1))
