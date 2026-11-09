"""Reader for MITT tracking results.

MITT is the classical speckle-tracking method this work is compared against. Its
output is the richest reference available for the in-vivo cohort: tracked
contour points for every frame, the end-systolic frame index, a 6-segment map,
and the strain curves MITT itself derived from those points.

That last pair is what makes Stage A checkable. Points and curves are stored
side by side, so a correct port of the strain arithmetic must turn one into the
other -- an exact target, on real data, available today for all 236 clips.

Layout, per ``loadMITTresults.m`` and ``loadOptions.m``::

    <root>/<variant>/<dataset>/<patient>/<view>/<patient>_<view>.mat

``variant`` is the tracking configuration: ``MITT_uni`` (unidirectional),
``ResultsMITT`` (bidirectional, corrected) or ``MITT_uncorrectedbi``.

Units: ``output.pointsX`` / ``pointsY`` are in **millimetres**. The MATLAB
pipeline divides by the pixel spacing and adds 1 to get pixel coordinates
(``loadIntervendorData.m:20-21``), then multiplies back out before computing
arc length (``calculateLongStrainInvivo.m``) -- a round trip that leaves the
values unchanged. Arc length is therefore computed on these values as stored.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..images import _matio

__all__ = ["MittResult", "MITT_VARIANTS", "mitt_path", "read_mitt"]

# loadOptions.m:160-172 -- the folder per tracking configuration.
MITT_VARIANTS = {
    "uni": "MITT_uni",
    "bi_corrected": "ResultsMITT",
    "bi_uncorrected": "MITT_uncorrectedbi",
}

DEFAULT_MITT_ROOT = Path(r"F:\PHD\DataSets")


@dataclass
class MittResult:
    """One clip's MITT tracking output.

    ``points_x`` / ``points_y`` are ``(n_frames, n_points)`` in millimetres,
    the orientation ``computeArclength`` expects. ``es_frame`` and
    ``ref_frame`` are 1-based, as MATLAB stores them. ``segments`` maps each
    contour point to one of 6 segments, converted to 0-based.
    """

    points_x: np.ndarray
    points_y: np.ndarray
    es_frame: int
    ref_frame: int
    segments: np.ndarray
    strain_curve: np.ndarray | None
    segmental_curves: list[np.ndarray]
    pixel_spacing: np.ndarray | None
    drift_compensated: bool
    peak_value: float | None
    peak_frame: int | None
    source: Path | None = None

    @property
    def n_frames(self) -> int:
        return int(self.points_x.shape[0])

    @property
    def n_points(self) -> int:
        return int(self.points_x.shape[1])

    @property
    def n_segments(self) -> int:
        return int(self.segments.max()) + 1 if self.segments.size else 0


def mitt_path(
    patient: str,
    view: str,
    dataset: str,
    variant: str = "uni",
    root: str | Path = DEFAULT_MITT_ROOT,
) -> Path:
    """Where one clip's MITT result lives."""
    if variant not in MITT_VARIANTS:
        raise ValueError(
            f"variant must be one of {sorted(MITT_VARIANTS)}, got {variant!r}"
        )
    folder = MITT_VARIANTS[variant]
    return (
        Path(root) / folder / dataset / patient / view / f"{patient}_{view}.mat"
    )


def read_mitt(path: str | Path) -> MittResult:
    """Parse one MITT ``.mat``."""
    path = Path(path)
    variables = _matio.load_mat(path)
    if "output" not in variables:
        raise ValueError(
            f"{path.name}: no 'output' variable; found {sorted(variables)}"
        )
    output = variables["output"]

    points_x = np.asarray(_matio.field(output, "pointsX"), dtype=float)
    points_y = np.asarray(_matio.field(output, "pointsY"), dtype=float)
    if points_x.shape != points_y.shape:
        raise ValueError(
            f"{path.name}: pointsX {points_x.shape} != pointsY {points_y.shape}"
        )
    if points_x.ndim != 2:
        raise ValueError(
            f"{path.name}: expected (n_frames, n_points), got {points_x.shape}"
        )

    return MittResult(
        points_x=points_x,
        points_y=points_y,
        es_frame=int(_matio.scalar(_matio.field(output, "esFrame"))),
        ref_frame=int(_matio.scalar(_matio.field(output, "refFrame"))),
        segments=_read_segments(output),
        strain_curve=_first_cell(variables.get("strainCurves")),
        segmental_curves=_segmental_curves(variables.get("segmentalStrainCurves")),
        pixel_spacing=_optional_vector(variables.get("Info"), "pixelSpacing"),
        drift_compensated=_optional_flag(variables.get("drift_comp")),
        peak_value=_timing(variables.get("timings"), "peakval"),
        peak_frame=_timing(variables.get("timings"), "frame", as_int=True),
        source=path,
    )


def _read_segments(output) -> np.ndarray:
    """``segments.indexes`` -> 0-based segment id per contour point."""
    if not _matio.has_field(output, "segments"):
        return np.empty(0, dtype=int)
    segments = _matio.field(output, "segments")
    if not _matio.has_field(segments, "indexes"):
        return np.empty(0, dtype=int)
    indexes = np.asarray(_matio.field(segments, "indexes")).ravel().astype(int)
    return indexes - 1


def _first_cell(value) -> np.ndarray | None:
    """MATLAB stores the strain curve inside a 1x1 cell."""
    if value is None:
        return None
    array = np.asarray(value, dtype=object)
    if array.dtype == object:
        if array.size == 0:
            return None
        return np.asarray(_matio.unwrap(array.ravel()[0]), dtype=float).ravel()
    return np.asarray(value, dtype=float).ravel()


def _segmental_curves(value) -> list[np.ndarray]:
    """``segmentalStrainCurves{1}.curves`` -> one curve per segment."""
    if value is None:
        return []
    outer = np.asarray(value, dtype=object).ravel()
    if outer.size == 0:
        return []
    entry = _matio.unwrap(outer[0])
    if not _matio.has_field(entry, "curves"):
        return []
    cells = np.asarray(_matio.field(entry, "curves"), dtype=object).ravel()
    return [
        np.asarray(_matio.unwrap(cell), dtype=float).ravel() for cell in cells
    ]


def _optional_vector(struct, name: str) -> np.ndarray | None:
    if struct is None or not _matio.has_field(struct, name):
        return None
    return np.asarray(_matio.field(struct, name), dtype=float).ravel()


def _optional_flag(value) -> bool:
    if value is None:
        return False
    return bool(np.asarray(value).ravel()[0])


def _timing(value, field: str, as_int: bool = False):
    """``timings{1}.peakval`` / ``.frame`` -- the peak-systolic annotation."""
    if value is None:
        return None
    cells = np.asarray(value, dtype=object).ravel()
    if cells.size == 0:
        return None
    entry = _matio.unwrap(cells[0])
    if not _matio.has_field(entry, field):
        return None
    scalar = _matio.scalar(_matio.field(entry, field))
    return int(scalar) if as_int else float(scalar)
