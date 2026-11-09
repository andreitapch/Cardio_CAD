"""Per-view analysis from ``Data/IV##/Analysis/scar_struct.mat``.

``loadIntervendorData.m`` only ever reads ``cycle_frame_limits`` out of this
file -- the contours it works with come from MITT tracking instead. The
contours stored here are loaded faithfully and exposed, but **their coordinate
convention is unverified**: nothing in the MATLAB pipeline consumed them, so
there is no reference that pins down whether they index the raw ``tissue3``
frame or the clinical view produced by
:func:`motion_estimator.loaders.images.image_prep.to_clinical_view`, nor whether they are
1-based. Settle that with a visual overlay before trusting them for anything
quantitative. The strain targets do not depend on them -- those come from
:mod:`motion_estimator.loaders.annotations.gt_strain`.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..images import _matio

__all__ = ["ViewAnalysis", "read_scar_struct", "read_view_analysis", "available_views"]

_INFO_FIELD = re.compile(r"^info_(?P<view>\w+)$")
_N_SEGMENTS = 6


@dataclass
class ViewAnalysis:
    """One cardiac view's entry in ``scar_struct``.

    ``cycle_frame_limits`` is 1-based and inclusive, exactly as MATLAB indexes
    it. ``endo_sample`` / ``epi_sample`` are ``(2, n_samples)`` with x in row 0
    and y in row 1. ``segments`` holds :data:`_N_SEGMENTS` arrays of **0-based**
    indices into those samples, converted from MATLAB's 1-based cells.
    """

    view: str
    cycle_frame_limits: tuple[int, int]
    surf_endo: np.ndarray
    surf_epi: np.ndarray
    surf_endo_ind: np.ndarray
    surf_epi_ind: np.ndarray
    seg_definition: np.ndarray
    endo_sample: np.ndarray
    epi_sample: np.ndarray
    segments: list[np.ndarray]
    source: Path | None = None

    @property
    def n_cycle_frames(self) -> int:
        """Frame count after cropping -- inclusive, so 51..94 is 44 frames."""
        first, last = self.cycle_frame_limits
        return last - first + 1

    @property
    def n_samples(self) -> int:
        return int(self.endo_sample.shape[1])


def available_views(path: str | Path) -> list[str]:
    """View names present in *path*, e.g. ``['2ch', '4ch']``."""
    struct = _load_scar_struct(Path(path))
    views = []
    for name in _matio.fields(struct):
        match = _INFO_FIELD.match(name)
        if match:
            views.append(match.group("view"))
    return views


def read_scar_struct(path: str | Path) -> dict[str, ViewAnalysis]:
    """Every view in *path*, keyed by view name."""
    path = Path(path)
    struct = _load_scar_struct(path)
    return {
        view: _parse_view(struct, view, path)
        for view in available_views(path)
    }


def read_view_analysis(path: str | Path, view: str) -> ViewAnalysis:
    """Just *view* from *path*; mirrors ``loadScarStruct`` in the MATLAB."""
    path = Path(path)
    struct = _load_scar_struct(path)
    if not _matio.has_field(struct, f"info_{view}"):
        raise KeyError(
            f"{path.name}: no view {view!r}; available: {available_views(path)}"
        )
    return _parse_view(struct, view, path)


def _load_scar_struct(path: Path):
    variables = _matio.load_mat(path)
    if "scar_struct" not in variables:
        raise ValueError(
            f"{path.name}: no 'scar_struct' variable; found {sorted(variables)}"
        )
    return variables["scar_struct"]


def _parse_view(struct, view: str, path: Path) -> ViewAnalysis:
    info = _matio.field(struct, f"info_{view}")

    limits = np.asarray(_matio.field(info, "cycle_frame_limits")).ravel()
    if limits.size != 2:
        raise ValueError(
            f"{path.name}/{view}: cycle_frame_limits has {limits.size} values, expected 2"
        )
    # Stored as uint8, so widen before anything arithmetic touches it.
    first, last = int(limits[0]), int(limits[1])
    if first > last:
        raise ValueError(
            f"{path.name}/{view}: cycle_frame_limits [{first} {last}] is reversed"
        )

    contour = _matio.field(info, "contour")
    sampled = _matio.field(info, "contour_sampled")

    endo_sample = _stack_xy(sampled, "Endo_sample_x", "Endo_sample_y")
    epi_sample = _stack_xy(sampled, "Epi_sample_x", "Epi_sample_y")
    segments = _parse_segments(sampled, endo_sample.shape[1], path, view)

    return ViewAnalysis(
        view=view,
        cycle_frame_limits=(first, last),
        surf_endo=np.asarray(_matio.field(contour, "surf_endo"), dtype=float),
        surf_epi=np.asarray(_matio.field(contour, "surf_epi"), dtype=float),
        surf_endo_ind=np.asarray(_matio.field(contour, "surf_endo_ind"), dtype=float),
        surf_epi_ind=np.asarray(_matio.field(contour, "surf_epi_ind"), dtype=float),
        seg_definition=np.asarray(_matio.field(contour, "seg_definition")).ravel(),
        endo_sample=endo_sample,
        epi_sample=epi_sample,
        segments=segments,
        source=path,
    )


def _stack_xy(sampled, x_field: str, y_field: str) -> np.ndarray:
    x = np.asarray(_matio.field(sampled, x_field), dtype=float).ravel()
    y = np.asarray(_matio.field(sampled, y_field), dtype=float).ravel()
    if x.size != y.size:
        raise ValueError(f"{x_field}/{y_field} length mismatch: {x.size} vs {y.size}")
    return np.vstack([x, y])


def _parse_segments(sampled, n_samples: int, path: Path, view: str) -> list[np.ndarray]:
    """MATLAB's 1-based index cells -> 0-based arrays, with coverage checked.

    A wrong segment split would quietly mislabel segmental strain, so the
    indices are validated here rather than at the point of use.
    """
    cells = np.asarray(_matio.field(sampled, "segment_definition"), dtype=object).ravel()
    if cells.size != _N_SEGMENTS:
        raise ValueError(
            f"{path.name}/{view}: segment_definition has {cells.size} segments, "
            f"expected {_N_SEGMENTS} to match the GE segmental strain vector"
        )

    segments = []
    for position, cell in enumerate(cells, start=1):
        indices = np.asarray(_matio.unwrap(cell)).ravel().astype(int)
        if indices.size == 0:
            raise ValueError(f"{path.name}/{view}: segment {position} is empty")
        if indices.min() < 1 or indices.max() > n_samples:
            raise ValueError(
                f"{path.name}/{view}: segment {position} indices span "
                f"{indices.min()}..{indices.max()}, outside 1..{n_samples}"
            )
        segments.append(indices - 1)

    combined = np.concatenate(segments)
    duplicated = combined.size - np.unique(combined).size
    if duplicated:
        raise ValueError(
            f"{path.name}/{view}: {duplicated} sample(s) assigned to more than "
            "one segment"
        )
    return segments
