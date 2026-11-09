"""Tests for the per-view analysis reader.

Run with:  python -m pytest motion_estimator/tests/test_scar_struct.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import scipy.io as sio

from motion_estimator.loaders.annotations.scar_struct import (
    available_views,
    read_scar_struct,
    read_view_analysis,
)
from .conftest import (
    IV02_CYCLE_LIMITS,
    IV02_N_CONTOUR_POINTS,
    IV02_N_SAMPLES,
    make_scar_struct_mat,
)


@pytest.fixture
def scar_path(tmp_path: Path) -> Path:
    return make_scar_struct_mat(tmp_path / "scar_struct.mat")


def test_discovers_views_from_the_info_fields(scar_path: Path):
    assert available_views(scar_path) == ["2ch", "4ch"]


def test_cycle_limits_are_python_ints_not_uint8(scar_path: Path):
    """uint8 limits silently wrap under arithmetic, so they must be widened."""
    view = read_view_analysis(scar_path, "2ch")

    first, last = view.cycle_frame_limits
    assert (first, last) == IV02_CYCLE_LIMITS["2ch"]
    assert type(first) is int and type(last) is int


def test_cycle_frame_count_is_inclusive(scar_path: Path):
    assert read_view_analysis(scar_path, "2ch").n_cycle_frames == 44
    assert read_view_analysis(scar_path, "4ch").n_cycle_frames == 44


def test_contour_and_sample_shapes(scar_path: Path):
    view = read_view_analysis(scar_path, "2ch")

    assert view.surf_endo.shape == (2, IV02_N_CONTOUR_POINTS)
    assert view.surf_epi.shape == (2, IV02_N_CONTOUR_POINTS)
    assert view.endo_sample.shape == (2, IV02_N_SAMPLES)
    assert view.epi_sample.shape == (2, IV02_N_SAMPLES)
    assert view.n_samples == IV02_N_SAMPLES


def test_sample_rows_are_x_then_y(scar_path: Path):
    """The fixture writes y = 2x, which fixes the row order unambiguously."""
    view = read_view_analysis(scar_path, "2ch")

    np.testing.assert_allclose(view.endo_sample[1], view.endo_sample[0] * 2)


def test_segments_are_zero_based_and_cover_every_sample(scar_path: Path):
    view = read_view_analysis(scar_path, "2ch")

    assert len(view.segments) == 6
    combined = np.concatenate(view.segments)
    assert combined.min() == 0
    assert combined.max() == IV02_N_SAMPLES - 1
    np.testing.assert_array_equal(np.sort(combined), np.arange(IV02_N_SAMPLES))


def test_read_scar_struct_returns_every_view(scar_path: Path):
    views = read_scar_struct(scar_path)

    assert sorted(views) == ["2ch", "4ch"]
    assert views["4ch"].cycle_frame_limits == IV02_CYCLE_LIMITS["4ch"]


def test_unknown_view_lists_what_is_available(scar_path: Path):
    with pytest.raises(KeyError, match="3ch"):
        read_view_analysis(scar_path, "3ch")


def test_missing_scar_struct_variable_is_rejected(tmp_path: Path):
    sio.savemat(tmp_path / "wrong.mat", {"other": np.zeros((2, 2))})

    with pytest.raises(ValueError, match="scar_struct"):
        read_scar_struct(tmp_path / "wrong.mat")


def test_wrong_segment_count_is_rejected(tmp_path: Path):
    """Five segments cannot be matched against GE's six strain values."""
    segment_definition = np.empty((1, 5), dtype=object)
    for index in range(5):
        segment_definition[0, index] = np.arange(
            index * 2 + 1, index * 2 + 3, dtype=np.uint8
        ).reshape(1, -1)

    sio.savemat(
        tmp_path / "scar_struct.mat",
        {
            "scar_struct": {
                "info_2ch": {
                    "cycle_frame_limits": np.asarray([[1, 10]], dtype=np.uint8),
                    "contour": {
                        "surf_endo": np.zeros((2, 3)),
                        "surf_epi": np.zeros((2, 3)),
                        "surf_endo_ind": np.zeros((2, 3)),
                        "surf_epi_ind": np.zeros((2, 3)),
                        "seg_definition": np.ones((3, 1), dtype=np.uint8),
                    },
                    "contour_sampled": {
                        "Endo_sample_x": np.arange(10.0).reshape(1, -1),
                        "Endo_sample_y": np.arange(10.0).reshape(1, -1),
                        "Epi_sample_x": np.arange(10.0).reshape(1, -1),
                        "Epi_sample_y": np.arange(10.0).reshape(1, -1),
                        "segment_definition": segment_definition,
                    },
                }
            }
        },
    )

    with pytest.raises(ValueError, match="expected 6"):
        read_view_analysis(tmp_path / "scar_struct.mat", "2ch")


def test_real_iv02_scar_struct(iv02_dir: Path):
    """The pinned values from the actual file, when the drive is mounted."""
    views = read_scar_struct(iv02_dir / "Analysis" / "scar_struct.mat")

    assert sorted(views) == ["2ch", "4ch"]
    for view, limits in IV02_CYCLE_LIMITS.items():
        assert views[view].cycle_frame_limits == limits
        assert views[view].surf_endo.shape == (2, IV02_N_CONTOUR_POINTS)
        assert views[view].n_samples == IV02_N_SAMPLES
        assert len(views[view].segments) == 6
