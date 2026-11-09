"""Tests for the ground-truth strain reader.

Run with:  python -m pytest motion_estimator/tests/test_gt_strain.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import scipy.io as sio

from motion_estimator.loaders.annotations.gt_strain import (
    N_SEGMENTS,
    PHASES,
    gt_path,
    read_gt_strain,
    vendor_location,
)
from .conftest import IV02_GLS_ES, make_gt_strain_mat


@pytest.fixture
def gt_file(tmp_path: Path) -> Path:
    return make_gt_strain_mat(tmp_path / "GE_strainIV02.mat")


def test_reads_all_three_phases_for_both_views(gt_file: Path):
    gt = read_gt_strain(gt_file)

    assert gt.views == ["2ch", "4ch"]
    for phase in PHASES:
        for view in ("2ch", "4ch"):
            assert gt.segmental[phase][view].shape == (N_SEGMENTS,)


def test_gls_matches_the_pinned_iv02_values(gt_file: Path):
    gt = read_gt_strain(gt_file)

    assert gt.gls_es == pytest.approx(IV02_GLS_ES)


def test_patient_and_vendor_recovered_from_the_path(tmp_path: Path):
    """Labels come from the filename and its ``<Vendor>_mat`` parent folder."""
    vendor_dir = tmp_path / "GE_mat"
    vendor_dir.mkdir()
    gt = read_gt_strain(make_gt_strain_mat(vendor_dir / "GE_strainIV02.mat"))

    assert gt.patient == "IV02"
    assert gt.vendor == "GE"


def test_explicit_labels_win_over_the_path(tmp_path: Path):
    gt = read_gt_strain(
        make_gt_strain_mat(tmp_path / "GE_strainIV02.mat"),
        patient="IV99",
        vendor="Philips",
    )

    assert (gt.patient, gt.vendor) == ("IV99", "Philips")


def test_peak_and_end_systolic_differ_only_where_expected(gt_file: Path):
    """IV02's 2ch vectors differ in segment 4 alone.

    ``valPeakS = min(strainCurve(1:ED))`` and ``valES = strainCurve(ED)``
    coincide wherever a segment has not started recovering, so the two phases
    look nearly identical -- which is exactly why confusing them is easy.
    """
    gt = read_gt_strain(gt_file)

    peak = gt.segmental["peak_systolic"]["2ch"]
    end = gt.segmental["end_systolic"]["2ch"]
    differing = np.flatnonzero(~np.isclose(peak, end))

    np.testing.assert_array_equal(differing, [3])
    assert peak[3] == pytest.approx(3.59)
    assert end[3] == pytest.approx(-1.06)


def test_target_vector_is_six_segments_then_gls(gt_file: Path):
    gt = read_gt_strain(gt_file)

    target = gt.target_vector("2ch")

    assert target.shape == (N_SEGMENTS + 1,)
    np.testing.assert_allclose(target[:N_SEGMENTS], gt.segmental["end_systolic"]["2ch"])
    assert target[-1] == pytest.approx(IV02_GLS_ES["2ch"])


def test_target_vector_defaults_to_end_systolic(gt_file: Path):
    gt = read_gt_strain(gt_file)

    np.testing.assert_allclose(
        gt.target_vector("2ch"), gt.target_vector("2ch", phase="end_systolic")
    )
    assert not np.allclose(
        gt.target_vector("2ch"), gt.target_vector("2ch", phase="peak_systolic")
    )


def test_target_vector_rejects_an_unknown_phase(gt_file: Path):
    with pytest.raises(ValueError, match="unknown phase"):
        read_gt_strain(gt_file).target_vector("2ch", phase="systolic")


def test_target_vector_reports_a_missing_view(gt_file: Path):
    with pytest.raises(KeyError, match="3ch"):
        read_gt_strain(gt_file).target_vector("3ch")


def test_single_view_file_is_handled(tmp_path: Path):
    """Not every patient has both views."""
    gt = read_gt_strain(make_gt_strain_mat(tmp_path / "GE_strainIV07.mat", views=("2ch",)))

    assert gt.views == ["2ch"]
    assert gt.target_vector("2ch").shape == (N_SEGMENTS + 1,)


def test_wrong_segment_count_is_rejected(tmp_path: Path):
    sio.savemat(
        tmp_path / "GE_strainIV99.mat",
        {
            "strain": {
                "segm_2ch_end_systolic_strain": np.zeros((1, 5)),
                "GLS_2ch_ES": np.asarray([[-10.0]]),
            }
        },
    )

    with pytest.raises(ValueError, match="expected 6"):
        read_gt_strain(tmp_path / "GE_strainIV99.mat")


def test_file_without_gls_is_rejected(tmp_path: Path):
    sio.savemat(
        tmp_path / "GE_strainIV98.mat",
        {"strain": {"segm_2ch_end_systolic_strain": np.zeros((1, 6))}},
    )

    with pytest.raises(ValueError, match="GLS"):
        read_gt_strain(tmp_path / "GE_strainIV98.mat")


def test_missing_strain_variable_is_rejected(tmp_path: Path):
    sio.savemat(tmp_path / "other.mat", {"nope": np.zeros((1, 1))})

    with pytest.raises(ValueError, match="strain"):
        read_gt_strain(tmp_path / "other.mat")


@pytest.mark.parametrize(
    "vendor,folder,prefix",
    [
        ("GE", "GE_mat", "GE_strain"),
        ("Hitachi", "Hitachi_mat", "Hitachi_strain"),
        # The two that break the <Vendor>_mat / <Vendor>_strain pattern,
        # per loadOptions.m:58-62.
        ("Average", "Average_mat", "Mean_vendors_strain"),
        ("Not_GE_Mean", "Not_GE_Mean", "NOT_GE_Mean"),
    ],
)
def test_vendor_folder_and_prefix(vendor, folder, prefix):
    assert vendor_location(vendor) == (folder, prefix)


def test_gt_path_layout():
    path = gt_path(Path("root") / "Retest_dataset", "IV02", "GE")

    assert path == Path("root/Retest_dataset/GT_STRAIN/GE_mat/GE_strainIV02.mat")


def test_real_iv02_gt_strain(dataset_root: Path):
    """The pinned values from the actual file, when the drive is mounted."""
    path = gt_path(dataset_root / "Retest_dataset", "IV02", "GE")
    if not path.is_file():
        pytest.skip(f"{path} not present")

    gt = read_gt_strain(path)

    assert gt.patient == "IV02"
    assert gt.vendor == "GE"
    assert gt.gls_es == pytest.approx(IV02_GLS_ES)
    np.testing.assert_allclose(
        gt.segmental["end_systolic"]["2ch"],
        [-15.42, -9.26, -2.28, -1.06, -5.13, -9.70],
    )
    np.testing.assert_allclose(
        gt.segmental["end_systolic"]["4ch"],
        [-9.07, -11.36, -0.78, 13.87, 0.05, -9.24],
    )
