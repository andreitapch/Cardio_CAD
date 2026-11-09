"""Tests for the clinical-view transform.

These assert the *indexing identities* of loadIntervendorData.m rather than
just shapes, because a wrong-axis flip or transpose preserves the shape and
would only show up much later as a mirrored image and unusable contours.

Run with:  python -m pytest motion_estimator/tests/test_image_prep.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from motion_estimator.loaders.images.image_prep import (
    crop_to_cycle,
    frames_first,
    grid_size_clinical,
    pixel_spacing_mm,
    prepare_clip,
    to_clinical_view,
)
from motion_estimator.loaders.images.srd_reader import read_clip
from .conftest import make_clip_mat


def test_to_clinical_view_matches_the_matlab_indexing():
    """``permute(A(end:-1:1,:,:),[2 1 3])`` is ``out[i,j,k] == A[rows-1-j,i,k]``."""
    tissue = np.arange(4 * 3 * 2).reshape(4, 3, 2)
    out = to_clinical_view(tissue)

    assert out.shape == (3, 4, 2)
    rows = tissue.shape[0]
    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            for k in range(out.shape[2]):
                assert out[i, j, k] == tissue[rows - 1 - j, i, k]


def test_to_clinical_view_is_not_a_plain_transpose():
    """Guards against dropping the flip, which keeps the shape but mirrors it."""
    tissue = np.arange(4 * 3 * 2).reshape(4, 3, 2)

    assert not np.array_equal(
        to_clinical_view(tissue), np.transpose(tissue, (1, 0, 2))
    )


def test_to_clinical_view_applied_twice_is_a_vertical_flip():
    """Sanity anchor: the transform is an involution up to a mirror."""
    tissue = np.arange(3 * 3 * 2).reshape(3, 3, 2)

    np.testing.assert_array_equal(
        to_clinical_view(to_clinical_view(tissue)), tissue[::-1, ::-1, :]
    )


def test_to_clinical_view_rejects_volumetric():
    with pytest.raises(ValueError, match="3-D"):
        to_clinical_view(np.zeros((2, 3, 4, 5)))


def test_crop_to_cycle_is_one_based_and_inclusive():
    """IV02's 2ch limits ``[51 94]`` must yield 44 frames, not 43."""
    tissue = np.arange(100).reshape(1, 1, 100)

    cropped = crop_to_cycle(tissue, (51, 94))

    assert cropped.shape[-1] == 44
    assert cropped[0, 0, 0] == 50
    assert cropped[0, 0, -1] == 93


def test_crop_to_cycle_accepts_matlab_uint8_limits():
    """``cycle_frame_limits`` is stored as uint8; arithmetic must widen first."""
    tissue = np.zeros((1, 1, 100))

    cropped = crop_to_cycle(tissue, np.asarray([[51, 94]], dtype=np.uint8))

    assert cropped.shape[-1] == 44


def test_crop_to_cycle_single_frame_cycle():
    tissue = np.arange(10).reshape(1, 1, 10)

    assert crop_to_cycle(tissue, (5, 5)).shape[-1] == 1


@pytest.mark.parametrize("limits", [(0, 10), (1, 101), (60, 50)])
def test_crop_to_cycle_rejects_out_of_range(limits):
    with pytest.raises(ValueError, match="cycle_frame_limits"):
        crop_to_cycle(np.zeros((1, 1, 100)), limits)


def test_pixel_spacing_reorders_and_converts_to_mm():
    """``voxelsize([2 1 3]) .* 1000`` -- the swap of the first two matters."""
    spacing = pixel_spacing_mm([0.0005, 0.0006, 0.001])

    np.testing.assert_allclose(spacing, [0.6, 0.5, 1.0])


def test_grid_size_reorders_to_match_the_image_axes():
    np.testing.assert_array_equal(grid_size_clinical([300, 245, 267]), [245, 300, 267])


def test_pixel_spacing_axes_agree_with_the_transformed_image(tmp_path: Path):
    """Row/column spacing must line up with the rows/columns after the permute."""
    tissue = np.zeros((300, 245, 8), dtype=np.uint8)
    clip = read_clip(
        make_clip_mat(tmp_path / "c.srd", tissue, voxelsize=(0.0005, 0.0006, 0.001))
    )

    prepared = prepare_clip(clip)

    assert prepared.images.shape[:2] == (245, 300)
    # axis 0 has 245 entries and came from the original columns, whose spacing
    # is voxelsize[1] = 0.6 mm.
    np.testing.assert_allclose(prepared.pixel_spacing_mm[:2], [0.6, 0.5])


def test_prepare_clip_crops_then_transforms(tmp_path: Path):
    tissue = np.arange(4 * 3 * 10, dtype=np.uint8).reshape(4, 3, 10)
    clip = read_clip(make_clip_mat(tmp_path / "c.srd", tissue))

    prepared = prepare_clip(clip, cycle_frame_limits=(3, 6))

    assert prepared.n_frames == 4
    assert prepared.cycle_frame_limits == (3, 6)
    assert prepared.source_shape == (4, 3, 10)
    np.testing.assert_array_equal(
        prepared.images, to_clinical_view(tissue[:, :, 2:6])
    )


def test_prepare_clip_without_limits_keeps_every_frame(tmp_path: Path):
    clip = read_clip(
        make_clip_mat(tmp_path / "c.srd", np.zeros((4, 3, 10), dtype=np.uint8))
    )

    prepared = prepare_clip(clip)

    assert prepared.n_frames == 10
    assert prepared.cycle_frame_limits is None


def test_prepare_clip_refuses_volumetric(tmp_path: Path):
    clip = read_clip(
        make_clip_mat(tmp_path / "v.srd", np.zeros((4, 3, 5, 2), dtype=np.uint8))
    )

    with pytest.raises(NotImplementedError, match="3D\\+t"):
        prepare_clip(clip)


def test_frames_first_puts_frames_on_axis_zero():
    images = np.zeros((245, 300, 44))

    assert frames_first(images).shape == (44, 245, 300)


def test_frames_first_preserves_each_frame():
    images = np.arange(2 * 3 * 4).reshape(2, 3, 4)
    stacked = frames_first(images)

    for k in range(images.shape[2]):
        np.testing.assert_array_equal(stacked[k], images[:, :, k])


def test_end_to_end_on_the_speqle_demo_clip(demo_srd: Path):
    """The real file, through the real path, with a plausible cycle window."""
    prepared = prepare_clip(read_clip(demo_srd), cycle_frame_limits=(51, 94))

    assert prepared.images.shape == (245, 300, 44)
    assert prepared.images.dtype == np.uint8
    np.testing.assert_allclose(
        prepared.pixel_spacing_mm[:2], [0.53149, 0.53149], rtol=1e-3
    )
