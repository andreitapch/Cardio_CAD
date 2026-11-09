"""Tests for the Speqle clip reader.

Run with:  python -m pytest motion_estimator/tests/test_srd_reader.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from motion_estimator.loaders.images.srd_reader import UltrasoundClip, read_clip
from .conftest import make_clip_mat


def test_demo_srd_is_a_matlab_v5_mat_file(demo_srd: Path):
    """The premise of the whole reader: .srd is a MAT-file with another suffix."""
    assert demo_srd.read_bytes()[:19] == b"MATLAB 5.0 MAT-file"


def test_reads_the_speqle_demo_clip(demo_srd: Path):
    clip = read_clip(demo_srd)

    assert clip.layout == "tissue3"
    assert clip.tissue.shape == (300, 245, 267)
    assert clip.tissue.dtype == np.uint8
    assert clip.n_frames == 267
    assert not clip.is_volumetric
    np.testing.assert_allclose(clip.gridsize, [300, 245, 267])
    np.testing.assert_allclose(clip.voxelsize, [5.3149e-4, 5.3149e-4, 1e-3], rtol=1e-3)
    assert clip.info["srd_format"] == "V1.3"
    assert clip.info["type"] == "GEVivid7"
    assert clip.ecg is not None and clip.ecg.size == 2669


def test_frames_come_from_the_array_not_usinfo(demo_srd: Path):
    """``USInfo.frames`` reads 1 on this clip while it holds 267 frames.

    A reader trusting that field would silently truncate every clip to one
    frame, so n_frames must derive from the array.
    """
    clip = read_clip(demo_srd)
    assert clip.info["frames"] == 1
    assert clip.n_frames == 267


def test_volumetric_clips_are_flagged_not_rejected(tmp_path: Path):
    """Speqle stores 3D+t in ``tissue3`` too -- two bundled demos are 4-D."""
    path = make_clip_mat(
        tmp_path / "vol.srd", np.zeros((5, 6, 7, 3), dtype=np.uint8)
    )
    clip = read_clip(path)

    assert clip.is_volumetric
    assert clip.n_frames == 3


def test_synthetic_clip_round_trips(tmp_path: Path):
    tissue = np.arange(2 * 3 * 4, dtype=np.uint8).reshape(2, 3, 4)
    clip = read_clip(make_clip_mat(tmp_path / "clip.srd", tissue))

    np.testing.assert_array_equal(clip.tissue, tissue)
    np.testing.assert_allclose(clip.voxelsize, [5e-4, 6e-4, 1e-3])
    assert isinstance(clip, UltrasoundClip)
    assert clip.source == tmp_path / "clip.srd"


def test_repr_omits_the_pixel_data(tmp_path: Path):
    clip = read_clip(
        make_clip_mat(tmp_path / "clip.srd", np.zeros((80, 90, 30), dtype=np.uint8))
    )
    text = repr(clip)

    assert "(80, 90, 30)" in text
    assert len(text) < 200


def test_h5_is_refused_because_it_is_polar(tmp_path: Path):
    """The .h5 export is un-scan-converted, so it is not a drop-in for the MAT."""
    path = tmp_path / "IV02_4ch.h5"
    path.write_bytes(b"")

    with pytest.raises(NotImplementedError, match="polar"):
        read_clip(path)


def test_non_speqle_mat_is_rejected_by_name(tmp_path: Path):
    import scipy.io as sio

    sio.savemat(tmp_path / "other.mat", {"something_else": np.zeros((2, 2))})

    with pytest.raises(ValueError, match="USData"):
        read_clip(tmp_path / "other.mat")


def test_two_dimensional_tissue_is_rejected(tmp_path: Path):
    import scipy.io as sio

    sio.savemat(
        tmp_path / "flat.srd",
        {"USInfo": {"voxelsize": np.ones((1, 3))}, "USData": {"tissue3": np.zeros((4, 4))}},
    )

    with pytest.raises(ValueError, match="3-D or 4-D"):
        read_clip(tmp_path / "flat.srd")


def test_real_data_mat_is_a_500_square_scan_converted_grid(iv02_dir: Path):
    """Every ``data.mat`` in this cohort is scan-converted onto a 500x500 grid."""
    clip = read_clip(iv02_dir / "2ch" / "data.mat")

    assert clip.tissue.shape[:2] == (500, 500)
    assert clip.tissue.dtype == np.uint8
    assert clip.n_frames == 144
    assert (clip.voxelsize[:2] > 0).all()


def test_real_srd_is_a_coarser_grid_than_data_mat(iv02_dir: Path):
    """The ``.srd`` is the same acquisition at a coarser scan conversion.

    Same frame count, different pixel grid and spacing -- so the two are not
    interchangeable even though both are Cartesian.
    """
    from_srd = read_clip(iv02_dir / "IV02_2ch.srd")
    from_mat = read_clip(iv02_dir / "2ch" / "data.mat")

    assert from_srd.n_frames == from_mat.n_frames
    assert from_srd.tissue.shape[:2] != from_mat.tissue.shape[:2]
    assert from_srd.voxelsize[0] > from_mat.voxelsize[0]


def test_real_h5_is_polar_with_the_frame_axis_first(iv02_dir: Path):
    """What the ``.h5`` export actually is -- not a transpose of the MAT files.

    ``Tissue`` is ``(n_frames, n_beams, n_samples)`` straight off the Vivid E9,
    so no reordering turns it into the scan-converted image in ``data.mat``.
    This is why ``read_clip`` refuses ``.h5``.
    """
    from motion_estimator.loaders.images.srd_reader import read_h5_tissue

    raw, attrs = read_h5_tissue(iv02_dir / "IV02_2ch.h5")
    from_mat = read_clip(iv02_dir / "2ch" / "data.mat").tissue

    assert attrs["_dataset"] == "Tissue"
    assert raw.ndim == 3
    # Same acquisition: the frame count agrees, and it leads rather than trails.
    assert raw.shape[0] == from_mat.shape[2] == 144
    # Nothing else agrees -- the beam/sample extent is not the 500x500 grid.
    assert raw.shape[1:] != from_mat.shape[:2]
    assert sorted(raw.shape) != sorted(from_mat.shape)


def test_real_h5_carries_metadata_the_mat_exports_drop(iv02_dir: Path):
    from motion_estimator.loaders.images.srd_reader import read_h5_tissue

    _, attrs = read_h5_tissue(iv02_dir / "IV02_2ch.h5")

    assert attrs["PatientID"] == "IV02"
    assert attrs["ManufacturerModelName"] == "Vivid E9"
    assert "ProbeName" in attrs
    # ViewName is present but blank throughout the cohort, so the view has to
    # keep coming from the filename.
    assert attrs.get("ViewName", "") == ""


def test_read_clip_refuses_the_real_h5(iv02_dir: Path):
    with pytest.raises(NotImplementedError, match="polar"):
        read_clip(iv02_dir / "IV02_2ch.h5")
