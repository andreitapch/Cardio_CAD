"""End-to-end composition of the readers: index -> clip -> cycle -> target.

The individual modules are covered by their own tests; this one exists to catch
the seams between them, which is where the axis-order and 1-vs-0-based bugs
live. It is the shape the embedding extractor will consume, so if this walk
works, the model side has a stable contract to build on.

Run with:  python -m pytest motion_estimator/tests/test_pipeline.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from motion_estimator.loaders.dataset_index import index_dataset
from motion_estimator.loaders.annotations.gt_strain import N_SEGMENTS, read_gt_strain
from motion_estimator.loaders.images.image_prep import frames_first, prepare_clip
from motion_estimator.loaders.annotations.scar_struct import read_view_analysis
from motion_estimator.loaders.images.srd_reader import read_clip


def load_example(ref):
    """One training example's worth of data, from a :class:`ClipRef`."""
    analysis = read_view_analysis(ref.scar_struct_path, ref.key.view)
    prepared = prepare_clip(
        read_clip(ref.clip_path), cycle_frame_limits=analysis.cycle_frame_limits
    )
    target = read_gt_strain(ref.gt_strain_path).target_vector(ref.key.view)
    return prepared, analysis, target


def test_every_complete_clip_yields_frames_and_a_target(fake_cohort: Path):
    complete = [ref for ref in index_dataset(fake_cohort) if ref.is_complete]
    assert complete, "fixture should provide complete clips"

    for ref in complete:
        prepared, analysis, target = load_example(ref)

        assert prepared.n_frames == analysis.n_cycle_frames
        assert target.shape == (N_SEGMENTS + 1,)
        assert np.isfinite(target).all()


def test_frame_count_follows_the_view_specific_cycle(fake_cohort: Path):
    """2ch and 4ch have different cycle windows, so must not share a length."""
    refs = {
        ref.key.view: ref
        for ref in index_dataset(fake_cohort, datasets=("Retest_dataset",))
        if ref.key.patient == "IV02"
    }

    for view, expected in (("2ch", (51, 94)), ("4ch", (50, 93))):
        prepared, analysis, _ = load_example(refs[view])
        assert analysis.cycle_frame_limits == expected
        assert prepared.cycle_frame_limits == expected
        assert prepared.n_frames == expected[1] - expected[0] + 1


def test_target_differs_between_views(fake_cohort: Path):
    """A per-view target that came out identical would mean a lookup bug."""
    ref = next(
        ref
        for ref in index_dataset(fake_cohort, datasets=("Retest_dataset",))
        if ref.key.patient == "IV02" and ref.key.view == "2ch"
    )
    gt = read_gt_strain(ref.gt_strain_path)

    assert not np.allclose(gt.target_vector("2ch"), gt.target_vector("4ch"))


def test_frames_first_gives_the_per_frame_view_a_model_needs(fake_cohort: Path):
    ref = next(ref for ref in index_dataset(fake_cohort) if ref.is_complete)
    prepared, _, _ = load_example(ref)

    stacked = frames_first(prepared.images)

    assert stacked.shape == (prepared.n_frames,) + prepared.images.shape[:2]
    np.testing.assert_array_equal(stacked[0], prepared.images[:, :, 0])


def test_incomplete_clips_are_excluded_by_the_completeness_flag(fake_cohort: Path):
    refs = index_dataset(fake_cohort)
    usable = [ref for ref in refs if ref.is_complete]

    assert len(usable) < len(refs)
    assert all(ref.gt_strain_path is not None for ref in usable)


def test_groups_partition_the_usable_clips(fake_cohort: Path):
    """Every clip belongs to exactly one group, and groups never span datasets."""
    usable = [ref for ref in index_dataset(fake_cohort) if ref.is_complete]

    by_group: dict[str, list] = {}
    for ref in usable:
        by_group.setdefault(ref.group_id, []).append(ref)

    assert sum(len(clips) for clips in by_group.values()) == len(usable)
    for group_id, clips in by_group.items():
        assert len({ref.key.dataset for ref in clips}) == 1
        assert group_id.startswith(clips[0].key.dataset)


def test_real_iv02_walk(dataset_root: Path):
    """The same walk over the actual IV02 data, when the drive is mounted."""
    refs = [
        ref
        for ref in index_dataset(dataset_root, datasets=("Retest_dataset",))
        if ref.key.patient == "IV02"
    ]
    if not refs:
        pytest.skip("IV02 not present")

    for ref in refs:
        if not ref.is_complete:
            pytest.skip(f"{ref.key} incomplete: {ref.missing}")
        prepared, analysis, target = load_example(ref)

        assert prepared.n_frames == 44
        assert prepared.images.ndim == 3
        assert target.shape == (N_SEGMENTS + 1,)
        assert (prepared.pixel_spacing_mm[:2] > 0).all()
        assert analysis.n_samples == 159
