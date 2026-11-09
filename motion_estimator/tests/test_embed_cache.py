"""Tests for the embedding cache: resumability, staleness, assembly.

A stub encoder stands in for EchoCLIP so these run with no GPU and no network.
The stub returns embeddings derived from the pixels, so a mixed-up clip shows
up as a value mismatch rather than passing on shape alone.

Run with:  python -m pytest motion_estimator/tests/test_embed_cache.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from motion_estimator.loaders.dataset_index import index_dataset
from motion_estimator.echoclip.config import EncoderConfig
from motion_estimator.echoclip.embed_cache import (
    build_cache,
    clip_cache_path,
    load_embeddings,
    read_cache_meta,
)


class StubEncoder:
    """Deterministic stand-in: embedding rows encode the clip's own pixels."""

    def __init__(self, config=None, embed_dim=8):
        self.config = config or EncoderConfig()
        self._embed_dim = embed_dim
        self.calls = 0

    @property
    def embed_dim(self):
        return self._embed_dim

    def encode_clip(self, frames):
        self.calls += 1
        n = self.config.n_frames
        seed = int(np.asarray(frames).sum()) % (2**31)
        rows = np.linspace(seed, seed + n, n)[:, None] + np.arange(self._embed_dim)
        rows = rows.astype(np.float32)
        return rows / np.linalg.norm(rows, axis=-1, keepdims=True)


@pytest.fixture
def cohort(fake_cohort: Path):
    return index_dataset(fake_cohort)


def test_builds_one_file_per_complete_clip(cohort, tmp_path: Path):
    encoder = StubEncoder()

    report = build_cache(cohort, encoder, tmp_path / "cache")

    assert len(report.written) == 4
    assert len(report.skipped) == 4  # the IV03 clips have no ground truth
    assert not report.failed
    assert encoder.calls == 4


def test_skips_record_why(cohort, tmp_path: Path):
    report = build_cache(cohort, StubEncoder(), tmp_path / "cache")

    reasons = {str(key): reason for key, reason in report.skipped}
    assert all("gt_strain" in reason for reason in reasons.values())
    assert all("IV03" in key for key in reasons)


def test_rerun_reuses_and_does_not_recompute(cohort, tmp_path: Path):
    cache = tmp_path / "cache"
    build_cache(cohort, StubEncoder(), cache)

    second = StubEncoder()
    report = build_cache(cohort, second, cache)

    assert len(report.reused) == 4
    assert not report.written
    assert second.calls == 0


def test_overwrite_forces_recompute(cohort, tmp_path: Path):
    cache = tmp_path / "cache"
    build_cache(cohort, StubEncoder(), cache)

    second = StubEncoder()
    report = build_cache(cohort, second, cache, overwrite=True)

    assert len(report.written) == 4
    assert second.calls == 4


def test_an_interrupted_build_resumes(cohort, tmp_path: Path):
    """Deleting one cached clip must recompute exactly that clip."""
    cache = tmp_path / "cache"
    build_cache(cohort, StubEncoder(), cache)
    victim = [ref for ref in cohort if ref.is_complete][1]
    clip_cache_path(cache, victim.key).unlink()

    encoder = StubEncoder()
    report = build_cache(cohort, encoder, cache)

    assert encoder.calls == 1
    assert report.written == [victim.key]
    assert len(report.reused) == 3


def test_no_tmp_files_are_left_behind(cohort, tmp_path: Path):
    cache = tmp_path / "cache"
    build_cache(cohort, StubEncoder(), cache)

    assert not list(cache.glob("*.tmp"))


def test_a_failing_clip_does_not_abort_the_pass(cohort, tmp_path: Path):
    class Flaky(StubEncoder):
        def encode_clip(self, frames):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("simulated CUDA hiccup")
            return super().encode_clip(frames)

    report = build_cache(cohort, Flaky(), tmp_path / "cache")

    assert len(report.failed) == 1
    assert "simulated CUDA hiccup" in report.failed[0][1]
    assert len(report.written) == 3  # the rest still got done


def test_meta_records_the_fingerprint(cohort, tmp_path: Path):
    cache = tmp_path / "cache"
    build_cache(cohort, StubEncoder(EncoderConfig(sampling="upstream")), cache)

    meta = read_cache_meta(cache)

    assert meta["config"]["sampling"] == "upstream"
    assert meta["embed_dim"] == 8
    assert "written_at" in meta


def test_settings_change_is_refused_rather_than_mixed(cohort, tmp_path: Path):
    """Mixing fingerprints would give a dataset whose rows are incomparable."""
    cache = tmp_path / "cache"
    build_cache(cohort, StubEncoder(), cache)

    with pytest.raises(ValueError, match="different encoder settings"):
        build_cache(cohort, StubEncoder(EncoderConfig(sampling="upstream")), cache)


def test_settings_change_is_allowed_with_overwrite(cohort, tmp_path: Path):
    cache = tmp_path / "cache"
    build_cache(cohort, StubEncoder(), cache)

    report = build_cache(
        cohort, StubEncoder(EncoderConfig(n_frames=5)), cache, overwrite=True
    )

    assert len(report.written) == 4
    assert read_cache_meta(cache)["config"]["n_frames"] == 5


# ------------------------------------------------------------------- loading
def test_load_assembles_aligned_arrays(cohort, tmp_path: Path):
    cache = tmp_path / "cache"
    build_cache(cohort, StubEncoder(), cache)

    data = load_embeddings(cohort, cache)

    assert len(data) == 4
    assert data.embeddings.shape == (4, 20, 8)
    assert data.targets.shape == (4, 7)
    assert data.groups.shape == (4,)
    assert set(data.views) == {"2ch", "4ch"}


def test_loaded_rows_stay_matched_to_their_clip(cohort, tmp_path: Path):
    """Row i's embedding, target and group must all describe the same clip."""
    cache = tmp_path / "cache"
    build_cache(cohort, StubEncoder(), cache)

    data = load_embeddings(cohort, cache)

    for index, key in enumerate(data.keys):
        np.testing.assert_array_equal(
            data.embeddings[index], np.load(clip_cache_path(cache, key))
        )
        assert data.groups[index] == f"{key.dataset}/{key.patient}"
        assert data.views[index] == key.view
        assert data.datasets[index] == key.dataset


def test_targets_default_to_end_systolic_and_track_the_view(cohort, tmp_path: Path):
    from motion_estimator.loaders.annotations.gt_strain import read_gt_strain

    cache = tmp_path / "cache"
    build_cache(cohort, StubEncoder(), cache)

    data = load_embeddings(cohort, cache)

    assert data.phase == "end_systolic"
    for index, key in enumerate(data.keys):
        ref = next(r for r in cohort if r.key == key)
        expected = read_gt_strain(ref.gt_strain_path).target_vector(key.view)
        np.testing.assert_allclose(data.targets[index], expected)


def test_phase_is_selectable_without_rebuilding(cohort, tmp_path: Path):
    """Changing the strain phase must not invalidate a GPU pass."""
    cache = tmp_path / "cache"
    build_cache(cohort, StubEncoder(), cache)

    end = load_embeddings(cohort, cache)
    peak = load_embeddings(cohort, cache, phase="peak_systolic")

    np.testing.assert_array_equal(end.embeddings, peak.embeddings)
    assert not np.allclose(end.targets, peak.targets)


def test_default_grouping_scopes_groups_by_dataset(cohort, tmp_path: Path):
    """Under 'dataset_patient', Retest/IV02 and Test/IV02 are separate groups.

    ``n_patients`` counts distinct people and so stays at 1 here -- the two
    folders reuse the same IV## ids. That gap between 1 patient and 2 groups is
    exactly what the grouping choice is about.
    """
    cache = tmp_path / "cache"
    build_cache(cohort, StubEncoder(), cache)

    data = load_embeddings(cohort, cache)

    assert data.n_groups == 2
    assert data.n_patients == 1
    for group in np.unique(data.groups):
        assert len(np.unique(data.datasets[data.groups == group])) == 1


def test_patient_grouping_merges_the_two_folders(cohort, tmp_path: Path):
    cache = tmp_path / "cache"
    build_cache(cohort, StubEncoder(), cache)

    data = load_embeddings(cohort, cache, grouping="patient")

    assert data.n_groups == 1
    assert set(np.unique(data.datasets)) == {"Retest_dataset", "Test_dataset"}


def test_subset_keeps_everything_aligned(cohort, tmp_path: Path):
    cache = tmp_path / "cache"
    build_cache(cohort, StubEncoder(), cache)
    data = load_embeddings(cohort, cache)

    retest = data.subset(data.datasets == "Retest_dataset")

    assert len(retest) == 2
    assert set(retest.datasets) == {"Retest_dataset"}
    assert retest.embeddings.shape[0] == len(retest.keys) == len(retest.targets)


def test_load_without_a_cache_says_so(cohort, tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="build_cache"):
        load_embeddings(cohort, tmp_path / "empty")


def test_load_with_no_matching_clips_says_so(cohort, tmp_path: Path):
    cache = tmp_path / "cache"
    build_cache(cohort, StubEncoder(), cache)

    with pytest.raises(ValueError, match="no cached clips matched"):
        load_embeddings([], cache)


def test_inconsistent_shapes_are_caught(cohort, tmp_path: Path):
    cache = tmp_path / "cache"
    build_cache(cohort, StubEncoder(), cache)
    key = next(ref.key for ref in cohort if ref.is_complete)
    np.save(clip_cache_path(cache, key), np.zeros((7, 8), dtype=np.float32))

    with pytest.raises(ValueError, match="inconsistent shapes"):
        load_embeddings(cohort, cache)
