"""Tests for pooling, feature construction and grouped cross-validation.

The leakage guarantees are the point here: no group may appear in both sides of
a split, and a clip with a partly-missing target must still train the targets
it does have.

Run with:  python -m pytest motion_estimator/tests/test_probe.py
"""
from __future__ import annotations

import numpy as np
import pytest

from motion_estimator.loaders.dataset_index import ClipKey
from motion_estimator.echoclip.embed_cache import EmbeddingDataset, group_keys
from motion_estimator.echoclip.probe import build_features, pool_frames, run_cv


def make_dataset(n_patients=10, n_frames=4, dim=6, seed=0, grouping="dataset_patient"):
    """A synthetic cohort where the target really is a function of the features."""
    rng = np.random.default_rng(seed)
    keys, embeddings, targets, views, datasets = [], [], [], [], []

    weights = rng.normal(size=(dim, 7))
    for patient in range(n_patients):
        for dataset in ("Retest_dataset", "Test_dataset"):
            for view in ("2ch", "4ch"):
                key = ClipKey(dataset, f"IV{patient:02d}", view)
                base = rng.normal(size=dim)
                frames = base + 0.01 * rng.normal(size=(n_frames, dim))
                keys.append(key)
                embeddings.append(frames)
                targets.append(base @ weights + 0.1 * rng.normal(size=7))
                views.append(view)
                datasets.append(dataset)

    return EmbeddingDataset(
        keys=keys,
        embeddings=np.asarray(embeddings),
        targets=np.asarray(targets),
        groups=group_keys(keys, grouping),
        views=np.asarray(views),
        datasets=np.asarray(datasets),
        phase="end_systolic",
        config={},
    )


# ------------------------------------------------------------------- pooling
def test_pool_frames_collapses_the_frame_axis():
    assert pool_frames(np.ones((5, 4, 6))).shape == (5, 6)


def test_pooled_vectors_are_unit_norm():
    rng = np.random.default_rng(0)
    pooled = pool_frames(rng.normal(size=(7, 4, 6)))

    np.testing.assert_allclose(np.linalg.norm(pooled, axis=-1), 1.0)


def test_pool_frames_survives_an_all_zero_clip():
    """Renormalising must not divide by zero on a degenerate clip."""
    pooled = pool_frames(np.zeros((2, 4, 6)))

    assert np.isfinite(pooled).all()


# ------------------------------------------------------------------ features
def test_features_append_the_view_as_one_column():
    data = make_dataset()

    features = build_features(data)

    assert features.shape == (len(data), data.embeddings.shape[-1] + 1)


def test_view_column_marks_4ch():
    data = make_dataset()

    features = build_features(data)

    np.testing.assert_array_equal(features[:, -1], (data.views == "4ch").astype(float))


# ------------------------------------------------------------------------ cv
def test_no_group_appears_in_both_sides_of_a_split():
    """The leakage guarantee, asserted directly rather than trusted."""
    data = make_dataset()

    result = run_cv(data, "ridge", alpha=1.0, n_splits=5)

    all_test = np.concatenate([fold.indices for fold in result.folds])
    for fold in result.folds:
        test_groups = set(data.groups[fold.indices])
        train_groups = set(data.groups[np.setdiff1d(np.arange(len(data)), fold.indices)])
        assert not (test_groups & train_groups)
    # and every clip is tested exactly once
    np.testing.assert_array_equal(np.sort(all_test), np.arange(len(data)))


def test_both_views_of_a_clip_stay_on_the_same_side():
    data = make_dataset()

    result = run_cv(data, "ridge", alpha=1.0, n_splits=5)

    for fold in result.folds:
        held = {data.keys[i] for i in fold.indices}
        for key in held:
            partner = ClipKey(key.dataset, key.patient, "4ch" if key.view == "2ch" else "2ch")
            if partner in set(data.keys):
                assert partner in held


def test_patient_grouping_keeps_both_scans_together():
    data = make_dataset(grouping="patient")

    result = run_cv(data, "ridge", alpha=1.0, n_splits=5)

    for fold in result.folds:
        held_patients = {data.keys[i].patient for i in fold.indices}
        for index in range(len(data)):
            if data.keys[index].patient in held_patients:
                assert index in set(fold.indices)


def test_ridge_beats_the_mean_baseline_on_learnable_data():
    """A sanity check on the machinery, not a claim about the real cohort."""
    data = make_dataset(n_patients=30)

    ridge = run_cv(data, "ridge", alpha=0.1, n_splits=5)
    mean = run_cv(data, "mean", n_splits=5)

    assert ridge.metrics[-1].mae < mean.metrics[-1].mae


def test_every_clip_gets_an_out_of_fold_prediction():
    data = make_dataset()

    result = run_cv(data, "ridge", alpha=1.0, n_splits=5)

    assert np.isfinite(result.predictions).all()


def test_partially_missing_targets_still_train_their_other_columns():
    """A clip missing seg5 must not be dropped from seg1-4 and GLS."""
    data = make_dataset(n_patients=12)
    data.targets[::3, 4] = np.nan

    result = run_cv(data, "ridge", alpha=0.1, n_splits=4)

    assert result.metrics[4].n < len(data)  # seg5 lost the NaN rows
    assert result.metrics[0].n == len(data)  # seg1 kept everything
    assert np.isfinite(result.predictions[:, 0]).all()


def test_a_fully_missing_target_is_reported_not_crashed():
    data = make_dataset(n_patients=12)
    data.targets[:, 3] = np.nan

    result = run_cv(data, "ridge", alpha=0.1, n_splits=4)

    assert result.metrics[3].n == 0
    assert np.isnan(result.predictions[:, 3]).all()


def test_mean_baseline_ignores_the_features_entirely():
    data = make_dataset()
    scrambled = make_dataset()
    scrambled.embeddings[:] = np.random.default_rng(1).normal(
        size=scrambled.embeddings.shape
    )

    first = run_cv(data, "mean", n_splits=5)
    second = run_cv(scrambled, "mean", n_splits=5)

    np.testing.assert_allclose(first.predictions, second.predictions)


def test_mean_baseline_uses_only_training_fold_values():
    """Using the test fold's own mean would be leakage that flatters the floor."""
    data = make_dataset(n_patients=8)

    result = run_cv(data, "mean", n_splits=4)

    for fold in result.folds:
        test_mean = np.nanmean(data.targets[fold.indices, -1])
        predicted = result.predictions[fold.indices, -1][0]
        assert not np.isclose(predicted, test_mean, atol=1e-9)


def test_unknown_estimator_is_rejected():
    with pytest.raises(ValueError, match="unknown estimator"):
        run_cv(make_dataset(), "magic")


def test_grouping_choices_are_validated():
    with pytest.raises(ValueError, match="grouping must be"):
        group_keys([ClipKey("Retest_dataset", "IV02", "2ch")], "by-vibes")


def test_regrouped_changes_only_the_group_key():
    data = make_dataset()

    patient_grouped = data.regrouped("patient")

    assert data.n_groups == 2 * patient_grouped.n_groups
    np.testing.assert_array_equal(data.embeddings, patient_grouped.embeddings)
    np.testing.assert_array_equal(data.targets, patient_grouped.targets)
