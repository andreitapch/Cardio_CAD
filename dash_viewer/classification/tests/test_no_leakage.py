"""Regression tests for the classifier evaluation pipeline.

The central guarantee: on data with no signal, reported metrics must sit at
chance. Before nested CV and training-only threshold selection were introduced,
the cutoff was fitted on each test fold's own labels, so MCC and F1 came out
well above chance even on pure noise.

Run with:  python -m pytest dash_viewer/classification/tests/test_no_leakage.py

Optional heavy deps (dash, xgboost, shap, lightgbm, plotly, numba) are stubbed
if absent so these tests run without the full application environment.
"""
import sys
import types

import numpy as np
import pandas as pd
import pytest


def _stub_missing_dependencies():
    """Insert no-op stand-ins for deps that aren't needed by the code under test."""
    class _Dummy:
        def __init__(self, *a, **k):
            pass

    stubs = {
        'dash': dict(dash_table=types.SimpleNamespace(DataTable=lambda **k: k)),
        'plotly': {},
        'plotly.figure_factory': dict(create_annotated_heatmap=lambda *a, **k: None),
        'xgboost': dict(XGBClassifier=_Dummy),
        'lightgbm': dict(LGBMClassifier=_Dummy),
        'shap': dict(Explainer=_Dummy,
                     plots=types.SimpleNamespace(beeswarm=lambda *a, **k: None)),
        'numba': dict(njit=lambda f=None, **k: (f if f else (lambda g: g))),
    }
    for name, attrs in stubs.items():
        try:
            __import__(name)
        except ImportError:
            module = types.ModuleType(name)
            for key, value in attrs.items():
                setattr(module, key, value)
            sys.modules[name] = module


_stub_missing_dependencies()

from dash_viewer.classification.model_params import ModelParams, DEFAULT_SEARCH_BUDGET
from dash_viewer.classification.machine_learning.utils_classifier import (
    DropUninformativeColumns, _augment_with_imputed, build_estimator_pipeline,
    choose_threshold, find_best_class_params, make_splitter,
    prepare_train_test_data, resolve_exam_groups)


def make_frame(n=200, informative=True, seed=0, n_features=8, duplicate_exams=0,
               missing_frac=0.0):
    """Build a frame shaped the way prepare_train_test_data expects.

    Columns are ['study_nr', 'Total_Remodeling', f0..fN]; everything from index
    2 onwards is treated as a feature.
    """
    if informative:
        from sklearn.datasets import make_classification
        X, y = make_classification(n_samples=n, n_features=n_features, n_informative=4,
                                   n_redundant=0, weights=[0.75, 0.25], random_state=seed)
    else:
        rng = np.random.default_rng(seed)
        X = rng.normal(size=(n, n_features))
        y = rng.integers(0, 2, n)  # labels independent of X

    if missing_frac:
        rng = np.random.default_rng(seed + 99)
        X = X.copy()
        X[rng.random(X.shape) < missing_frac] = np.nan

    frame = pd.DataFrame({'study_nr': np.arange(n), 'Total_Remodeling': y})
    for i in range(n_features):
        frame[f'f{i}'] = X[:, i]

    if duplicate_exams:
        # A few exams contribute two rows, as echo_biomarkers_view does for
        # studies recorded with two EDV sources: same study_nr, same label.
        extra = frame.head(duplicate_exams).copy()
        frame = pd.concat([frame, extra], ignore_index=True)
    return frame


def run_pipeline(frame, seed=0, **kwargs):
    options = dict(n_outer_splits=3, n_inner_splits=2, search_budget=3, random_state=seed)
    options.update(kwargs)
    return find_best_class_params(frame, 'f0', None, None, 'None', 'RF_class',
                                  'class_features', **options)


@pytest.mark.parametrize('class_type', [
    'KNN_class', 'DT_class', 'RF_class', 'GBM_class', 'LogisticRegression_class',
])
def test_every_param_grid_builds(class_type):
    """LogisticRegression used a list-of-dicts grid that raised AttributeError."""
    params = ModelParams()
    params.get_param_grid(class_type, 'None')
    assert params.param_combinations
    assert len(params.param_combinations) <= DEFAULT_SEARCH_BUDGET
    assert all(isinstance(combo, dict) for combo in params.param_combinations)


def test_search_budget_can_be_disabled():
    params = ModelParams()
    params.get_param_grid('RF_class', 'None', n_iter=None)
    assert len(params.param_combinations) == 6144


def test_choose_threshold_stays_in_bounds():
    """precision_recall_curve returns len(thresholds) == len(precision) - 1."""
    rng = np.random.default_rng(0)
    for _ in range(200):
        n = int(rng.integers(10, 60))
        y = rng.integers(0, 2, n)
        if len(set(y)) < 2:
            continue
        threshold = choose_threshold(y, rng.random(n))
        assert np.isfinite(threshold)


def test_imputed_rows_for_held_out_studies_are_dropped():
    """Imputed copies of evaluated patients must never reach the training fold."""
    X_train, y_train = np.zeros((4, 3)), np.zeros(4, dtype=int)
    X_imp = np.arange(15, dtype=float).reshape(5, 3)
    y_imp = np.array([0, 1, 0, 1, 0])
    imp_studies = np.array([10, 11, 12, 13, 14])

    X_aug, y_aug = _augment_with_imputed(X_train, y_train, X_imp, y_imp,
                                         imp_studies, held_out_keys={11, 13})

    assert X_aug.shape[0] == 4 + 3
    assert y_aug.shape[0] == 4 + 3
    kept_rows = X_aug[:3]
    for excluded in (X_imp[1], X_imp[3]):
        assert not any(np.array_equal(excluded, row) for row in kept_rows)


def test_learns_real_signal():
    result = run_pipeline(make_frame(informative=True))
    metrics = result[3]
    assert isinstance(metrics, dict)
    assert metrics['AUC_global'] > 0.7, metrics


def test_no_signal_scores_at_chance():
    """The leakage guard.

    AUC and MCC are the honest indicators here. F1 is not: a model that predicts
    almost everything positive scores ~0.66 on balanced random labels, which is
    why F1 alone was never evidence that the old pipeline worked.
    """
    aucs, mccs = [], []
    for seed in range(4):
        metrics = run_pipeline(make_frame(n=200, informative=False, seed=seed), seed=seed)[3]
        assert isinstance(metrics, dict)
        aucs.append(metrics['AUC_global'])
        mccs.append(metrics['MCC'])

    assert 0.35 < np.mean(aucs) < 0.65, f'AUC on noise was {np.mean(aucs):.3f}'
    assert abs(np.mean(mccs)) < 0.15, f'MCC on noise was {np.mean(mccs):.3f}'


def test_pipeline_preprocessing_is_fit_on_training_data_only():
    """The scaler must not see held-out rows.

    Fit on a training block whose values are tiny, then transform a block whose
    values are huge: if the scaler had been fit over both, the transformed
    values would be near zero.
    """
    X_train = np.random.default_rng(0).normal(loc=0.0, scale=1.0, size=(60, 4))
    X_held_out = np.full((10, 4), 1000.0)
    y_train = np.tile([0, 1], 30)

    from sklearn.tree import DecisionTreeClassifier
    pipeline = build_estimator_pipeline(DecisionTreeClassifier(random_state=0), 'None')
    pipeline.fit(X_train, y_train)

    scaler = pipeline.named_steps['scale']
    assert np.all(np.abs(scaler.mean_) < 1.0), scaler.mean_

    transformed = pipeline[:-1].transform(X_held_out)
    assert np.all(transformed > 10), transformed.max()


def test_drop_uninformative_columns_learns_from_training_data():
    X = np.array([
        [1.0, 5.0, np.nan, 2.0],
        [2.0, 5.0, np.nan, 3.0],
        [3.0, 5.0, np.nan, 4.0],
        [4.0, 5.0, 1.0, 5.0],
    ])
    dropper = DropUninformativeColumns(missing_threshold=30.0).fit(X)
    # column 1 is constant, column 2 is 75% missing
    assert list(dropper.columns_to_keep_) == [True, False, False, True]
    assert dropper.transform(X).shape == (4, 2)


def test_pipeline_imputes_missing_values():
    frame = make_frame(n=160, informative=True, missing_frac=0.15)
    metrics = run_pipeline(frame)[3]
    assert isinstance(metrics, dict), 'NaNs should be handled inside the pipeline'
    assert 'AUC_global' in metrics


def test_no_grouping_when_every_row_is_a_distinct_exam():
    """Patient_ID and study_nr are 1:1 here, so grouping is normally unnecessary.

    Returning None keeps the plain stratified splitter, which balances classes
    better than the grouped one.
    """
    assert resolve_exam_groups(make_frame(n=20)) is None


def test_grouping_kicks_in_when_an_exam_has_several_rows():
    frame = make_frame(n=20, duplicate_exams=3)
    groups = resolve_exam_groups(frame)
    assert groups is not None
    assert len(groups) == 23
    assert len(set(groups)) == 20


def test_duplicate_exam_rows_never_straddle_a_fold():
    frame = make_frame(n=120, duplicate_exams=12)
    groups = resolve_exam_groups(frame)
    y = frame['Total_Remodeling'].values
    X = frame[[c for c in frame.columns if c.startswith('f')]].values

    splitter, split_groups = make_splitter(3, groups)
    for train_idx, test_idx in splitter.split(X, y, groups=split_groups):
        assert not set(groups[train_idx]) & set(groups[test_idx])


def test_run_with_duplicate_exams_still_produces_metrics():
    frame = make_frame(n=180, informative=True, duplicate_exams=8)
    metrics = run_pipeline(frame)[3]
    assert isinstance(metrics, dict)
    assert metrics['AUC_global'] > 0.6, metrics


def test_identifier_columns_are_never_features():
    """Identifier columns must not reach the classifier.

    Patient_ID is 1:1 with study_nr here, so it is effectively a row index and
    a near-perfect label proxy if a frame ever carries it through.
    """
    frame = make_frame(n=40)
    frame['Patient_ID'] = np.arange(40)

    *_, feature_names_orig, _ = prepare_train_test_data(
        frame, 'f0', None, None, 'class_features')

    assert 'Patient_ID' not in feature_names_orig
    assert 'study_nr' not in feature_names_orig
    assert 'Total_Remodeling' not in feature_names_orig
