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
    _augment_with_imputed, choose_threshold, find_best_class_params)


def make_frame(n=200, informative=True, seed=0, n_features=8):
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

    frame = pd.DataFrame({'study_nr': np.arange(n), 'Total_Remodeling': y})
    for i in range(n_features):
        frame[f'f{i}'] = X[:, i]
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
                                         imp_studies, held_out_studies={11, 13})

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
