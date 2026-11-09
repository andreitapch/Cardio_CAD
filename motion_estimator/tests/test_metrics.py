"""Tests for the masked regression metrics.

Run with:  python -m pytest motion_estimator/tests/test_metrics.py
"""
from __future__ import annotations

import numpy as np
import pytest

from motion_estimator.echoclip.metrics import TARGET_LABELS, masked_metrics


def test_perfect_prediction_scores_zero_error():
    truth = np.arange(21, dtype=float).reshape(3, 7)

    results = masked_metrics(truth, truth.copy())

    for result in results:
        assert result.n == 3
        assert result.mae == pytest.approx(0.0)
        assert result.rmse == pytest.approx(0.0)
        assert result.bias == pytest.approx(0.0)


def test_nan_truth_is_excluded_and_n_reflects_it():
    """A clip missing seg5 must still contribute to every other target."""
    truth = np.zeros((4, 7))
    truth[1, 4] = np.nan
    truth[2, 4] = np.nan

    results = masked_metrics(truth, np.zeros((4, 7)))

    assert [r.n for r in results] == [4, 4, 4, 4, 2, 4, 4]


def test_all_nan_column_reports_zero_n_not_a_crash():
    truth = np.zeros((3, 7))
    truth[:, 3] = np.nan

    results = masked_metrics(truth, np.zeros((3, 7)))

    assert results[3].n == 0
    assert np.isnan(results[3].mae)


def test_nan_prediction_is_excluded_too():
    """A target the model could not fit is not scored as if it predicted zero."""
    prediction = np.zeros((3, 7))
    prediction[:, 2] = np.nan

    results = masked_metrics(np.zeros((3, 7)), prediction)

    assert results[2].n == 0
    assert results[0].n == 3


def test_bias_is_signed_and_mae_is_not():
    truth = np.zeros((4, 7))
    prediction = np.full((4, 7), -2.0)

    results = masked_metrics(truth, prediction)

    assert results[0].bias == pytest.approx(-2.0)
    assert results[0].mae == pytest.approx(2.0)


def test_limits_of_agreement_are_bias_plus_minus_1_96_sd():
    truth = np.zeros((6, 7))
    prediction = np.zeros((6, 7))
    prediction[:, 0] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]

    result = masked_metrics(truth, prediction)[0]

    difference = np.array([1.0, 2, 3, 4, 5, 6])
    assert result.bias == pytest.approx(difference.mean())
    assert result.loa_lower == pytest.approx(
        difference.mean() - 1.96 * difference.std(ddof=1)
    )
    assert result.loa_upper == pytest.approx(
        difference.mean() + 1.96 * difference.std(ddof=1)
    )


def test_constant_prediction_gives_nan_correlation_not_a_warning():
    """The mean-only baseline predicts a constant; r is undefined, not zero."""
    truth = np.tile(np.arange(7, dtype=float), (5, 1))
    truth[:, 0] = [1.0, 2, 3, 4, 5]

    result = masked_metrics(truth, np.zeros((5, 7)))[0]

    assert np.isnan(result.pearson)


def test_correlation_is_computed_on_the_masked_subset():
    truth = np.zeros((5, 7))
    truth[:, 0] = [1.0, 2, 3, 4, np.nan]
    prediction = np.zeros((5, 7))
    prediction[:, 0] = [2.0, 4, 6, 8, 999]

    result = masked_metrics(truth, prediction)[0]

    assert result.n == 4
    assert result.pearson == pytest.approx(1.0)


def test_shape_mismatch_is_rejected():
    with pytest.raises(ValueError, match="shape mismatch"):
        masked_metrics(np.zeros((3, 7)), np.zeros((3, 6)))


def test_labels_are_six_segments_then_gls():
    assert TARGET_LABELS == ("seg1", "seg2", "seg3", "seg4", "seg5", "seg6", "GLS")
    assert [r.label for r in masked_metrics(np.zeros((2, 7)), np.zeros((2, 7)))] == list(
        TARGET_LABELS
    )
