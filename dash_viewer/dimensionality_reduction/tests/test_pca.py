"""PCA behaviour that the dashboard depends on.

calculate_pca runs on a Dash callback worker, not the main thread, so anything
that needs a GUI event loop takes the tab down with it. It also has to hand back
components whose count matches what the caller asked for, because the viewer
stores that number and labels plots with it.
"""
import threading

import numpy as np
import pytest

from dash_viewer.dimensionality_reduction import pca_viewer
from dash_viewer.dimensionality_reduction.pca_viewer import calculate_pca, reconstruct_traces_pca

from .conftest import make_traces


@pytest.fixture
def curves(rng):
    # 60 studies x 120 samples, correlated so the components mean something
    return rng.normal(size=(60, 120)) @ rng.normal(size=(120, 120))


def test_does_not_open_a_gui_window(curves, monkeypatch):
    """plt.show() needs the main thread; opening one from a callback kills the tab."""
    calls = []
    monkeypatch.setattr(pca_viewer, 'plot_cum_variance_with_k',
                        lambda *a, **k: calls.append(a))
    monkeypatch.delenv('CARDIO_PCA_PLOTS', raising=False)

    calculate_pca(3, curves)

    assert calls == [], 'variance plot must stay opt-in inside the server'


def test_plot_is_opt_in_for_scripts(curves, monkeypatch):
    calls = []
    monkeypatch.setattr(pca_viewer, 'plot_cum_variance_with_k',
                        lambda *a, **k: calls.append(a))
    monkeypatch.setenv('CARDIO_PCA_PLOTS', '1')

    calculate_pca(3, curves)

    assert len(calls) == 1, 'CARDIO_PCA_PLOTS=1 should still give a script its plot'


def test_runs_on_a_worker_thread(curves):
    """Dash callbacks never run on the main thread."""
    outcome = {}

    def run():
        try:
            outcome['result'] = calculate_pca(3, curves)
        except BaseException as exc:  # noqa: BLE001 - the point is to report it
            outcome['error'] = exc

    thread = threading.Thread(target=run)
    thread.start()
    thread.join(timeout=120)

    assert 'error' not in outcome, f"PCA failed off the main thread: {outcome.get('error')}"
    pca_result, _, n_used = outcome['result']
    assert pca_result.shape == (60, n_used)


def test_requested_components_are_respected(curves):
    _, _, n_used = calculate_pca(3, curves)
    assert n_used <= 3


def test_full_rank_reconstruction_is_lossless(curves):
    """n_components=100 means 'all of them', so nothing should be lost."""
    pca_result, pca_var, n_used = calculate_pca(100, curves)
    reconstructed = reconstruct_traces_pca(pca_result, pca_var)

    assert reconstructed.shape == curves.shape
    assert np.abs(reconstructed - curves).max() < 1e-6


def test_rejects_non_numeric_input(rng):
    """A 1-D array of strings is what the old echo branch built; it must not pass silently."""
    with pytest.raises(ValueError):
        calculate_pca(3, np.array(['1.0 2.0', '3.0 4.0']))


def test_truncating_a_fit_matches_refitting(curves):
    """The truncated model must be indistinguishable from PCA(n_components=k)."""
    from sklearn.decomposition import PCA

    full = PCA().fit(curves)
    truncated = pca_viewer.truncate_pca(full, 5)
    refitted = PCA(n_components=5, svd_solver='full').fit(curves)

    assert truncated.n_components_ == refitted.n_components_
    assert np.allclose(np.abs(truncated.components_), np.abs(refitted.components_))
    assert np.allclose(truncated.explained_variance_ratio_, refitted.explained_variance_ratio_)
    assert np.allclose(np.abs(truncated.transform(curves)), np.abs(refitted.transform(curves)))
    assert np.allclose(truncated.inverse_transform(truncated.transform(curves)),
                       refitted.inverse_transform(refitted.transform(curves)))


def test_requested_components_are_not_silently_capped(curves):
    """Asking for more than the 95% count used to hand back the 95% count."""
    _, _, ninety_five = calculate_pca(0, curves)
    requested = ninety_five + 10

    _, _, n_used = calculate_pca(requested, curves)

    assert n_used == requested, 'the slider value must be what actually gets used'


def test_request_beyond_available_components_is_clamped(curves):
    _, _, n_used = calculate_pca(99, curves)

    assert n_used <= min(curves.shape)


def test_zero_means_the_ninety_five_percent_count(curves):
    _, pca_var, n_used = calculate_pca(0, curves)

    assert float(np.sum(pca_var.explained_variance_ratio_)) >= 0.95
    smaller = pca_viewer.truncate_pca(pca_viewer.PCA().fit(curves), n_used - 1)
    assert float(np.sum(smaller.explained_variance_ratio_)) < 0.95, 'should be the smallest such count'


def test_holdout_r2_is_reported_separately_from_in_sample(curves):
    """In-sample explained variance says nothing about an unseen study."""
    train_r2, held_out_r2, k = pca_viewer.holdout_r2(curves, n_components=5)

    assert k == 5
    assert train_r2 > held_out_r2, 'in-sample fit should exceed held-out fit'
    assert -5 < held_out_r2 <= 1


def test_holdout_is_skipped_for_a_tiny_cohort(rng):
    assert pca_viewer.holdout_r2(rng.normal(size=(3, 10)), n_components=2) is None


def test_r2_of_a_full_rank_reconstruction_is_one(curves):
    from sklearn.decomposition import PCA

    pca = PCA().fit(curves)

    assert pca_viewer.r2_score_reconstruction(pca, curves) == pytest.approx(1.0)
