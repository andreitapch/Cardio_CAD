"""Combining strain traces with echo biomarkers for PCA.

The two blocks arrive in different shapes and units: traces are one row per
study in strain units, echo params are one row per *view* and already
standardised. Getting the alignment or the scaling wrong is silent - the PCA
still runs, the plots still render, and the numbers are wrong.
"""
import numpy as np
import pytest

from dash_viewer.dimensionality_reduction.dimen_reduction_viewer import DimensionalityRedViewer
from dash_viewer.dimensionality_reduction.pca_viewer import calculate_pca

from .conftest import (N_FEATURES, N_SEGMENTS, make_concat_tracesDF, make_echo_params,
                       make_study_nrs, make_traces, reassemble)

N_STUDIES = 12


@pytest.fixture
def viewer(rng):
    study_nrs = make_study_nrs(N_STUDIES)
    instance = DimensionalityRedViewer('/tab-PCA')
    instance.concat_traces_by = 'full_traces_echo'
    instance.study_nr = np.array(study_nrs)
    instance.array_curves = make_traces(rng, N_STUDIES)
    instance.concat_tracesDF = make_concat_tracesDF(study_nrs)
    instance.concat_echo_params = make_echo_params(rng, study_nrs)
    return instance


def test_echo_rows_collapse_to_one_per_study(viewer):
    """The DB returns a row per view; PCA needs a row per study."""
    order = viewer.concat_tracesDF['study_nr'].tolist()
    echo = viewer.build_echo_matrix(order)

    assert echo.shape == (N_STUDIES, 4)


def test_echo_rows_follow_the_trace_order(viewer, rng):
    """reconstruct_traces indexes study_nr positionally - order is load-bearing."""
    order = viewer.concat_tracesDF['study_nr'].tolist()
    shuffled = list(reversed(order))

    forward = viewer.build_echo_matrix(order)
    backward = viewer.build_echo_matrix(shuffled)

    assert np.allclose(forward, backward[::-1])


def test_study_without_echo_data_sits_at_the_cohort_mean(viewer, rng):
    """Dropping it would desynchronise the echo rows from the trace rows."""
    order = viewer.concat_tracesDF['study_nr'].tolist()
    viewer.concat_echo_params = make_echo_params(rng, order[:-1])

    echo = viewer.build_echo_matrix(order)

    assert echo.shape == (N_STUDIES, 4)
    assert np.allclose(echo[-1], 0.0), 'missing study should land on the standardised mean'


def test_missing_single_biomarker_does_not_produce_nan(viewer):
    viewer.concat_echo_params.loc[0, 'biomarker_2'] = np.nan

    echo = viewer.build_echo_matrix(viewer.concat_tracesDF['study_nr'].tolist())

    assert np.isfinite(echo).all()


def test_combined_matrix_is_numeric_and_wider_than_the_traces(viewer):
    combined = viewer.combine_traces_with_echo(
        viewer.array_curves, viewer.concat_tracesDF['study_nr'].tolist())

    assert combined.shape == (N_STUDIES, N_FEATURES + 4)
    assert combined.dtype.kind == 'f', 'features must be numeric, not joined strings'
    assert np.isfinite(combined).all()
    assert viewer.n_trace_features == N_FEATURES


def test_reconstruction_returns_strain_units(viewer):
    """The traces are standardised for the fit; plots and the DB need them back."""
    combined = viewer.combine_traces_with_echo(
        viewer.array_curves, viewer.concat_tracesDF['study_nr'].tolist())
    viewer.pca_result, viewer.pca_var_comp, viewer.n_components_used = calculate_pca(100, combined)

    viewer.reconstruct_traces(None)
    reconstructed = reassemble(viewer.DM_tracesDF)

    assert len(viewer.DM_tracesDF) == N_STUDIES * N_SEGMENTS
    assert reconstructed.shape == viewer.array_curves.shape
    # full rank in, full rank out: only the scaler round-trip is under test here
    assert np.abs(reconstructed - viewer.array_curves).max() < 1e-4


def test_plain_traces_path_is_left_unscaled(viewer):
    """Without echo params there is no scaler, and reconstruction must not invert one."""
    viewer.concat_traces_by = 'full_traces'
    viewer.load_pca_data = lambda: None  # no DB
    viewer.trace_scaler, viewer.n_trace_features = None, None

    viewer.pca_result, viewer.pca_var_comp, viewer.n_components_used = calculate_pca(
        100, viewer.array_curves)
    viewer.reconstruct_traces(None)
    reconstructed = reassemble(viewer.DM_tracesDF)

    assert viewer.trace_scaler is None
    assert np.abs(reconstructed - viewer.array_curves).max() < 1e-4


def test_run_reports_what_the_components_bought(viewer):
    """The tab showed how many components were used but never what they explain."""
    from dash_viewer.dimensionality_reduction.pca_viewer import calculate_pca as fit

    viewer.pca_result, viewer.pca_var_comp, viewer.n_components_used = fit(5, viewer.array_curves)
    viewer.describe_pca(viewer.array_curves)

    assert len(viewer.variance_summary) == 1
    line = viewer.variance_summary[0]
    assert 'components explain' in line
    assert 'held-out' in line, 'in-sample variance alone is not comparable to the autoencoder'


def test_summary_is_cleared_between_runs(viewer):
    viewer.variance_summary = ['stale line from a previous run']
    viewer.concat_traces_by = 'full_traces'
    viewer.prepare_all_traces = lambda: None

    viewer.load_pca_data()

    assert viewer.variance_summary == []
