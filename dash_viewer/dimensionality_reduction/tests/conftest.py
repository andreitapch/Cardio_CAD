"""Shared fixtures for the dimensionality-reduction tests.

Everything here is synthetic: these tests must run without a database, without
a GPU and without a Dash server.
"""
import numpy as np
import pandas as pd
import pytest

N_SEGMENTS = 18
SEGMENT_LEN = 20
N_FEATURES = N_SEGMENTS * SEGMENT_LEN


@pytest.fixture
def rng():
    return np.random.default_rng(20260905)


def make_traces(rng, n_studies):
    """Strain-like curves: negative, a few units wide, one row per study."""
    return rng.normal(size=(n_studies, N_FEATURES)) * 3.5 - 12.0


def make_study_nrs(n_studies, first=100):
    return list(range(first, first + n_studies))


def make_concat_tracesDF(study_nrs):
    """The frame CurveConcatenator returns: one row per study, segments intact."""
    return pd.DataFrame({
        'study_nr': study_nrs,
        'concat_curve_len': [np.full(N_SEGMENTS, SEGMENT_LEN) for _ in study_nrs],
        'concat_curve_segm': [[f'seg{i}' for i in range(N_SEGMENTS)] for _ in study_nrs],
        'concat_id_segment': [list(range(N_SEGMENTS)) for _ in study_nrs],
    })


def make_echo_params(rng, study_nrs, n_biomarkers=4, views_per_study=2):
    """Standardised echo biomarkers, several view rows per study as the DB returns them."""
    rows = []
    for study_nr in study_nrs:
        for _ in range(views_per_study):
            row = {'study_nr': study_nr}
            row.update({f'biomarker_{i}': rng.normal() for i in range(n_biomarkers)})
            rows.append(row)
    return pd.DataFrame(rows)


def reassemble(dm_tracesDF):
    """Glue the per-segment reconstruction back into one row per study."""
    return np.array([np.concatenate(group['strain_trace'].tolist())
                     for _, group in dm_tracesDF.groupby('study_nr', sort=True)])
