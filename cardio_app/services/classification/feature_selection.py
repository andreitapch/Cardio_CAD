import gc
import os
from datetime import datetime

import shap
import numpy as np
import pandas as pd
import xgboost as xgb
import lime.lime_tabular

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.feature_selection import f_classif, mutual_info_classif, RFE

from cardio_app.services.classification.data import get_echoParams_data
from cardio_app.services.classification.engine import (
    build_preprocessor, convert_echoParam_values, make_splitter, preprocess_data,
    resolve_exam_groups, surviving_feature_names)
from pattern_recognition.utils.parameters_dics import dbTableNames, dbConParameters
from pattern_recognition.data.database_connection.db_con import DatabaseConnection as dbc

gc.enable()

import random
random.seed(42)
np.random.seed(42)


# Five folds is the same partition count the classifier uses, and it is what
# makes selection frequency readable: a feature is picked in 0/5 .. 5/5 folds.
DEFAULT_N_SELECTION_FOLDS = 5

# A feature has to survive the majority of folds to count as selected. 0.6 of 5
# folds means 3/5 - below that the feature is being chosen by whichever patients
# happened to land in one training half.
DEFAULT_MIN_SELECTION_FREQUENCY = 0.6

# Unchanged from the single-pass version so runs stay comparable with the
# results already stored in calculated_features_remodeling_T.
CONSENSUS_WEIGHTS = {
    'SHAP': 0.3,
    'Random Forest': 0.25,
    'Mutual Information': 0.2,
    'F-statistic': 0.15,
    'RFE Importance': 0.1,
}
MEASURE_COLUMNS = list(CONSENSUS_WEIGHTS)

# Min-max normalised per fold; RFE is already a bounded 1/rank score and was
# never normalised, which is kept so the weighting means the same thing.
NORMALISED_COLUMNS = ['F-statistic', 'Mutual Information', 'Random Forest', 'SHAP']

STABILITY_COLUMNS = ['Weighted Consensus Score', 'Score SD', 'Selection Frequency',
                     'Folds Selected', 'Folds Present']


def _download_payload(feature_scores, selected_features, data_source, k, threshold,
                      min_frequency):
    """Snapshot of the ranking for the CSV download.

    The on-screen table shows five of the columns; the store keeps the whole
    frame, because the per-measure scores and the stability columns are the
    result of the per-fold rework and were otherwise lost when the page closed.
    Never raises - an export convenience must not be able to kill the run.
    """
    try:
        frame = feature_scores.reset_index().rename(columns={'index': 'Feature'})
        return {
            'stamp': datetime.now().strftime('%Y%m%d-%H%M%S'),
            'config': {
                'data_source': data_source,
                'k_features': k,
                'threshold': threshold,
                'min_selection_frequency': _as_frequency(min_frequency),
                'n_selection_folds': DEFAULT_N_SELECTION_FOLDS,
                'run_at': datetime.now().isoformat(timespec='seconds'),
            },
            'features': frame.to_dict('records'),
            'selected': list(selected_features or []),
        }
    except Exception as exc:
        print(f'[WARN] could not build the CSV download payload: {exc}')
        return None


def _build_download_frame(payload):
    """The full ranking, plus which features the run actually kept.

    Column order follows the table so a downloaded file reads like the screen,
    and the run settings ride along as constant columns - a score without the k,
    threshold and frequency that produced it is not reusable in the thesis.
    """
    frame = pd.DataFrame(payload.get('features') or [])
    if frame.empty:
        return frame

    ordered = [c for c in ['Feature'] + MEASURE_COLUMNS + STABILITY_COLUMNS
               if c in frame.columns]
    frame = frame[ordered + [c for c in frame.columns if c not in ordered]]

    selected = set(payload.get('selected') or [])
    if 'Feature' in frame.columns:
        frame.insert(1, 'Selected', frame['Feature'].isin(selected))

    for key, value in (payload.get('config') or {}).items():
        frame[f'config_{key}'] = value
    return frame


def display_matrix(X, selected_features):
    """Z-scored copy of the selected columns, for the class-average bar chart only.

    Never feeds selection - it is fit on the whole cohort on purpose, because the
    figure describes the cohort rather than estimating anything. Raw units would
    put a 200 ml volume next to a 0.1 ratio on one axis and hide every small
    feature; NaNs stay NaN and the chart's groupby-mean skips them.
    """
    subset = X.loc[:, [c for c in selected_features if c in X.columns]].astype(float)
    return (subset - subset.mean()) / (subset.std(ddof=0) + 1e-10)


def compute_feature_importance_cv(X, y, k, threshold, groups=None,
                                  n_splits=DEFAULT_N_SELECTION_FOLDS,
                                  min_frequency=DEFAULT_MIN_SELECTION_FREQUENCY,
                                  random_state=42):
    """Cross-validated, stability-aware feature ranking.

    The single-pass version fit the imputer, the scaler and all five importance
    measures on every patient at once, so the ranking knew the labels of the
    patients the classifier is later graded on. Here each fold fits its own
    preprocessing on its own training half and scores the measures on that half
    alone, and the folds are grouped exactly like the classifier's so an exam's
    rows never straddle a split.

    The headline number is SELECTION FREQUENCY: the fraction of folds in which
    the feature made the top-k / cleared the threshold. A feature picked in 5/5
    folds is signal; one picked in 2/5 is the training half talking. The mean
    consensus score and its spread are reported alongside it.

    A feature dropped as uninformative in a fold counts as not selected there -
    that is a real instability, not a missing observation - so the denominator is
    always the number of folds that ran.

    Returns (scores DataFrame indexed by feature, selected feature names).
    """
    feature_names = list(X.columns)
    y_arr = np.asarray(y).ravel()

    splitter, split_groups = make_splitter(n_splits, groups, shuffle=True,
                                           random_state=random_state)

    per_fold_scores, per_fold_selected, n_folds = [], [], 0
    for fold, (train_idx, _) in enumerate(splitter.split(X, y_arr, groups=split_groups)):
        try:
            fold_scores = _fold_feature_scores(X.iloc[train_idx], y_arr[train_idx],
                                               feature_names)
        except Exception as e:
            print(f"[WARN] feature selection fold {fold + 1} failed ({e}); skipped.")
            continue

        n_folds += 1
        per_fold_scores.append(fold_scores)
        # The k / threshold controls are applied INSIDE each fold, so selection
        # frequency answers "would this run have picked the feature", not "is it
        # somewhere in the ranking".
        per_fold_selected.append(set(_apply_controls(fold_scores, k, threshold)))

    if not per_fold_scores:
        print("[ERROR] no selection fold produced a ranking.")
        return pd.DataFrame(columns=MEASURE_COLUMNS + STABILITY_COLUMNS), []

    print(f"[INFO] feature selection ran on {n_folds} fold(s)"
          + (" with exam-grouped splits" if split_groups is not None else ""))

    aggregated = _aggregate_folds(per_fold_scores, per_fold_selected, n_folds)
    selected_features = _apply_controls(
        aggregated[aggregated['Selection Frequency'] >= _as_frequency(min_frequency)],
        k, threshold)

    return aggregated, selected_features


def _as_frequency(value):
    """Minimum selection frequency as a usable number.

    The dcc.Input hands back None while it is empty and a string mid-edit, and a
    None comparison against a float raises inside the callback.
    """
    try:
        freq = float(value)
    except (TypeError, ValueError):
        return DEFAULT_MIN_SELECTION_FREQUENCY
    return min(max(freq, 0.0), 1.0)


def _apply_controls(feature_scores, k, threshold):
    """The existing threshold-then-top-k rule, unchanged."""
    if feature_scores.empty:
        return []
    if threshold is not None:
        feature_scores = feature_scores[feature_scores['Weighted Consensus Score'] >= threshold]
    if k and k < len(feature_scores):
        feature_scores = feature_scores.head(k)
    return feature_scores.index.tolist()


def _fold_feature_scores(X_train, y_train, feature_names):
    """All five importance measures on one training fold, normalised within it.

    Preprocessing is build_preprocessor() - the same drop/impute/scale steps the
    classifier fits per fold - rather than a second implementation that could
    drift from it. f_classif, mutual_info_classif and RandomForest all reject
    NaN, so the imputation genuinely has to happen before scoring; the point is
    that it happens HERE, on this fold's training half, not once over everyone.

    Normalising per fold before aggregating keeps each fold's min-max on its own
    scale, so one fold with an unusually large F-statistic cannot dominate the
    pooled mean.
    """
    preprocessor = build_preprocessor(scale=True)
    X_transformed = preprocessor.fit_transform(X_train.values)
    names = surviving_feature_names(preprocessor, feature_names)

    X_fold = pd.DataFrame(X_transformed, columns=names)
    if np.isnan(X_fold.values).any():
        raise ValueError("preprocessing left NaNs; the importance measures reject them")

    fold_scores = pd.DataFrame(index=X_fold.columns)
    fold_scores['F-statistic'], _ = f_classif(X_fold, y_train)
    fold_scores['Mutual Information'] = mutual_info_classif(X_fold, y_train,
                                                            random_state=42)

    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_fold, y_train)
    fold_scores['Random Forest'] = rf.feature_importances_

    lr = LogisticRegression(max_iter=500, solver='liblinear')
    rfe = RFE(lr, n_features_to_select=1, step=_rfe_step(X_fold.shape[1]))
    rfe.fit(X_fold, y_train)
    fold_scores['RFE Importance'] = 1 / (rfe.ranking_ + 1e-10)

    shap_values, _ = compute_shap_values(X_fold, y_train)
    fold_scores['SHAP'] = np.abs(shap_values.values).mean(axis=0)

    # F-statistic is NaN for a column with no within-class variance in this
    # fold; treat it as uninformative rather than poisoning the consensus.
    fold_scores = fold_scores.fillna(0.0)

    for col in NORMALISED_COLUMNS:
        span = fold_scores[col].max() - fold_scores[col].min()
        fold_scores[col] = (fold_scores[col] - fold_scores[col].min()) / (span + 1e-10)

    fold_scores['Weighted Consensus Score'] = sum(
        weight * fold_scores[col] for col, weight in CONSENSUS_WEIGHTS.items())

    return fold_scores.sort_values(by='Weighted Consensus Score', ascending=False)


def _rfe_step(n_features):
    """How many features RFE drops per round.

    RFE is the slowest measure here and now runs once per fold: with
    n_features_to_select=1 and step=1 it refits the logistic regression
    n_features - 1 times. On this cohort's ~30 biomarkers that is ~30 fits of a
    liblinear model on ~200 rows - milliseconds - so the step stays 1 and the
    ranking is bit-for-bit what the single-pass version produced.

    It only grows for wide inputs (PCA/autoencoder components, strain time
    points), where 5% per round turns a quadratic number of fits into ~20. The
    ranking then comes in blocks of tied features instead of a strict order,
    which is a coarser version of the same ordering - and RFE carries the
    smallest consensus weight (0.1), so the blocking moves the ranking very
    little.
    """
    return 1 if n_features <= 50 else max(1, int(0.05 * n_features))


def _aggregate_folds(per_fold_scores, per_fold_selected, n_folds):
    """Pool per-fold rankings into one table with a stability column.

    Mean and SD are taken over the folds in which the feature SURVIVED the drop
    step, because a fold that never scored it has no score to average. Selection
    frequency divides by every fold that ran, so being dropped counts against
    the feature - which is the point of the number.
    """
    all_features = sorted({f for fold in per_fold_scores for f in fold.index})

    rows = {}
    for feature in all_features:
        present = [fold.loc[feature] for fold in per_fold_scores if feature in fold.index]
        consensus = np.array([row['Weighted Consensus Score'] for row in present], dtype=float)
        n_selected = sum(1 for chosen in per_fold_selected if feature in chosen)

        row = {col: float(np.mean([r[col] for r in present])) for col in MEASURE_COLUMNS}
        row['Weighted Consensus Score'] = float(consensus.mean())
        # ddof=1 needs two folds; one fold has no spread to report, and 0.0 is
        # less misleading here than a NaN that would break the plots.
        row['Score SD'] = float(np.std(consensus, ddof=1)) if len(consensus) > 1 else 0.0
        row['Selection Frequency'] = n_selected / float(n_folds)
        row['Folds Selected'] = f"{n_selected}/{n_folds}"
        row['Folds Present'] = f"{len(present)}/{n_folds}"
        rows[feature] = row

    aggregated = pd.DataFrame.from_dict(rows, orient='index')
    aggregated = aggregated[MEASURE_COLUMNS + STABILITY_COLUMNS]
    aggregated = aggregated.sort_values(
        by=['Selection Frequency', 'Weighted Consensus Score'], ascending=False)

    numeric = MEASURE_COLUMNS + ['Weighted Consensus Score', 'Score SD', 'Selection Frequency']
    aggregated[numeric] = aggregated[numeric].round(3)
    return aggregated


def compute_shap_values(X, y):
    # use_label_encoder was removed in xgboost 2.0 and now raises; aucpr is the
    # imbalance-aware objective the rest of the project ranks on.
    model = xgb.XGBClassifier(eval_metric="aucpr", random_state=42)
    model.fit(X, y)
    explainer = shap.Explainer(model, seed=42)
    shap_values = explainer(X)

    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    # Opt-in only. beeswarm draws through matplotlib, and matplotlib rendering a
    # SHAP plot is what pulled conda's libomp.dll after torch had bound
    # libiomp5md.dll and aborted the server with "OMP: Error #15" (see the header
    # of dash_main.py). Set CARDIO_SHAP_PLOTS=1 when calling from a script.
    if os.environ.get('CARDIO_SHAP_PLOTS') == '1':
        shap.plots.beeswarm(shap_values)
    return shap_values, X.columns


LEAKAGE_NOTE = (
    "What this does and does not fix: the ranking below is computed fold by fold. "
    "Each fold fits its own imputer and scaler on its own training patients, so "
    "the preprocessing no longer sees the patients it is scored on, and the "
    "selection frequency tells you how often a feature survived a different "
    "training half - a feature picked in 5/5 folds is stable, one picked in 2/5 "
    "is mostly noise. It does NOT make the saved list bias-free. The list is "
    "still chosen using the labels of the whole cohort and then reused as the "
    "predictors the classifier is evaluated on, so any performance measured with "
    "it is still optimistic. The only complete fix is to run feature selection "
    "inside each of the classifier's own cross-validation folds. Read these "
    "numbers as 'which features are stable', not as a validated result."
)


