import copy
import gc
import json
import re
import traceback
import warnings
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed, parallel_backend

from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import ADASYN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline

from sklearn.base import clone, BaseEstimator, TransformerMixin
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.metrics import f1_score, accuracy_score
from sklearn.metrics import matthews_corrcoef, recall_score
from sklearn.metrics import roc_curve, auc, confusion_matrix, brier_score_loss
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import precision_score, roc_auc_score, average_precision_score

from cardio_app.services.classification.shap_explainer import (
    compute_shap_timepoints, compute_shap_reduced, is_shap_supported_model,
    reduce_shap_to_class_1, select_important_features)
from cardio_app.services.classification.model_params import ModelParams, DEFAULT_SEARCH_BUDGET
from pattern_recognition.utils.parameters_dics import dbTableNames


# One repeat reproduces the single nested-CV run this module has always done.
# More repeats re-draw the outer folds and are what make the confidence
# intervals mean anything; they cost n_repeats times as much.
DEFAULT_N_REPEATS = 1

# Imbalance strategies the inner loop may choose between when search_sampler is
# on. 'None' is included so "do nothing" has to win on merit.
SAMPLER_CHOICES = ('None', 'SMOTE', 'ADASYN', 'Balanced')

# Threshold objective. F2 weights recall four times precision: in this cohort a
# missed remodeling case costs more than a false alarm.
DEFAULT_BETA = 2.0

# In-fold dimensionality reduction. 'none' is the default so nothing changes for
# any existing caller. Components default to 8: enough for UMAP to keep more
# than a 2-D picture's worth of structure, few enough that SMOTE interpolates
# somewhere meaningful.
REDUCER_CHOICES = (None, 'none', 'pca', 'umap')
DEFAULT_REDUCER_COMPONENTS = 8


class DropUninformativeColumns(BaseEstimator, TransformerMixin):
    """Drop mostly-missing and constant columns, learned from training data only.

    impute_missing_values() made the same two decisions, but it ran once over
    the whole cohort before any split, so which columns survived was informed
    by the patients being held out.
    """

    def __init__(self, missing_threshold=30.0):
        self.missing_threshold = missing_threshold

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        missing_pct = np.isnan(X).mean(axis=0) * 100.0
        keep = missing_pct <= self.missing_threshold

        with warnings.catch_warnings():
            # All-NaN columns are already excluded by `keep`; nanmin/nanmax warn
            # on them anyway.
            warnings.simplefilter('ignore', category=RuntimeWarning)
            col_min = np.nanmin(X, axis=0)
            col_max = np.nanmax(X, axis=0)
        constant = np.isclose(col_min, col_max, equal_nan=True)
        keep &= ~constant

        if not keep.any():
            # Nothing survives; keep everything rather than hand an empty
            # matrix to the imputer.
            print("[WARN] DropUninformativeColumns would drop every column; keeping all.")
            keep = np.ones(X.shape[1], dtype=bool)

        self.columns_to_keep_ = keep
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        return X[:, self.columns_to_keep_]


def make_sampler(unbalanced_type):
    """Resampler for the imbalance strategy, as pipeline steps.

    Returned as steps rather than applied directly so imblearn only runs them
    during fit. Applying them by hand (the old get_samples call) risked
    resampling data that was about to be predicted on.
    """
    if unbalanced_type == 'SMOTE':
        return [('sampler', SMOTE(sampling_strategy='auto', k_neighbors=3, random_state=42))]
    if unbalanced_type == 'ADASYN':
        return [('sampler', ADASYN(sampling_strategy='auto', random_state=42))]
    if unbalanced_type == 'Balanced':
        return [('oversampler', ADASYN(sampling_strategy='auto', random_state=42)),
                ('undersampler', RandomUnderSampler(sampling_strategy='auto', random_state=42))]
    return []


class _SafeReducer(BaseEstimator, TransformerMixin):
    """A dimensionality reducer that a small or degenerate fold cannot kill.

    Two jobs, both of which have to happen at FIT time because they depend on
    the size of the training half and nothing else knows it:

    * clamp the neighbourhood and the component count against the fold. UMAP
      needs n_neighbors < n_samples and n_components < n_samples; on a 40-row
      inner fold the defaults are simply invalid and it raises or warns.
    * survive failure. A reducer that blows up on one split must degrade to "no
      reduction for this split" with a printed warning, exactly as
      _prepare_inner_splits() already treats a failed preprocessing fit - it
      must not take the whole run down.

    output_feature_names_ is the names of the columns this step emits, or None
    when it fell back to passthrough; surviving_feature_names() reads it so the
    SHAP matrix and its column labels cannot drift apart.
    """

    # Prefix per reducer family, so a SHAP axis says which embedding it is.
    _NAME_PREFIX = {'UMAP': 'UMAP', 'PCA': 'PC'}

    def __init__(self, reducer=None):
        self.reducer = reducer

    def _clamped(self, n_samples, n_features):
        """A copy of the reducer with its sizes made legal for this fold."""
        fitted = clone(self.reducer)
        params = fitted.get_params()

        if 'n_components' in params and params['n_components'] is not None:
            # UMAP's spectral init needs strictly fewer components than
            # samples; PCA cannot exceed min(n_samples, n_features) either.
            ceiling = max(1, min(int(n_features), int(n_samples) - 2))
            wanted = int(params['n_components'])
            if wanted > ceiling:
                print(f"[WARN] reducer n_components {wanted} > {ceiling} for a fold of "
                      f"{n_samples} rows; using {ceiling}.")
                fitted.set_params(n_components=ceiling)

        if 'n_neighbors' in params and params['n_neighbors'] is not None:
            neighbours = max(2, min(int(params['n_neighbors']), int(n_samples) - 1))
            if neighbours != int(params['n_neighbors']):
                print(f"[WARN] reducer n_neighbors clamped to {neighbours} for a fold of "
                      f"{n_samples} rows.")
                fitted.set_params(n_neighbors=neighbours)

        return fitted

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.n_features_in_ = X.shape[1]
        self.reducer_ = None
        self.output_feature_names_ = None
        self.n_fit_rows_ = int(X.shape[0])

        if self.reducer is None:
            return self

        try:
            candidate = self._clamped(X.shape[0], X.shape[1])
            candidate.fit(X)
        except Exception as e:
            print(f"[WARN] dimensionality reduction failed on this split "
                  f"({type(e).__name__}: {e}); continuing without reduction.")
            return self

        self.reducer_ = candidate
        n_out = int(getattr(candidate, 'n_components_', None)
                    or candidate.get_params().get('n_components'))
        prefix = self._NAME_PREFIX.get(type(candidate).__name__, 'COMP')
        self.output_feature_names_ = [f'{prefix}_{i + 1}' for i in range(n_out)]
        return self

    def transform(self, X):
        if getattr(self, 'reducer_', None) is None:
            return np.asarray(X, dtype=float)
        return self.reducer_.transform(np.asarray(X, dtype=float))


def make_reducer(reducer, n_components, random_state=42):
    """Dimensionality reduction for the CV pipeline, as pipeline steps.

    Steps rather than a pre-transformed matrix for the same reason as
    make_sampler(): the embedding must be FIT on the training half of a split
    and only ever APPLIED to the held-out half. UMAP in particular is flexible
    enough to bend itself around whatever points it is shown, so fitting it on
    the whole cohort before cross-validating would let held-out patients shape
    the space the model is then scored in.

    umap is imported lazily: it pulls numba in and costs seconds to import, and
    this module has to stay importable on machines that never use it.
    """
    if reducer is None or reducer == 'none':
        return []

    # A reducer with no component count is a caller that only chose the method;
    # UMAP(n_components=None) raises, so fall back rather than blow up.
    n_components = int(n_components or DEFAULT_REDUCER_COMPONENTS)

    if reducer == 'pca':
        from sklearn.decomposition import PCA
        return [('reduce', _SafeReducer(PCA(n_components=n_components,
                                            random_state=random_state)))]

    if reducer == 'umap':
        try:
            import umap
        except ImportError as e:
            raise ImportError(
                "reducer='umap' needs the 'umap-learn' package, which is not "
                "installed (pip install umap-learn)."
            ) from e
        return [('reduce', _SafeReducer(umap.UMAP(n_components=n_components,
                                                  random_state=random_state)))]

    raise ValueError(f"unknown reducer {reducer!r}; expected None, 'none', 'pca' or 'umap'")


# Inputs whose columns are successive time points of one strain curve rather
# than independent measurements.
TRACE_INPUTS = frozenset({'reconstructed_traces', 'reconstructed_traces_vac'})


def default_scaling_for(class_input):
    """Whether to standardise features for this input type.

    Off for raw strain traces. Their columns are time points of a single curve,
    so z-scoring each one independently rescales every point by a different
    factor and flattens the amplitude differences along the curve. Tree models
    are indifferent (a monotonic per-feature transform does not change their
    splits) but KNN is not: it changes the distance metric outright.

    On for tabular biomarkers and PCA/autoencoder components, which are
    genuinely separate quantities on different units.
    """
    return class_input not in TRACE_INPUTS


def build_estimator_pipeline(clf, unbalanced_type, n_neighbors=15, missing_threshold=30.0,
                             scale=True, reducer=None, reducer_components=None):
    """Estimator with every fitted preprocessing step inside it.

    Column dropping, KNN imputation and standardisation used to run once in
    preprocess_data() over the full cohort, so the imputer's neighbours and the
    scaler's mean/variance were computed partly from held-out patients. As
    pipeline steps they are refit inside each fold on training data alone.

    scale=False omits the scaler; see default_scaling_for().

    ORDER, and it is deliberate: drop -> impute -> scale -> REDUCE -> sample ->
    classify. The reducer comes after scaling because UMAP and PCA are both
    distance/variance based and would otherwise be dominated by whichever raw
    column happens to have the largest units. It comes BEFORE the sampler
    because SMOTE interpolates between neighbours, and interpolating in a
    2304-dimensional raw trace space produces synthetic patients that lie
    nowhere near the data manifold; in an 8-dimensional embedding the same
    interpolation is meaningful. The sampler still sits inside the fitted
    estimator, so imblearn keeps it confined to fit.
    """
    steps = [
        ('drop_uninformative', DropUninformativeColumns(missing_threshold=missing_threshold)),
        ('impute', KNNImputer(n_neighbors=n_neighbors)),
    ]
    if scale:
        steps.append(('scale', StandardScaler()))
    steps += make_reducer(reducer, reducer_components)
    steps += make_sampler(unbalanced_type)
    steps.append(('clf', clf))
    return ImbPipeline(steps)


def build_preprocessor(n_neighbors=15, missing_threshold=30.0, scale=True,
                       reducer=None, reducer_components=None):
    """The hyperparameter-INDEPENDENT half of build_estimator_pipeline().

    Dropping, imputing and scaling do not depend on the classifier's
    hyperparameters, yet the inner loop refit them once per candidate - for a
    budget of 60 that is 59 wasted KNN imputations per inner split. Fitting this
    once per inner split and reusing the transformed matrices is exactly
    equivalent, because it is still fit on that split's training half alone.

    The sampler deliberately stays OUT of here: it must run inside the fitted
    estimator so it only ever touches data being trained on.

    The REDUCER, by contrast, belongs in here and nowhere else. It does not
    depend on the classifier's hyperparameters either, and it is by far the most
    expensive step in the pipeline - a UMAP fit on a 250x72 matrix takes about
    25 seconds. Fitting it here means once per inner split; putting it in
    build_estimator_pipeline()'s per-candidate path instead would mean once per
    CANDIDATE, i.e. ~60 x 25s per inner split, which is not a run anyone can
    wait for. It is still fit on that split's training half alone, so the
    guarantee is unchanged. Same placement as in build_estimator_pipeline():
    after scaling, before the (absent here) sampler.
    """
    steps = [
        ('drop_uninformative', DropUninformativeColumns(missing_threshold=missing_threshold)),
        ('impute', KNNImputer(n_neighbors=n_neighbors)),
    ]
    if scale:
        steps.append(('scale', StandardScaler()))
    steps += make_reducer(reducer, reducer_components)
    return SkPipeline(steps)


def build_sampler_estimator(clf, unbalanced_type):
    """Sampler + classifier, for data that is already preprocessed.

    The counterpart of build_preprocessor(). Resampling stays a pipeline step so
    imblearn still confines it to fit.
    """
    return ImbPipeline(make_sampler(unbalanced_type) + [('clf', clf)])


def apply_imbalance_correction(clf, unbalanced_type, y_train):
    """Give XGBoost the imbalance correction model_params.py cannot give it.

    Every other estimator in this project either takes class_weight='balanced'
    from its grid or is balanced by a sampler. XGBClassifier takes neither, so
    "XGBoost + None" trained unweighted on a 3:1 cohort and simply learned the
    majority class. scale_pos_weight is xgboost's equivalent knob.

    Computed from this fold's TRAINING labels only - the ratio is a statistic of
    the data, and reading it off the full cohort would leak the held-out class
    balance into the fit.

    Not applied when a sampler is active: the classes are already balanced by
    the time the booster sees them, and correcting twice over-predicts the
    minority class (same reasoning as the RF class_weight note).

    The parameter is found by suffix rather than by an exact name, because the
    booster is not always at the top level: it is 'scale_pos_weight' on a bare
    XGBClassifier, 'estimator__scale_pos_weight' inside a CalibratedClassifierCV
    and 'xgb__scale_pos_weight' for the XGB base learner of a stack. Estimators
    that have no such knob (KNN, the spline/logistic pipeline) are returned
    untouched - they take class_weight from their grid instead.
    """
    if unbalanced_type != 'None':
        return clf
    try:
        keys = [k for k in clf.get_params() if k.split('__')[-1] == 'scale_pos_weight']
    except Exception:
        return clf
    if not keys:
        return clf

    y_train = np.asarray(y_train)
    n_pos = int(np.sum(y_train == 1))
    n_neg = int(np.sum(y_train != 1))
    if n_pos == 0 or n_neg == 0:
        return clf

    ratio = float(n_neg) / float(n_pos)
    return clone(clf).set_params(**{key: ratio for key in keys})


def shap_unavailable_reason(class_type, features):
    """Why SHAP will not be produced for this run, or None if it will be.

    The viewer shows this instead of leaving the explanation panel silently
    blank, which gave no clue whether SHAP had failed or was never attempted.
    """
    if features in ['global_strain']:
        return ("SHAP is not produced for a single global-strain feature - "
                "there is nothing to rank.")

    try:
        probe = ModelParams()
        probe.get_param_grid(class_type, 'None', n_iter=1)
    except ValueError:
        return f"Unknown classifier type '{class_type}'."

    if not is_shap_supported_model(probe.clf):
        label = type(probe.clf).__name__
        if hasattr(probe.clf, 'steps'):
            label = type(probe.clf.steps[-1][1]).__name__

        # The wrapper models hide a tree ensemble behind a meta-estimator, so
        # the generic "not a tree ensemble" wording reads as wrong to anyone who
        # knows a booster is in there. Say what actually blocks it.
        wrapped = {
            'Champion_class': ("the calibrated Champion model wraps its booster in "
                               "CalibratedClassifierCV, which fits several boosters and "
                               "exposes none of them"),
            'Stacking_class': ("the Stacking ensemble combines a logistic regression, a "
                               "forest and a booster, so there is no single tree model "
                               "to explain"),
        }.get(class_type)
        if wrapped:
            return (f"SHAP is not produced for this model: {wrapped}. The per-model "
                    f"rankings (Random Forest, Gradient Boosting, XGBoost) are the "
                    f"place to read feature influence.")

        return (f"SHAP is only computed for tree ensembles (Random Forest, "
                f"Gradient Boosting, XGBoost, LightGBM). {label} is not covered, "
                f"so no feature ranking is shown.")
    return None


def surviving_feature_names(pipeline, feature_names):
    """Names of the columns the pipeline's transforms actually emit.

    Without a reducer that is just the original names minus the ones the drop
    step removed. WITH a reducer the transformed matrix has n_components
    columns that correspond to no original feature at all, so returning the
    surviving raw names would hand SHAP a label list of the wrong length - it
    either raises or, worse, silently labels component 3 with the name of some
    unrelated time point. The reducer reports its own names ('UMAP_1', 'PC_1',
    ...), and reports None when it fell back to passthrough, in which case the
    raw names are correct again.
    """
    if feature_names is None:
        return None
    try:
        mask = pipeline.named_steps['drop_uninformative'].columns_to_keep_
        names = [name for name, keep in zip(feature_names, mask) if keep]
    except (AttributeError, KeyError):
        names = list(feature_names)

    try:
        reduce_step = pipeline.named_steps['reduce']
    except (AttributeError, KeyError):
        return names

    reduced_names = getattr(reduce_step, 'output_feature_names_', None)
    return list(reduced_names) if reduced_names else names


def predict_scores(model, X):
    """Continuous scores for the positive class, whatever the estimator exposes."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        return model.decision_function(X)
    raise ValueError("The model does not support probability prediction.")


def get_best_metrics(X_test, X_train, model, unbalanced_type, y_test, y_train, scale=True,
                     reducer=None, reducer_components=None):
    """Threshold-free score for one train/test pair, used to rank hyperparameters.

    Preprocessing and resampling happen inside the pipeline, so they are fit on
    X_train alone and X_test is only ever transformed.
    """
    pipeline = build_estimator_pipeline(
        apply_imbalance_correction(model, unbalanced_type, y_train),
        unbalanced_type, scale=scale, reducer=reducer,
        reducer_components=reducer_components)
    pipeline.fit(X_train, y_train)
    y_pred_proba = predict_scores(pipeline, X_test)
    return average_precision_score(y_test, y_pred_proba)


def score_on_preprocessed(X_test, X_train, model, unbalanced_type, y_test, y_train):
    """get_best_metrics() for matrices a fitted preprocessor already transformed.

    Identical selection score, minus the redundant drop/impute/scale refit; see
    build_preprocessor().
    """
    pipeline = build_sampler_estimator(
        apply_imbalance_correction(model, unbalanced_type, y_train), unbalanced_type)
    pipeline.fit(X_train, y_train)
    return average_precision_score(y_test, predict_scores(pipeline, X_test))


def choose_threshold(y_true, y_scores, beta=1.0):
    """Cutoff maximising F-beta on the data given.

    beta > 1 weights recall over precision; beta defaults to 1 (plain F1) so
    callers that do not care keep the old behaviour.

    Must only ever be called on TRAINING data. Calling it on the fold being
    scored fits the cutoff to the labels it is then graded against, which
    inflates every threshold-dependent metric.
    """
    prec, rec, thr = precision_recall_curve(y_true, y_scores)
    b2 = beta * beta
    f_scores = ((1 + b2) * prec * rec) / (b2 * prec + rec + 1e-12)
    best_idx = int(np.nanargmax(f_scores))
    # precision_recall_curve returns len(thr) == len(prec) - 1, so the last
    # point of the curve has no threshold of its own. Clamp instead of the
    # previous best_idx - 1, which shifted every other index by one.
    return thr[min(best_idx, len(thr) - 1)]


def fit_and_threshold(model, X_train, y_train, unbalanced_type, groups=None,
                      random_state=42, scale=True, beta=1.0,
                      reducer=None, reducer_components=None):
    """Fit on the training fold and pick a cutoff without touching held-out data.

    The cutoff is chosen on out-of-fold scores pooled over ALL of a 3-fold inner
    split of the training data: every training row is scored exactly once, by a
    model that did not train on it, and one threshold is chosen on the whole
    pooled set. The previous version used a single 1/3 slice, so the cutoff was
    read off ~30 patients of which ~8 were positive - a curve with that few
    positives moves several percentage points on one patient changing side.

    The pipeline is then refit on the full training fold so no data is wasted.
    When groups are supplied the calibration splits keep an exam's rows on one
    side, matching how the outer folds are built.

    Returns (fitted_pipeline, threshold, n_fits).
    """
    threshold, n_fits = 0.5, 0
    try:
        splitter, split_groups = make_splitter(3, groups, shuffle=True,
                                               random_state=random_state)
        oof_true, oof_scores = [], []
        for cal_train, cal_test in splitter.split(X_train, y_train, groups=split_groups):
            # The reducer is part of the calibration pipeline too. A threshold
            # is a property of one model's score distribution, so it has to be
            # read off a model built exactly like the one it will be applied
            # to - a calibration fit on unreduced features scores a different
            # model entirely and its cutoff means nothing.
            calibration_pipeline = build_estimator_pipeline(
                apply_imbalance_correction(clone(model), unbalanced_type, y_train[cal_train]),
                unbalanced_type, scale=scale, reducer=reducer,
                reducer_components=reducer_components)
            calibration_pipeline.fit(X_train[cal_train], y_train[cal_train])
            n_fits += 1
            oof_scores.append(predict_scores(calibration_pipeline, X_train[cal_test]))
            oof_true.append(y_train[cal_test])

        pooled_true = np.concatenate(oof_true)
        pooled_scores = np.concatenate(oof_scores)
        if len(np.unique(pooled_true)) < 2:
            raise ValueError("calibration folds produced a single class")
        threshold = choose_threshold(pooled_true, pooled_scores, beta=beta)
    except Exception as e:
        print(f"[WARN] threshold calibration failed ({e}); falling back to 0.5")

    pipeline = build_estimator_pipeline(
        apply_imbalance_correction(model, unbalanced_type, y_train),
        unbalanced_type, scale=scale, reducer=reducer,
        reducer_components=reducer_components)
    pipeline.fit(X_train, y_train)
    n_fits += 1
    return pipeline, threshold, n_fits


# Columns that identify a row rather than describe it. Patient_ID in
# particular is now carried on some frames so folds can be grouped by patient;
# without this guard the positional columns[2:] slice below would hand it to the
# classifier as a predictor, which is a near-perfect label proxy in a cohort
# where outcome is recorded per patient.
ID_COLUMNS = frozenset({'study_nr', 'Patient_ID', 'patientID', 'data_source',
                        'EDV_source', 'reconstructed_type', 'isImputed',
                        'n_components', 'encoding'})
LABEL_COLUMN = 'Total_Remodeling'


def prepare_train_test_data(orig_data, orig_feature_col_name, imp_data, imp_feature_col_name, feature):
    feature_names_orig, feature_names_imp = None, None
    X_imp, y_imp, X_orig, y_orig = None, None, None, None

    if feature == 'all_features' or feature == 'gls'or feature == 'class_features':
        isMultiFeatures = True
    else:
        isMultiFeatures = False

    if not isMultiFeatures:
        if imp_data is not None:

            if imp_feature_col_name == 'global_strain':
                X_imp = imp_data[imp_feature_col_name].values.reshape(-1, 1)
                feature_names_imp = [imp_feature_col_name]
            else:
                X_imp = np.stack(imp_data[imp_feature_col_name].values)
                feature_names_imp = [f"{imp_feature_col_name}_{i}" for i in range(X_imp.shape[1])]
            y_imp = imp_data['Total_Remodeling'].values

        if orig_feature_col_name == 'global_strain':
            X_orig = orig_data[orig_feature_col_name].values.reshape(-1, 1)
            feature_names_orig = [orig_feature_col_name]
        else:
            X_orig = np.stack(orig_data[orig_feature_col_name].values)
            feature_names_orig = [f"{orig_feature_col_name}_{i}" for i in range(X_orig.shape[1])]
        y_orig = orig_data['Total_Remodeling'].values

        return X_imp, y_imp, X_orig, y_orig, feature_names_orig, feature_names_imp

    else:
        X_imp_concat, X_concat = None, None

        features = [c for c in orig_data.columns[2:] if c not in ID_COLUMNS]
        feature_names = features.copy()

        if LABEL_COLUMN in features:
            features.remove(LABEL_COLUMN)
        elif feature == 'gls':
            features = 'global_strain'

        if feature == 'all_features' or feature == 'gls':
            X_concat = np.stack(orig_data[orig_feature_col_name].values)
            X_orig = np.column_stack((X_concat, orig_data[features].values))
            concat_cols = [f"{orig_feature_col_name}_{i}" for i in range(X_concat.shape[1])]
            feature_names_orig = concat_cols + features

            if imp_data is not None:
                X_imp_concat = np.stack(imp_data[orig_feature_col_name].values)
                feature_names_imp = concat_cols + features
                X_imp = np.column_stack((X_imp_concat, imp_data[features].values))

        else:
            feature_names_orig = features.copy()
            X_orig = orig_data[features].values
            if imp_data is not None:
                feature_names_imp = features.copy()
                X_imp = imp_data[features].values

        y = orig_data['Total_Remodeling'].values
        if imp_data is not None:
            y_imp = imp_data['Total_Remodeling'].values

        return X_imp, y_imp, X_orig, y, feature_names_orig, feature_names_imp


def _study_ids(df):
    """Study identifiers aligned row-for-row with the feature matrix.

    prepare_train_test_data() never reorders rows, so positional indexing into
    this array matches the X it returns.
    """
    if df is None or 'study_nr' not in df.columns:
        return None
    return df['study_nr'].values


def resolve_exam_groups(data):
    """Group ids that keep one exam on one side of a split, or None if unneeded.

    The unit of analysis is the exam, not the patient: Patient_ID and study_nr
    are 1:1 in this cohort (252 patients, 252 studies), so grouping by patient
    would buy nothing. What does vary is the number of ROWS per exam. The
    multiple rows an exam has upstream - one per cardiac view in
    ste_strainValues, one per segment in ste_segmentalStrain - are aggregated
    to a single row before they reach here, but not always: a handful of
    studies appear twice in echo_biomarkers_view with two EDV sources
    (3DEcho/2DEcho2 and CMR), same label and same strain, differing only in the
    volume measurements.

    Two rows of one exam sharing a label must not straddle a split, so grouping
    kicks in only when duplicates are actually present. When every row is a
    distinct exam this returns None and the caller uses a plain stratified
    split, which stratifies better than the grouped splitter.
    """
    if data is None or 'study_nr' not in data.columns:
        return None

    studies = data['study_nr'].values
    n_duplicated = len(studies) - len(pd.unique(studies))
    if n_duplicated == 0:
        return None

    print(f"[INFO] {n_duplicated} duplicate exam row(s) found; folds will be "
          f"grouped by study_nr so no exam straddles a split.")
    return studies


def make_splitter(n_splits, groups, shuffle=True, random_state=42):
    """Cross-validation splitter, group-aware when group ids are supplied."""
    if groups is None:
        return StratifiedKFold(n_splits=n_splits, shuffle=shuffle,
                               random_state=random_state), None
    return StratifiedGroupKFold(n_splits=n_splits, shuffle=shuffle,
                                random_state=random_state), groups


def _augment_with_imputed(X_train, y_train, X_imp, y_imp, imp_keys, held_out_keys):
    """Add imputed rows to a training fold, minus any belonging to held-out patients.

    The imputed frame covers the same cohort as the original one, so adding it
    wholesale (the previous behaviour) put a near-duplicate of every held-out
    patient into training. Rows keyed to a patient being evaluated are dropped.
    """
    if X_imp is None:
        return X_train, y_train

    if imp_keys is None:
        print("[WARN] imputed frame has no study_nr/Patient_ID - cannot verify patient "
              "separation; imputed rows are being dropped rather than leaked.")
        return X_train, y_train

    keep = ~np.isin(imp_keys, list(held_out_keys))
    if not keep.any():
        return X_train, y_train

    return np.vstack((X_imp[keep], X_train)), np.concatenate((y_imp[keep], y_train))


def build_search_candidates(class_type, unbalanced_type, search_budget, random_state,
                            search_sampler):
    """Candidate (hyperparameters, imbalance strategy) pairs for the inner loop.

    With search_sampler off the strategy is fixed and the candidates are exactly
    what ModelParams draws today.

    With it on, the strategy is searched as an OUTER RING with a reduced
    per-strategy budget: search_budget // 4 parameter draws for each of the four
    strategies, so the total stays at ~search_budget fits rather than 4x it.
    Sampling (params, sampler) jointly from one flat space would be tidier, but
    the grid itself depends on the strategy - class_weight is dropped when a
    sampler already balances the classes - so each strategy has to draw from its
    own grid. Each draws with a different seed so the four rings are not the
    same parameter sets four times over.

    Returns (ModelParams prototype, [(params, sampler), ...]).
    """
    parameters = ModelParams()
    if not search_sampler:
        parameters.get_param_grid(class_type, unbalanced_type, n_iter=search_budget,
                                  random_state=random_state)
        return parameters, [(p, unbalanced_type) for p in parameters.param_combinations]

    per_sampler = max(1, int(search_budget) // len(SAMPLER_CHOICES))
    candidates = []
    for offset, sampler in enumerate(SAMPLER_CHOICES):
        ring = ModelParams()
        ring.get_param_grid(class_type, sampler, n_iter=per_sampler,
                            random_state=random_state + offset)
        candidates.extend((params, sampler) for params in ring.param_combinations)

    # The estimator prototype is strategy-independent; any grid call sets it.
    parameters.get_param_grid(class_type, unbalanced_type, n_iter=1, random_state=random_state)
    return parameters, candidates


# Parameters that make a model simpler as they GROW (more smoothing, stronger
# penalty) and as they SHRINK (more capacity). Used only to break ties under the
# 1-SE rule, so a rough ordering is enough.
_COMPLEXITY_UP = ('n_estimators', 'max_depth', 'max_samples', 'subsample', 'C', 'clf__C')
_COMPLEXITY_DOWN = ('min_samples_leaf', 'min_samples_split', 'min_child_weight',
                    'n_neighbors', 'ccp_alpha', 'reg_alpha', 'reg_lambda', 'gamma')


def _complexity_score(params):
    """Rough "how much capacity does this ask for" number; lower is simpler.

    A heuristic, deliberately: the point of the 1-SE rule is only to prefer the
    plainer of two models that scored the same, not to rank them precisely.
    max_depth=None counts as very deep, which is what it means.
    """
    score = 0.0
    for key in _COMPLEXITY_UP:
        if key not in params:
            continue
        value = params[key]
        if value is None:
            score += np.log1p(50.0) if 'depth' in key else 0.0
        elif isinstance(value, (int, float)):
            score += np.log1p(abs(float(value)))
    for key in _COMPLEXITY_DOWN:
        if key in params and isinstance(params[key], (int, float)):
            score -= np.log1p(abs(float(params[key])))
    return float(score)


def select_best_candidate(candidates, per_candidate_scores, use_one_se=True):
    """Pick one candidate from the inner-loop scores, tempering the winner's curse.

    argmax over ~60 noisy estimates from 5 small folds systematically picks the
    candidate whose noise happened to be largest, and that advantage does not
    survive to the outer fold. The 1-SE rule takes every candidate within one
    standard error of the best mean and returns the SIMPLEST of them
    (_complexity_score), which is the usual remedy.

    Pass use_one_se=False for the raw argmax.

    Returns (params, sampler, mean_score) or None if nothing scored.
    """
    summaries = []
    for (params, sampler), scores in zip(candidates, per_candidate_scores):
        valid = [s for s in scores if s is not None and not np.isnan(s)]
        if not valid:
            continue
        # Standard error over inner folds; a single fold has no spread to speak
        # of, so the rule degenerates to argmax, which is correct.
        se = float(np.std(valid, ddof=1) / np.sqrt(len(valid))) if len(valid) > 1 else 0.0
        summaries.append((float(np.mean(valid)), se, params, sampler))

    if not summaries:
        return None

    best_mean, best_se, _, _ = max(summaries, key=lambda s: s[0])
    if not use_one_se:
        chosen = max(summaries, key=lambda s: s[0])
        return chosen[2], chosen[3], chosen[0]

    within = [s for s in summaries if s[0] >= best_mean - best_se]
    chosen = min(within, key=lambda s: (_complexity_score(s[2]), -s[0]))
    return chosen[2], chosen[3], chosen[0]


def _prepare_inner_splits(X_tr, y_tr, inner_splits, outer_keys, held_out_keys,
                          X_imp, y_imp, imp_keys, scale,
                          reducer=None, reducer_components=None):
    """Transformed (train, test) matrices for every inner split, fit once each.

    Imputed augmentation and preprocessing are both hyperparameter-independent,
    so they are done here rather than once per candidate. The preprocessor is
    still fit on the inner TRAINING half only and the inner test half is only
    ever transformed, which is the same guarantee build_estimator_pipeline gave
    - the sampler is the one step that must stay inside the per-candidate
    estimator, and it does.
    """
    prepared = []
    for inner_train, inner_test in inner_splits:
        Xi_tr, yi_tr = X_tr[inner_train], y_tr[inner_train]
        Xi_te, yi_te = X_tr[inner_test], y_tr[inner_test]

        inner_held_out = set(held_out_keys)
        if outer_keys is not None:
            inner_held_out |= set(outer_keys[inner_test])
        Xi_tr, yi_tr = _augment_with_imputed(Xi_tr, yi_tr, X_imp, y_imp,
                                             imp_keys, inner_held_out)

        try:
            # ONE preprocessor fit per inner split, reducer included, reused by
            # every candidate scored on this split - see build_preprocessor().
            preprocessor = build_preprocessor(scale=scale, reducer=reducer,
                                              reducer_components=reducer_components)
            Xi_tr_t = preprocessor.fit_transform(Xi_tr)
            Xi_te_t = preprocessor.transform(Xi_te)
        except Exception as e:
            print(f"[WARN] inner-split preprocessing failed ({e}); split skipped.")
            continue
        prepared.append((Xi_tr_t, yi_tr, Xi_te_t, yi_te))
    return prepared


def _single_threaded(model):
    """Stop an estimator from grabbing every core while it is one of many threads.

    The candidates run in a thread pool, and the XGBoost grid asks for
    n_jobs=-1. Twelve candidate threads each spawning twelve booster threads is
    144 threads fighting over twelve cores, which is slower than running the
    search serially. Only affects speed, never the fitted result.
    """
    try:
        keys = [k for k in model.get_params() if k.split('__')[-1] in ('n_jobs', 'nthread')]
    except Exception:
        return model
    return model.set_params(**{k: 1 for k in keys}) if keys else model


def _score_candidate(params, sampler, clf_prototype, prepared_splits):
    """Inner PR-AUC of one candidate across every prepared inner split.

    Errors are captured per candidate rather than raised: one unfittable
    parameter set must not lose the other fifty-nine. The message is returned so
    the caller can surface it - a candidate that silently scores NaN is how a
    whole fold ends up unexplained.
    """
    scores, errors = [], []
    for Xi_tr, yi_tr, Xi_te, yi_te in prepared_splits:
        try:
            model = _single_threaded(clone(clf_prototype).set_params(**params))
            scores.append(score_on_preprocessed(Xi_te, Xi_tr, model, sampler, yi_te, yi_tr))
        except Exception as e:
            scores.append(np.nan)
            errors.append(f"{type(e).__name__}: {e}")
    return scores, errors


def _run_candidate_search(candidates, clf_prototype, prepared, n_jobs):
    """Score every candidate, in parallel when that is safe and serially when not.

    The 'threading' backend, deliberately, NOT loky's process pool.

    Loky spawns bare interpreters on Windows, and each one re-imports this
    module - and with it sklearn, xgboost and shap - with no conftest.py to run
    first. That ordering is load-bearing here: the repo-root conftest imports
    torch before anything else precisely because torch and conda's MKL both ship
    a libiomp5md.dll, Windows binds that name once per process, and whichever
    loads first wins. A worker therefore binds a different OpenMP runtime than
    the parent did, which is the "random crashes or deadlocks" threadpoolctl
    warns about; a worker that dies that way surfaces as a joblib
    TerminatedWorkerError and took the whole run down with it. It was
    intermittent because a DLL-binding race is intermittent.

    Threads have no such problem: they run in the parent process, with the
    libraries already imported in the order conftest established, and they cost
    nothing to start. The fitting inside a candidate is sklearn/numpy, which
    releases the GIL for nearly all of its runtime, so most of the speedup
    survives - and the process-spawn overhead that dominated small folds (14-29s
    to run six fits) disappears outright.

    A failure of the parallel layer itself must never lose the run: it falls
    back to a serial loop and says so, because a slow search is worth far more
    than a viewer that reports "no result".
    """
    try:
        with parallel_backend('threading'):
            return Parallel(n_jobs=n_jobs)(
                delayed(_score_candidate)(params, sampler, clf_prototype, prepared)
                for params, sampler in candidates)
    except Exception as e:
        print(f"[WARN] parallel candidate search failed ({type(e).__name__}: {e}); "
              f"falling back to a serial loop for this fold.")
        return [_score_candidate(params, sampler, clf_prototype, prepared)
                for params, sampler in candidates]


def _empty_result(class_type, error=None):
    """The contract's dict shape with nothing in it; returned on any failure."""
    return {
        'class_type': class_type,
        'error': error,
        'metrics': {},
        'ci': {},
        'pooled': {'y_true': [], 'y_proba': [], 'y_pred': []},
        'pooled_metrics': {},
        'cm': None,
        'best_params': {},
        'sampler': None,
        'reducer': None,
        'threshold': None,
        'n_fits': 0,
        'n_folds': 0,
        'shap_values': None,
        'shap_input_df': None,
        'important_features': [],
        'shap_df': pd.DataFrame(),
        # Not part of the contract; find_best_class_params reports it as the
        # inner-loop selection score, which is not a performance estimate.
        'inner_score': None,
    }


# A confidence interval over fewer than this many fold values says nothing, and
# a normal approximation on 2 points is actively misleading.
MIN_FOLDS_FOR_CI = 3


def _confidence_intervals(per_fold_metrics, confidence=1.96):
    """95% normal-approximation CI of the mean, per metric, over per-fold values.

    Normal approximation rather than bootstrap: with 5-25 fold values the
    bootstrap adds resampling noise of its own without buying accuracy, and this
    runs inside an interactive viewer. Metrics with too few values get None
    rather than a fabricated interval.
    """
    intervals = {}
    for name, values in per_fold_metrics.items():
        clean = [v for v in values if v is not None and np.isfinite(v)]
        if len(clean) < MIN_FOLDS_FOR_CI:
            intervals[name] = None
            continue
        mean = float(np.mean(clean))
        half = confidence * float(np.std(clean, ddof=1)) / np.sqrt(len(clean))
        intervals[name] = (round(float(mean - half), 4), round(float(mean + half), 4))
    return intervals


def evaluate_classifier(orig_data, orig_feature_col_name, imp_data, imp_feature_col_name,
                        unbalanced_type, class_type, features, *,
                        n_repeats=DEFAULT_N_REPEATS, n_outer_splits=5, n_inner_splits=5,
                        search_budget=DEFAULT_SEARCH_BUDGET, random_state=42,
                        scale_features=None, class_input=None,
                        beta=DEFAULT_BETA, search_sampler=False, n_jobs=-1,
                        reducer=None, reducer_components=DEFAULT_REDUCER_COMPONENTS,
                        autosave=False):
    """Repeated nested cross-validation with exam-safe folds. Never raises.

    Five things are kept inside the training half of every split: the
    hyperparameter search (inner loop), the imbalance strategy when it is
    searched, the preprocessing steps (imputer and scaler, refit per fold), the
    dimensionality reducer when one is asked for, and the decision threshold.

    reducer is None/'none', 'pca' or 'umap' and reducer_components how many
    dimensions it maps to. UMAP is far more flexible than a linear basis, so
    fitting it once on the whole cohort and cross-validating afterwards would
    let held-out patients shape the space the model is scored in; it is fit per
    split instead. The fit lives in build_preprocessor(), the
    hyperparameter-independent prefix, so it costs one embedding per split and
    not one per candidate. Folds are grouped by study_nr when an exam
    contributes more than one row, so its rows cannot straddle a split.

    n_repeats re-runs the whole outer loop with a different fold partition
    (random_state + repeat). One partition of 252 patients into 5 folds is a
    single draw, and the spread between models on one draw is often smaller than
    the spread of one model across draws - which is why a leaderboard built on a
    single run ranks noise. Per-fold values from every repeat feed both the
    reported mean and its confidence interval.

    Returns a dict; see the module README of keys in _empty_result(). On failure
    the same shape comes back with 'error' set.
    """
    try:
        result = _evaluate_classifier_inner(
            orig_data, orig_feature_col_name, imp_data, imp_feature_col_name,
            unbalanced_type, class_type, features, n_repeats=n_repeats,
            n_outer_splits=n_outer_splits, n_inner_splits=n_inner_splits,
            search_budget=search_budget, random_state=random_state,
            scale_features=scale_features, class_input=class_input,
            beta=beta, search_sampler=search_sampler, n_jobs=n_jobs,
            reducer=reducer, reducer_components=reducer_components)

        # Persisted the moment the folds are done, BEFORE the caller builds any
        # table or figure. autosave=False for callers that write their own
        # aggregated report, so a sweep does not also litter one file per model.
        if autosave:
            save_result_snapshot(result, {
                'class_type': class_type, 'class_input': class_input,
                'features': features, 'unbalanced_type': unbalanced_type,
                'search_sampler': search_sampler, 'reducer': reducer,
                'reducer_components': reducer_components if reducer else None,
                'scale_features': scale_features, 'beta': beta,
                'n_repeats': n_repeats, 'n_outer_splits': n_outer_splits,
                'n_inner_splits': n_inner_splits, 'search_budget': search_budget,
                'random_state': random_state,
            })
        return result
    except Exception as e:
        # The caller only ever sees this string, so it has to be enough to
        # diagnose from. str(e) alone is routinely empty (KeyError('x') prints
        # as "'x'", some joblib errors print as nothing at all), which turned
        # every failure into an unactionable "no result" in the viewer.
        detail = traceback.format_exc()
        print(detail)
        message = f"{type(e).__name__}: {e}" if str(e) else type(e).__name__
        where = traceback.extract_tb(e.__traceback__)
        if where:
            last = where[-1]
            filename = last.filename.replace('\\', '/').rsplit('/', 1)[-1]
            message += f" (at {filename}:{last.lineno} in {last.name})"
        print(f"[ERROR] evaluate_classifier failed - {message}")
        return _empty_result(class_type, error=message)
    finally:
        gc.collect()


def _evaluate_classifier_inner(orig_data, orig_feature_col_name, imp_data,
                               imp_feature_col_name, unbalanced_type, class_type,
                               features, *, n_repeats, n_outer_splits, n_inner_splits,
                               search_budget, random_state, scale_features,
                               class_input, beta, search_sampler, n_jobs,
                               reducer=None,
                               reducer_components=DEFAULT_REDUCER_COMPONENTS):
    """The body of evaluate_classifier(); exceptions are handled by its caller."""
    if scale_features is None:
        scale_features = default_scaling_for(class_input if class_input is not None else class_type)
    print(f"[INFO] feature standardisation: {'on' if scale_features else 'off'}")

    reducer = None if reducer in (None, 'none') else reducer
    if reducer is not None:
        reducer_components = int(reducer_components or DEFAULT_REDUCER_COMPONENTS)
        print(f"[INFO] in-fold dimensionality reduction: {reducer} "
              f"-> {reducer_components} components, fit on each split's training half only")
    else:
        print("[INFO] in-fold dimensionality reduction: none")

    X_imp, y_imp, X_orig, y_orig, feature_names_orig, feature_names_imp = prepare_train_test_data(
        orig_data, orig_feature_col_name, imp_data, imp_feature_col_name, features)

    groups = resolve_exam_groups(orig_data)

    # Keys that keep imputed duplicates out of evaluated folds. Always the
    # study identity: an imputed row is a reconstruction of one exam.
    orig_keys = _study_ids(orig_data)
    imp_keys = _study_ids(imp_data)

    parameters, candidates = build_search_candidates(
        class_type, unbalanced_type, search_budget, random_state, search_sampler)
    if not candidates:
        return _empty_result(class_type, error=f"no candidates for {class_type}")
    print(f"[INFO] inner search: {len(candidates)} candidate(s)"
          + (f" across {len(SAMPLER_CHOICES)} imbalance strategies" if search_sampler else ""))

    per_fold_metrics = {}
    per_fold_params, per_fold_inner_scores, per_fold_samplers, per_fold_thresholds = [], [], [], []
    # Pooled predictions and SHAP come from repeat 0 only: pooling folds from
    # different partitions would count every patient n_repeats times, so the PR
    # curve and the confusion matrix would no longer describe one pass over the
    # cohort.
    pooled_true, pooled_pred, pooled_proba, fold_records = [], [], [], []
    n_fits = 0

    for repeat in range(max(1, int(n_repeats))):
        seed = random_state + repeat
        outer_splitter, outer_groups = make_splitter(n_outer_splits, groups, random_state=seed)
        outer_splits = list(outer_splitter.split(X_orig, y_orig, groups=outer_groups))

        for fold, (outer_train, outer_test) in enumerate(outer_splits):
            X_tr, y_tr = X_orig[outer_train], y_orig[outer_train]
            X_te, y_te = X_orig[outer_test], y_orig[outer_test]

            held_out_keys = set(orig_keys[outer_test]) if orig_keys is not None else set()
            train_groups = groups[outer_train] if groups is not None else None

            # ---- inner loop: selection on outer training data only ----
            inner_splitter, inner_groups = make_splitter(n_inner_splits, train_groups,
                                                         random_state=seed)
            inner_splits = list(inner_splitter.split(X_tr, y_tr, groups=inner_groups))
            prepared = _prepare_inner_splits(
                X_tr, y_tr, inner_splits,
                orig_keys[outer_train] if orig_keys is not None else None,
                held_out_keys, X_imp, y_imp, imp_keys, scale_features,
                reducer=reducer, reducer_components=reducer_components)
            if not prepared:
                print(f"[WARN] repeat {repeat + 1} fold {fold + 1}: no usable inner split.")
                continue

            # Parallel over candidates, not over outer folds: one fold's data is
            # in memory at a time and the per-fold progress line stays readable.
            scored = _run_candidate_search(candidates, parameters.clf, prepared, n_jobs)
            n_fits += len(candidates) * len(prepared)

            first_error = next((errs[0] for _, errs in scored if errs), None)
            if first_error:
                print(f"[WARN] repeat {repeat + 1} fold {fold + 1}: some candidates "
                      f"failed, e.g. {first_error}")

            selected = select_best_candidate(candidates, [s for s, _ in scored])
            if selected is None:
                print(f"[WARN] repeat {repeat + 1} fold {fold + 1}: nothing scored; skipping.")
                continue
            best_params, best_sampler, best_inner_score = selected

            # ---- outer evaluation: refit + threshold on training data only ----
            X_tr_aug, y_tr_aug = _augment_with_imputed(X_tr, y_tr, X_imp, y_imp,
                                                       imp_keys, held_out_keys)
            # Imputed rows are prepended, so the group vector no longer lines up
            # with the augmented matrix; threshold calibration falls back to a
            # plain stratified split in that case.
            threshold_groups = train_groups if X_tr_aug.shape[0] == X_tr.shape[0] else None

            model = clone(parameters.clf).set_params(**best_params)
            pipeline, threshold, fits = fit_and_threshold(
                model, X_tr_aug, y_tr_aug, best_sampler, groups=threshold_groups,
                random_state=seed, scale=scale_features, beta=beta,
                reducer=reducer, reducer_components=reducer_components)
            n_fits += fits

            y_pred_proba = predict_scores(pipeline, X_te)
            y_pred = (y_pred_proba >= threshold).astype(int)

            calculate_final_metrics(per_fold_metrics, y_te, y_pred, y_pred_proba)
            per_fold_params.append(best_params)
            per_fold_inner_scores.append(round(best_inner_score, 4))
            per_fold_samplers.append(best_sampler)
            per_fold_thresholds.append(float(threshold))

            if repeat == 0:
                pooled_true.extend(y_te)
                pooled_pred.extend(y_pred)
                pooled_proba.extend(y_pred_proba)
                fold_records.append((pipeline, X_te))

            print(f"[Repeat {repeat + 1} fold {fold + 1}] inner PR-AUC {best_inner_score:.4f} | "
                  f"sampler {best_sampler} | threshold {threshold:.3f} | "
                  f"outer PR-AUC {per_fold_metrics['PR-AUC'][-1]:.4f} | "
                  f"predicted {np.bincount(y_pred, minlength=2).tolist()}")

    if not per_fold_params:
        return _empty_result(class_type, error="no outer fold produced predictions")

    mean_metrics = {m: round(float(np.nanmean(v)), 4)
                    for m, v in per_fold_metrics.items() if v}
    ci = _confidence_intervals(per_fold_metrics)

    pooled_true_arr = np.asarray(pooled_true)
    pooled_proba_arr = np.asarray(pooled_proba)
    pooled_metrics, cm = {}, None
    if len(pooled_true_arr) and len(np.unique(pooled_true_arr)) > 1:
        # A PR-AUC is only readable next to the positive rate: a classifier that
        # ranks at random scores the prevalence, not 0.5. AUC's baseline is 0.5.
        prevalence = round(float(np.mean(pooled_true_arr)), 4)
        pooled_metrics = {
            'PR-AUC_global': round(average_precision_score(pooled_true_arr, pooled_proba_arr), 4),
            'AUC_global': round(roc_auc_score(pooled_true_arr, pooled_proba_arr), 4),
            'Prevalence': prevalence,
            'PR-AUC_baseline': prevalence,
        }
        cm = confusion_matrix(pooled_true_arr, pooled_pred).tolist()

    shap_values, shap_input_df, important_features, shap_df = _compute_shap_across_folds(
        fold_records, class_type, features, feature_names_orig)

    # Hyperparameters and strategy vary per outer fold, so there is no single
    # "best" set. The modal choice is reported; the honest performance number is
    # mean_metrics, not a refit on all data.
    modal_params = _most_common_params(per_fold_params)
    modal_sampler = Counter(per_fold_samplers).most_common(1)[0][0]

    result = _empty_result(class_type)
    result.update({
        'metrics': mean_metrics,
        'ci': ci,
        'pooled': {'y_true': [int(v) for v in pooled_true],
                   'y_proba': [float(v) for v in pooled_proba],
                   'y_pred': [int(v) for v in pooled_pred]},
        'pooled_metrics': pooled_metrics,
        'cm': cm,
        'best_params': modal_params,
        'sampler': modal_sampler,
        'reducer': reducer,
        'threshold': round(float(np.mean(per_fold_thresholds)), 4),
        'n_fits': int(n_fits),
        'n_folds': len(per_fold_params),
        'shap_values': shap_values,
        'shap_input_df': shap_input_df,
        'important_features': important_features,
        'shap_df': shap_df,
        'inner_score': round(float(np.mean(per_fold_inner_scores)), 4),
    })
    return result


def find_best_class_params(orig_data, orig_feature_col_name, imp_data,
                           imp_feature_col_name, unbalanced_type, class_type, features,
                           n_outer_splits=5, n_inner_splits=5,
                           search_budget=DEFAULT_SEARCH_BUDGET, random_state=42,
                           scale_features=None, class_input=None,
                           reducer=None,
                           reducer_components=DEFAULT_REDUCER_COMPONENTS,
                           autosave=False):
    """Nested cross-validation, as a 9-tuple for the existing viewers.

    A thin adapter over evaluate_classifier(); everything of substance lives
    there. The metrics dict flattens the per-fold means together with the pooled
    globals, which is the shape best_params_table renders.

    final_model is FITTED on the whole cohort, so it can actually predict. Its
    reported performance is the cross-validated estimate in test_metrics, NOT a
    score of this refit - a model graded on the data it was fit to is not an
    estimate of anything.
    """
    result = evaluate_classifier(
        orig_data, orig_feature_col_name, imp_data, imp_feature_col_name,
        unbalanced_type, class_type, features, n_outer_splits=n_outer_splits,
        n_inner_splits=n_inner_splits, search_budget=search_budget,
        random_state=random_state, scale_features=scale_features,
        class_input=class_input, reducer=reducer,
        reducer_components=reducer_components, autosave=autosave,
        # F1, not F2: this entry point is what the existing viewers call, and
        # changing what their numbers mean is not this wrapper's decision.
        beta=1.0)

    if result['error'] or not result['best_params']:
        # This tuple of Nones is all the viewers get, so the reason has to reach
        # the log in full. "not best_params" with no error set would otherwise
        # fail completely silently - which is exactly how the intermittent
        # failure in test_learns_real_signal managed to hide.
        reason = result['error'] or (
            f"the run completed but selected no hyperparameters "
            f"(n_folds={result['n_folds']}, n_fits={result['n_fits']})")
        print(f"[ERROR] classification run produced no result: {reason}")
        return None, None, None, None, None, None, None, None, None

    test_metrics = dict(result['metrics'])
    if 'AUC' in result['metrics'] and result['ci'].get('AUC'):
        lo, hi = result['ci']['AUC']
        test_metrics['STD-AUC'] = round((hi - lo) / (2 * 1.96), 4)
    test_metrics.update(result['pooled_metrics'])

    cm = np.asarray(result['cm']) if result['cm'] is not None else None

    scale = scale_features
    if scale is None:
        scale = default_scaling_for(class_input if class_input is not None else class_type)
    final_model = _fit_final_model(orig_data, orig_feature_col_name, imp_data,
                                   imp_feature_col_name, features, class_type,
                                   result['best_params'], result['sampler'], scale,
                                   reducer=result.get('reducer'),
                                   reducer_components=reducer_components)

    return (final_model, result['best_params'], result['inner_score'], test_metrics, cm,
            result['shap_values'], result['shap_input_df'],
            result['important_features'], result['shap_df'])


def _fit_final_model(orig_data, orig_feature_col_name, imp_data, imp_feature_col_name,
                     features, class_type, best_params, sampler, scale,
                     reducer=None, reducer_components=DEFAULT_REDUCER_COMPONENTS):
    """Refit the modal configuration on the whole cohort, ready to predict.

    This used to be returned unfitted, which meant anything calling .predict()
    on it raised NotFittedError. It is a deployment artefact only: its
    performance is the CV estimate, never a score taken on this fit.
    """
    try:
        _, _, X_orig, y_orig, _, _ = prepare_train_test_data(
            orig_data, orig_feature_col_name, imp_data, imp_feature_col_name, features)
        prototype = ModelParams()
        prototype.get_param_grid(class_type, sampler, n_iter=1)
        model = clone(prototype.clf).set_params(**best_params)
        pipeline = build_estimator_pipeline(
            apply_imbalance_correction(model, sampler, y_orig), sampler, scale=scale,
            reducer=reducer, reducer_components=reducer_components)
        pipeline.fit(X_orig, y_orig)
        return pipeline
    except Exception as e:
        print(f"[WARN] final refit on the full cohort failed ({e}); returning None.")
        return None


def _most_common_params(per_fold_params):
    """Most frequently selected parameter set across outer folds."""
    counts = Counter(json.dumps(p, sort_keys=True, default=str) for p in per_fold_params)
    return json.loads(counts.most_common(1)[0][0])


def apply_preprocessing(pipeline, X):
    """Run X through the pipeline's transform steps, skipping the samplers.

    Samplers only act during fit and expose no transform, so they are stepped
    over here.
    """
    for _, step in pipeline.steps[:-1]:
        if hasattr(step, 'transform'):
            X = step.transform(X)
    return X


def _shap_for_one_fold(pipeline, X_val, class_type, feature_names_orig):
    """SHAP explanation for a single fitted fold, reduced to the positive class.

    The estimator is unwrapped from the pipeline and handed the preprocessed
    matrix it was actually trained on, with names narrowed to the columns that
    survived that fold's drop step.
    """
    model = pipeline.steps[-1][1]
    X_transformed = apply_preprocessing(pipeline, X_val)
    names = surviving_feature_names(pipeline, feature_names_orig)

    if class_type in ['reconstructed_traces', 'reconstructed_traces_vac']:
        shap_values, shap_input_df = compute_shap_timepoints(model, X_transformed, names)
    else:
        X_val_df = pd.DataFrame(X_transformed, columns=names)
        shap_values, shap_input_df = compute_shap_reduced(model, X_val_df, names)

    if shap_values is None or shap_input_df is None:
        return None

    return reduce_shap_to_class_1(shap_values), shap_input_df


def _concatenate_fold_explanations(fold_explanations):
    """Stack per-fold explanations over the features common to every fold.

    Each fold drops its own uninformative columns, so the surviving feature
    sets can differ. Only features present in all folds can be stacked; the
    rest are reported separately by the caller. Column order follows the first
    fold.

    The stacked object is a copy of the first fold's Explanation with its arrays
    replaced, rather than a freshly constructed one, so this does not depend on
    a particular shap constructor signature.
    """
    common = set(fold_explanations[0][1].columns)
    for _, shap_input_df in fold_explanations[1:]:
        common &= set(shap_input_df.columns)
    if not common:
        return None, None, []

    ordered = [c for c in fold_explanations[0][1].columns if c in common]
    dropped_somewhere = [c for _, df in fold_explanations for c in df.columns
                         if c not in common]

    values, data = [], []
    for shap_values, shap_input_df in fold_explanations:
        idx = [shap_input_df.columns.get_loc(c) for c in ordered]
        values.append(np.asarray(shap_values.values)[:, idx])
        data.append(np.asarray(shap_input_df.values)[:, idx])

    stacked = copy.deepcopy(fold_explanations[0][0])
    stacked.values = np.vstack(values)
    stacked.data = np.vstack(data)
    stacked.feature_names = list(ordered)

    base = getattr(stacked, 'base_values', None)
    if base is not None:
        base = np.asarray(base)
        if base.ndim >= 1 and base.shape[0] == len(fold_explanations[0][1]):
            # One base value per row: repeat it to match the stacked rows.
            stacked.base_values = np.concatenate(
                [np.asarray(getattr(sv, 'base_values', base)).reshape(-1)
                 for sv, _ in fold_explanations])

    stacked_df = pd.DataFrame(stacked.data, columns=ordered)
    return stacked, stacked_df, sorted(set(dropped_somewhere))


def _compute_shap_across_folds(fold_records, class_type, features, feature_names_orig,
                               top_n=10):
    """SHAP over every held-out fold, not just the last one.

    Each outer fold fits its own hyperparameters and its own preprocessing, so
    a single fold's ranking is one model's view of 1/k of the cohort. Stacking
    the folds explains every patient exactly once, on the model that did not
    train on them, which is the ranking that answers "what drove the
    classifier".
    """
    empty = (None, None, [], pd.DataFrame())
    if not fold_records:
        return empty

    if features in ['global_strain']:
        return empty
    if not is_shap_supported_model(fold_records[0][0]):
        # KNN, decision trees and the spline/logistic pipeline are not covered
        # by the explainers this project uses.
        return empty

    fold_explanations = []
    for i, (pipeline, X_val) in enumerate(fold_records):
        try:
            result = _shap_for_one_fold(pipeline, X_val, class_type, feature_names_orig)
            if result is not None:
                fold_explanations.append(result)
        except Exception as e:
            print(f"[WARN] SHAP failed on fold {i + 1}: {e}")

    if not fold_explanations:
        print("[Error] SHAP produced nothing on any fold.")
        return empty

    try:
        shap_values, shap_input_df, partial = _concatenate_fold_explanations(fold_explanations)
        if shap_values is None:
            print("[Error] folds share no common features; cannot stack SHAP.")
            return empty

        if partial:
            print(f"[INFO] {len(partial)} feature(s) dropped in at least one fold and "
                  f"excluded from the pooled ranking: {', '.join(map(str, partial[:8]))}"
                  + (' ...' if len(partial) > 8 else ''))

        important_features, shap_df = select_important_features(
            shap_values, shap_input_df.columns, top_n)

        # select_important_features returns (None, None) on failure.
        important_features = important_features or []
        if shap_df is None:
            shap_df = pd.DataFrame()

        print(f"[INFO] SHAP pooled over {len(fold_explanations)} folds "
              f"({shap_values.values.shape[0]} held-out rows); "
              f"kept top {len(important_features)} of {len(shap_input_df.columns)} features")
        return shap_values, shap_input_df, important_features, shap_df

    except Exception as e:
        print(f"[Error] SHAP aggregation failed: {e}")
        return empty


def calculate_final_metrics(metrics, y_true, y_pred, y_pred_proba):
    def record(name, value):
        metrics.setdefault(name, []).append(value)

    # average='binary' reports the positive (remodeling) class. The previous
    # average='weighted' blended both classes by prevalence, so on an imbalanced
    # cohort it was dominated by the majority class and read far too high.
    record('MCC', round(matthews_corrcoef(y_true, y_pred), 4))
    record('Precision', round(precision_score(y_true, y_pred, average='binary', pos_label=1, zero_division=0), 4))
    record('Recall', round(recall_score(y_true, y_pred, average='binary', pos_label=1, zero_division=0), 4))
    record('F1', round(f1_score(y_true, y_pred, average='binary', pos_label=1, zero_division=0), 4))
    record('Accuracy', round(accuracy_score(y_true, y_pred), 4))
    record('PR-AUC', round(average_precision_score(y_true, y_pred_proba), 4))
    record('AUC', round(roc_auc_score(y_true, y_pred_proba), 4))

    # Specificity is recall of the NEGATIVE class. On a 3:1 cohort it is the
    # counterweight to Recall: a model that flags everyone scores Recall 1.0 and
    # Specificity 0.0, and BalancedAccuracy (their mean) reports the 0.5 that
    # Accuracy hides.
    record('Specificity', round(recall_score(y_true, y_pred, average='binary', pos_label=0,
                                             zero_division=0), 4))
    record('BalancedAccuracy', round(balanced_accuracy_score(y_true, y_pred), 4))

    # Brier is a calibration score and only means anything on probabilities.
    # predict_scores() falls back to decision_function for estimators without
    # predict_proba, and those are unbounded margins, not probabilities - scoring
    # them would produce a number that looks like calibration and is not.
    scores = np.asarray(y_pred_proba, dtype=float)
    if scores.size and np.nanmin(scores) >= 0.0 and np.nanmax(scores) <= 1.0:
        record('Brier', round(brier_score_loss(y_true, scores), 4))
    else:
        record('Brier', float('nan'))

    return metrics


def _json_safe(value):
    """Coerce a result dict into types json.dump accepts.

    The dicts come straight out of sklearn, so they hold np.float64/np.int64
    scalars and ndarray confusion matrices. str()-ing them would write a file
    that loads but whose numbers are no longer numbers.
    """
    if value is None or isinstance(value, str):
        return value
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        value = float(value)
        # NaN/Infinity are not valid JSON; null survives a strict load.
        return value if np.isfinite(value) else None
    if isinstance(value, np.ndarray):
        return [_json_safe(v) for v in value.tolist()]
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    return str(value)


# Everything needed to rebuild the leaderboard row and the PR curve. The SHAP
# objects are deliberately excluded - megabytes of per-row arrays that do not
# survive JSON anyway.
_SAVED_RESULT_KEYS = ('class_type', 'error', 'metrics', 'ci', 'pooled', 'pooled_metrics',
                      'cm', 'best_params', 'sampler', 'threshold', 'reducer',
                      'n_fits', 'n_folds', 'inner_score', 'important_features')


def save_result_snapshot(result, context=None, reports_dir=None):
    """Write a finished run to reports/ before anything tries to render it.

    A nested-CV run costs hours; the figures built from it cost milliseconds and
    are the fragile part. A removed plotly helper once discarded a complete
    five-fold run at the final heatmap call, after every fold and the pooled
    SHAP had already been computed. Persisting here means a rendering bug costs
    a page refresh rather than an afternoon.

    Never raises: a failed save must not take down a run that actually worked.
    Returns the path written, or None.
    """
    try:
        root = Path(reports_dir) if reports_dir else Path(__file__).resolve().parents[3] / 'reports'
        root.mkdir(parents=True, exist_ok=True)

        context = dict(context or {})
        stamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        parts = [str(context.get(k, '')) for k in ('class_input', 'input_type')]
        slug = '_'.join(re.sub(r'[^A-Za-z0-9-]+', '', p) for p in parts if p)
        name = f"classification_{stamp}_{result.get('class_type', 'run')}" + (f'_{slug}' if slug else '')

        payload = {
            'report_type': 'classification_run',
            'schema_version': 1,
            'saved_at': datetime.now().isoformat(timespec='seconds'),
            'config': _json_safe(context),
            'result': {k: _json_safe(result.get(k)) for k in _SAVED_RESULT_KEYS},
        }
        path = root / f'{name}.json'
        with open(path, 'w', encoding='utf-8') as handle:
            json.dump(payload, handle, indent=2)
        print(f'[INFO] run saved to {path}')
        return path
    except Exception as e:
        print(f'[WARN] could not save the run to reports/ ({type(e).__name__}: {e}); '
              f'the results below are still valid.')
        return None


def impute_missing_values(df, n_neighbors=15, missing_threshold=30):
    imputed_df = df.copy()

    if isinstance(imputed_df, pd.Series):
        imputed_df = imputed_df.to_frame()

    imputed_df = imputed_df.apply(pd.to_numeric, errors='coerce')

    num_cols = imputed_df.select_dtypes(include=['int64', 'float64']).columns
    missing_per_column = imputed_df[num_cols].isnull().sum() / len(imputed_df) * 100
    cols_to_keep = missing_per_column[missing_per_column <= missing_threshold].index
    imputed_df = imputed_df[cols_to_keep]

    imputer = KNNImputer(n_neighbors=n_neighbors)
    imputed_df[cols_to_keep] = imputer.fit_transform(imputed_df[cols_to_keep])

    constant_features = imputed_df.columns[imputed_df.nunique() <= 1]
    imputed_df.drop(columns=constant_features, inplace=True)

    if imputed_df.isnull().values.any():
        raise ValueError("Imputation failed, there are still NaN values in the dataframe.")

    return imputed_df


def convert_echoParam_values(echo_paramDF):
    for col in echo_paramDF.columns:
        if echo_paramDF[col].apply(lambda x: isinstance(x, str)).any():
            echo_paramDF[col] = echo_paramDF[col].replace({'True': True, 'False': False})

    echo_paramDF = echo_paramDF.astype(
        {col: int for col in echo_paramDF.select_dtypes(include=['bool']).columns}
    )

    echo_paramDF = echo_paramDF.replace({None: np.nan})

    for col in echo_paramDF.select_dtypes(include=['object']).columns:
        try:
            echo_paramDF[col] = pd.to_numeric(echo_paramDF[col], errors='coerce')
        except Exception as e:
            print(f"Could not convert column {col}: {e}")

    echo_paramDF = echo_paramDF.apply(
        lambda col: col.map(
            lambda x: int(x) if isinstance(x, float) and x.is_integer() else round(x, 4) if isinstance(x, float) else x
        )
    )

    return echo_paramDF


# def load_data_traces(data_source):
#     filter_by_type = select_data_type('gls_params_vac', 32)
#
#     tracesDF = load_queries_table_db('pca_results_view',
#                                      'data_source', 'in', [data_source])
#     filtered_tracesDF = tracesDF[tracesDF['reconstructed_type'].str.contains(filter_by_type, na=False)]
#     echoParam_tracesDF, feature_col_name = get_echoParams_data(filtered_tracesDF, data_source, None)
#     return echoParam_tracesDF, feature_col_name


def preprocess_data(echo_paramDF, isStandardized, fit_preprocessing=True):
    """Split a biomarker frame into features, labels and patient identifiers.

    fit_preprocessing=False returns the features raw - missing values intact,
    unscaled - so the KNN imputer and the scaler can be fit per fold inside
    build_estimator_pipeline() instead of once over the whole cohort. Fitting
    them here means the imputer's neighbours and the scaler's mean and variance
    are computed partly from patients that later land in a test fold.
    """
    y = echo_paramDF[['Total_Remodeling']].reset_index(drop=True)
    scaler = StandardScaler()
    if isStandardized:
        patient_study_info = echo_paramDF[['Patient_ID', 'study_nr']]
        columns_to_drop = ['Patient_ID', 'study_nr', 'data_source', 'EDV_source', 'Total_Remodeling', 'Height',
                           'Weight']
        if fit_preprocessing:
            imputed_X, num_cols = impute_tab_data(columns_to_drop, echo_paramDF)
            X_scaled = imputed_X.copy()
            X_scaled[num_cols] = scaler.fit_transform(X_scaled[num_cols])
        else:
            scaler = None
            X_scaled = (echo_paramDF
                        .drop(columns=columns_to_drop, errors='ignore')
                        .apply(pd.to_numeric, errors='coerce'))
            X_scaled = X_scaled.select_dtypes(include=['int64', 'float64'])
        X_scaled = X_scaled.reset_index(drop=True)
        patient_study_info = patient_study_info.reset_index(drop=True)

    else:
        patient_study_info = None
        columns_to_drop = ['study_nr', 'Total_Remodeling']
        imputed_X, num_cols = impute_tab_data(columns_to_drop, echo_paramDF)
        X_scaled = imputed_X.copy()
        X_scaled = X_scaled.reset_index(drop=True)
        X_scaled = pd.concat([echo_paramDF[columns_to_drop].reset_index(drop=True), X_scaled], axis=1)

    return X_scaled, y, patient_study_info, scaler


def impute_tab_data(columns_to_drop, echo_paramDF):
    X = echo_paramDF.drop(columns=columns_to_drop)
    imputed_X = impute_missing_values(X)
    num_cols = imputed_X.select_dtypes(include=['int64', 'float64']).columns
    return imputed_X, num_cols


def load_selected_features():
    from cardio_app.services.classification.data import load_queries_table_db
    featuresDF = load_queries_table_db(dbTableNames['calculated_features_remodeling_T'], None, None, None)
    unique_featuresDF = featuresDF.drop_duplicates(subset=['feature'], keep='first')
    unique_features_list = unique_featuresDF['feature'].tolist()
    return featuresDF, unique_features_list


