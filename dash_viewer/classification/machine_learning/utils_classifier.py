import gc
import json
import traceback
from collections import Counter

import numpy as np
import pandas as pd
from numba import njit
from dash import dash_table
import plotly.figure_factory as ff

from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import ADASYN
from imblearn.under_sampling import RandomUnderSampler

from sklearn.base import clone
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score
from sklearn.metrics import matthews_corrcoef, recall_score
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.metrics import precision_score, roc_auc_score, average_precision_score

from dash_viewer.classification.machine_learning.utils.shap_explainer import compute_shap_timepoints, \
    compute_shap_reduced, is_shap_supported_model, select_important_features
from dash_viewer.classification.model_params import ModelParams, DEFAULT_SEARCH_BUDGET
from pattern_recognition.utils.parameters_dics import dbTableNames


def predict_scores(model, X):
    """Continuous scores for the positive class, whatever the estimator exposes."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        return model.decision_function(X)
    raise ValueError("The model does not support probability prediction.")


def get_best_metrics(X_test, X_train, model, unbalanced_type, y_test, y_train):
    """Threshold-free score for one train/test pair, used to rank hyperparameters."""
    X_resampled, y_resampled = get_samples(X_train, unbalanced_type, y_train)
    model.fit(X_resampled, y_resampled)
    y_pred_proba = predict_scores(model, X_test)
    return average_precision_score(y_test, y_pred_proba)


def choose_threshold(y_true, y_scores):
    """Cutoff maximising F1 on the data given.

    Must only ever be called on TRAINING data. Calling it on the fold being
    scored fits the cutoff to the labels it is then graded against, which
    inflates every threshold-dependent metric.
    """
    prec, rec, thr = precision_recall_curve(y_true, y_scores)
    f_scores = (2 * prec * rec) / (prec + rec + 1e-12)
    best_idx = int(np.nanargmax(f_scores))
    # precision_recall_curve returns len(thr) == len(prec) - 1, so the last
    # point of the curve has no threshold of its own. Clamp instead of the
    # previous best_idx - 1, which shifted every other index by one.
    return thr[min(best_idx, len(thr) - 1)]


def fit_and_threshold(model, X_train, y_train, unbalanced_type, random_state=42):
    """Fit on the training fold and pick a cutoff without touching held-out data.

    The cutoff is chosen on an inner split of the training fold, then the model
    is refit on the full training fold so no training data is wasted.
    """
    X_res, y_res = get_samples(X_train, unbalanced_type, y_train)

    threshold = 0.5
    try:
        inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state)
        inner_train, inner_cal = next(iter(inner.split(X_train, y_train)))

        X_inner_res, y_inner_res = get_samples(X_train[inner_train], unbalanced_type,
                                               y_train[inner_train])
        model.fit(X_inner_res, y_inner_res)
        cal_scores = predict_scores(model, X_train[inner_cal])
        threshold = choose_threshold(y_train[inner_cal], cal_scores)
    except Exception as e:
        print(f"[WARN] threshold calibration failed ({e}); falling back to 0.5")

    model.fit(X_res, y_res)
    return model, threshold


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

        features = orig_data.columns[2:].tolist()
        feature_names = features.copy()

        if 'Total_Remodeling' in features:
            features.remove('Total_Remodeling')
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


def get_samples(X_train, unbalanced_type, y_train):
    X_resampled, y_resampled = X_train, y_train

    if unbalanced_type == 'SMOTE':
        oversample = SMOTE(sampling_strategy='auto', k_neighbors=3, random_state=42)
        X_resampled, y_resampled = oversample.fit_resample(X_train, y_train)

    elif unbalanced_type == 'ADASYN':
        oversample = ADASYN(sampling_strategy='auto', random_state=42)
        X_resampled, y_resampled = oversample.fit_resample(X_train, y_train)

    elif unbalanced_type == 'Balanced':
        oversample = ADASYN(sampling_strategy='auto', random_state=42)
        smote = SMOTE(sampling_strategy='auto', k_neighbors=3, random_state=42)
        under = RandomUnderSampler(sampling_strategy='auto', random_state=42)
        X_resampled, y_resampled = oversample.fit_resample(X_train, y_train)
        X_resampled, y_resampled = under.fit_resample(X_resampled, y_resampled)

    return X_resampled, y_resampled


def _study_ids(df):
    """Patient/study identifiers aligned row-for-row with the feature matrix.

    prepare_train_test_data() never reorders rows, so positional indexing into
    this array matches the X it returns.
    """
    if df is None or 'study_nr' not in df.columns:
        return None
    return df['study_nr'].values


def _augment_with_imputed(X_train, y_train, X_imp, y_imp, imp_studies, held_out_studies):
    """Add imputed rows to a training fold, minus any belonging to held-out studies.

    The imputed frame covers the same cohort as the original one, so adding it
    wholesale (the previous behaviour) put a near-duplicate of every held-out
    patient into training. Rows whose study_nr is being evaluated are dropped.
    """
    if X_imp is None:
        return X_train, y_train

    if imp_studies is None:
        print("[WARN] imputed frame has no study_nr column - cannot verify patient "
              "separation; imputed rows are being dropped rather than leaked.")
        return X_train, y_train

    keep = ~np.isin(imp_studies, list(held_out_studies))
    if not keep.any():
        return X_train, y_train

    return np.vstack((X_imp[keep], X_train)), np.concatenate((y_imp[keep], y_train))


def find_best_class_params(orig_data, orig_feature_col_name, imp_data,
                           imp_feature_col_name, unbalanced_type, class_type, features,
                           n_outer_splits=5, n_inner_splits=5,
                           search_budget=DEFAULT_SEARCH_BUDGET, random_state=42):
    """Nested cross-validation: hyperparameters chosen inside each outer fold.

    The previous implementation selected hyperparameters on one set of 5 folds
    and then reported metrics on those same folds, so the best-of-N maximum was
    reported as if it were an unbiased estimate. Here the inner loop only ever
    sees the outer training split, and the reported metrics come from outer test
    folds that took no part in selection.
    """
    try:
        X_imp, y_imp, X_orig, y_orig, feature_names_orig, feature_names_imp = prepare_train_test_data(
            orig_data, orig_feature_col_name, imp_data, imp_feature_col_name, features)

        orig_studies = _study_ids(orig_data)
        imp_studies = _study_ids(imp_data)

        parameters = ModelParams()
        parameters.get_param_grid(class_type, unbalanced_type, n_iter=search_budget,
                                  random_state=random_state)

        metrics = {'MCC': [], 'Precision': [], 'Recall': [], 'F1': [],
                   'Accuracy': [], 'AUC': [], 'PR-AUC': []}
        all_y_true, all_y_pred, all_y_proba = [], [], []
        per_fold_params, per_fold_inner_scores = [], []
        last_fold = None

        outer = StratifiedKFold(n_splits=n_outer_splits, shuffle=True, random_state=random_state)
        outer_splits = list(outer.split(X_orig, y_orig))

        for fold, (outer_train, outer_test) in enumerate(outer_splits):
            X_tr, y_tr = X_orig[outer_train], y_orig[outer_train]
            X_te, y_te = X_orig[outer_test], y_orig[outer_test]

            outer_test_studies = set(orig_studies[outer_test]) if orig_studies is not None else set()

            # ---- inner loop: hyperparameter selection, outer training data only ----
            inner = StratifiedKFold(n_splits=n_inner_splits, shuffle=True,
                                    random_state=random_state)
            inner_splits = list(inner.split(X_tr, y_tr))

            best_inner_score, best_params = -np.inf, None

            for param_combination in parameters.param_combinations:
                inner_scores = []
                for inner_train, inner_test in inner_splits:
                    Xi_tr, yi_tr = X_tr[inner_train], y_tr[inner_train]
                    Xi_te, yi_te = X_tr[inner_test], y_tr[inner_test]

                    held_out = set(outer_test_studies)
                    if orig_studies is not None:
                        held_out |= set(orig_studies[outer_train][inner_test])
                    Xi_tr, yi_tr = _augment_with_imputed(Xi_tr, yi_tr, X_imp, y_imp,
                                                         imp_studies, held_out)

                    model = clone(parameters.clf).set_params(**param_combination)
                    try:
                        inner_scores.append(
                            get_best_metrics(Xi_te, Xi_tr, model, unbalanced_type, yi_te, yi_tr))
                    except Exception as e:
                        print(f"[WARN] fold {fold + 1} params {param_combination} failed: {e}")
                        inner_scores.append(np.nan)

                valid = [s for s in inner_scores if not np.isnan(s)]
                if not valid:
                    continue
                mean_inner = float(np.mean(valid))

                if mean_inner > best_inner_score:
                    best_inner_score, best_params = mean_inner, param_combination

            if best_params is None:
                print(f"[WARN] outer fold {fold + 1}: no parameter combination scored; skipping.")
                continue

            # ---- outer evaluation: refit + threshold on training data only ----
            X_tr_aug, y_tr_aug = _augment_with_imputed(X_tr, y_tr, X_imp, y_imp,
                                                       imp_studies, outer_test_studies)

            model = clone(parameters.clf).set_params(**best_params)
            model, threshold = fit_and_threshold(model, X_tr_aug, y_tr_aug, unbalanced_type,
                                                 random_state=random_state)

            y_pred_proba = predict_scores(model, X_te)
            y_pred = (y_pred_proba >= threshold).astype(int)

            calculate_final_metrics(metrics, y_te, y_pred, y_pred_proba)
            all_y_true.extend(y_te)
            all_y_pred.extend(y_pred)
            all_y_proba.extend(y_pred_proba)

            per_fold_params.append(best_params)
            per_fold_inner_scores.append(round(best_inner_score, 4))
            last_fold = (model, X_te)

            print(f"[Fold {fold + 1}] inner PR-AUC {best_inner_score:.4f} | "
                  f"threshold {threshold:.3f} | outer PR-AUC {metrics['PR-AUC'][-1]:.4f} | "
                  f"predicted {np.bincount(y_pred, minlength=2).tolist()}")

        if not all_y_true:
            print("[ERROR] no outer fold produced predictions.")
            return None, None, None, None, None, None, None, None, None

        all_y_true = np.asarray(all_y_true)
        all_y_proba = np.asarray(all_y_proba)

        mean_metrics = {m: round(float(np.mean(v)), 4) for m, v in metrics.items() if v}
        mean_metrics['STD-AUC'] = round(float(np.std(metrics['AUC'])), 4)
        mean_metrics['PR-AUC_global'] = round(average_precision_score(all_y_true, all_y_proba), 4)
        mean_metrics['AUC_global'] = round(roc_auc_score(all_y_true, all_y_proba), 4)

        cm = confusion_matrix(all_y_true, all_y_pred)

        # Hyperparameters now vary per outer fold, so there is no single "best"
        # set. The modal choice is reported for display; the honest performance
        # number is mean_metrics, not a refit on all data.
        best_params = _most_common_params(per_fold_params)
        best_test_score = float(np.mean(per_fold_inner_scores))

        shap_values, shap_input_df, important_features, shap_df = _compute_shap_for_fold(
            last_fold, class_type, features, feature_names_orig)

        final_model = clone(parameters.clf).set_params(**best_params)

        return (final_model, best_params, best_test_score, mean_metrics, cm,
                shap_values, shap_input_df, important_features, shap_df)

    except Exception as e:
        traceback.print_exc()
        print(f"An error occurred in find_best_class_params: {e}")
        return None, None, None, None, None, None, None, None, None

    finally:
        gc.collect()


def _most_common_params(per_fold_params):
    """Most frequently selected parameter set across outer folds."""
    counts = Counter(json.dumps(p, sort_keys=True, default=str) for p in per_fold_params)
    return json.loads(counts.most_common(1)[0][0])


def _compute_shap_for_fold(last_fold, class_type, features, feature_names_orig):
    """SHAP for the final outer fold's model.

    Note: this explains one fold, not the ensemble of folds. Averaging SHAP
    across folds would be more faithful but the models differ in shape.
    """
    empty = (None, None, [], pd.DataFrame())
    if last_fold is None:
        return empty

    model, X_val = last_fold
    if not is_shap_supported_model(model) or features in ['global_strain']:
        return empty

    try:
        if class_type in ['reconstructed_traces', 'reconstructed_traces_vac']:
            shap_values, shap_input_df = compute_shap_timepoints(model, X_val, feature_names_orig)
        else:
            X_val_df = pd.DataFrame(X_val, columns=feature_names_orig)
            shap_values, shap_input_df = compute_shap_reduced(model, X_val_df, feature_names_orig)

        if shap_input_df is None:
            return empty

        important_features, shap_df = select_important_features(shap_values, shap_input_df.columns, 10)
        print(f"[INFO] Retained {len(important_features)} important features "
              f"out of {len(shap_input_df.columns)}")
        return shap_values, shap_input_df, important_features, shap_df

    except Exception as e:
        print(f"[Error] SHAP computation failed: {e}")
        return empty


def calculate_final_metrics(metrics, y_true, y_pred, y_pred_proba):
    # average='binary' reports the positive (remodeling) class. The previous
    # average='weighted' blended both classes by prevalence, so on an imbalanced
    # cohort it was dominated by the majority class and read far too high.
    metrics['MCC'].append(round(matthews_corrcoef(y_true, y_pred), 4))
    metrics['Precision'].append(round(precision_score(y_true, y_pred, average='binary', pos_label=1, zero_division=0), 4))
    metrics['Recall'].append(round(recall_score(y_true, y_pred, average='binary', pos_label=1, zero_division=0), 4))
    metrics['F1'].append(round(f1_score(y_true, y_pred, average='binary', pos_label=1, zero_division=0), 4))
    metrics['Accuracy'].append(round(accuracy_score(y_true, y_pred), 4))
    metrics['PR-AUC'].append(round(average_precision_score(y_true, y_pred_proba), 4))

    auc_score = roc_auc_score(y_true, y_pred_proba)
    metrics['AUC'].append(round(auc_score, 4))

    # auc_mean = np.mean(metrics['AUC'])
    # auc_std = np.std(metrics['AUC'])
    #
    # metrics['STD-AUC'].append(round(auc_std, 4))

    return metrics


def get_confusion_matrix_table(cm):
    x = ['0', '1']
    y = ['0', '1']
    cm_fig = ff.create_annotated_heatmap(cm, x=x, y=y, colorscale='Blues')
    cm_fig.update_layout(
        xaxis=dict(title='Predicted value'),
        yaxis=dict(title='True value')
    )
    return cm_fig


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


def preprocess_data(echo_paramDF, isStandardized):
    y = echo_paramDF[['Total_Remodeling']].reset_index(drop=True)
    # study_nr = echo_paramDF[['study_nr']].reset_index(drop=True)
    scaler = StandardScaler()
    if isStandardized:
        patient_study_info = echo_paramDF[['Patient_ID', 'study_nr']]
        columns_to_drop = ['Patient_ID', 'study_nr', 'data_source', 'EDV_source', 'Total_Remodeling', 'Height',
                           'Weight']
        imputed_X, num_cols = impute_tab_data(columns_to_drop, echo_paramDF)
        X_scaled = imputed_X.copy()
        X_scaled[num_cols] = scaler.fit_transform(X_scaled[num_cols])
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
    from dash_viewer.classification.load_prepare_data import load_queries_table_db
    featuresDF = load_queries_table_db(dbTableNames['calculated_features_remodeling_T'], None, None, None)
    unique_featuresDF = featuresDF.drop_duplicates(subset=['feature'], keep='first')
    unique_features_list = unique_featuresDF['feature'].tolist()
    return featuresDF, unique_features_list


def best_params_table(orig_concat_tracesDF, orig_feature_col_name,
                      imp_concat_tracesDF, imp_feature_col_name, unbalanced_type, class_type, features):

    (best_model, best_param, best_test_score, test_metrics,
     cm,shap_values, shap_input_df, important_features, shap_df) = find_best_class_params(orig_concat_tracesDF,
                                                                                          orig_feature_col_name,
                                                                                         imp_concat_tracesDF,
                                                                                         imp_feature_col_name,
                                                                                         unbalanced_type,
                                                                                         class_type,
                                                                                         features)

    if best_param is None or test_metrics is None:
        print("[ERROR] classification run produced no result; see traceback above.")
        return None, None, None, None, None, None, None, [], pd.DataFrame()

    best_param_df = pd.DataFrame([best_param], columns=best_param.keys())
    test_metrics_df = pd.DataFrame([test_metrics],
                                   columns=test_metrics.keys()).round(4)

    # Inner-loop selection score, NOT the reported performance. The honest
    # numbers are in test_metrics, which come from held-out outer folds.
    best_param_df['Inner PR-AUC (selection):'] = [round(best_test_score, 4)]

    best_param_table = dash_table.DataTable(
        data=best_param_df.to_dict('records'),
        columns=[{'name': i, 'id': i} for i in best_param_df.columns],
        style_cell={'textAlign': 'left'},
    )

    test_metrics_table = dash_table.DataTable(
        data=test_metrics_df.to_dict('records'),
        columns=[{'name': i, 'id': i} for i in test_metrics_df.columns],
        style_cell={'textAlign': 'left'},
    )
    cm_table_fig = get_confusion_matrix_table(cm) if cm is not None else None


    return best_param, best_param_table, test_metrics_table, cm, cm_table_fig, shap_values, shap_input_df, important_features, shap_df
