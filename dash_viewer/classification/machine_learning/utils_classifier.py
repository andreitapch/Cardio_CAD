import gc
import numpy as np
import pandas as pd
from numba import njit
from dash import dash_table
import plotly.figure_factory as ff

from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import ADASYN
from imblearn.under_sampling import RandomUnderSampler

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
from dash_viewer.classification.model_params import ModelParams
from pattern_recognition.utils.parameters_dics import dbTableNames


def get_best_metrics(X_test, X_train, cv_scores, model, unbalanced_type, y_test, y_train):
    X_resampled, y_resampled = get_samples(X_train, unbalanced_type, y_train)
    model.fit(X_resampled, y_resampled)
    y_pred = model.predict(X_test)
    if hasattr(model, "predict_proba"):
        y_pred_proba = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        y_pred_proba = model.decision_function(X_test)
    else:
        raise ValueError("The model does not support probability prediction.")

    metric = average_precision_score(y_test, y_pred_proba)
    # metric = roc_auc_score(y_test, y_pred_proba)
    # metric = f1_score(y_test, y_pred)
    return metric


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


def find_best_class_params(orig_data, orig_feature_col_name, imp_data,
                           imp_feature_col_name, unbalanced_type, class_type, features):
    try:
        parameters = ModelParams()
        parameters.get_param_grid(class_type, unbalanced_type)
        X_imp, y_imp, X_orig, y_orig = None, None, None, None
        skf, splits = None, None
        best_params = None
        best_test_score = -np.inf
        best_model = None

        X_imp, y_imp, X_orig, y_orig, feature_names_orig, feature_names_imp = prepare_train_test_data(
            orig_data, orig_feature_col_name, imp_data, imp_feature_col_name, features)

        for param_combination in parameters.param_combinations:
            model = parameters.clf.set_params(**param_combination)
            cv_scores = []

            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            splits = list(skf.split(X_orig, y_orig))

            # early_stop_threshold = 0.35

            for train_index, test_index in splits:
                X_train, X_test = X_orig[train_index], X_orig[test_index]
                y_train, y_test = y_orig[train_index], y_orig[test_index]

                if imp_data is not None:
                    X_train = np.vstack((X_imp, X_train))
                    y_train = np.concatenate((y_imp, y_train))

                score = get_best_metrics(X_test, X_train, cv_scores, model, unbalanced_type, y_test, y_train)
                cv_scores.append(score)

                # if np.mean(cv_scores) < early_stop_threshold:
                #     break

            # mean_cv_score = np.mean(cv_scores)
            # in find_best_class_params, after the fold loop:
            mean_cv_score = np.mean([s for s in cv_scores if not np.isnan(s)])


            if mean_cv_score > best_test_score:
                best_test_score = mean_cv_score
                best_params = param_combination
                best_model = model

        if best_model:
            if imp_data is not None:
                isImp = True
                test_metrics, cm, shap_values, shap_input_df, important_features, shap_df = apply_best_params_with_metrics(best_params, X_imp, y_imp, unbalanced_type,
                                                                  class_type, splits, X_orig, y_orig, isImp,
                                                                  feature_names_orig, feature_names_imp, features)
            else:
                isImp = False
                var = None
                test_metrics, cm, shap_values, shap_input_df, important_features, shap_df  = apply_best_params_with_metrics(best_params, var, var, unbalanced_type,
                                                                  class_type, var, X_orig, y_orig, isImp,
                                                                  feature_names_orig, feature_names_imp, features)



            return best_model, best_params, best_test_score, test_metrics, cm, shap_values, shap_input_df, important_features, shap_df
        else:
            return None, None, None, None, None, None, None, None, None

    except Exception as e:
        print(f"An error occurred: {e}")
        return None, None, None, None, None, None, None, None, None

    finally:
        for var in ['X_imp', 'y_imp', 'X_orig', 'y_orig', 'X_train', 'y_train', 'X_test', 'y_test']:
            if var in locals():
                del locals()[var]
        gc.collect()


def apply_best_params_with_metrics(best_params, X, y, unbalanced_type,class_type,
                                   splits, X_orig, y_orig, isImp, feature_names_orig,
                                   feature_names_imp, features):
    shap_values = None
    shap_input_df = None

    parameters = ModelParams()
    parameters.get_param_grid(class_type, unbalanced_type)
    model = parameters.clf.set_params(**best_params)
    return_shap = is_shap_supported_model(model)
    metrics = {
        'MCC': [], 'Precision': [], 'Recall': [], 'F1': [],
        'Accuracy': [], 'AUC': [], 'STD-AUC': [], 'PR-AUC': []
    }

    all_y_true = []
    all_y_pred = []
    all_y_proba = []

    try:
        if isImp:
            for i, (train_index, test_index) in enumerate(splits):
                X_train = np.vstack((X, X_orig[train_index]))
                y_train = np.concatenate((y, y_orig[train_index]))
                X_val = X_orig[test_index]
                y_val = y_orig[test_index]

                X_resampled, y_resampled = get_samples(X_train, unbalanced_type, y_train)
                model.fit(X_resampled, y_resampled)

                # y_pred_val = model.predict(X_val)
                y_pred_proba = model.predict_proba(X_val)[:, 1]
                # fpr, tpr, thresholds = roc_curve(y_val, y_pred_proba)
                # j_scores = tpr - fpr
                # best_thresh = thresholds[np.argmax(j_scores)]

                prec, rec, thr = precision_recall_curve(y_val, y_pred_proba)
                f_scores = (2 * prec * rec) / (prec + rec + 1e-12)
                best_idx = np.nanargmax(f_scores)
                best_thresh = thr[max(best_idx - 1, 0)]

                y_pred_val = (y_pred_proba >= best_thresh).astype(int)

                # print(f"[Fold {i + 1}] Probabilities stats - min: {np.min(y_pred_proba):.3f}, max: {np.max(y_pred_proba):.3f}")
                # print(f"Predictions: {np.bincount(y_pred_val)}")

                calculate_final_metrics(metrics, y_val, y_pred_val, y_pred_proba)
                all_y_true.extend(y_val)
                all_y_pred.extend(y_pred_val)
                all_y_proba.extend(y_pred_proba)

        else:
            # skf = StratifiedKFold(n_splits=5, shuffle=False)
            # for i, (train_index, test_index) in enumerate(skf.split(X_orig, y_orig)):
            for i, (train_index, test_index) in enumerate(splits):

                X_train, X_val = X_orig[train_index], X_orig[test_index]
                y_train, y_val = y_orig[train_index], y_orig[test_index]

                X_resampled, y_resampled = get_samples(X_train, unbalanced_type, y_train)
                model.fit(X_resampled, y_resampled)

                # Get predicted probabilities
                y_pred_proba = model.predict_proba(X_val)[:, 1]

                # fpr, tpr, thresholds = roc_curve(y_val, y_pred_proba)
                # j_scores = tpr - fpr
                # best_thresh = thresholds[np.argmax(j_scores)]

                prec, rec, thr = precision_recall_curve(y_val, y_pred_proba)
                f_scores = (2 * prec * rec) / (prec + rec + 1e-12)
                best_idx = np.nanargmax(f_scores)
                best_thresh = thr[max(best_idx - 1, 0)]

                y_pred_val = (y_pred_proba >= best_thresh).astype(int)

                print(
                    f"[Fold {i + 1}] Probabilities stats - min: {np.min(y_pred_proba):.3f}, max: {np.max(y_pred_proba):.3f}")
                print(f"Predictions: {np.bincount(y_pred_val)}")  # Check balance

                calculate_final_metrics(metrics, y_val, y_pred_val, y_pred_proba)
                all_y_true.extend(y_val)
                all_y_pred.extend(y_pred_val)
                all_y_proba.extend(y_pred_proba)

        all_y_true = np.array(all_y_true)
        all_y_proba = np.array(all_y_proba)

        global_pr_auc = average_precision_score(all_y_true, all_y_proba)
        global_roc_auc = roc_auc_score(all_y_true, all_y_proba)
        # mean_metrics = {metric: np.mean(scores) for metric, scores in metrics.items()}

        mean_metrics = {m: np.mean(v) for m, v in metrics.items()}
        mean_metrics['STD-AUC'] = round(np.std(metrics['AUC']), 4)
        mean_metrics['PR-AUC_global'] = round(global_pr_auc, 4)
        mean_metrics['AUC_global'] = round(global_roc_auc, 4)

        cm = confusion_matrix(all_y_true, all_y_pred)

        if return_shap and features not in ['global_strain']:#'class_features',
            try:
                if class_type in ['reconstructed_traces', 'reconstructed_traces_vac']:
                    shap_values, shap_input_df = compute_shap_timepoints(model, X_val, feature_names_orig)
                else:
                    X_val_df = pd.DataFrame(X_val, columns=feature_names_orig)
                    shap_values, shap_input_df = compute_shap_reduced(model, X_val_df, feature_names_orig)

                if shap_input_df is not None:
                    important_features, shap_df = select_important_features(shap_values, shap_input_df.columns, 10)
                else:
                    important_features, shap_df = [], pd.DataFrame()

                print(
                    f"[INFO] Retained {len(important_features)} important features out of {len(shap_input_df.columns)}")

            except Exception as e:
                print(f"[Error] SHAP computation failed: {e}")

            return mean_metrics, cm, shap_values, shap_input_df, important_features, shap_df
        else:
            return  mean_metrics, cm, None, None, [], pd.DataFrame()

    except Exception as e:
        print(f"[ERROR] apply_best_params_with_metrics failed: {e}")
        return {}, None, None, None, None, None

    finally:
        for var in ['X', 'y', 'X_orig', 'y_orig', 'X_train', 'X_val', 'y_train', 'y_val']:
            if var in locals():
                del locals()[var]
        gc.collect()


def calculate_final_metrics(metrics, y_true, y_pred, y_pred_proba):
    metrics['MCC'].append(round(matthews_corrcoef(y_true, y_pred), 4))
    metrics['Precision'].append(round(precision_score(y_true, y_pred, average='weighted', zero_division=0), 4))
    metrics['Recall'].append(round(recall_score(y_true, y_pred, average='weighted', zero_division=0), 4))
    metrics['F1'].append(round(f1_score(y_true, y_pred, average='weighted', zero_division=0), 4))
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

    best_param_df = pd.DataFrame([best_param], columns=best_param.keys())
    test_metrics_df = pd.DataFrame([test_metrics],
                                   columns=test_metrics.keys()).round(4)

    best_score = round(best_test_score, 4)
    best_param_df['PR-AUC:'] = [best_score]

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
