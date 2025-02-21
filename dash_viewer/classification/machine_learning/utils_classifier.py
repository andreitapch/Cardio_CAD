import gc
import numpy as np
import pandas as pd
import plotly.figure_factory as ff

from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import ADASYN
from sklearn.impute import KNNImputer
from sklearn.metrics import precision_score
from imblearn.combine import SMOTEENN, SMOTETomek
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score, accuracy_score
from sklearn.metrics import matthews_corrcoef, recall_score
from sklearn.metrics import roc_curve, auc, confusion_matrix

from dash_viewer.classification.load_prepare_data import select_data_type
from pattern_recognition.data.loadTraces import LoadTraces as load_traces
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, KFold
from dash_viewer.classification.load_prepare_data import select_data_type, load_segment_table_db, get_echoParams_data
from sklearn.preprocessing import StandardScaler
from sklearn.utils import parallel_backend

from dash_viewer.classification.model_params import ModelParams
from pattern_recognition.utils.parameters_dics import dbTableNames


def get_best_metrics_(X_test, X_train, cv_scores, model, unbalanced_type, y_test, y_train):
    X_resampled, y_resampled = get_samples(X_train, unbalanced_type, y_train)
    model.fit(X_resampled, y_resampled)

    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    auc_score = auc(fpr, tpr)
    return auc_score


def get_best_metrics(X_test, X_train, cv_scores, model, unbalanced_type, y_test, y_train):
    X_resampled, y_resampled = get_samples(X_train, unbalanced_type, y_train)
    model.fit(X_resampled, y_resampled)

    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    pr_auc_score = round(auc(recall, precision), 4)
    f1 = round(f1_score(y_test, y_pred), 4)
    mcc = round(matthews_corrcoef(y_test, y_pred), 4)

    weighted_score = 0.4 * pr_auc_score + 0.3 * f1 + 0.3 * mcc

    return weighted_score


def prepare_train_test_data(orig_data, orig_feature_col_name, imp_data, imp_feature_col_name):
    if imp_data is not None:
        if imp_feature_col_name == 'global_strain':
            X_imp = imp_data[imp_feature_col_name].values.reshape(-1, 1)
        else:
            X_imp = np.stack(imp_data[imp_feature_col_name].values)
        y_imp = imp_data['Total_Remodeling'].values

        if orig_feature_col_name == 'global_strain':
            X_orig = orig_data[orig_feature_col_name].values.reshape(-1, 1)
        else:
            X_orig = np.stack(orig_data[orig_feature_col_name].values)
        y_orig = orig_data['Total_Remodeling'].values

        return X_imp, y_imp, X_orig, y_orig
    else:
        if orig_feature_col_name == 'global_strain':
            X = orig_data[orig_feature_col_name].values.reshape(-1, 1)
        else:
            X = np.stack(orig_data[orig_feature_col_name].values)
        y = orig_data['Total_Remodeling'].values

        return None, None, X, y


def get_samples(X_train, unbalanced_type, y_train):
    X_resampled = None
    y_resampled = None

    if unbalanced_type == 'SMOTE':
        oversample = SMOTE(random_state=42)
        X_resampled, y_resampled = oversample.fit_resample(X_train, y_train)
    elif unbalanced_type == 'ADASYN':
        oversample = ADASYN(random_state=42)
        X_resampled, y_resampled = oversample.fit_resample(X_train, y_train)
    elif unbalanced_type == 'SMOTEENN':
        oversample = SMOTEENN(random_state=42)
        X_resampled, y_resampled = oversample.fit_resample(X_train, y_train)
    elif unbalanced_type == 'SMOTETomek':
        oversample = SMOTETomek(random_state=42)
        X_resampled, y_resampled = oversample.fit_resample(X_train, y_train)
    else:
        oversample = None
        X_resampled = X_train
        y_resampled = y_train
    return X_resampled, y_resampled


def find_best_class_params(orig_data, orig_feature_col_name, imp_data,
                           imp_feature_col_name, unbalanced_type, class_type):
    try:
        parameters = ModelParams()
        parameters.get_param_grid(class_type, unbalanced_type)
        X_imp, y_imp, X_orig, y_orig = None, None, None, None
        splits = None
        skf = None
        best_params = None
        best_test_score = -np.inf
        best_model = None

        X_imp, y_imp, X_orig, y_orig = prepare_train_test_data(
            orig_data, orig_feature_col_name, imp_data, imp_feature_col_name)

        for param_combination in parameters.param_combinations:
            model = parameters.clf.set_params(**param_combination)
            cv_scores = []

            if imp_data is not None:
                skf = StratifiedKFold(n_splits=5, shuffle=False)
                splits = list(skf.split(X_orig, y_orig))

                for i, (train_index, test_index) in enumerate(splits):
                    X_train_orig = X_orig[train_index]
                    y_train_orig = y_orig[train_index]
                    X_test = X_orig[test_index]
                    y_test = y_orig[test_index]

                    X_train = np.vstack((X_imp, X_train_orig))
                    y_train = np.concatenate((y_imp, y_train_orig))

                    score = get_best_metrics(X_test, X_train, cv_scores, model, unbalanced_type, y_test,
                                             y_train)
                    cv_scores.append(score)
            else:
                skf = StratifiedKFold(n_splits=5, shuffle=False)
                for train_index, test_index in skf.split(X_orig, y_orig):
                    X_train, X_test = X_orig[train_index], X_orig[test_index]
                    y_train, y_test = y_orig[train_index], y_orig[test_index]

                    score = get_best_metrics(X_test, X_train, cv_scores, model, unbalanced_type, y_test,
                                             y_train)
                    cv_scores.append(score)

            mean_cv_score = np.mean(cv_scores)

            if mean_cv_score > best_test_score:
                best_test_score = mean_cv_score
                best_params = param_combination
                best_model = model

        if imp_data is not None:
            test_metrics, cm = apply_best_params_with_metrics(best_params, X_imp, y_imp, unbalanced_type,
                                                              class_type, splits, X_orig, y_orig, isImp=True)
        else:

            test_metrics, cm = apply_best_params_with_metrics(best_params, None, None,
                                                              unbalanced_type,
                                                              class_type, None, X_orig, y_orig, isImp=False)

        return best_model, best_params, best_test_score, test_metrics, cm

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        del X_imp, y_imp, X_orig, y_orig, X_train, y_train, X_test, y_test
        gc.collect()


def apply_best_params_with_metrics(best_params, X, y, unbalanced_type, class_type, splits, X_orig, y_orig,
                                   isImp):
    try:
        parameters = ModelParams()
        parameters.get_param_grid(class_type, unbalanced_type)
        model = parameters.clf.set_params(**best_params)
        metrics = {
            'MCC': [], 'Precision': [], 'Recall': [], 'F1': [],
            'Accuracy': [], 'AUC': [], 'STD-AUC': []
        }

        all_y_true = []
        all_y_pred = []

        if isImp:
            for i, (train_index, test_index) in enumerate(splits):
                X_train = np.vstack((X, X_orig[train_index]))
                y_train = np.concatenate((y, y_orig[train_index]))
                X_val = X_orig[test_index]
                y_val = y_orig[test_index]

                X_resampled, y_resampled = get_samples(X_train, unbalanced_type, y_train)
                model.fit(X_resampled, y_resampled)

                y_pred_val = model.predict(X_val)
                y_pred_proba = model.predict_proba(X_val)[:, 1]

                calculate_final_metrics(metrics, y_val, y_pred_val, y_pred_proba)
                all_y_true.extend(y_val)
                all_y_pred.extend(y_pred_val)
        else:
            skf = StratifiedKFold(n_splits=5, shuffle=False)
            for train_index, test_index in skf.split(X_orig, y_orig):
                X_train, X_val = X_orig[train_index], X_orig[test_index]
                y_train, y_val = y_orig[train_index], y_orig[test_index]

                X_resampled, y_resampled = get_samples(X_train, unbalanced_type, y_train)
                model.fit(X_resampled, y_resampled)

                y_pred_val = model.predict(X_val)
                y_pred_proba = model.predict_proba(X_val)[:, 1]

                metrics = calculate_final_metrics(metrics, y_val, y_pred_val, y_pred_proba)
                all_y_true.extend(y_val)
                all_y_pred.extend(y_pred_val)

        mean_metrics = {metric: np.mean(scores) for metric, scores in metrics.items()}

        cm = confusion_matrix(all_y_true, all_y_pred)
        return mean_metrics, cm
    finally:
        del X, y, X_train, X_val, y_train, y_val
        gc.collect()


def calculate_final_metrics(metrics, y_true, y_pred, y_pred_proba):
    metrics['MCC'].append(round(matthews_corrcoef(y_true, y_pred), 4))
    metrics['Precision'].append(round(precision_score(y_true, y_pred, average='weighted', zero_division=0), 4))
    metrics['Recall'].append(round(recall_score(y_true, y_pred, average='weighted', zero_division=0), 4))
    metrics['F1'].append(round(f1_score(y_true, y_pred, average='weighted', zero_division=0), 4))
    metrics['Accuracy'].append(round(accuracy_score(y_true, y_pred), 4))

    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    auc_score = auc(fpr, tpr)
    metrics['AUC'].append(round(auc_score, 4))

    auc_mean = np.mean(metrics['AUC'])
    auc_std = np.std(metrics['AUC'])

    # Append results
    # metrics['MEAN-AUC'].append(round(auc_mean, 4))
    metrics['STD-AUC'].append(round(auc_std, 4))

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
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns
    imputed_df = df.copy()

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
    echo_paramDF.replace({'True': True, 'False': False}, inplace=True)
    echo_paramDF = echo_paramDF.astype(
        {col: int for col in echo_paramDF.select_dtypes(include=['bool']).columns}
    )

    for col in echo_paramDF.select_dtypes(include=['object']).columns:
        try:
            echo_paramDF[col] = pd.to_numeric(echo_paramDF[col], errors='raise')
        except ValueError:
            pass

    echo_paramDF = echo_paramDF.apply(
        lambda col: col.map(
            lambda x: int(x) if isinstance(x, float) and x.is_integer() else x) if col.dtype == float else col
    )
    echo_paramDF = echo_paramDF.apply(lambda col: col.map(lambda x: round(x, 4) if isinstance(x, float) else x))
    echo_paramDF = echo_paramDF.replace({None: np.nan})

    return echo_paramDF

def load_data_traces(data_source):
    filter_by_type = select_data_type('gls_remod_class_vac', 32)

    tracesDF = load_segment_table_db('pca_results_view',
                                     'data_source', 'in', [data_source])
    filtered_tracesDF = tracesDF[tracesDF['reconstructed_type'].str.contains(filter_by_type, na=False)]
    echoParam_tracesDF, feature_col_name = get_echoParams_data(filtered_tracesDF, data_source, None)
    return echoParam_tracesDF, feature_col_name


def preprocess_data(echo_paramDF):
    y = echo_paramDF['Total_Remodeling']
    patient_study_info = echo_paramDF[['patientID', 'study_nr']]
    columns_to_drop = ['patientID', 'study_nr', 'data_source', 'EDV_source', 'Total_Remodeling', 'Height', 'Weight']
    X = echo_paramDF.drop(columns=columns_to_drop)
    imputed_X = impute_missing_values(X)
    num_cols = imputed_X.select_dtypes(include=['int64', 'float64']).columns
    scaler = StandardScaler()
    X_scaled = imputed_X.copy()
    X_scaled[num_cols] = scaler.fit_transform(X_scaled[num_cols])
    X_scaled = X_scaled.reset_index(drop=True)
    patient_study_info = patient_study_info.reset_index(drop=True)

    return X_scaled, y, patient_study_info


def load_selected_features():
    table_name = dbTableNames['calculated_features_remodeling_T']
    featuresDF = load_segment_table_db(table_name,None, None, None)
    unique_featuresDF = featuresDF.drop_duplicates(subset=['feature'], keep='first')
    unique_features_list = unique_featuresDF['feature'].tolist()
    return featuresDF, unique_features_list


def evaluate_feature_subset(X, y, feature_subset, classifier=None):
    X_subset = X[feature_subset]
    skf = StratifiedKFold(n_splits=5, shuffle=True)

    precision_list, recall_list, auc_list = [], [], []

    for train_index, test_index in skf.split(X_subset, y):
        X_train, X_test = X_subset.iloc[train_index], X_subset.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        classifier.fit(X_train, y_train)

        y_prob = classifier.predict_proba(X_test)[:, 1]
        precision, recall, _ = precision_recall_curve(y_test, y_prob)
        auc_score = auc(recall, precision)

        precision_list.append(precision)
        recall_list.append(recall)
        auc_list.append(auc_score)

    avg_auc = np.mean(auc_list)

    return avg_auc, np.mean(precision_list), np.mean(recall_list)
