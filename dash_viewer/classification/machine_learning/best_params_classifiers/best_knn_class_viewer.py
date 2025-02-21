import os
import gc
import numpy as np
import pandas as pd

import plotly.figure_factory as ff

from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import ADASYN

import dash_bootstrap_components as dash_bc
from dash import html, dcc, Output, Input, ctx
from dash import callback, State, dash, dash_table

from imblearn.combine import SMOTEENN, SMOTETomek
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.metrics import f1_score, accuracy_score
from sklearn.metrics import precision_score, roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import matthews_corrcoef, recall_score
from pattern_recognition.data.loadTraces import LoadTraces as load_traces
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.utils import parallel_backend

from dash_viewer.classification.load_prepare_data import load_imputed_data, get_concat_18_segments, select_data_type, \
    load_segment_table_db, get_feature_components, get_echoParams_data

from dash_viewer.classification.model_params import ModelParams

# os.environ['MKL_NUM_THREADS'] = '1'
# os.environ['OMP_NUM_THREADS'] = '1'

gc.enable()


def safe_eval(val):
    try:
        return eval(val)
    except Exception:
        return val


@callback(
    Output('id-table-best-params', 'children'),
    Output('id-table-test-scores', 'children'),
    Output('id-table-cm-scores', 'figure'),
    Input('isDataImputed-id', 'value'),
    Input('id-best-knn-btn', 'n_clicks'),
    Input('id-data-source-dropdown_bc', 'value'),
    Input('id-unbalanced_KnnBest-dropdown', 'value'),
    State('id-concat_Type-dropdown', 'value'),
    State('id-class_type-dropdown', 'value'),
)
def update_output(isImputed, best_neighbors, data_source, unbalanced_type, knn_type, class_type):
    global instance_clf, param_table
    instance_clf = BestParamClassViewer()
    studyType = 'BL'
    if "id-best-knn-btn" == ctx.triggered_id and best_neighbors > 0:
        if data_source:
            data_source_mapping = {
                'Speqle': 'speqle_segmentalStrainInterp_T',
                'GE': 'st_segmentalStrainInterp_T',
                'Philips': 'phi_segmentalStrainInterp_T',
                'GE_Philips': ['phi_segmentalStrainInterp_T', 'st_segmentalStrainInterp_T']
            }
            if data_source in data_source_mapping:
                table_name = data_source_mapping[data_source]
                instance_clf.table_name = table_name

                print('-----------------------------------------------------------------------------------------------')
                print(f'------------ Find the best parameters for classification in table: {data_source}--------------')
                print(f'unbalanced method: {unbalanced_type}')
                print(f'Experiment type: {knn_type}')
                print(f'Class type: {class_type}')
                print(f'Imputed data: {isImputed}')
                print('-----------------------------------------------------------------------------------------------')

                param_table, test_metrics_table, cm_table = load_data_traces(knn_type, isImputed, unbalanced_type,
                                                                             class_type,
                                                                             data_source)
                gc.collect()
                return param_table, test_metrics_table, cm_table
        else:
            table_name = None
            return dash.no_update, dash.no_update, dash.no_update

    else:
        gc.collect()
        return dash.no_update, dash.no_update, dash.no_update


def best_params_table(orig_concat_tracesDF, orig_feature_col_name,
                      imp_concat_tracesDF, imp_feature_col_name, unbalanced_type, class_type):
    (best_model, best_param, best_test_score,
     test_metrics, cm) = instance_clf.find_best_class_params(orig_concat_tracesDF, orig_feature_col_name,
                                                             imp_concat_tracesDF,
                                                             imp_feature_col_name,
                                                             unbalanced_type,
                                                             class_type)

    best_param_df = pd.DataFrame([best_param], columns=best_param.keys())
    test_metrics_df = pd.DataFrame([test_metrics],
                                   columns=test_metrics.keys()).round(4)

    best_score = round(best_test_score, 4)

    best_param_df['Best Score:'] = [best_score]

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
    cm_table_fig = instance_clf.get_confusion_matrix_table(cm)

    return best_param, best_param_table, test_metrics_table, cm_table_fig


def load_data_traces(knn_type, isImputed, unbalanced_type, class_type, data_source):
    imp_concat_tracesDF = None
    imp_feature_col_name = None
    filter_by_type = select_data_type(knn_type, None)

    # if data_source == 'GE_Philips':
    #     data_label = ['GE', 'Philips']
    # elif data_source == 'GE':
    #     data_label = ['GE']
    # elif data_source == 'Philips':
    #     data_label = ['Philips']
    # else:
    #     data_label = ['Speqle']

    tracesDF = load_segment_table_db('pca_results_view',
                                     'data_source', 'in', [data_source])

    #get reconstructed traces from PCA
    original_tracesDF = tracesDF[tracesDF['reconstructed_type'] == f'{filter_by_type}-0']

    if isImputed:
        imputed_tracesDF = tracesDF[tracesDF['reconstructed_type'] == f'{filter_by_type}-1']
    else:
        imputed_tracesDF = None

    if knn_type == 'concat_18_segments_100' or knn_type == 'concat_18_segments_95':

        orig_concat_tracesDF, orig_feature_col_name = get_concat_18_segments(original_tracesDF)
        if imputed_tracesDF is not None:
            imp_concat_tracesDF, imp_feature_col_name = get_concat_18_segments(imputed_tracesDF)

    elif knn_type == 'gls_remod_class':
        orig_concat_tracesDF, orig_feature_col_name = get_echoParams_data(original_tracesDF, data_source,
                                                                          'global_strain')
        if imputed_tracesDF is not None:
            imp_concat_tracesDF, imp_feature_col_name = get_echoParams_data(imputed_tracesDF, data_source,
                                                                            'global_strain')

    elif knn_type == 'echo_remod_class':
        orig_concat_tracesDF, orig_feature_col_name = get_echoParams_data(original_tracesDF, data_source, None)
        if imputed_tracesDF is not None:
            imp_concat_tracesDF, imp_feature_col_name = get_echoParams_data(imputed_tracesDF, data_source, None)

    else:
        table_name = 'calculated_pca_results_T'
        feature_col_name = 'pca_results'
        orig_concat_tracesDF, orig_feature_col_name = get_feature_components(original_tracesDF, data_source,
                                                                             filter_by_type, feature_col_name,
                                                                             table_name)
        if imputed_tracesDF is not None:
            imp_concat_tracesDF, imp_feature_col_name = get_feature_components(imputed_tracesDF, data_source,
                                                                               filter_by_type, feature_col_name,
                                                                               table_name)

    best_param, param_table, test_metrics_table, cm_table_fig = best_params_table(orig_concat_tracesDF,
                                                                                  orig_feature_col_name,
                                                                                  imp_concat_tracesDF,
                                                                                  imp_feature_col_name,
                                                                                  unbalanced_type, class_type)
    return param_table, test_metrics_table, cm_table_fig


class BestParamClassViewer:

    def __init__(self):
        self.cm = None
        self.X_resampled = None
        self.parameters = None
        self.y_resampled = None
        self.metrics = None
        self.param_grid = None
        self.clf = None
        self.tracesDF = None
        self.df_isImputed_0 = None
        self.studyType = 'BL'
        self.pca_resultsDF = None
        self.test, self.train, self.train = None, None, None
        self.concat_array_curves = None
        self.reconstructed_tracesDF = None
        self.calc_remod_df = None

    def get_content(self):
        dropdown_options = [
            {'label': '18 segments concat', 'value': 'concat_18_segments_100'},
            {'label': '18 segments concat PCA: 95% Variance', 'value': 'concat_18_segments_95'},
            {'label': '100% of PCA variance components', 'value': '100_variance_components'},
            {'label': '95% of PCA variance components', 'value': '95_Variance_components'},
            {'label': '100% of PCA variance components per segment', 'value': '100_components_per_segment'},
            {'label': '95% of PCA variance per segment concatenated', 'value': '95_components_per_segment'},
            {'label': 'GLS', 'value': 'gls_remod_class'}
        ]

        return html.Div([
            dash_bc.Row([
                dash_bc.Col([
                    html.Br(),
                    dash_bc.Row([
                        html.Label('Select Data Source:', style={'fontWeight': 'bold', 'marginLeft': '20px'}),
                        dcc.Dropdown(
                            id='id-data-source-dropdown_bc',
                            options=[
                                {'label': 'Speqle', 'value': 'Speqle'},
                                {'label': 'GE', 'value': 'GE'},
                                {'label': 'Philips', 'value': 'Philips'},
                                {'label': 'GE + Philips', 'value': 'GE_Philips'}
                            ],
                            value='GE_Philips',
                            style={'width': '300px', 'marginLeft': '10px'}
                        ),
                    ]),

                    html.Br(),
                    html.Br(),

                    dash_bc.Row([
                        dash_bc.Checkbox(
                            id='isDataImputed-id',
                            label='Include imputed data',
                            value=True,
                            style={'marginLeft': '20px', 'width': '70%'}
                        ),
                    ]),
                ], width=2),
                dash_bc.Col([
                    # html.Br(),
                    # html.Br(),
                    html.Br(),
                    html.Label('Classify by:', style={'marginLeft': '35px', 'fontWeight': 'bold'}),
                    dcc.Dropdown(
                        id='id-concat_Type-dropdown',
                        options=dropdown_options,
                        value='concat_18_segments_100',
                        style={'marginLeft': '20px', 'height': 'Auto', 'width': '80%'}
                    ),
                ], width=5),
                dash_bc.Col([
                    html.Br(),
                    html.Label('Machine Learning Classifiers:', style={'fontWeight': 'bold'}),
                    dcc.Dropdown(
                        id='id-class_type-dropdown',
                        options=[
                            {'label': 'KNN', 'value': 'KNN_class'},
                            {'label': 'Decision Tree', 'value': 'DT_class'},
                            {'label': 'Random Forest', 'value': 'RF_class'},
                            # {'label': 'Gradient Boosting', 'value': 'GBM_class'},
                            # {'label': 'XGBoost', 'value': 'XGBoost_class'},
                            # {'label': 'LightGBM', 'value': 'LightGBM_class'},
                            {'label': 'SVM', 'value': 'SVM_class'},
                            {'label': 'MLP', 'value': 'MLP_class'},
                            # {'label': 'Naive Bayes', 'value': 'NaiveBayes_class'}
                        ],
                        value='KNN_class',
                        style={'height': 'Auto', 'width': '100%'}
                    ),
                ], style={'width': '100%'}),

                dash_bc.Col([
                    html.Br(),
                    html.Label('Unbalanced:', style={'fontWeight': 'bold'}),
                    dcc.Dropdown(
                        id='id-unbalanced_KnnBest-dropdown',
                        options=[
                            {'label': 'None', 'value': 'None'},
                            {'label': 'SMOTE', 'value': 'SMOTE'},
                            {'label': 'ADASYN', 'value': 'ADASYN'},
                            {'label': 'SMOTEENN', 'value': 'SMOTEENN'},
                            {'label': 'SMOTETomek', 'value': 'SMOTETomek'}
                        ],
                        value='None',
                        style={'height': 'Auto', 'width': '70%'}
                    ),
                ], style={'width': '100%'}),
            ]),
            dash_bc.Row([
                html.Br(),
                html.Label('Find the best parameters:',
                           style={'fontWeight': 'bold', 'marginLeft': '20px'}),
                dash_bc.Button('Classify', id='id-best-knn-btn', n_clicks=0,
                               style={'marginLeft': '30px', 'width': '120px'})
            ]),

            html.Br(),
            html.Br(),

            dash_bc.Row([
                html.Div(id='id-table-best-params',
                         style={'marginLeft': '20px'})
            ], justify="center", style={'width': '50%', 'margin': 'auto'}),
            html.Br(),
            dash_bc.Row([
                html.Div(id='id-table-test-scores',
                         style={'marginLeft': '20px'})
            ], justify="center", style={'width': '50%', 'margin': 'auto'}),
            html.Br(),
            html.Br(),
            dash_bc.Row([
                dcc.Graph(id='id-table-cm-scores')
            ], justify="center", style={'width': '70%', 'margin': 'auto'}),
        ])

    def find_best_class_params(self, orig_data, orig_feature_col_name, imp_data,
                               imp_feature_col_name, unbalanced_type, class_type):
        try:
            self.parameters = ModelParams()
            self.parameters.get_param_grid(class_type, unbalanced_type)
            X_imp, y_imp, X_orig, y_orig = None, None, None, None
            splits = None
            skf = None
            best_params = None
            best_test_score = -np.inf
            best_model = None

            X_imp, y_imp, X_orig, y_orig = self.prepare_train_test_data(
                orig_data, orig_feature_col_name, imp_data, imp_feature_col_name)

            for param_combination in self.parameters.param_combinations:
                model = self.parameters.clf.set_params(**param_combination)
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

                        score = self.get_best_metrics(X_test, X_train, cv_scores, model, unbalanced_type, y_test,
                                                      y_train)
                        cv_scores.append(score)
                else:
                    skf = StratifiedKFold(n_splits=5, shuffle=False)
                    for train_index, test_index in skf.split(X_orig, y_orig):
                        X_train, X_test = X_orig[train_index], X_orig[test_index]
                        y_train, y_test = y_orig[train_index], y_orig[test_index]

                        score = self.get_best_metrics(X_test, X_train, cv_scores, model, unbalanced_type, y_test,
                                                      y_train)
                        cv_scores.append(score)

                mean_cv_score = np.mean(cv_scores)

                if mean_cv_score > best_test_score:
                    best_test_score = mean_cv_score
                    best_params = param_combination
                    best_model = model

            if imp_data is not None:
                test_metrics, cm = self.apply_best_params_with_metrics(best_params, X_imp, y_imp, unbalanced_type,
                                                                       class_type, splits, X_orig, y_orig, isImp=True)
            else:

                test_metrics, cm = self.apply_best_params_with_metrics(best_params, None, None,
                                                                       unbalanced_type,
                                                                       class_type, None, X_orig, y_orig, isImp=False)

            return best_model, best_params, best_test_score, test_metrics, cm

        except Exception as e:
            print(f"An error occurred: {e}")
        finally:
            del X_imp, y_imp, X_orig, y_orig, X_train, y_train, X_test, y_test
            gc.collect()

    def get_best_metrics(self, X_test, X_train, cv_scores, model, unbalanced_type, y_test, y_train):
        self.get_samples(X_train, unbalanced_type, y_train)
        model.fit(self.X_resampled, self.y_resampled)

        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
        pr_auc_score = round(auc(recall, precision), 4)
        f1 = round(f1_score(y_test, y_pred), 4)
        mcc = round(matthews_corrcoef(y_test, y_pred), 4)

        weighted_score = 0.5 * pr_auc_score + 0.3 * f1 + 0.2 * mcc

        return weighted_score

    def prepare_train_test_data(self, orig_data, orig_feature_col_name, imp_data, imp_feature_col_name):
        if imp_data is not None:
            if imp_feature_col_name == 'Global Strain %':
                X_imp = imp_data[imp_feature_col_name].values.reshape(-1, 1)
            else:
                X_imp = np.stack(imp_data[imp_feature_col_name].values)
            y_imp = imp_data['Total_Remodeling'].values

            if orig_feature_col_name == 'Global Strain %':
                X_orig = orig_data[orig_feature_col_name].values.reshape(-1, 1)
            else:
                X_orig = np.stack(orig_data[orig_feature_col_name].values)
            y_orig = orig_data['Total_Remodeling'].values

            return X_imp, y_imp, X_orig, y_orig
        else:
            if orig_feature_col_name == 'Global Strain %':
                X = orig_data[orig_feature_col_name].values.reshape(-1, 1)
            else:
                X = np.stack(orig_data[orig_feature_col_name].values)
            y = orig_data['Total_Remodeling'].values

            return None, None, X, y

    def get_samples(self, X_train, unbalanced_type, y_train):
        self.X_resampled = None
        self.y_resampled = None

        if unbalanced_type == 'SMOTE':
            oversample = SMOTE(random_state=42)
            self.X_resampled, self.y_resampled = oversample.fit_resample(X_train, y_train)
        elif unbalanced_type == 'ADASYN':
            oversample = ADASYN(random_state=42)
            self.X_resampled, self.y_resampled = oversample.fit_resample(X_train, y_train)
        elif unbalanced_type == 'SMOTEENN':
            oversample = SMOTEENN(random_state=42)
            self.X_resampled, self.y_resampled = oversample.fit_resample(X_train, y_train)
        elif unbalanced_type == 'SMOTETomek':
            oversample = SMOTETomek(random_state=42)
            self.X_resampled, self.y_resampled = oversample.fit_resample(X_train, y_train)
        else:
            oversample = None
            self.X_resampled = X_train
            self.y_resampled = y_train

    def apply_best_params_with_metrics(self, best_params, X, y, unbalanced_type, class_type, splits, X_orig, y_orig,
                                       isImp):
        try:
            self.parameters = ModelParams()
            self.parameters.get_param_grid(class_type, unbalanced_type)
            model = self.parameters.clf.set_params(**best_params)
            metrics = {
                'MCC': [], 'Precision': [], 'Recall': [], 'F1': [],
                'Accuracy': [], 'AUC': [], 'PR-AUC': []
            }

            all_y_true = []
            all_y_pred = []

            if isImp:
                for i, (train_index, test_index) in enumerate(splits):
                    X_train = np.vstack((X, X_orig[train_index]))
                    y_train = np.concatenate((y, y_orig[train_index]))
                    X_val = X_orig[test_index]
                    y_val = y_orig[test_index]

                    self.get_samples(X_train, unbalanced_type, y_train)
                    model.fit(self.X_resampled, self.y_resampled)
                    # print(f"Actual number of iterations: {self.parameters.clf.n_iter_}")
                    y_pred_val = model.predict(X_val)
                    y_pred_proba = model.predict_proba(X_val)[:, 1]

                    self.calculate_final_metrics(metrics, y_val, y_pred_val, y_pred_proba)
                    all_y_true.extend(y_val)
                    all_y_pred.extend(y_pred_val)
            else:
                skf = StratifiedKFold(n_splits=5, shuffle=False)
                for train_index, test_index in skf.split(X_orig, y_orig):
                    X_train, X_val = X_orig[train_index], X_orig[test_index]
                    y_train, y_val = y_orig[train_index], y_orig[test_index]

                    self.get_samples(X_train, unbalanced_type, y_train)
                    model.fit(self.X_resampled, self.y_resampled)

                    y_pred_val = model.predict(X_val)
                    y_pred_proba = model.predict_proba(X_val)[:, 1]

                    self.calculate_final_metrics(metrics, y_val, y_pred_val, y_pred_proba)
                    all_y_true.extend(y_val)
                    all_y_pred.extend(y_pred_val)

            mean_metrics = {metric: np.mean(scores) for metric, scores in metrics.items()}

            cm = confusion_matrix(all_y_true, all_y_pred)
            return mean_metrics, cm
        finally:
            del X, y, X_train, X_val, y_train, y_val
            gc.collect()

    def calculate_final_metrics(self, metrics, y_true, y_pred, y_pred_proba):
        metrics['MCC'].append(round(matthews_corrcoef(y_true, y_pred), 4))
        metrics['Precision'].append(round(precision_score(y_true, y_pred, average='weighted', zero_division=0), 4))
        metrics['Recall'].append(round(recall_score(y_true, y_pred, average='weighted', zero_division=0), 4))
        metrics['F1'].append(round(f1_score(y_true, y_pred, average='weighted', zero_division=0), 4))
        metrics['Accuracy'].append(round(accuracy_score(y_true, y_pred), 4))

        fpr, tpr, _ = roc_curve(y_true, y_pred)
        auc_score = auc(fpr, tpr)
        metrics['AUC'].append(round(auc_score, 4))

        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
        pr_auc = round(auc(recall, precision), 4)
        metrics['PR-AUC'].append(pr_auc)

    def get_confusion_matrix_table(self, cm):
        x = ['0', '1']
        y = ['0', '1']
        cm_fig = ff.create_annotated_heatmap(cm, x=x, y=y, colorscale='Blues')
        cm_fig.update_layout(
            xaxis=dict(title='Predicted value'),
            yaxis=dict(title='True value')
        )
        return cm_fig
