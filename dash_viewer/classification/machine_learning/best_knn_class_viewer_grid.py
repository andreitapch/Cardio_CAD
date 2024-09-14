import gc
import numpy as np
import pandas as pd

from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import ADASYN

import dash_bootstrap_components as dash_bc
from dash import html, dcc, Output, Input, ctx
from dash import callback, State, dash, dash_table
from imblearn.combine import SMOTEENN, SMOTETomek

from sklearn.metrics import f1_score, accuracy_score
from sklearn.metrics import precision_score, roc_auc_score
from sklearn.metrics import matthews_corrcoef, recall_score
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold

from dash_viewer.classification.load_prepare_data import load_imputed_data, get_concat_18_segments, select_data_type, \
    load_segment_table_db, get_echoParams_data, get_pca_components

from dash_viewer.classification.model_params import ModelParams

gc.enable()


def safe_eval(val):
    try:
        return eval(val)
    except Exception:
        return val


@callback(
    Output('id-table-best-params', 'children'),
    Output('id-table-test-scores', 'children'),
    Input('isDataImputed-id', 'value'),
    Input('id-best-knn-btn', 'n_clicks'),
    Input('id-unbalanced_KnnBest-dropdown', 'value'),
    State('id-concat_Type-dropdown', 'value'),
    State('id-class_type-dropdown', 'value'),
)
def update_output(isImputed, best_neighbors, unbalanced_type, knn_type, class_type):
    global instance_clf, param_table
    instance_clf = BestParamClassViewer()

    if "id-best-knn-btn" == ctx.triggered_id and best_neighbors > 0:
        print('---------------------------------------------------------------------------')
        print('-------------- Find the best parameters for classification: ---------------')
        print(f'unbalanced method: {unbalanced_type}')
        print(f'Experiment type: {knn_type}')
        print(f'Class type: {class_type}')
        print(f'Imputed data: {isImputed}')
        print('---------------------------------------------------------------------------')
        tracesDF, df_isImputed_1, df_isImputed_0 = load_imputed_data(studyType='BL')
        param_table, test_score_table = load_data_traces(knn_type, isImputed, unbalanced_type, class_type,
                                                         tracesDF, df_isImputed_1, df_isImputed_0)
        gc.collect()
        return param_table, test_score_table
    else:
        gc.collect()
        return dash.no_update, dash.no_update


def best_params_table(concat_tracesDF, feature_col_name, unbalanced_type, class_type):
    best_param_df = pd.DataFrame()

    best_param, avg_scores, best_score = instance_clf.find_best_class_params(concat_tracesDF, feature_col_name,
                                                                             unbalanced_type, class_type)
    best_param_df = pd.DataFrame([best_param], columns=best_param.keys())
    rename_dict = {col: col.replace('knn__', '').replace('knn__n_', '') for col in best_param_df.columns}
    best_param_df = best_param_df.rename(columns=rename_dict)

    best_param_df['Best Score'] = [round(best_score, 4)]

    best_param_table = dash_table.DataTable(
        data=best_param_df.to_dict('records'),
        columns=[{'name': i, 'id': i} for i in best_param_df.columns],
        style_cell={'textAlign': 'left'},
    )

    test_score_table = dash_table.DataTable(
        data=[{'Metrics': k, 'Values': v} for k, v in avg_scores[best_param].items()],
        columns=[{'name': 'Metrics', 'id': 'Metrics'}, {'name': 'Values', 'id': 'Values'}],
        style_cell={'textAlign': 'left'},
    )
    return best_param_table, test_score_table



def load_data_traces(knn_type, isImputed, unbalanced_type, class_type,
                     tracesDF, df_isImputed_1, df_isImputed_0):
    filter_by_type = select_data_type(knn_type, isImputed)
    concat_tracesDF = pd.DataFrame()
    reconstructed_tracesDF = load_segment_table_db('reconstructed_remodeling_view',
                                                   'reconstructed_type', '=', filter_by_type)
    if not isImputed:
        study_nr_df = df_isImputed_0
        reconstructed_tracesDF = reconstructed_tracesDF[
            reconstructed_tracesDF['study_nr'].isin(study_nr_df['study_nr'])]

    if knn_type == 'concat_18_segments_100' or knn_type == 'concat_18_segments_95':
        concat_tracesDF, feature_col_name = get_concat_18_segments(reconstructed_tracesDF)
    elif knn_type == 'gls_remod_class':
        concat_tracesDF, feature_col_name = get_echoParams_data(concat_tracesDF, reconstructed_tracesDF, True)
    elif knn_type == 'echo_remod_class':
        concat_tracesDF, feature_col_name = get_echoParams_data(concat_tracesDF, reconstructed_tracesDF, False)

    else:
        concat_tracesDF, feature_col_name = get_pca_components(concat_tracesDF, df_isImputed_0, filter_by_type,
                                                               isImputed, reconstructed_tracesDF)

    best_param, param_table, test_score_table = best_params_table(concat_tracesDF, feature_col_name,
                                                                  unbalanced_type, class_type)
    return param_table, test_score_table


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
            {'label': 'Concatenate 18 segments', 'value': 'concat_18_segments_100'},
            {'label': 'Concatenate 18 segments PCA: 95% Variance', 'value': 'concat_18_segments_95'},
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
                        dash_bc.Checkbox(
                            id='isDataImputed-id',
                            label='Include imputed data',
                            value=True,
                            style={'marginLeft': '30px', 'width': '70%'}
                        ),
                    ]),
                ], width=2),
                dash_bc.Col([
                    html.Br(),
                    html.Label('Classify using PCA Results by:', style={'fontWeight': 'bold'}),
                    dcc.Dropdown(
                        id='id-concat_Type-dropdown',
                        options=dropdown_options,
                        value='concat_18_segments_100',
                        style={'height': 'Auto', 'width': '100%'}
                    ),
                ],width=4),
                dash_bc.Col([
                    html.Br(),
                    html.Label('Machine Learning Classifiers:', style={'fontWeight': 'bold'}),
                    dcc.Dropdown(
                        id='id-class_type-dropdown',
                        options=[
                            {'label': 'KNN', 'value': 'KNN_class'},
                            {'label': 'Decision Tree', 'value': 'DT_class'},
                            {'label': 'Random Forest', 'value': 'RF_class'},
                            {'label': 'Gradient Boosting', 'value': 'GBM_class'},
                            {'label': 'XGBoost', 'value': 'XGBoost_class'},
                            {'label': 'LightGBM', 'value': 'LightGBM_class'},
                            {'label': 'CatBoost', 'value': 'CatBoost_class'},
                            {'label': 'SVM', 'value': 'SVM_class'},
                            {'label': 'Logistic Regression', 'value': 'LogisticRegression_class'},
                            {'label': 'Naive Bayes', 'value': 'NaiveBayes_class'}
                        ],
                        value='KNN_class',
                        style={'height': 'Auto', 'width': '70%'}
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
                html.Label('Find the best parameters for KNN:',
                           style={'fontWeight': 'bold', 'marginLeft': '20px'}),
                dash_bc.Button('Classify', id='id-best-knn-btn', n_clicks=0,
                               style={'marginLeft': '20px', 'width': '120px'})
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

        ])

    def find_best_class_params(self, data, feature_col_name, unbalanced_type, class_type):
        try:
            if feature_col_name == 'Global Strain %':
                X = data[feature_col_name].values.reshape(-1, 1)
            else:
                X = np.stack(data[feature_col_name].values)
            y = data['Total_Remodeling'].values

            self.parameters = ModelParams()
            self.parameters.get_param_grid(class_type)

            stratified_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            test_scores = []
            best_params_list = []

            for train_idx, test_idx in stratified_kfold.split(X, y):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]

                # Handle imbalanced classes
                self.get_samples(X_train, unbalanced_type, y_train)

                # Train using grid search
                scoring = {'f1_weighted': 'f1_weighted', 'roc_auc': 'roc_auc'}
                grid = GridSearchCV(self.parameters.clf, self.parameters.param_grid, cv=5, scoring='roc_auc',
                                    n_jobs=-1, verbose=1)
                grid.fit(self.X_resampled, self.y_resampled)

                # Evaluate on the test fold
                best_model = grid.best_estimator_
                y_pred = best_model.predict(X_test)

                # Calculate metrics
                self.calculate_metrics(grid, y_pred, y_test, X_test)
                test_scores.append(self.metrics)
                best_params_list.append(grid.best_params_)

            # Average scores across the folds
            avg_scores = self.aggregate_scores(test_scores)

            # Select the best parameters based on test performance
            best_param, best_score = self.select_best_params(best_params_list, avg_scores)

            return best_param, avg_scores, best_score

        finally:
            del X, y
            gc.collect()

    def aggregate_scores(self, test_scores):
        # Aggregate the scores across all folds
        aggregated_scores = {}
        for params, metrics in test_scores:
            if params not in aggregated_scores:
                aggregated_scores[params] = metrics
            else:
                for metric_name in metrics:
                    aggregated_scores[params][metric_name] += metrics[metric_name]

        for params in aggregated_scores:
            for metric_name in aggregated_scores[params]:
                aggregated_scores[params][metric_name] /= len(test_scores)

        return aggregated_scores

    def select_best_params(self, avg_scores):
        best_param = None
        best_score = -float('inf')

        for params, metrics in avg_scores.items():
            score = metrics['AUC']  # Or another metric of choice
            if score > best_score:
                best_score = score
                best_param = params

        return best_param, best_score

    def get_samples(self, X_train, unbalanced_type, y_train):
        oversample = None
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

    def calculate_metrics(self, grid, y_pred, y_test, X_test):
        try:
            self.metrics = None
            self.metrics = {
                'MCC Score': round(matthews_corrcoef(y_test, y_pred), 4),
                'Precision': round(precision_score(y_test, y_pred, average='weighted', zero_division=0), 4),
                'Recall': round(recall_score(y_test, y_pred, average='weighted', zero_division=0), 4),
                'F1 Score': round(f1_score(y_test, y_pred, average='weighted', zero_division=0), 4),
                'Accuracy': round(accuracy_score(y_test, y_pred), 4),
                'AUC': round(roc_auc_score(y_test, y_pred), 4)
            }

            print(f"-------------------------------------------------")
            # print(f"Grid search results: {grid.cv_results_}")
            print(f"Best score (training): {grid.best_score_}")
            print(f"Best params: {grid.best_params_}")
            print(f"-------------------------------------------------")
            print(f"-------------------------------------------------")
            print(f"Test score: {grid.score(X_test, y_test)}")
            for metric_name, metric_value in self.metrics.items():
                print(f"{metric_name}: {metric_value:.4f}")
            print(f"-------------------------------------------------")
        finally:
            gc.collect()
