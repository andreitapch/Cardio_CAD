import os
import gc
import numpy as np
import pandas as pd

import dash_bootstrap_components as dash_bc
from dash import html, dcc, Output, Input, ctx
from dash import callback, State, dash, dash_table

from dash_viewer.classification.load_prepare_data import get_concat_18_segments, select_data_type, \
    load_segment_table_db,  get_feature_components, get_echoParams_data


from dash_viewer.classification.machine_learning.utils_classifier import get_confusion_matrix_table, find_best_class_params

gc.enable()


def safe_eval(val):
    try:
        return eval(val)
    except Exception:
        return val


@callback(
    Output('id-table-best-params-vac', 'children'),
    Output('id-table-test-scores-vac', 'children'),
    Output('id-table-cm-scores-vac', 'figure'),
    Input('isDataImputed-id-vac', 'value'),
    Input('id-best-knn-btn-vac', 'n_clicks'),
    Input('id-data-source-dropdown_bc-vac', 'value'),
    Input('id-unbalanced_KnnBest-dropdown-vac', 'value'),
    Input('id-slider-vac', 'value'),
    State('id-concat_Type-dropdown-vac', 'value'),
    State('id-class_type-dropdown-vac', 'value'),
)
def update_output(isImputed, best_neighbors, data_source, unbalanced_type, encoding, knn_type, class_type):
    global instance_clf, param_table
    instance_clf = BestParamClassVacViewer()
    studyType = 'BL'
    if "id-best-knn-btn-vac" == ctx.triggered_id and best_neighbors > 0:
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
                                                                             class_type, data_source, encoding)

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
     test_metrics, cm) = find_best_class_params(orig_concat_tracesDF, orig_feature_col_name,
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
    cm_table_fig = get_confusion_matrix_table(cm)

    return best_param, best_param_table, test_metrics_table, cm_table_fig


def load_data_traces(knn_type, isImputed, unbalanced_type, class_type, data_source, encoding):
    imp_concat_tracesDF = None
    imp_feature_col_name = None
    filter_by_type = select_data_type(knn_type, encoding)

    if data_source == 'GE_Philips':
        data_label = ['GE', 'Philips']
    elif data_source == 'GE':
        data_label = ['GE']
    elif data_source == 'Philips':
        data_label = ['Philips']
    else:
        data_label = ['Speqle']

    tracesDF = load_segment_table_db('pca_results_view',
                                     'data_source', 'in', [data_source])

    original_tracesDF = tracesDF[tracesDF['reconstructed_type'] == f'{filter_by_type}-0']

    if isImputed:
        imputed_tracesDF = tracesDF[tracesDF['reconstructed_type'] == f'{filter_by_type}-1']
    else:
        imputed_tracesDF = None

    if knn_type == 'reconstructed_traces_vac':
        orig_concat_tracesDF, orig_feature_col_name = get_concat_18_segments(original_tracesDF)
        if imputed_tracesDF is not None:
            imp_concat_tracesDF, imp_feature_col_name = get_concat_18_segments(imputed_tracesDF)

    elif knn_type == 'gls_remod_class_vac':
        orig_concat_tracesDF, orig_feature_col_name = get_echoParams_data(original_tracesDF, data_source, 'global_strain')
        if imputed_tracesDF is not None:
            imp_concat_tracesDF, imp_feature_col_name = get_echoParams_data(imputed_tracesDF, data_source, 'global_strain')

    elif knn_type == 'echo_remod_class':
        orig_concat_tracesDF, orig_feature_col_name = get_echoParams_data(original_tracesDF, data_source, None)
        if imputed_tracesDF is not None:
            imp_concat_tracesDF, imp_feature_col_name = get_echoParams_data(imputed_tracesDF, data_source, None)

    elif knn_type == '18_segments_vac' or knn_type == 'per_segment_vac':
        table_name = 'calculated_dr_encoded_data_T'
        feature_col_name = 'encoded_data'
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


class BestParamClassVacViewer:

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
            {'label': 'Reconstructed traces', 'value': 'reconstructed_traces_vac'},
            {'label': '18 Segments Autoencoder', 'value': '18_segments_vac'},
            {'label': 'Per Segments Autoencoder', 'value': 'per_segment_vac'},
            {'label': 'Echo-parameters', 'value': 'echo-params_vac'},
            {'label': 'GLS', 'value': 'gls_remod_class_vac'}
        ]

        return html.Div([
            dash_bc.Row([
                dash_bc.Col([
                    html.Br(),
                    dash_bc.Row([
                        html.Label('Select Data Source:', style={'fontWeight': 'bold', 'marginLeft': '20px'}),
                        dcc.Dropdown(
                            id='id-data-source-dropdown_bc-vac',
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
                            id='isDataImputed-id-vac',
                            label='Include imputed data',
                            value=True,
                            style={'marginLeft': '30px', 'width': '90%'}
                        ),
                    ]),
                ], width=2),

                dash_bc.Col([
                    html.Br(),
                    html.Label('Classify by:',
                               style={'marginLeft': '20px', 'fontWeight': 'bold'}),
                    dcc.Dropdown(
                        id='id-concat_Type-dropdown-vac',
                        options=dropdown_options,
                        value='reconstructed_traces_vac',
                        style={'marginLeft': '10px', 'height': 'Auto', 'width': '100%'}
                    ),
                ], width=2),

                dash_bc.Col([
                    html.Br(),
                    html.Label('Machine Learning Classifiers:',
                               style={'fontWeight': 'bold'}),
                    dcc.Dropdown(
                        id='id-class_type-dropdown-vac',
                        options=[
                            {'label': 'KNN', 'value': 'KNN_class'},
                            {'label': 'Decision Tree', 'value': 'DT_class'},
                            {'label': 'Random Forest', 'value': 'RF_class'},
                            {'label': 'Gradient Boosting', 'value': 'GBM_class'},
                            {'label': 'XGBoost', 'value': 'XGBoost_class'},
                            {'label': 'LightGBM', 'value': 'LightGBM_class'},
                            {'label': 'SVM', 'value': 'SVM_class'},
                            {'label': 'MLP', 'value': 'MLP_class'},
                        ],
                        value='KNN_class',
                        style={'height': 'Auto', 'width': '100%'}
                    ),
                ], width=3),

                dash_bc.Col([
                    html.Br(),
                    html.Label('Unbalanced:', style={'fontWeight': 'bold'}),
                    dcc.Dropdown(
                        id='id-unbalanced_KnnBest-dropdown-vac',
                        options=[
                            {'label': 'None', 'value': 'None'},
                            {'label': 'SMOTE', 'value': 'SMOTE'},
                            {'label': 'ADASYN', 'value': 'ADASYN'},
                            {'label': 'SMOTEENN', 'value': 'SMOTEENN'},
                            {'label': 'SMOTETomek', 'value': 'SMOTETomek'}
                        ],
                        value='None',
                        style={'height': 'Auto', 'width': '100%'}
                    ),
                ], width=2),

                dash_bc.Col([
                    html.Br(),
                    html.P("Encoding Dimensions:"),
                    dcc.Slider(
                        id='id-slider-vac',
                        min=2,
                        max=100,
                        value=32,
                        step=None,
                        marks={2: '2', 4: '4', 8: '8', 16: '16', 32: '32', 64: '64', 100: '100'},
                    )
                ], width=3),

            ]),
            dash_bc.Row([
                html.Br(),
                html.Label('Find the best parameters:',
                           style={'fontWeight': 'bold', 'marginLeft': '20px'}),
                dash_bc.Button('Classify', id='id-best-knn-btn-vac', n_clicks=0,
                               style={'marginLeft': '30px', 'width': '120px'})
            ]),

            html.Br(),
            html.Br(),

            dash_bc.Row([
                html.Div(id='id-table-best-params-vac',
                         style={'marginLeft': '20px'})
            ], justify="center", style={'width': '50%', 'margin': 'auto'}),
            html.Br(),
            dash_bc.Row([
                html.Div(id='id-table-test-scores-vac',
                         style={'marginLeft': '20px'})
            ], justify="center", style={'width': '50%', 'margin': 'auto'}),
            html.Br(),
            html.Br(),
            dash_bc.Row([
                dcc.Graph(id='id-table-cm-scores-vac')
            ], justify="center", style={'width': '70%', 'margin': 'auto'}),
        ])