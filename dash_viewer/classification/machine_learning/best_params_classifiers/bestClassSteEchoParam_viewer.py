import os
import gc
from datetime import datetime

import numpy as np
import pandas as pd

import dash_bootstrap_components as dash_bc
from dash import html, dcc, Output, Input, ctx
from dash import callback, State, dash, dash_table


from dash_viewer.classification.load_prepare_data import get_concat_18_segments, select_data_type, \
    load_queries_table_db,  get_feature_components, get_echoParams_data


from dash_viewer.classification.machine_learning.utils_classifier import get_confusion_matrix_table, \
    best_params_table, convert_echoParam_values, preprocess_data, \
    load_selected_features

gc.enable()

@callback(
    Output('id-table-best-params-steEcho', 'children'),
    Output('id-table-test-scores-steEcho', 'children'),
    Output('id-table-cm-scores-steEcho', 'figure'),
    Output('id-steEcho-results-store', 'data'),
    Input('isDataImputed-id-steEcho', 'value'),
    Input('id-best-knn-btn-steEcho', 'n_clicks'),
    Input('id-data-source-dropdown_bc-steEcho', 'value'),
    Input('id-unbalanced_KnnBest-dropdown-steEcho', 'value'),
    Input('id-slider-steEcho', 'value'),
    State('id-concat_Type-dropdown-steEcho', 'value'),
    State('id-class_type-dropdown-steEcho', 'value'),
)
def update_output(isImputed, best_neighbors, data_source, unbalanced_type, encoding, knn_type, class_type):
    global instance_steEchoClass, param_table
    instance_steEchoClass = BestClassSteEchoParamViewer()
    studyType = 'BL'
    if "id-best-knn-btn-steEcho" == ctx.triggered_id and best_neighbors > 0:
        if data_source:
            data_source_mapping = {
                'Speqle': 'speqle_segmentalStrainInterp_T',
                'GE': 'st_segmentalStrainInterp_T',
                'Philips': 'phi_segmentalStrainInterp_T',
                'GE_Philips': ['phi_segmentalStrainInterp_T', 'st_segmentalStrainInterp_T']
            }
            if data_source in data_source_mapping:
                table_name = data_source_mapping[data_source]
                instance_steEchoClass.table_name = table_name

                print('-----------------------------------------------------------------------------------------------')
                print(f'------------ Find the best parameters for classification in table: {data_source}--------------')
                print(f'unbalanced method: {unbalanced_type}')
                print(f'Experiment type: {knn_type}')
                print(f'Class type: {class_type}')
                print(f'Imputed data: {isImputed}')
                print('-----------------------------------------------------------------------------------------------')

                param_table, test_metrics_table, cm_table = load_traces_df(knn_type, isImputed, unbalanced_type,
                                                                             class_type, data_source, encoding)

                gc.collect()
                return (param_table, test_metrics_table, cm_table,
                        _download_payload(data_source, knn_type, class_type,
                                          unbalanced_type, isImputed, encoding,
                                          param_table, test_metrics_table))
            return dash.no_update, dash.no_update, dash.no_update, dash.no_update
        else:
            table_name = None
            return dash.no_update, dash.no_update, dash.no_update, dash.no_update

    else:
        gc.collect()
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update


def _safe_token(value, fallback='na'):
    token = ''.join(ch if ch.isalnum() or ch in '-_' else '-' for ch in str(value or ''))
    return token.strip('-') or fallback


def _download_payload(data_source, knn_type, class_type, unbalanced_type, isImputed,
                      encoding, param_table, test_metrics_table):
    """Snapshot of the finished run, for the CSV download.

    Plain records in a store, not the rendered DataTables: those are components,
    and parsing them back would break with the next styling change. Never
    raises - an export convenience must not be able to lose the run.
    """
    try:
        return {
            'stamp': datetime.now().strftime('%Y%m%d-%H%M%S'),
            'config': {
                'data_source': data_source,
                'classify_by': knn_type,
                'model': str(class_type).replace('_class', ''),
                'imbalance': unbalanced_type,
                'imputed': bool(isImputed),
                'encoding_dimensions': encoding,
                'run_at': datetime.now().isoformat(timespec='seconds'),
            },
            'best_params': list(getattr(param_table, 'data', None) or []),
            'metrics': list(getattr(test_metrics_table, 'data', None) or []),
        }
    except Exception as exc:
        print(f'[WARN] could not build the CSV download payload: {exc}')
        return None


def _build_download_frame(payload):
    """One tidy long frame: section, field, value.

    One file rather than one per table - a metrics row is only reusable next to
    the hyperparameters and the setup that produced it. Long rather than wide
    because the sections share no column names.
    """
    rows = []
    sections = (
        ('run_config', [payload.get('config') or {}]),
        ('metrics', payload.get('metrics') or []),
        ('best_params', payload.get('best_params') or []),
    )
    for section, records in sections:
        for record in records:
            for field, value in (record or {}).items():
                rows.append({'section': section, 'field': field, 'value': value})
    return pd.DataFrame(rows, columns=['section', 'field', 'value'])


@callback(
    Output('id-steEcho-download', 'data'),
    Output('id-steEcho-download-status', 'children'),
    Input('id-steEcho-download-btn', 'n_clicks'),
    State('id-steEcho-results-store', 'data'),
    prevent_initial_call=True,
)
def download_steEcho_csv(n_clicks, payload):
    if not n_clicks or not payload:
        return dash.no_update, dash_bc.Alert(
            'Nothing to download yet - press Classify first.',
            color='secondary', className='mb-0')

    # A malformed stored result must produce a message, not a dead callback.
    try:
        frame = _build_download_frame(payload)
    except Exception as exc:
        return dash.no_update, dash_bc.Alert(
            f'Could not build the CSV from the last run: {exc}',
            color='danger', className='mb-0')

    if frame.empty:
        return dash.no_update, dash_bc.Alert(
            'The last run produced no rows to export.',
            color='warning', className='mb-0')

    config = payload.get('config') or {}
    stamp = payload.get('stamp') or datetime.now().strftime('%Y%m%d-%H%M%S')
    filename = (f"classification_steEcho_{_safe_token(stamp)}_"
                f"{_safe_token(config.get('model'))}_"
                f"{_safe_token(config.get('classify_by'))}.csv")
    return dcc.send_data_frame(frame.to_csv, filename, index=False), ''


def load_traces_df(dataOrig_type, isImputed, unbalanced_type, class_type, data_source, encoding):
    imp_concat_tracesDF, orig_concat_tracesDF,echoParam_feat = None, None, None
    filter_by_type = select_data_type(dataOrig_type, encoding)

    tracesDF = load_queries_table_db('pca_results_view', 'data_source', 'in', [data_source])
    filtered_tracesDF = tracesDF[tracesDF['reconstructed_type'].str.contains(filter_by_type, na=False)]

    echoParam_tracesDF, feature_col_name = get_echoParams_data(filtered_tracesDF, data_source, None)
    echoParam_tracesDF = convert_echoParam_values(echoParam_tracesDF)

    if tracesDF is not None and echoParam_tracesDF is not None:

        # fit_preprocessing=False keeps missing values and raw scales: the KNN
        # imputer and the scaler are refit inside each CV fold by
        # build_estimator_pipeline() rather than once over every patient here.
        X_orig, y,patient_study_info, scaler = preprocess_data(
            echoParam_tracesDF, True, fit_preprocessing=False)
        featuresDF, unique_features_list = load_selected_features()
        X_filtered = X_orig.loc[:, [col for col in X_orig.columns if col in unique_features_list]]

        echoParam_feat = X_filtered.copy()
        echoParam_feat.insert(0, 'study_nr', patient_study_info['study_nr'])

    original_tracesDF = tracesDF[tracesDF['reconstructed_type'] == f'{filter_by_type}-0']

    if isImputed:
        imputed_tracesDF = tracesDF[tracesDF['reconstructed_type'] == f'{filter_by_type}-1']
    else:
        imputed_tracesDF = None

    if dataOrig_type == 'reconstructed_traces_vac':
        orig_concat_tracesDF, orig_feature_col_name = get_concat_18_segments(original_tracesDF)
        if imputed_tracesDF is not None:
            imp_concat_tracesDF, imp_feature_col_name = get_concat_18_segments(imputed_tracesDF)

    elif dataOrig_type == '18_segments_vac' or dataOrig_type == 'per_segment_vac':
        table_name = 'calculated_dr_encoded_data_T'
        feature_col_name = 'encoded_data'
        orig_concat_tracesDF, orig_feature_col_name = get_feature_components(original_tracesDF, data_source,
                                                                             filter_by_type, feature_col_name,
                                                                             table_name,False)
        if imputed_tracesDF is not None:
            imp_concat_tracesDF, imp_feature_col_name = get_feature_components(imputed_tracesDF, data_source,
                                                                               filter_by_type, feature_col_name,
                                                                               table_name,True)

    orig_merged_df = orig_concat_tracesDF.merge(echoParam_feat, on='study_nr', how='left', suffixes=('_orig', '_echo'))
    imp_merged_df = imp_concat_tracesDF.merge(echoParam_feat, on='study_nr', how='left', suffixes=('_orig', '_echo'))
    features = None

    (best_param, param_table, test_metrics_table, cm, cm_table_fig,
     shap_values, shap_input_df, important_features, shap_df) = best_params_table(
        orig_merged_df, orig_feature_col_name, imp_merged_df, imp_feature_col_name,
        unbalanced_type, class_type, features)
    return param_table, test_metrics_table, cm_table_fig




class BestClassSteEchoParamViewer:

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
            {'label': '18 Segments Encoded', 'value': '18_segments_vac'},
            {'label': 'Per Segments Encoded', 'value': 'per_segment_vac'},
            {'label': 'Echo-parameters', 'value': 'echo-params_vac'},
            {'label': 'GLS', 'value': 'gls_params_vac'}
        ]

        return html.Div([
            dash_bc.Row([
                dash_bc.Col([
                    html.Br(),
                    dash_bc.Row([
                        html.Label('Select Data Source:', style={'fontWeight': 'bold', 'marginLeft': '20px'}),
                        dcc.Dropdown(
                            id='id-data-source-dropdown_bc-steEcho',
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
                            id='isDataImputed-id-steEcho',
                            label='Include imputed data',
                            value=False,
                            style={'marginLeft': '30px', 'width': '90%'}
                        ),
                    ]),
                ], width=2),

                dash_bc.Col([
                    html.Br(),
                    html.Label('Classify by:',
                               style={'marginLeft': '20px', 'fontWeight': 'bold'}),
                    dcc.Dropdown(
                        id='id-concat_Type-dropdown-steEcho',
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
                        id='id-class_type-dropdown-steEcho',
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
                        id='id-unbalanced_KnnBest-dropdown-steEcho',
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
                        id='id-slider-steEcho',
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
                dash_bc.Button('Classify', id='id-best-knn-btn-steEcho', n_clicks=0,
                               style={'marginLeft': '30px', 'width': '120px'})
            ]),

            html.Br(),
            html.Br(),

            dash_bc.Row([
                dash_bc.Col(dash_bc.Button('Download CSV', id='id-steEcho-download-btn',
                                           n_clicks=0, color='secondary', outline=True,
                                           style={'marginLeft': '20px', 'width': '140px'}),
                            width=2),
                dash_bc.Col(html.Div(id='id-steEcho-download-status'), width=6),
            ]),
            dcc.Store(id='id-steEcho-results-store', data=None),
            dcc.Download(id='id-steEcho-download'),
            html.Br(),

            dash_bc.Row([
                html.Div(id='id-table-best-params-steEcho',
                         style={'marginLeft': '20px'})
            ], justify="center", style={'width': '50%', 'margin': 'auto'}),
            html.Br(),
            dash_bc.Row([
                html.Div(id='id-table-test-scores-steEcho',
                         style={'marginLeft': '20px'})
            ], justify="center", style={'width': '50%', 'margin': 'auto'}),
            html.Br(),
            html.Br(),
            dash_bc.Row([
                dcc.Graph(id='id-table-cm-scores-steEcho')
            ], justify="center", style={'width': '70%', 'margin': 'auto'}),
        ])