import gc
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import dash_bootstrap_components as dash_bc
from dash import html, dcc, Output, Input, ctx
from dash import callback, State, dash, dash_table



from dash_viewer.classification.machine_learning.utils.utils_clinical_relevance import split_important_features, \
    plot_pca_loading_segments
from dash_viewer.classification.machine_learning.utils_classifier import (
    best_params_table, default_scaling_for, shap_unavailable_reason)
from dash_viewer.classification.load_prepare_data import load_data_traces, get_component_options, load_loadings
from dash_viewer.classification.machine_learning.utils.shap_explainer import shap_beeswarm_base64, save_shap_beeswarm_svg
from large_language_models.MedicalLLMInterpreter import MedicalLLMInterpreter
from pattern_recognition.utils.parameters_dics import openai_key

gc.enable()


def safe_eval(val):
    try:
        return eval(val)
    except Exception:
        return val


@callback(
    Output('id-number_components-dropdown', 'options'),
    Output('id-number_components-dropdown', 'value'),
    Input('isDataImputed-id', 'value'),
    Input('id-concat_Type-dropdown', 'value'),
    Input('id-classify-dropdown', 'value'),
)
def fill_Components_dropdown(isImputed, input_type, class_input):

    if class_input in ['autoencoder_class']:
        if input_type is None:
            return [], None
        else:
            reconstructed_type = f'AutoencoderDM-{input_type}'
        options_list = get_component_options(reconstructed_type, isImputed, 'calculated_dr_encoded_data_T', 'like',
                                             'encoding')
    elif class_input in ['reconstructed_traces','components','features_class', 'gls_class']:
        options_list = get_component_options([input_type], isImputed, 'calculated_pca_params_T', 'in', 'n_components_')
    else:
        return [], None
    default_value = options_list[0]['value'] if options_list else None
    return options_list, default_value


@callback(
    Output('id-table-best-params', 'children'),
    Output('id-table-test-scores', 'children'),
    Output('id-table-cm-scores', 'figure'),
    Output('id-beeswarm-plot', 'src'),
    Output('id-shap-status', 'children'),
    Output('id-run-summary', 'children'),
    Output('dropdown-pc-selector', 'options'),
    Output('dropdown-pc-selector', 'value'),
    Output('pca-loadings-store', 'data'),
    Output('clinical_feats-store', 'data'),
    Input('isDataImputed-id', 'value'),  #isImputed
    Input('id-best-knn-btn', 'n_clicks'),  #btn_neighbors
    Input('id-data-source-dropdown_bc', 'value'),  #data_source
    Input('id-unbalanced_KnnBest-dropdown', 'value'),  #unbalanced_type
    Input('addGLS-id', 'value'),  #addGLS
    Input('addFeatEcho-id', 'value'),  #addFeatEcho
    State('id-concat_Type-dropdown', 'value'),  #input_type
    State('id-class_type-dropdown', 'value'),  #class_type
    Input('id-number_components-dropdown', 'value'),  #n_components
    Input('id-classify-dropdown', 'value'),  #classify_type
    Input('id-scaling-dropdown', 'value')  #scaling_mode
)
def update_output(isImputed, btn_neighbors, data_source, unbalanced_type,
                  addGLS, addFeatEcho, input_type, class_type, n_components, class_input,
                  scaling_mode):
    global instance_clf, param_table
    pc_number = None
    instance_clf = BestParamClassViewer()
    studyType = 'BL'

    # Ten outputs; keep every early exit the same width.
    no_change = (dash.no_update,) * 10

    if "id-best-knn-btn" != ctx.triggered_id or not data_source:
        return no_change

    data_source_mapping = {
        'Speqle': 'speqle_segmentalStrainInterp_T',
        'GE': 'st_segmentalStrainInterp_T',
        'Philips': 'phi_segmentalStrainInterp_T',
        'GE_Philips': ['phi_segmentalStrainInterp_T', 'st_segmentalStrainInterp_T']
    }
    if data_source not in data_source_mapping:
        return no_change

    instance_clf.table_name = data_source_mapping[data_source]

    if scaling_mode == 'on':
        scale_features = True
    elif scaling_mode == 'off':
        scale_features = False
    else:
        scale_features = default_scaling_for(class_input)

    print('-----------------------------------------------------------------------------------------------')
    print(f'------------ Find the best parameters for classification in table: {data_source}--------------')
    print(f'Balanced Tech   : {unbalanced_type}')
    print(f'Classify by     : {class_input}')
    print(f'Use             : {input_type}')
    print(f'Components      : {n_components}')
    print(f'Is Imputed      : {isImputed}')
    print(f'ML Class        : {class_type}')
    print(f'With GLS        : {addGLS}')
    print(f'With Echo Params: {addFeatEcho}')
    print(f'Standardise     : {scale_features} ({scaling_mode})')
    print('-----------------------------------------------------------------------------------------------')

    component = n_components

    orig_concat_tracesDF, orig_feature_col_name, imp_concat_tracesDF, imp_feature_col_name, features = \
        load_data_traces(input_type, isImputed, unbalanced_type, class_type, data_source, component, addGLS,
                         addFeatEcho, pc_number, class_input)

    (best_param, param_table, test_metrics_table,
     cm, cm_table_fig, shap_values, shap_input_df,
     important_features, shap_df) = best_params_table(orig_concat_tracesDF,
                                                      orig_feature_col_name,
                                                      imp_concat_tracesDF,
                                                      imp_feature_col_name,
                                                      unbalanced_type, class_type, features,
                                                      scale_features=scale_features,
                                                      class_input=class_input)

    if best_param is None:
        gc.collect()
        return (
            html.Div(), html.Div(), go.Figure(), None,
            dash_bc.Alert('No SHAP: the run produced no result.', color='secondary'),
            dash_bc.Alert('The classification run failed. See the console log for the traceback.',
                          color='danger'),
            [], None, dash.no_update, dash.no_update,
        )

    dash_beeswarm = shap_beeswarm_base64(shap_values, class_input)

    print('---------------------------------------- Results ----------------------------------------------')
    test_metrics_df = pd.DataFrame(test_metrics_table.data)
    print(test_metrics_df.to_string(index=False))

    print("\nConfusion Matrix:")
    print(pd.DataFrame(cm, columns=["Predicted 0", "Predicted 1"], index=["Actual 0", "Actual 1"]))

    # Say why the explanation panel is empty rather than leaving it blank.
    reason = shap_unavailable_reason(class_type, features)
    if reason:
        shap_status = dash_bc.Alert(reason, color='secondary', className='mb-2')
    elif not important_features:
        shap_status = dash_bc.Alert(
            'SHAP ran but produced no ranking. See the console log for the reason.',
            color='warning', className='mb-2')
    else:
        n_total = len(shap_input_df.columns) if shap_input_df is not None else len(important_features)
        shap_status = dash_bc.Alert(
            f'SHAP pooled across all held-out folds - every patient explained once, by the '
            f'model that did not train on them. Showing the top {len(important_features)} '
            f'of {n_total} features by mean |SHAP|.',
            color='info', className='mb-2')

    run_summary = _build_run_summary(data_source, class_input, input_type, class_type,
                                     unbalanced_type, n_components, isImputed,
                                     addGLS, addFeatEcho, scale_features, scaling_mode, cm)

    if dash_beeswarm and important_features:
        save_shap_beeswarm_svg(shap_values, class_input, "shap_beeswarm.svg")
        pca_feats, clinical_feats = split_important_features(important_features)
        pca_loadings = load_loadings(data_source, input_type, isImputed, n_components,
                                     'calculated_pca_params_T')
        print("\nTop Important Features:")
        print(important_features)

        if isinstance(pca_feats, list) and len(pca_feats) > 0:
            dropdown_options = [
                {"label": feat.replace("pca_results_", "PC "), "value": int(feat.split("_")[-1])}
                for feat in pca_feats if "pca_results_" in feat
            ]
            default_pc_value = int(pca_feats[0].split("_")[-1])
        else:
            dropdown_options = []
            default_pc_value = None
    else:
        dropdown_options = dash.no_update
        pca_loadings = dash.no_update
        pca_feats = dash.no_update
        clinical_feats = dash.no_update
        default_pc_value = None

    print('-----------------------------------------------------------------------------------------------')

    gc.collect()
    return (
        param_table,
        test_metrics_table,
        cm_table_fig if cm_table_fig is not None else dash.no_update,
        dash_beeswarm,
        shap_status,
        run_summary,
        dropdown_options,
        default_pc_value,
        pca_loadings,
        clinical_feats
    )


def _build_run_summary(data_source, class_input, input_type, class_type, unbalanced_type,
                       n_components, isImputed, addGLS, addFeatEcho, scale_features,
                       scaling_mode, cm):
    """Badges describing the run, plus a note on how to read the metrics table.

    The distinction matters: the inner PR-AUC is what hyperparameters were
    chosen on, and is optimistically biased by construction. Everything else
    comes from outer folds that took no part in selection.
    """
    def badge(label, value, colour='light'):
        return dash_bc.Badge(f'{label}: {value}', color=colour, text_color='dark',
                             className='me-2 mb-1')

    scaling_label = f"{'on' if scale_features else 'off'}"
    if scaling_mode == 'auto':
        scaling_label += ' (auto)'

    badges = [
        badge('Source', data_source),
        badge('Target', class_input),
        badge('Input', input_type),
        badge('Model', class_type.replace('_class', '')),
        badge('Imbalance', unbalanced_type),
        badge('Components', n_components),
        badge('Imputed', 'yes' if isImputed else 'no'),
        badge('Standardised', scaling_label),
    ]
    if addGLS:
        badges.append(badge('+GLS', 'yes'))
    if addFeatEcho:
        badges.append(badge('+Echo features', 'yes'))

    n_evaluated = int(np.sum(cm)) if cm is not None else 0

    return html.Div([
        html.Div(badges, className='mb-2'),
        html.Small([
            html.B('How to read this: '),
            f'metrics are pooled over held-out outer folds ({n_evaluated} patients, each '
            f'predicted once by a model that did not train on them). ',
            html.B('Inner PR-AUC (selection)'),
            ' in the parameters table is what the hyperparameter search scored - it is '
            'optimistically biased and is not a performance estimate.'
        ], className='text-muted'),
    ])


@callback(
    Output('pca-loading-plot', 'figure'),
    Output('id-llm-class-div', 'children'),
    Output('data_table-class-div', 'children'),
    Input('dropdown-pc-selector', 'value'),
    State('pca-loadings-store', 'data'),
    Input('id-number_components-dropdown', 'value'),
    Input('clinical_feats-store', 'data'),
)
def update_pca_loading_plot(component_idx, loadings_data, n_components, selected_features):
    if component_idx is None or loadings_data is None:
        return go.Figure(), html.Div("No component selected."), html.Div()

    if isinstance(n_components, list):
        n_components = n_components[0] if n_components else None

    if n_components is None:
        return go.Figure(), html.Div("Invalid number of components."), html.Div()

    n_segments = 18

    fig, df_summary = plot_pca_loading_segments(
        loadings_data[component_idx],
        component_idx,
        n_segments,
        int(n_components)
    )

    table = dash_table.DataTable(
        columns=[{"name": col, "id": col} for col in df_summary.columns],
        data=df_summary.to_dict('records'),
        style_table={"overflowX": "auto"},
        style_cell={"textAlign": "center", "padding": "6px", "fontSize": "14px"},
        style_header={"fontWeight": "bold", "backgroundColor": "#f9f9f9"},
        page_size=10,
    )

    return fig, dash.no_update, table


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
        return html.Div([
            dash_bc.Row([
                dash_bc.Col([
                    html.Br(),
                    dash_bc.Row([
                        html.Label('Select Data Source:', style={'fontWeight': 'bold', 'marginLeft': '20px'}),
                        dcc.Dropdown(
                            id='id-data-source-dropdown_bc',
                            options=[
                                # {'label': 'Speqle', 'value': 'Speqle'},
                                # {'label': 'GE', 'value': 'GE'},
                                # {'label': 'Philips', 'value': 'Philips'},
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
                    html.Br(),
                    html.Label('Classify by:', style={'marginLeft': '35px', 'fontWeight': 'bold'}),
                    dcc.Dropdown(
                        id='id-classify-dropdown',
                        options=[
                            {'label': 'Reconstructed traces', 'value': 'reconstructed_traces'},
                            {'label': 'Components', 'value': 'components'},
                            {'label': 'Features', 'value': 'features_class'},
                            {'label': 'GLS', 'value': 'gls_class'},
                            # {'label': 'Autoencoder Reconstructed', 'value': 'autoencoder_reconstructed'},
                            # {'label': 'Autoencoder Encodings', 'value': 'autoencoder_encodings'},
                        ],
                        value=None,
                        style={'marginLeft': '20px', 'height': 'Auto', 'width': '180'}
                    ),
                ], width=3),
                dash_bc.Col([
                    html.Br(),
                    dash_bc.Row([
                        html.Label('Use:', style={'marginLeft': '35px', 'fontWeight': 'bold'}),
                        dcc.Dropdown(
                            id='id-concat_Type-dropdown',
                            options=[
                                {'label': 'Full traces', 'value': 'full_traces'},
                                {'label': 'Segments', 'value': 'segment'},
                            ],
                            value=None,
                            style={'marginLeft': '20px', 'height': 'Auto', 'width': '150px'}
                        ),
                    ]),
                    html.Br(),
                    html.Br(),
                    dash_bc.Row([
                        html.Label('Number of components:', style={'marginLeft': '35px', 'fontWeight': 'bold'}),
                        dcc.Dropdown(
                            id='id-number_components-dropdown',
                            options=[],
                            value=None,
                            style={'marginLeft': '20px', 'height': 'Auto', 'width': '150px'}
                        ),
                    ]),
                ], width=2),
                dash_bc.Col([
                    html.Br(),
                    html.Label('Machine Learning Classifiers:', style={'fontWeight': 'bold'}),
                    dcc.Dropdown(
                        id='id-class_type-dropdown',
                        options=[
                            # {'label': 'KNN', 'value': 'KNN_class'},
                            {'label': 'Decision Tree', 'value': 'DT_class'},
                            {'label': 'Random Forest', 'value': 'RF_class'},
                            {'label': 'Gradient Boosting', 'value': 'GBM_class'},
                            {'label': 'XGBoost', 'value': 'XGBoost_class'},
                            {'label': 'Logistic Regression', 'value': 'LogisticRegression_class'},
                        ],
                        value='RF_class',
                        style={'height': 'Auto', 'width': '100%'}
                    ),
                    html.Br(),
                    html.Br(),
                    dash_bc.Row([
                        dash_bc.Checkbox(
                            id='addGLS-id',
                            label='Add GLS',
                            value=False,
                            style={'marginLeft': '20px', 'width': '70%'}
                        ),
                    ]),
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
                            {'label': 'Balanced', 'value': 'Balanced'}
                        ],
                        value='None',
                        style={'height': 'Auto', 'width': '70%'}
                    ),
                    html.Br(),
                    dash_bc.Row([
                        dash_bc.Checkbox(
                            id='addFeatEcho-id',
                            label='Add selected features',
                            value=False,
                            style={'marginLeft': '20px', 'width': '70%'}
                        ),
                    ]),
                    html.Br(),
                    html.Label('Standardise features:', style={'fontWeight': 'bold'}),
                    dcc.Dropdown(
                        id='id-scaling-dropdown',
                        options=[
                            {'label': 'Auto (off for raw traces)', 'value': 'auto'},
                            {'label': 'Always', 'value': 'on'},
                            {'label': 'Never', 'value': 'off'},
                        ],
                        value='auto',
                        style={'height': 'Auto', 'width': '70%'}
                    ),
                    html.Small(
                        'Raw strain traces are time points of one curve; z-scoring each '
                        'point separately changes KNN distances and the SHAP value axis.',
                        className='text-muted'),
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

            dcc.Loading(
                id='id-classify-loading',
                type='default',
                children=html.Div([
                    dash_bc.Row([
                        html.Div(id='id-run-summary', style={'marginLeft': '20px'})
                    ], style={'width': '90%', 'margin': 'auto'}),
                    html.Br(),

                    html.H5('Selected hyperparameters',
                            style={'marginLeft': '20px', 'fontWeight': 'bold'}),
                    dash_bc.Row([
                        html.Div(id='id-table-best-params',
                                 style={'marginLeft': '20px'})
                    ], justify="center", style={'width': '90%', 'margin': 'auto'}),
                    html.Br(),

                    html.H5('Held-out performance',
                            style={'marginLeft': '20px', 'fontWeight': 'bold'}),
                    dash_bc.Row([
                        html.Div(id='id-table-test-scores',
                                 style={'marginLeft': '20px'})
                    ], justify="center", style={'width': '90%', 'margin': 'auto'}),
                    html.Br(),

                    html.H5('Confusion matrix',
                            style={'marginLeft': '20px', 'fontWeight': 'bold'}),
                    dash_bc.Row([
                        dcc.Graph(id='id-table-cm-scores')
                    ], justify="center", style={'width': '70%', 'margin': 'auto'}),
                    html.Br(),

                    html.H5('Feature influence (SHAP)',
                            style={'marginLeft': '20px', 'fontWeight': 'bold'}),
                    dash_bc.Row([
                        html.Div(id='id-shap-status', style={'marginLeft': '20px'})
                    ], style={'width': '90%', 'margin': 'auto'}),
                    dash_bc.Row([
                        html.Img(id='id-beeswarm-plot', style={'maxWidth': '100%'})
                    ], justify="center", style={'width': '70%', 'margin': 'auto'}),
                ])
            ),

            dash_bc.Row([
                html.Br(),
                dash_bc.Row([
                    html.Label('Select PCA Component:', style={'marginLeft': '20px', 'fontWeight': 'bold'}),
                    dcc.Dropdown(
                        id='dropdown-pc-selector',
                        options=[],
                        value=None,
                        style={'marginLeft': '20px', 'width': '200px'}
                    )
                ]),
                html.Br(),

                dash_bc.Row([
                    html.Div(id='data_table-class-div', style={'marginLeft': '20px', 'marginRight': '20px'})
                ], justify="center", style={'width': '80%', 'margin': 'auto'}),

                dcc.Store(id='pca-loadings-store', data=None),
                html.Br(),
                dash_bc.Row([
                    dcc.Graph(id='pca-loading-plot')
                ], justify="center", style={'width': '70%', 'margin': 'auto'}),

                dash_bc.Row([
                    html.Div(id='id-llm-class-div',
                             style={'marginLeft': '10px', 'marginRight': '10px'}),
                ]),

                dcc.Store(id='clinical_feats-store', data=None),

            ])
        ])
