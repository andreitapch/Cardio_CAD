# import numpy as np
# import pandas as pd
#
# import plotly.graph_objects as go
# import dash_bootstrap_components as dbc
#
# from dash import html, dcc, Output, Input, callback, State, ctx, dash
# import xgboost as xgb
#
# from dash_viewer.classification.load_prepare_data import load_imputed_data
# from dash_viewer.classification.test_classifiers import create_data_table_results, create_cm, count_data, \
#     create_scatter_plot, extract_sex_data, classify_per_fold, load_data_traces
#
#
# @callback(
#     Output('id-metrics-output_xgb', 'children'),
#     Output('id-cm-plot_xgb', 'figure'),
#     Output('id-data-count-plot_xgb', 'figure'),
#     Output('id-class-feature-plot_xgb', 'figure'),
#     Output('id-cm-female_xgb', 'figure'),
#     Output('id-cm-male_xgb', 'figure'),
#     Output('id-metrics-male_xgb', 'children'),
#     Output('id-metrics-female_xgb', 'children'),
#     Input('id-button_xgb', 'n_clicks'),
#     Input('isImputed-id_xgb', 'value'),
#     Input('id-data-source-dropdown_bc-xgb', 'value'),
#     State('id-test_type-dropdown_xgb', 'value'),
#     State('id-unbalanced_class-dropdown_xgb', 'value'),
#     State('textarea-n_estimators-id', 'value'),
#     State('textarea-max_depth-id', 'value'),
#     State('textarea-learning_rate-id', 'value'),
#     State('textarea-gamma-id', 'value'),
#     State('textarea-subsample-id', 'value'),
#     State('textarea-colsample_bytree-id', 'value')
# )
# def update_output(btn_click, isImputed, data_source, test_type, unbalanced_type, n_estimators, max_depth, learning_rate, gamma,
#                   subsample, colsample_bytree):
#     global instance_steEchoClass, param_table
#     studyType = 'BL'
#     instance_steEchoClass = XGBClassViewer()
#
#     if "id-button_xgb" == ctx.triggered_id and btn_click > 0:
#         if data_source:
#             data_source_mapping = {
#                 'Speqle': 'speqle_segmentalStrainInterp_T',
#                 'GE': 'st_segmentalStrainInterp_T',
#                 'Philips': 'phi_segmentalStrainInterp_T',
#                 'GE_Philips': ['phi_segmentalStrainInterp_T', 'st_segmentalStrainInterp_T']
#             }
#             if data_source in data_source_mapping:
#                 table_name = data_source_mapping[data_source]
#                 instance_steEchoClass.table_name = table_name
#
#         tracesDF, df_isImputed_1, df_isImputed_0 = load_imputed_data(studyType, isImputed, data_source)
#
#         (concat_tracesDF, feature_col_name, males_df,
#          females_df, over_50_df, under_50_df, reconstructed_tracesDF) = load_data_traces(test_type, isImputed,
#                                                                                          df_isImputed_0)
#
#         results, data_count_fig, scatter_fig, \
#             cm_female, cm_male, metrics_male, metrics_female = classify_data_loader(unbalanced_type,
#                                                                                     reconstructed_tracesDF, test_type,
#                                                                                     concat_tracesDF, feature_col_name,
#                                                                                     males_df, females_df, over_50_df,
#                                                                                     under_50_df,
#                                                                                     n_estimators, max_depth,
#                                                                                     learning_rate, gamma, subsample,
#                                                                                     colsample_bytree)
#
#         confusion_matrices, metrics_table = create_data_table_results(results, False)
#
#         cm_fig = create_cm(confusion_matrices, instance_steEchoClass.clf, False)
#         table_male = create_data_table_results(metrics_male, True)
#         table_female = create_data_table_results(metrics_female, True)
#
#         return metrics_table, cm_fig, data_count_fig, scatter_fig, cm_female, cm_male, table_male, table_female
#
#     else:
#         return (dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update,
#                 dash.no_update, dash.no_update)
#
#
# def classify_data_loader(unbalanced_type, reconstructed_tracesDF, test_type, concat_tracesDF, feature_col_name,
#                          males_df, females_df, over_50_df, under_50_df, n_estimators, max_depth, learning_rate, gamma,
#                          subsample, colsample_bytree):
#     results, pred_tracesDF = instance_steEchoClass.classify_data(concat_tracesDF, feature_col_name, unbalanced_type,
#                                                         n_estimators, max_depth, learning_rate, gamma, subsample,
#                                                         colsample_bytree)
#
#     cm_female, cm_male, metrics_male, metrics_female = extract_sex_data(females_df, males_df,
#                                                                         pred_tracesDF, instance_steEchoClass.clf)
#     cm_female_plot = create_cm(cm_female, instance_steEchoClass.clf, True)
#     cm_male_plot = create_cm(cm_male, instance_steEchoClass.clf, True)
#
#     data_count_fig = count_data(reconstructed_tracesDF)
#
#     if test_type != 'gls_remod_class':
#         scatter_fig = create_scatter_plot(instance_steEchoClass.all_X_test, instance_steEchoClass.all_y_pred)
#     else:
#         scatter_fig = go.Figure()
#
#     return results, data_count_fig, scatter_fig, cm_female_plot, cm_male_plot, metrics_male, metrics_female
#
#
# class XGBClassViewer:
#
#     def __init__(self):
#         self.all_y_pred = []
#         self.all_X_test = []
#         self.scatter_fig = go.Figure()
#         self.y_pred = None
#         self.y_test = None
#         self.clf = None
#         self.tracesDF = None
#         self.df_isImputed_0 = None
#         self.studyType = 'BL'
#         self.pca_resultsDF = None
#         self.test, self.train, self.train = None, None, None
#         self.concat_array_curves = None
#         self.reconstructed_tracesDF = None
#         self.calc_remod_df = None
#
#     def classify_data(self, concat_tracesDF, feature_col_name, unbalanced_type, n_estimators, max_depth, learning_rate,
#                       gamma, subsample, colsample_bytree):
#         fold = 0
#         results = []
#         predictions_df = pd.DataFrame(columns=['study_nr', 'prediction'])
#         self.clf = xgb.XGBClassifier(random_state=42, n_estimators=n_estimators, max_depth=max_depth,
#                                      learning_rate=learning_rate, gamma=gamma, subsample=subsample,
#                                      colsample_bytree=colsample_bytree)
#
#         results, pred_tracesDF = classify_per_fold(concat_tracesDF, feature_col_name, fold, predictions_df,
#                                                    results, unbalanced_type, self.clf, self.all_y_pred, self.all_X_test)
#
#         return results, pred_tracesDF
#
#     def get_content(self):
#         return html.Div([
#             dbc.Col([
#                 dbc.Row([
#                     html.Br(),
#                     dbc.Checkbox(
#                         id='isImputed-id_xgb',
#                         label='Include imputed data',
#                         value=True,
#                         style={'marginLeft': '10px'}
#                     ),
#                 ], style={'marginLeft': '20px'}),
#             ]),
#             dbc.Col([
#                 dbc.Row([
#                     html.Label('Select Data Source:', style={'fontWeight': 'bold', 'marginLeft': '20px'}),
#                     dcc.Dropdown(
#                         id='id-data-source-dropdown_bc-xgb',
#                         options=[
#                             {'label': 'Speqle', 'value': 'Speqle'},
#                             {'label': 'GE', 'value': 'GE'},
#                             {'label': 'Philips', 'value': 'Philips'},
#                             {'label': 'GE + Philips', 'value': 'GE_Philips'}
#                         ],
#                         value='GE_Philips',
#                         style={'width': '300px', 'marginLeft': '10px'}
#                     ),
#                 ]),
#             ]),
#
#             dbc.Row([
#                 dbc.Col([
#                     html.Br(),
#                     html.Label('Classify by:', style={'fontWeight': 'bold'}),
#                     dcc.Dropdown(
#                         id='id-test_type-dropdown_xgb',
#                         options=[
#                             {'label': 'Concatenating 18 segments using 100% Variance',
#                              'value': 'concat_18_segments_100'},
#                             {'label': 'Concatenating 18 segments using 95% Variance',
#                              'value': 'concat_18_segments_95'},
#                             {'label': '100% of PCA variance components',
#                              'value': '100_variance_components'},
#                             {'label': '95% of PCA variance components',
#                              'value': '95_Variance_components'},
#                             {'label': '100% of PCA variance components per segment',
#                              'value': '100_components_per_segment'},
#                             {'label': '95% of PCA variance per segment concatenated',
#                              'value': '95_components_per_segment'},
#                             {'label': 'Classify using GLS',
#                              'value': 'gls_remod_class'},
#                             {'label': 'Classify using Echo Params',
#                              'value': 'echo_remod_class'},
#                         ],
#                         value='concat_18_segments_100',
#                     ),
#                     html.Br(),
#                     html.Button('Classify',
#                                 id='id-button_xgb',
#                                 n_clicks=0,
#                                 style={'width': '100px'})
#                 ], style={'marginLeft': '20px'}),
#
#                 dbc.Col([
#                     html.Br(),
#                     html.Label('Unbalanced:', style={'fontWeight': 'bold'}),
#                     dcc.Dropdown(
#                         id='id-unbalanced_class-dropdown_xgb',
#                         options=[
#                             {'label': 'None', 'value': 'None'},
#                             {'label': 'SMOTE', 'value': 'SMOTE'},
#                             {'label': 'ADASYN', 'value': 'ADASYN'},
#                             {'label': 'SMOTEENN', 'value': 'SMOTEENN'},
#                             {'label': 'SMOTETomek', 'value': 'SMOTETomek'}
#                         ],
#                         value='None',
#                         style={'height': 'Auto', 'width': '70%'}
#                     ),
#                 ], style={'width': '100%'}),
#             ]),
#
#             dbc.Row([
#                 dbc.Col([
#                     html.Label('Estimators:', style={'fontWeight': 'bold', 'marginLeft': '20px'}),
#                     dcc.Input(
#                         id="textarea-n_estimators-id",
#                         type="number",
#                         debounce=True,
#                         value=100,
#                         style={'width': '100px', 'marginLeft': '20px'}
#                     ),
#                     html.Br(),
#                     html.Label('Max depth:', style={'fontWeight': 'bold', 'marginLeft': '20px'}),
#                     dcc.Input(
#                         id="textarea-max_depth-id",
#                         type="number",
#                         debounce=True,
#                         value=None,
#                         style={'width': '100px', 'marginLeft': '20px'}
#                     ),
#                     html.Br(),
#                     html.Label('Learning rate:', style={'fontWeight': 'bold', 'marginLeft': '20px'}),
#                     dcc.Input(
#                         id="textarea-learning_rate-id",
#                         type="number",
#                         debounce=True,
#                         value=0.1,
#                         style={'width': '100px', 'marginLeft': '20px'}
#                     ),
#                 ]),
#
#                 dbc.Col([
#                     html.Label('Gamma:', style={'fontWeight': 'bold', 'marginLeft': '20px'}),
#                     dcc.Input(
#                         id="textarea-gamma-id",
#                         type="number",
#                         debounce=True,
#                         value=0,
#                         style={'width': '100px', 'marginLeft': '20px'}
#                     ),
#                     html.Br(),
#                     html.Label('Subsample:', style={'fontWeight': 'bold', 'marginLeft': '20px'}),
#                     dcc.Input(
#                         id="textarea-subsample-id",
#                         type="number",
#                         debounce=True,
#                         value=1,
#                         style={'width': '100px', 'marginLeft': '20px'}
#                     ),
#                     html.Br(),
#                     html.Label('Colsample bytree:', style={'fontWeight': 'bold', 'marginLeft': '20px'}),
#                     dcc.Input(
#                         id="textarea-colsample_bytree-id",
#                         type="number",
#                         debounce=True,
#                         value=1,
#                         style={'width': '100px', 'marginLeft': '20px'}
#                     ),
#                 ]),
#             ], style={'width': '100%'}),
#
#             html.Br(),
#             dbc.Accordion([
#                 dbc.AccordionItem(
#                     title="Data:",
#                     children=[
#                         dbc.Row([
#                             html.Label('Data Count:', style={'fontWeight': 'bold'}),
#                             dcc.Graph(id='id-data-count-plot_xgb')
#                         ], style={'width': '50%', 'margin': 'auto'}),
#                     ],
#                 ),
#                 dbc.AccordionItem(
#                     title="Metrics:",
#                     children=[
#                         dbc.Row([
#                             dbc.Col([
#                                 dbc.Row([
#                                     html.Div(id='id-metrics-output_xgb', style={'width': '60%', 'margin': 'auto'})
#                                 ], style={'width': '70%', 'margin': 'auto'}),
#                             ]),
#
#                             dbc.Col([
#                                 dbc.Row([
#                                     dcc.Graph(id='id-cm-plot_xgb')
#                                 ], style={'width': '70%', 'margin': 'auto'}),
#                             ]),
#                         ]),
#                         dbc.Row([
#                             html.Label('Scatter plot:', style={'fontWeight': 'bold'}),
#                             dcc.Graph(id='id-class-feature-plot_xgb')
#                         ], style={'width': '70%', 'margin': 'auto'}),
#                     ]),
#                 dbc.AccordionItem(
#                     title="Sex-specific Metrics:",
#                     children=[
#                         dbc.Row([
#                             dbc.Col([
#                                 html.Label('Female Confusion Matrix:', style={'fontWeight': 'bold'}),
#                                 dcc.Graph(id='id-cm-female_xgb')
#                             ], style={'width': '50%', 'margin': 'auto'}),
#                             dbc.Col([
#                                 html.Label('Male Confusion Matrix:', style={'fontWeight': 'bold'}),
#                                 dcc.Graph(id='id-cm-male_xgb')
#                             ], style={'width': '50%', 'margin': 'auto'}),
#                         ]),
#                         dbc.Row([
#                             dbc.Col([
#                                 html.Label('Female Metrics:', style={'fontWeight': 'bold'}),
#                                 html.Div(id='id-metrics-female_xgb', style={'width': '50%', 'margin': 'auto'})
#                             ], style={'width': '50%', 'margin': 'auto'}),
#                             dbc.Col([
#                                 html.Label('Male Metrics:', style={'fontWeight': 'bold'}),
#                                 html.Div(id='id-metrics-male_xgb', style={'width': '50%', 'margin': 'auto'})
#                             ], style={'width': '50%', 'margin': 'auto'}),
#                         ])
#                     ]
#                 ),
#             ], always_open=True)
#         ])
