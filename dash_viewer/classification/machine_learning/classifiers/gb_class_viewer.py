# import numpy as np
# import pandas as pd
#
# import plotly.graph_objects as go
# import dash_bootstrap_components as dash_bc
#
# from dash import html, dcc, Output, Input, callback, State, ctx, dash
#
# from sklearn.ensemble import GradientBoostingClassifier
#
# from dash_viewer.classification.load_prepare_data import select_data_type, load_imputed_data, get_concat_18_segments, \
#  get_feature_components
# from dash_viewer.classification.test_classifiers import create_data_table_results, get_clinical_data, create_cm, \
#     count_data, create_scatter_plot, split_data, calculate_metrics, extract_sex_data, load_data_traces, \
#     classify_per_fold
# from dash_viewer.dimensionality_reduction.dimen_reduction_viewer import load_table_db
#
#
# @callback(
#     Output('id-metrics-output_gb', 'children'),
#     Output('id-cm-plot_gb', 'figure'),
#     Output('id-data-count-plot_gb', 'figure'),
#     Output('id-class-feature-plot_gb', 'figure'),
#     Output('id-cm-male_gb', 'figure'),
#     Output('id-cm-female_gb', 'figure'),
#     Output('id-metrics-male_gb', 'children'),
#     Output('id-metrics-female_gb', 'children'),
#     Input('id-gb-button', 'n_clicks'),
#     Input('isImputed-id_gb', 'value'),
#     State('id-test_type-dropdown_gb', 'value'),
#     State('id-gb_n_estimators', 'value'),
#     State('id-gb_learning_rate', 'value'),
#     State('id-gb_max_depth', 'value'),
#     State('id-gb_subsample', 'value'),
#     State('id-gb_min_samples', 'value'),
#     State('id-unbalanced_class-dropdown_gb', 'value')
# )
# def update_output(gb_click, isImputed, test_type, gb_n_estimators, gb_learning_rate, gb_max_depth, gb_subsample,
#                   gb_min_samples, unbalanced_type):
#     global instance_steEchoClass, param_table
#     studyType = 'BL'
#     instance_steEchoClass = GBViewer()
#
#     if "id-gb-button" == ctx.triggered_id and gb_click > 0:
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
#             cm_female, cm_male, metrics_male, metrics_female = classify_data_loader(gb_n_estimators, gb_learning_rate,
#                                                                                     gb_max_depth, gb_subsample,
#                                                                                     gb_min_samples, unbalanced_type,
#                                                                                     reconstructed_tracesDF, test_type,
#                                                                                     concat_tracesDF, feature_col_name,
#                                                                                     males_df,
#                                                                                     females_df, over_50_df, under_50_df)
#
#         confusion_matrices, metrics_table = create_data_table_results(results, False)
#
#         cm_fig = create_cm(confusion_matrices, instance_steEchoClass.clf, False)
#         table_male = create_data_table_results(metrics_male, True)
#         table_female = create_data_table_results(metrics_female, True)
#
#         return metrics_table, cm_fig, data_count_fig, scatter_fig, cm_male, cm_female, table_male, table_female
#
#     else:
#         return (dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update,
#                 dash.no_update, dash.no_update)
#
#
# def classify_data_loader(gb_n_estimators, gb_learning_rate, gb_max_depth, gb_subsample, gb_min_samples, unbalanced_type,
#                          reconstructed_tracesDF, test_type,
#                          concat_tracesDF, feature_col_name,
#                          males_df,
#                          females_df, over_50_df, under_50_df):
#     results, pred_tracesDF = instance_steEchoClass.classify_data(concat_tracesDF, feature_col_name, gb_n_estimators,
#                                                         gb_learning_rate, gb_max_depth, gb_subsample,
#                                                         gb_min_samples, unbalanced_type)
#
#     cm_female, cm_male, metrics_male, metrics_female = extract_sex_data(females_df, males_df,
#                                                                         pred_tracesDF, instance_steEchoClass.clf)
#     cm_female_plot = create_cm(cm_female, instance_steEchoClass.clf, True)
#     cm_male_plot = create_cm(cm_male, instance_steEchoClass.clf, True)
#
#     data_count_fig = count_data(reconstructed_tracesDF)
#
#     if not test_type == 'gls_remod_class':
#         scatter_fig = create_scatter_plot(instance_steEchoClass.all_X_test, instance_steEchoClass.all_y_pred)
#     else:
#         scatter_fig = go.Figure()
#
#     return results, data_count_fig, scatter_fig, cm_male_plot, cm_female_plot, metrics_male, metrics_female
#
#
# class GBViewer:
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
#     def get_content(self):
#         return html.Div([
#             dash_bc.Row([
#                 html.Br(),
#                 dash_bc.Checkbox(
#                     id='isImputed-id_gb',
#                     label='Include imputed data',
#                     value=True,
#                     style={'marginLeft': '10px'}
#                 ),
#             ], style={'marginLeft': '20px'}),
#
#             dash_bc.Row([
#                 dash_bc.Col([
#                     html.Br(),
#                     html.Label('Classify by:', style={'fontWeight': 'bold'}),
#                     dcc.Dropdown(
#                         id='id-test_type-dropdown_gb',
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
#
#                         ],
#                         value='concat_18_segments_100',
#                     ),
#                     html.Br(),
#                     html.Button('Classify',
#                                 id='id-gb-button',
#                                 n_clicks=0,
#                                 style={'width': '100px'})
#                 ], style={'marginLeft': '20px'}),
#
#                 dash_bc.Col([
#                     html.Br(),
#                     html.Label('Unbalanced:', style={'fontWeight': 'bold'}),
#                     dcc.Dropdown(
#                         id='id-unbalanced_class-dropdown_gb',
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
#             dash_bc.Row([
#                 html.Br(),
#                 html.Br(),
#                 dash_bc.Col([
#                     html.Label('n_estimators:', style={'fontWeight': 'bold', 'marginLeft': '20px'}),
#                     html.Br(),
#                     dcc.Input(id='id-gb_n_estimators',
#                               type='number',
#                               style={'width': '150px', 'marginLeft': '20px'}
#                               ),
#
#                     html.Br(),
#                     html.Label('learning_rate:', style={'fontWeight': 'bold', 'marginLeft': '20px'}),
#                     html.Br(),
#                     dcc.Input(
#                         id='id-gb_learning_rate',
#                         type='number',
#                         style={'width': '150px', 'marginLeft': '20px'}
#                     ),
#
#                     html.Br(),
#                     html.Label('max_depth:', style={'fontWeight': 'bold', 'marginLeft': '20px'}),
#                     html.Br(),
#                     dcc.Input(
#                         id='id-gb_max_depth',
#                         type='number',
#                         style={'width': '150px', 'marginLeft': '20px'}
#                     ),
#                 ]),
#
#                 dash_bc.Col([
#                     html.Label('subsample:', style={'fontWeight': 'bold'}),
#                     html.Br(),
#                     dcc.Input(
#                         id='id-gb_subsample',
#                         type='number',
#                         style={'width': '150px'}
#                     ),
#
#                     html.Br(),
#                     html.Label('min_samples:', style={'fontWeight': 'bold'}),
#                     html.Br(),
#                     dcc.Input(
#                         id='id-gb_min_samples',
#                         type='number',
#                         style={'width': '150px'}
#                     ),
#                 ]),
#             ]),
#             html.Br(),
#             dash_bc.Accordion([
#                 dash_bc.AccordionItem(
#                     title="Data:",
#                     children=[
#                         dash_bc.Row([
#                             html.Label('Data Count:', style={'fontWeight': 'bold'}),
#                             dcc.Graph(id='id-data-count-plot_gb')
#                         ], justify="center", style={'width': '50%', 'margin': 'auto'}),
#                     ],
#                 ),
#                 dash_bc.AccordionItem(
#                     title="Metrics:",
#                     children=[
#                         dash_bc.Row([
#                             dash_bc.Col([
#                                 dash_bc.Row([
#                                     html.Div(id='id-metrics-output_gb', style={'width': '60%', 'margin': 'auto'})
#                                 ], justify="center", style={'width': '70%', 'margin': 'auto'}),
#                             ]),
#
#                             dash_bc.Col([
#                                 dash_bc.Row([
#                                     dcc.Graph(id='id-cm-plot_gb')
#                                 ], justify="center", style={'width': '70%', 'margin': 'auto'}),
#                             ]),
#                         ]),
#                         dash_bc.Row([
#                             html.Label('Scatter plot:', style={'fontWeight': 'bold'}),
#                             dcc.Graph(id='id-class-feature-plot_gb')
#                         ], justify="center", style={'width': '70%', 'margin': 'auto'}),
#
#                        dash_bc.AccordionItem(
#                             title="Metrics Per Gender:",
#                             children=[
#                                 dash_bc.Row([
#                                     dash_bc.Col([
#                                         html.Label('Male:', style={'fontWeight': 'bold'}),
#                                         dash_bc.Row([
#                                             html.Div(id='id-metrics-male_gb', style={'width': '60%', 'margin': 'auto'})
#                                         ], justify="center", style={'width': '70%', 'margin': 'auto'}),
#                                         html.Br(),
#                                         dash_bc.Row([
#                                             dcc.Graph(id='id-cm-male_gb')
#                                         ], justify="center", style={'width': '70%', 'margin': 'auto'}),
#                                     ]),
#
#                                     dash_bc.Col([
#                                         html.Label('Female:', style={'fontWeight': 'bold'}),
#                                         dash_bc.Row([
#                                             html.Div(id='id-metrics-female_gb',
#                                                      style={'width': '60%', 'margin': 'auto'})
#                                         ], justify="center", style={'width': '70%', 'margin': 'auto'}),
#                                         html.Br(),
#                                         dash_bc.Row([
#                                             dcc.Graph(id='id-cm-female_gb')
#                                         ], justify="center", style={'width': '70%', 'margin': 'auto'}),
#                                     ]),
#                                 ]),
#                             ]),
#                     ]),
#             ], always_open=True, )
#         ])
#
#     def classify_data(self, concat_tracesDF, feature_col_name, gb_n_estimators, gb_learning_rate, gb_max_depth,
#                       gb_subsample, gb_min_samples, unbalanced_type):
#         fold = 0
#         results = []
#         predictions_df = pd.DataFrame(columns=['study_nr', 'prediction'])
#         self.clf = GradientBoostingClassifier(n_estimators=gb_n_estimators, learning_rate=gb_learning_rate,
#                                               max_depth=gb_max_depth, subsample=gb_subsample,
#                                               min_samples_split=gb_min_samples,
#                                               random_state=42)
#
#         results, pred_tracesDF = classify_per_fold(concat_tracesDF, feature_col_name, fold, predictions_df,
#                                                    results, unbalanced_type, self.clf, self.all_y_pred,
#                                                    self.all_X_test)
#
#         return results, pred_tracesDF
