# import gc
# import dash_bootstrap_components as dash_bc
# from dash import html, dcc, Output, Input, ctx
# from dash import callback, State, dash
# from dash_viewer.classification.load_prepare_data import load_data_traces
#
#
# gc.enable()
#
#
# def safe_eval(val):
#     try:
#         return eval(val)
#     except Exception:
#         return val
#
#
# @callback(
#     Output('id-table-best-params-vac', 'children'),
#     Output('id-table-test-scores-vac', 'children'),
#     Output('id-table-cm-scores-vac', 'figure'),
#     Input('isDataImputed-id-vac', 'value'),
#     Input('id-best-knn-btn-vac', 'n_clicks'),
#     Input('id-data-source-dropdown_bc-vac', 'value'),
#     Input('id-unbalanced_KnnBest-dropdown-vac', 'value'),
#     Input('id-slider-vac', 'value'),
#     State('id-concat_Type-dropdown-vac', 'value'),
#     State('id-class_type-dropdown-vac', 'value'),
# )
# def update_output(isImputed, best_neighbors, data_source, unbalanced_type, encoding, knn_type, class_type):
#     global instance_clf, param_table
#     instance_clf = BestParamClassVacViewer()
#     studyType = 'BL'
#     if "id-best-knn-btn-vac" == ctx.triggered_id and best_neighbors > 0:
#         if data_source:
#             data_source_mapping = {
#                 'Speqle': 'speqle_segmentalStrainInterp_T',
#                 'GE': 'st_segmentalStrainInterp_T',
#                 'Philips': 'phi_segmentalStrainInterp_T',
#                 'GE_Philips': ['phi_segmentalStrainInterp_T', 'st_segmentalStrainInterp_T']
#             }
#             if data_source in data_source_mapping:
#                 table_name = data_source_mapping[data_source]
#                 instance_clf.table_name = table_name
#
#                 print('-----------------------------------------------------------------------------------------------')
#                 print(f'------------ Find the best parameters for classification in table: {data_source}--------------')
#                 print(f'unbalanced method: {unbalanced_type}')
#                 print(f'Experiment type: {knn_type}')
#                 print(f'Class type: {class_type}')
#                 print(f'Imputed data: {isImputed}')
#                 print('-----------------------------------------------------------------------------------------------')
#
#                 orig_concat_tracesDF, orig_feature_col_name, imp_concat_tracesDF, imp_feature_col_name, features = \
#                     load_data_traces(input_type, isImputed, unbalanced_type, class_type, data_source, component, addGLS,
#                                      addFeatEcho, pc_number, class_input)
#
#                 (best_param, param_table, test_metrics_table,
#                  cm, cm_table_fig, shap_values, shap_input_df,
#                  important_features, shap_df) = best_params_table(orig_concat_tracesDF,
#                                                                   orig_feature_col_name,
#                                                                   imp_concat_tracesDF,
#                                                                   imp_feature_col_name,
#                                                                   unbalanced_type, class_type, features)
#                 dash_beeswarm = shap_beeswarm_base64(shap_values, class_input)
#
#                 print('---------------------------------------- Results ----------------------------------------------')
#                 test_metrics_df = pd.DataFrame(test_metrics_table.data)
#                 print(test_metrics_df.to_string(index=False))
#
#                 print("\nConfusion Matrix:")
#                 print(pd.DataFrame(cm, columns=["Predicted 0", "Predicted 1"], index=["Actual 0", "Actual 1"]))
#
#                 if dash_beeswarm:
#                     print("\nTop Important Features:")
#                     print(important_features)
#
#                 print('-----------------------------------------------------------------------------------------------')
#
#                 gc.collect()
#                 return param_table, test_metrics_table, cm_table_fig, dash_beeswarm
#         else:
#             table_name = None
#             return dash.no_update, dash.no_update, dash.no_update
#
#     else:
#         gc.collect()
#         return dash.no_update, dash.no_update, dash.no_update
#
#
#
#
#
# # def load_data_traces(knn_type, isImputed, unbalanced_type, class_type, data_source, encoding):
# #     imp_feature_col_name, imp_concat_tracesDF = None, None
# #     orig_concat_tracesDF, fil_impDF, fil_origDF = None, None, None
# #
# #     filter_by_type = select_data_type(knn_type, encoding)
# #
# #     tracesDF = load_queries_table_db('pca_results_view',
# #                                      'data_source', 'in', [data_source])
# #
# #     if isImputed:
# #         original_tracesDF = tracesDF[tracesDF['reconstructed_type'] == f'{filter_by_type}-0']
# #         imputed_tracesDF = tracesDF[tracesDF['reconstructed_type'] == f'{filter_by_type}-1']
# #     else:
# #         imputed_tracesDF = None
# #         original_tracesDF = None
# #         imp_concat_tracesDF = None
# #
# #     ## FOR Strain Traces Classification
# #     if knn_type == 'reconstructed_traces_vac':
# #         if imputed_tracesDF is None:
# #             orig_concat_tracesDF, orig_feature_col_name = get_concat_18_segments(tracesDF)
# #         else:
# #             orig_concat_tracesDF, orig_feature_col_name = get_concat_18_segments(original_tracesDF)
# #             imp_concat_tracesDF, imp_feature_col_name = get_concat_18_segments(imputed_tracesDF)
# #
# #             features = imp_feature_col_name
# #             imp_concat_tracesDF = remove_outliers_by_class_hampel(imp_concat_tracesDF, features, 'Total_Remodeling',15,2)
# #
# #         features = orig_feature_col_name
# #         orig_concat_tracesDF = remove_outliers_by_class_hampel(orig_concat_tracesDF, features, 'Total_Remodeling',15,2)
# #
# #     # FOR GLS Classification
# #     elif knn_type == 'gls_params_vac':
# #         features = 'global_strain'
# #         if imputed_tracesDF is None:
# #             orig_concat_tracesDF, orig_feature_col_name = get_echoParams_data(tracesDF, data_source, features)
# #         else:
# #             orig_concat_tracesDF, orig_feature_col_name = get_echoParams_data(original_tracesDF, data_source, features)
# #             imp_concat_tracesDF, imp_feature_col_name = get_echoParams_data(imputed_tracesDF, data_source, features)
# #             imp_concat_tracesDF = remove_outliers_from_df(features, imp_concat_tracesDF, 2, 98)
# #
# #         orig_concat_tracesDF = remove_outliers_from_df(features, orig_concat_tracesDF, 2, 98)
# #
# #     elif knn_type == '18_segments_vac' or knn_type == 'per_segment_vac':
# #         table_name = 'calculated_dr_encoded_data_T'
# #         features = 'encoded_data'
# #         if imputed_tracesDF is None:
# #             orig_concat_tracesDF, orig_feature_col_name = get_feature_components(tracesDF, data_source,
# #                                                                                  filter_by_type, features,
# #                                                                                  table_name)
# #         else:
# #             orig_concat_tracesDF, orig_feature_col_name = get_feature_components(original_tracesDF, data_source,
# #                                                                                  filter_by_type, features,
# #                                                                                  table_name)
# #             imp_concat_tracesDF, imp_feature_col_name = get_feature_components(imputed_tracesDF, data_source,
# #                                                                                filter_by_type, features,
# #                                                                                table_name)
# #
# #             imp_concat_tracesDF = remove_outliers_by_class_hampel(imp_concat_tracesDF, features, 'Total_Remodeling',15,2)
# #
# #         features = orig_feature_col_name
# #         orig_concat_tracesDF = remove_outliers_by_class_hampel(orig_concat_tracesDF, features, 'Total_Remodeling',15,2)
# #
# #     best_param, param_table, test_metrics_table, cm_table_fig = best_params_table(orig_concat_tracesDF,
# #                                                                                   orig_feature_col_name,
# #                                                                                   imp_concat_tracesDF,
# #                                                                                   imp_feature_col_name,
# #                                                                                   unbalanced_type, class_type)
# #     return param_table, test_metrics_table, cm_table_fig
#
#
# class BestParamClassVacViewer:
#
#     def __init__(self):
#         self.cm = None
#         self.X_resampled = None
#         self.parameters = None
#         self.y_resampled = None
#         self.metrics = None
#         self.param_grid = None
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
#         dropdown_options = [
#             {'label': 'Reconstructed traces', 'value': 'reconstructed_traces_vac'},
#             {'label': '18 Segments Autoencoder', 'value': '18_segments_vac'},
#             {'label': 'Per Segments Autoencoder', 'value': 'per_segment_vac'},
#             {'label': 'GLS', 'value': 'gls_params_vac'}
#         ]
#
#         return html.Div([
#             dash_bc.Row([
#                 dash_bc.Col([
#                     html.Br(),
#                     dash_bc.Row([
#                         html.Label('Select Data Source:', style={'fontWeight': 'bold', 'marginLeft': '20px'}),
#                         dcc.Dropdown(
#                             id='id-data-source-dropdown_bc-vac',
#                             options=[
#                                 {'label': 'Speqle', 'value': 'Speqle'},
#                                 {'label': 'GE', 'value': 'GE'},
#                                 {'label': 'Philips', 'value': 'Philips'},
#                                 {'label': 'GE + Philips', 'value': 'GE_Philips'}
#                             ],
#                             value='GE_Philips',
#                             style={'width': '300px', 'marginLeft': '10px'}
#                         ),
#                     ]),
#
#                     html.Br(),
#                     html.Br(),
#
#                     dash_bc.Row([
#                         dash_bc.Checkbox(
#                             id='isDataImputed-id-vac',
#                             label='Include imputed data',
#                             value=False,
#                             style={'marginLeft': '30px', 'width': '90%'}
#                         ),
#                     ]),
#                 ], width=2),
#
#                 dash_bc.Col([
#                     html.Br(),
#                     html.Label('Classify by:',
#                                style={'marginLeft': '20px', 'fontWeight': 'bold'}),
#                     dcc.Dropdown(
#                         id='id-concat_Type-dropdown-vac',
#                         options=dropdown_options,
#                         value='reconstructed_traces_vac',
#                         style={'marginLeft': '10px', 'height': 'Auto', 'width': '100%'}
#                     ),
#                 ], width=3),
#
#                 dash_bc.Col([
#                     html.Br(),
#                     html.Label('Machine Learning Classifiers:',
#                                style={'fontWeight': 'bold'}),
#                     dcc.Dropdown(
#                         id='id-class_type-dropdown-vac',
#                         options=[
#                             {'label': 'KNN', 'value': 'KNN_class'},
#                             {'label': 'Decision Tree', 'value': 'DT_class'},
#                             {'label': 'Random Forest', 'value': 'RF_class'},
#                             {'label': 'Gradient Boosting', 'value': 'GBM_class'},
#                             {'label': 'XGBoost', 'value': 'XGBoost_class'},
#                             {'label': 'SVM', 'value': 'SVM_class'},
#                             {'label': 'MLP', 'value': 'MLP_class'},
#                             {'label': 'Logistic Regression', 'value': 'LogisticRegression_class'},
#                         ],
#                         value='KNN_class',
#                         style={'height': 'Auto', 'width': '100%'}
#                     ),
#                 ], width=2),
#
#                 dash_bc.Col([
#                     html.Br(),
#                     html.Label('Unbalanced:', style={'fontWeight': 'bold'}),
#                     dcc.Dropdown(
#                         id='id-unbalanced_KnnBest-dropdown-vac',
#                         options=[
#                             {'label': 'None', 'value': 'None'},
#                             {'label': 'SMOTE', 'value': 'SMOTE'},
#                             {'label': 'ADASYN', 'value': 'ADASYN'},
#                             {'label': 'Balanced', 'value': 'Balanced'}
#                         ],
#                         value='None',
#                         style={'height': 'Auto', 'width': '100%'}
#                     ),
#                 ], width=2),
#
#                 dash_bc.Col([
#                     html.Br(),
#                     html.P("Encoding Dimensions:"),
#                     dcc.Slider(
#                         id='id-slider-vac',
#                         min=2,
#                         max=100,
#                         value=32,
#                         step=None,
#                         marks={2: '2', 4: '4', 8: '8', 16: '16', 32: '32', 64: '64', 100: '100'},
#                     )
#                 ], width=3),
#
#             ]),
#             dash_bc.Row([
#                 html.Br(),
#                 html.Label('Find the best parameters:',
#                            style={'fontWeight': 'bold', 'marginLeft': '20px'}),
#                 dash_bc.Button('Classify', id='id-best-knn-btn-vac', n_clicks=0,
#                                style={'marginLeft': '30px', 'width': '120px'})
#             ]),
#
#             html.Br(),
#             html.Br(),
#
#             dash_bc.Row([
#                 html.Div(id='id-table-best-params-vac',
#                          style={'marginLeft': '20px'})
#             ], justify="center", style={'width': '50%', 'margin': 'auto'}),
#             html.Br(),
#             dash_bc.Row([
#                 html.Div(id='id-table-test-scores-vac',
#                          style={'marginLeft': '20px'})
#             ], justify="center", style={'width': '50%', 'margin': 'auto'}),
#             html.Br(),
#             html.Br(),
#             dash_bc.Row([
#                 dcc.Graph(id='id-table-cm-scores-vac')
#             ], justify="center", style={'width': '70%', 'margin': 'auto'}),
#         ])
