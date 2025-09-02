# import numpy as np
# import pandas as pd
# import plotly.graph_objects as go
# import plotly.figure_factory as ff
# import dash_bootstrap_components as dash_bc
# from sklearn.model_selection import train_test_split
# import torch
# from torch import nn, optim
# from torch.utils.tensorboard import SummaryWriter
# from dash import html, dcc, Output, Input, callback, dash, dash_table
# from sklearn.decomposition import PCA
#
# from sklearn.model_selection import StratifiedKFold
# from sklearn.metrics import roc_auc_score
# from sklearn.metrics import accuracy_score, precision_score, recall_score
# from sklearn.metrics import matthews_corrcoef, f1_score, confusion_matrix
#
# from classifiers.nBeats_classifier import NBEATSClassifier
# from pattern_recognition.data.database_connection import sql_queries
# from pattern_recognition.utils.parameters_dics import dbConParameters
# from pattern_recognition.data.database_connection.db_con import DatabaseConnection as dbc
# from pattern_recognition.data_preprocessing.speqle.concatenate_traces import CurveConcatenator
# from pattern_recognition.data.loadTraces import LoadTraces as load_traces
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
#     Output('id-metrics-ae-results', 'children'),
#     Output('id-cm-ae-plot', 'figure'),
#     Output('id-class-ae-feature-plot', 'figure'),
#     # Output('id-learning-plot', 'figure'),
#     Input('isImputed-id', 'value'),
#     Input('id-Type-dropdown', 'value'),
#     # State('id-knn_distance-dropdown', 'value'),
#     Input('textarea-k', 'value'),
#     # State('id-knn_weight-dropdown', 'value')
#
# )
# def update_output(isImputed, test_type, num_epochs):
#     global instance_autoEncoder
#     instance_autoEncoder = NBeats_classifier()
#
#     if num_epochs is not None:
#         instance_autoEncoder.load_imputed_data()
#         results, scatter_fig = load_data_traces(test_type, isImputed, num_epochs)
#
#         confusion_matrices, metrics_table = create_data_table_results(results)
#
#         agg_confusion_matrix = np.sum(confusion_matrices, axis=0)
#         class_labels = [0, 1]
#         z = agg_confusion_matrix
#         x = class_labels
#         y = class_labels
#
#         cm_fig = ff.create_annotated_heatmap(z, x=x, y=y, colorscale='Blues')
#
#         # Update layout
#         cm_fig.update_layout(
#             xaxis=dict(title='Predicted value'),
#             yaxis=dict(title='True value')
#         )
#
#         return metrics_table, cm_fig, scatter_fig
#
#     else:
#         return dash.no_update, dash.no_update, dash.no_update
#
#
# def create_data_table_results(results):
#     all_metrics = {'Metric': [], 'Fold 1': [], 'Fold 2': [], 'Fold 3': [], 'Fold 4': [], 'Fold 5': []}
#     confusion_matrices = []
#     for i, result in enumerate(results):
#         y_test, y_pred, metrics = result
#         for metric in ['Accuracy', 'Precision', 'Recall', 'F1-score', 'MCC', 'AUC']:
#             if i == 0:
#                 all_metrics['Metric'].append(metric)
#             all_metrics[f'Fold {i + 1}'].append(round(metrics[metric], 4))
#
#         confusion_matrices.append(metrics['Confusion Matrix'])
#
#     metrics_df = pd.DataFrame(all_metrics)
#     metrics_df['Average'] = metrics_df.iloc[:, 1:].mean(axis=1).round(4)
#
#     metrics_table = dash_table.DataTable(
#         data=metrics_df.to_dict('records'),
#         columns=[{'name': i, 'id': i} for i in metrics_df.columns],
#         style_cell={'textAlign': 'center'},
#         style_data_conditional=[
#             {
#                 'if': {'column_id': 'Average'},
#                 'fontWeight': 'bold'
#             }
#         ],
#     )
#     return confusion_matrices, metrics_table
#
#
# def load_queries_table_db(view_name, column_name,operator, value):
#     try:
#         dbConParams = dbConParameters
#         db = dbc(driver=dbConParams['driver'], server=dbConParams['server'], database=dbConParams['database'])
#         db.connect()
#         db.execute_query(sql_queries.select_all_data_where(view_name, column_name, operator ,value))
#         df = db.df_query()
#         db.close_connection()
#         return df
#     except Exception as e:
#         print(f"An error occurred: {e}")
#         return None
#
#
# def select_test_type(test_type,isImputed):
#     global filter_by
#     if test_type == 'concat_18_segments_100' or test_type == '100_variance_components':
#         filter_by = 'PCA-full_traces-1'
#     elif test_type == 'concat_18_segments_95' or test_type == '95_Variance_components':
#         filter_by = 'PCA-full_traces-0.95'
#     elif test_type == '100_components_per_segment':
#         filter_by = 'PCA-segment-1'
#     elif test_type == '95_components_per_segment':
#         filter_by = 'PCA-segment-0.95'
#     elif test_type == 'gls_remod_class':
#         filter_by = 'PCA-full_traces-1'
#
#     if isImputed:
#         filter_by += "-1"
#     else:
#         filter_by += "-0"
#     return filter_by
#
#
# def load_data_traces(test_type, isImputed, num_epochs):
#     filter_by_type = select_test_type(test_type, isImputed)
#     concat_tracesDF = pd.DataFrame()
#     reconstructed_tracesDF = load_queries_table_db('reconstructed_remodeling_view',
#                                                    'reconstructed_type', '=', filter_by_type)
#     if not isImputed:
#         study_nr_df = instance_autoEncoder.df_isImputed_0
#         reconstructed_tracesDF = reconstructed_tracesDF[
#             reconstructed_tracesDF['study_nr'].isin(study_nr_df['study_nr'])]
#
#     if test_type == 'concat_18_segments_100' or test_type == 'concat_18_segments_95':
#         feature_col_name = 'concatenated_trace'
#
#         concatenated_strain = CurveConcatenator(reconstructed_tracesDF, isReconstructed=True)
#         concat_tracesDF = concatenated_strain.concatenate_traces_par()
#         concat_tracesDF['concatenated_trace'] = np.array(concat_tracesDF['concatenated_trace'])
#         reconstructed_tracesDF_unique = reconstructed_tracesDF.drop_duplicates(subset='study_nr')[
#             ['Total_Remodeling', 'study_nr']]
#
#         concat_tracesDF = pd.merge(concat_tracesDF,
#                                    reconstructed_tracesDF_unique[['Total_Remodeling', 'study_nr']],
#                                    on='study_nr',
#                                    how='left')
#     elif test_type == 'classify_gls':
#         reconstructed_tracesDF_selected = (reconstructed_tracesDF.groupby('study_nr')['Total_Remodeling'].mean().
#                                            astype(int).reset_index())
#
#         unique_study_nr = reconstructed_tracesDF['study_nr'].unique()
#         unique_study_nr_list = unique_study_nr.tolist()
#         unique_study_nr_str = ', '.join(map(str, unique_study_nr_list))
#
#         echo_paramDF = load_queries_table_db('imgData_2DEcho_speckle_BL',
#                                              'study_nr', 'in', unique_study_nr_str)
#
#         echo_paramDF_grouped = echo_paramDF.groupby('study_nr')['Global Strain %'].mean().reset_index()
#         echo_paramDF_grouped = pd.merge(echo_paramDF_grouped, reconstructed_tracesDF_selected, on='study_nr',
#                                         how='left')
#
#         feature_col_name = 'Global Strain %'
#         concat_tracesDF = echo_paramDF_grouped
#     else:
#         feature_col_name = 'pca_results'
#
#         componentsDF = load_queries_table_db('pca_results_view', 'reconstructed_type',
#                                              '=', filter_by_type)
#         componentsDF['pca_results'] = componentsDF['pca_results'].apply(safe_eval)
#
#         if not isImputed:
#             study_nr_df = instance_autoEncoder.df_isImputed_0
#             componentsDF = componentsDF[componentsDF['study_nr'].isin(study_nr_df['study_nr'])]
#
#         concat_tracesDF = componentsDF
#
#     # results = instance_autoEncoder.auto_encoder_classifier(concat_tracesDF, feature_col_name, num_epochs)
#     results = instance_autoEncoder.nBeats_classifier(concat_tracesDF, feature_col_name, num_epochs)
#
#     scatter_fig = instance_autoEncoder.create_scatter_plot()
#     instance_autoEncoder.writer.close()
#     return results, scatter_fig
#
#
# class NBeats_classifier:
#
#     def __init__(self):
#         self.recall, self.f1, self.accuracy = [], [], []
#         self.cm, self.mcc, self.auc = [], [], []
#         self.precision = []
#         self.all_y_pred, self.all_X_test = [], []
#         self.writer = SummaryWriter('runs/nbeats_experiment')
#         # self.average_loss = []
#         # self.losses = []
#
#         self.scatter_fig = go.Figure()
#         self.y_pred = None
#         self.y_test = None
#         self.knn = None
#         self.tracesDF = None
#         self.df_isImputed_0 = None
#         self.studyType = 'BL'
#         self.pca_resultsDF = None
#         self.test, self.train, self.train = None, None, None
#         self.concat_array_curves = None
#         self.reconstructed_tracesDF = None
#         self.calc_remod_df = None
#
#     def load_imputed_data(self):
#         speqle = load_traces
#         df = speqle(self.studyType, 'speqle_segmentalStrainInterp_T', None)
#         self.tracesDF = df.tracesDF[['study_nr', 'curve_segment', 'isImputed']]
#         df_isImputed_1 = self.tracesDF[self.tracesDF['isImputed'] == 1]['study_nr'].unique()
#         self.df_isImputed_0 = self.tracesDF[self.tracesDF['isImputed'] == 0]
#         unique_study_nr_0 = self.df_isImputed_0['study_nr'].unique()
#         self.df_isImputed_0 = self.df_isImputed_0[~self.df_isImputed_0['study_nr'].isin(df_isImputed_1)]
#
#     def get_content(self):
#         return html.Div([
#             dash_bc.Row([
#                 html.Br(),
#                 dash_bc.Checkbox(
#                     id='isImputed-id',
#                     label='Include imputed data',
#                     value=True,
#                     style={'marginLeft': '10px'}
#                 ),
#             ], style={'marginLeft': '20px'}),
#
#             html.Label('Select number of epochs:', style={'fontWeight': 'bold', 'marginLeft': '20px'}),
#             dcc.Input(
#                 id="textarea-k", type="number", debounce=True,
#                 style={'width': '100px', 'marginLeft': '20px'}),
#
#             dash_bc.Row([
#                 dash_bc.Col([
#                     html.Br(),
#                     html.Label('Classify :', style={'fontWeight': 'bold'}),
#                     dcc.Dropdown(
#                         id='id-Type-dropdown',
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
#                              'value': 'classify_gls'}
#                         ],
#                         value='concat_18_segments_100',
#                     ),
#                 ], style={'marginLeft': '20px'}),
#             ]),
#             html.Br(),
#             dash_bc.Accordion([
#                 # dash_bc.AccordionItem(
#                 #     title="ML:",
#                 #     children=[
#                 #         dash_bc.Row([
#                 #             html.Label('Learning Curve:', style={'fontWeight': 'bold'}),
#                 #             dcc.Graph(id='id-learning-plot')
#                 #         ], justify="center", style={'width': '50%', 'margin': 'auto'}),
#                 #     ],
#                 # ),
#                 dash_bc.AccordionItem(
#                     title="Metrics:",
#                     children=[
#                         dash_bc.Row([html.Div(id='id-metrics-ae-results', style={'width': '60%', 'margin': 'auto'})]),
#                         html.Br(),
#                         dash_bc.Row([
#                             dash_bc.Col([
#                                 dash_bc.Row([
#                                     html.Label('Confusion Matrix:', style={'fontWeight': 'bold'}),
#                                     dcc.Graph(id='id-cm-ae-plot')
#                                 ], justify="center", style={'width': '100%', 'margin': 'auto'}),
#                             ]),
#                             dash_bc.Col([
#                                 dash_bc.Row([
#                                     html.Label('Scatter plot:', style={'fontWeight': 'bold'}),
#                                     dcc.Graph(id='id-class-ae-feature-plot')
#                                 ], justify="center", style={'width': '100%', 'margin': 'auto'}),
#                             ]),
#                         ])
#
#                     ],
#                 )
#             ],
#                 always_open=True,
#             )
#         ])
#
#     def split_data(self, data, feature_col_name):
#         if feature_col_name == 'Global Strain %':
#             X = data[feature_col_name].values.reshape(-1, 1)
#         else:
#             X = np.stack(data[feature_col_name].values)
#
#         y = data['Total_Remodeling'].values
#         skf = StratifiedKFold(n_splits=5, shuffle=False)
#
#         for train_index, test_index in skf.split(X, y):
#             X_train, X_test = X[train_index], X[test_index]
#             y_train, y_test = y[train_index], y[test_index]
#             yield X_train, y_train, X_test, y_test
#
#     def nBeats_classifier(self, concat_tracesDF, feature_col_name, num_epochs):
#         fold = 0
#         results = []
#         writer = SummaryWriter('runs/nbeats_experiment')
#
#         input_dim = concat_tracesDF[feature_col_name].values[0].shape[0]
#         hidden_size = input_dim // 2
#         num_blocks = 3  # Number of N-BEATS blocks
#         num_classes = 2
#
#         n_beats_classifier = NBEATSClassifier(input_dim, hidden_size, num_blocks, num_classes)
#         criterion = nn.CrossEntropyLoss()
#         optimizer = optim.Adam(n_beats_classifier.parameters())
#         fold_losses = []
#
#         for X_train, y_train, X_test, y_test in self.split_data(concat_tracesDF, feature_col_name):
#             fold += 1
#
#             # Split the training data into a training set and a validation set
#             X_train_torch, X_val_torch, y_train_torch, y_val_torch = train_test_split(
#                 torch.from_numpy(X_train).float(), torch.from_numpy(y_train).long(), test_size=0.2)
#
#             X_test_torch = torch.from_numpy(X_test).float()
#
#             for epoch in range(num_epochs):
#                 # Initialize a dictionary to store the sum of each metric for this epoch
#                 metrics_sum = {'Accuracy': 0, 'Precision': 0, 'Recall': 0, 'F1-score': 0, 'MCC': 0, 'AUC': 0}
#
#                 n_beats_classifier.train()
#                 class_probs = n_beats_classifier(X_train_torch)
#                 loss = criterion(class_probs, y_train_torch)
#                 optimizer.zero_grad()
#                 loss.backward()
#                 optimizer.step()
#
#                 self.writer.add_scalar('training loss', loss.item(), global_step=epoch)
#
#                 n_beats_classifier.eval()
#                 class_probs_val = n_beats_classifier(X_val_torch)
#                 y_val_pred = class_probs_val.argmax(dim=1).numpy()
#                 self.calculate_metrics(y_val_torch.numpy(), y_val_pred, fold)
#                 self.calculate_avg_metrics()
#
#             n_beats_classifier.eval()
#             class_probs_test = n_beats_classifier(X_test_torch)
#             y_pred = class_probs_test.argmax(dim=1).numpy()
#
#             metrics = self.calculate_metrics(y_test, y_pred, fold)
#             results.append((y_test, y_pred, metrics))
#
#             self.all_y_pred.extend(y_pred)
#             self.all_X_test.extend(X_test)
#
#         return results
#
#     def calculate_metrics(self, y_true, y_pred, fold):
#         self.precision[fold].extend(precision_score(y_true, y_pred, average='weighted', zero_division=0))
#         self.recall[fold].extend(recall_score(y_true, y_pred, average='weighted', zero_division=0))
#         self.f1[fold].extend(f1_score(y_true, y_pred, average='weighted'))
#         self.accuracy[fold].extend(accuracy_score(y_true, y_pred))
#         self.cm[fold].extend(confusion_matrix(y_true, y_pred))
#         self.mcc[fold].extend(matthews_corrcoef(y_true, y_pred))
#         self.auc[fold].extend(roc_auc_score(y_true, y_pred))
#
#     def calculate_avg_metrics(self):
#         avg_precision = np.mean([np.mean(self.precision[fold]) for fold in len(self.precision)])
#         avg_recall = np.mean([np.mean(self.recall[fold]) for fold in len(self.recall)])
#         avg_f1 = np.mean([np.mean(self.f1[fold]) for fold in len(self.f1)])
#         avg_accuracy = np.mean([np.mean(self.accuracy[fold]) for fold in len(self.accuracy)])
#         avg_mcc = np.mean([np.mean(self.mcc[fold]) for fold in len(self.mcc)])
#         avg_auc = np.mean([np.mean(self.auc[fold]) for fold in len(self.auc)])
#
#         return avg_precision, avg_recall, avg_f1, avg_accuracy, avg_mcc, avg_auc
#
#     def create_scatter_plot(self):
#         pca = PCA(n_components=2)
#         X_test_pca = pca.fit_transform(self.all_X_test)
#
#         scatter_fig = go.Figure()
#
#         for class_value in np.unique(self.all_y_pred):
#             scatter_fig.add_trace(go.Scatter(
#                 x=X_test_pca[self.all_y_pred == class_value, 0],
#                 y=X_test_pca[self.all_y_pred == class_value, 1],
#                 mode='markers',
#                 marker=dict(
#                     color='blue' if class_value == 0 else 'red',
#                     size=10
#                 ),
#                 name='LV Remodeling' if class_value == 1 else 'Non remodeling'
#             ))
#
#         return scatter_fig
