# import numpy as np
# import pandas as pd
# import plotly.express as px
# import plotly.graph_objects as go
# import plotly.figure_factory as ff
#
# from imblearn.over_sampling import ADASYN, SMOTE
# from imblearn.combine import SMOTETomek, SMOTEENN
#
# from sklearn.manifold import TSNE
# from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix, \
#     matthews_corrcoef, roc_auc_score
#
# from sklearn.model_selection import StratifiedKFold
#
# from dash import dash_table
#
# from dash_viewer.classification.load_prepare_data import (get_feature_components, get_concat_18_segments,
#                                                           select_data_type, get_echoParams_data)
# from dash_viewer.dimensionality_reduction.dimen_reduction_viewer import load_table_db
# from pattern_recognition.utils.parameters_dics import dbTableNames
#
#
# def split_data(data, feature_col_name, unbalanced_type):
#     if feature_col_name == 'Global Strain %':
#         X = data[feature_col_name].values.reshape(-1, 1)
#     else:
#         X = np.stack(data[feature_col_name].values)
#     y = data['Total_Remodeling'].values
#
#     if unbalanced_type == 'SMOTE':
#         oversample = SMOTE(random_state=42)
#     elif unbalanced_type == 'ADASYN':
#         oversample = ADASYN(random_state=42)
#     elif unbalanced_type == 'SMOTEENN':
#         oversample = SMOTEENN(random_state=42)
#     elif unbalanced_type == 'SMOTETomek':
#         oversample = SMOTETomek(random_state=42)
#     else:
#         oversample = None
#
#     skf = StratifiedKFold(n_splits=5, shuffle=False)
#     for train_index, test_index in skf.split(X, y):
#         X_train, X_test = X[train_index], X[test_index]
#         y_train, y_test = y[train_index], y[test_index]
#
#
#         if oversample is not None:
#             X_train_resampled, y_train_resampled = oversample.fit_resample(X_train, y_train)
#             yield X_train_resampled, y_train_resampled, X_test, y_test, test_index
#         else:
#             yield X_train, y_train, X_test, y_test, test_index
#
#
# def calculate_metrics(y_true, y_pred, y_pred_proba, fold):
#     precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
#     recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
#     f1 = f1_score(y_true, y_pred, average='weighted')
#     accuracy = accuracy_score(y_true, y_pred)
#     cm = confusion_matrix(y_true, y_pred)
#     mcc = matthews_corrcoef(y_true, y_pred)
#     auc_ = roc_auc_score(y_true, y_pred_proba)
#
#     return {'Fold': fold, 'Accuracy': accuracy, 'Precision': precision,
#             'Recall': recall, 'F1-score': f1, 'MCC': mcc, 'AUC': auc_,
#             'Confusion Matrix': cm}
#
#
# def create_scatter_plot(all_X_test, all_y_pred):
#     tsne = TSNE(n_components=2, random_state=42)
#     X_test_tsne = tsne.fit_transform(np.array(all_X_test))
#
#     scatter_fig = go.Figure()
#
#     for class_value in np.unique(all_y_pred):
#         scatter_fig.add_trace(go.Scatter(
#             x=X_test_tsne[all_y_pred == class_value, 0],
#             y=X_test_tsne[all_y_pred == class_value, 1],
#             mode='markers',
#             marker=dict(
#                 color='blue' if class_value == 0 else 'red',
#                 size=10
#             ),
#             name='LV Remodeling' if class_value == 1 else 'Non remodeling'
#         ))
#
#     return scatter_fig
#
#
# def create_data_table_results(results, isColumn):
#     Accuracies, Precisions, Recalls, F1_scores, MCCs, AUCs, = [], [], [], [], [], []
#     if not isColumn:
#         metrics_by_fold = {f'Fold {i + 1}': [] for i in range(5)}
#         confusion_matrices = []
#         for _, _, metrics in results:
#             fold = metrics['Fold']
#             fold_key = f'Fold {fold}'
#             if fold_key in metrics_by_fold:
#                 metrics_by_fold[fold_key].append(metrics)
#             else:
#                 metrics_by_fold[fold_key] = [metrics]
#
#             confusion_matrices.append(metrics['Confusion Matrix'])
#             Accuracies.append(metrics['Accuracy'])
#             Precisions.append(metrics['Precision'])
#             Recalls.append(metrics['Recall'])
#             F1_scores.append(metrics['F1-score'])
#             MCCs.append(metrics['MCC'])
#             AUCs.append(metrics['AUC'])
#     else:
#         confusion_matrices = None
#         Accuracies.append(results['Accuracy'])
#         Precisions.append(results['Precision'])
#         Recalls.append(results['Recall'])
#         F1_scores.append(results['F1-score'])
#         MCCs.append(results['MCC'])
#         AUCs.append(results['AUC'])
#
#     avg_metrics = {
#         'Accuracy': round(sum(Accuracies) / len(Accuracies), 4) if Accuracies else 0,
#         'Precision': round(sum(Precisions) / len(Precisions), 4) if Precisions else 0,
#         'Recall': round(sum(Recalls) / len(Recalls), 4) if Recalls else 0,
#         'F1-score': round(sum(F1_scores) / len(F1_scores), 4) if F1_scores else 0,
#         'MCC': round(sum(MCCs) / len(MCCs), 4) if MCCs else 0,
#         'AUC': round(sum(AUCs) / len(AUCs), 4) if AUCs else 0
#     }
#
#     # Prepare data for the DataTable
#     metrics_data = [{'Metric': metric, 'Average': value} for metric, value in avg_metrics.items()]
#     metrics_df = pd.DataFrame(metrics_data)
#
#     # Create the DataTable
#     metrics_table = dash_table.DataTable(
#         data=metrics_df.to_dict('records'),
#         columns=[{'name': 'Metric', 'id': 'Metric'}, {'name': 'Average', 'id': 'Average'}],
#         style_cell={'textAlign': 'center'},
#         style_data_conditional=[
#             {'if': {'column_id': 'Average'}, 'fontWeight': 'bold'}
#         ]
#     )
#
#     return confusion_matrices, metrics_table
#
#
# def load_data_traces(test_type, isImputed, df_isImputed_0):
#     filter_by_type = select_data_type(test_type, isImputed)
#     concat_tracesDF = pd.DataFrame()
#
#     # Load reconstructed traces
#     reconstructed_tracesDF = load_table_db('reconstructed_remodeling_view',
#                                            'reconstructed_type', '=', filter_by_type)
#
#     # Load clinical data (Sex, Age)
#     unique_patientID = reconstructed_tracesDF['patientID'].unique()
#     unique_patientID_list = unique_patientID.tolist()
#     unique_patientID_str = ', '.join(map(str, unique_patientID_list))
#     males_df, females_df, over_50_df, under_50_df = get_clinical_data(unique_patientID_str)
#
#     if not isImputed:
#         study_nr_df = df_isImputed_0
#         reconstructed_tracesDF = reconstructed_tracesDF[
#             reconstructed_tracesDF['study_nr'].isin(study_nr_df['study_nr'])]
#
#     if test_type == 'concat_18_segments_100' or test_type == 'concat_18_segments_95':
#         concat_tracesDF, feature_col_name = get_concat_18_segments(reconstructed_tracesDF)
#     elif test_type == 'gls_remod_class':
#         concat_tracesDF, feature_col_name = get_echoParams_data(concat_tracesDF, reconstructed_tracesDF, 'global_strain')
#     elif test_type == 'echoParams_remod_class':
#         concat_tracesDF, feature_col_name = get_echoParams_data(concat_tracesDF, reconstructed_tracesDF, None)
#     else:
#         concat_tracesDF, feature_col_name = get_feature_components(concat_tracesDF, df_isImputed_0, filter_by_type,
#                                                                    isImputed, reconstructed_tracesDF,None)
#     if 'patientID' not in concat_tracesDF.columns:
#         reconstructed_tracesDF_unique = reconstructed_tracesDF.drop_duplicates(subset='study_nr', keep='first')
#         concat_tracesDF = pd.merge(concat_tracesDF, reconstructed_tracesDF_unique[['study_nr', 'patientID']],
#                                    on='study_nr',
#                                    how='left')
#
#     return concat_tracesDF, feature_col_name, males_df, females_df, over_50_df, under_50_df, reconstructed_tracesDF
#
#
# def count_data(reconstructed_tracesDF):
#     reconstructed_tracesDF_unique = reconstructed_tracesDF.drop_duplicates(subset='study_nr')
#     data_count = reconstructed_tracesDF_unique['Total_Remodeling'].value_counts().reset_index(name='Count')
#
#     data_count.columns = ['Class', 'Count']
#     data_count_fig = px.bar(data_count, x='Class', y='Count',
#                             color={0: 'Non remodeling', 1: 'LV Remodeling'},
#                             text_auto=True
#                             )
#     return data_count_fig
#
#
# def extract_sex_data(females_df, males_df, pred_tracesDF, clf):
#
#     male_pred_df = pd.merge(pred_tracesDF, males_df, left_on='patientID', right_on='Patient_ID', how='left')
#     male_pred_df = male_pred_df.dropna(subset=['Patient_ID'])
#     male_pred_df['Patient_ID'] = male_pred_df['Patient_ID'].astype(int)
#     female_pred_df = pd.merge(pred_tracesDF, females_df, left_on='patientID', right_on='Patient_ID', how='left')
#     female_pred_df = female_pred_df.dropna(subset=['Patient_ID'])
#     female_pred_df['Patient_ID'] = female_pred_df['Patient_ID'].astype(int)
#
#     cm_male = confusion_matrix(male_pred_df['Total_Remodeling'], male_pred_df['prediction'])
#     cm_female = confusion_matrix(female_pred_df['Total_Remodeling'], female_pred_df['prediction'])
#
#     metrics_male = calculate_metrics(male_pred_df['Total_Remodeling'], male_pred_df['prediction'],
#                                      male_pred_df['pred_proba'],fold=None)
#
#     metrics_female = calculate_metrics(female_pred_df['Total_Remodeling'], female_pred_df['prediction'],
#                                        female_pred_df['pred_proba'],fold=None)
#
#     return cm_female, cm_male, metrics_male, metrics_female
#
#
# def get_clinical_data(unique_patientID_str):
#     clinical_dataDF_gender = pd.DataFrame()
#     clinical_dataDF = load_table_db(dbTableNames['clinical_Data_BL_T'], 'Patient_ID', 'in',
#                                     unique_patientID_str)
#     clinical_dataDF_gender['Gender'] = clinical_dataDF['Gender'].map(lambda x: 'Male' if x == 0 else 'Female')
#     no_subjects_sex = clinical_dataDF_gender['Gender'].value_counts()
#     males_df = clinical_dataDF[clinical_dataDF['Gender'] == 0]
#     females_df = clinical_dataDF[clinical_dataDF['Gender'] == 1]
#     age_counts = {
#         'over_50': (clinical_dataDF['Age'] >= 50).sum(),
#         'under_50': (clinical_dataDF['Age'] < 50).sum()
#     }
#     over_50_df = clinical_dataDF[clinical_dataDF['Age'] >= 50]
#     under_50_df = clinical_dataDF[clinical_dataDF['Age'] < 50]
#     return males_df, females_df, over_50_df, under_50_df
#
#
# def create_cm(confusion_matrices, instance_class, ifColumn):
#     if not ifColumn:
#         agg_confusion_matrix = np.sum(confusion_matrices, axis=0)
#         z = agg_confusion_matrix
#         x = instance_class.classes_.tolist()
#         y = instance_class.classes_.tolist()
#     else:
#         z = confusion_matrices
#         x = ['0', '1']
#         y = ['0', '1']
#     cm_fig = ff.create_annotated_heatmap(z, x=x, y=y, colorscale='Blues')
#     cm_fig.update_layout(
#         xaxis=dict(title='Predicted value'),
#         yaxis=dict(title='True value')
#     )
#     return cm_fig
#
#
# def classify_per_fold(concat_tracesDF, feature_col_name, fold, predictions_df, results,
#                       unbalanced_type, clf, all_y_pred, all_X_test):
#     for X_train, y_train, X_test, y_test, test_indices in split_data(concat_tracesDF, feature_col_name,
#                                                                      unbalanced_type):
#         fold += 1
#         clf.fit(X_train, y_train)
#         y_pred = clf.predict(X_test)
#         y_pred_proba = clf.predict_proba(X_test)[:, 1]
#         metrics = calculate_metrics(y_test, y_pred, y_pred_proba, fold)
#         results.append((y_test, y_pred, metrics))
#
#         fold_predictions = pd.DataFrame({
#             'study_nr': concat_tracesDF.iloc[test_indices]['study_nr'].values,
#             'prediction': y_pred,
#             'pred_proba': y_pred_proba
#
#         })
#
#         predictions_df = pd.concat([predictions_df, fold_predictions])
#
#         all_y_pred.extend(y_pred)
#         all_X_test.extend(X_test)
#
#     # final_predictions = predictions_df.groupby('study_nr')['prediction'].agg(lambda x: x.mode()[0]).reset_index()
#     final_predictions = predictions_df.groupby('study_nr').agg({
#         'prediction': lambda x: x.mode()[0],
#         'pred_proba': lambda x: x.mode()[0]
#     }).reset_index()
#     pred_tracesDF = pd.merge(concat_tracesDF, final_predictions[['study_nr', 'prediction','pred_proba']], on='study_nr',
#                              how='left')
#     return results, pred_tracesDF
