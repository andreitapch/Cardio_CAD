import numpy as np
import pandas as pd
import ast
from pattern_recognition.data.database_connection import sql_queries
from pattern_recognition.utils.parameters_dics import dbConParameters
from pattern_recognition.data.database_connection.db_con import DatabaseConnection as dbc
from pattern_recognition.data_preprocessing.speqle.concatenate_traces import CurveConcatenator
from pattern_recognition.data.SPEQLE_data.loadTracesSpeqle import LoadTracesSpeqle as load_traces_speqle


# def safe_eval(val):
#     try:
#         return eval(val)
#     except Exception:
#         return val
def safe_eval(val):
    try:
        return ast.literal_eval(val)
    except (ValueError, SyntaxError):
        return val

def load_segment_table_db(view_name, column_name, operator, value):
    try:
        dbConParams = dbConParameters
        db = dbc(driver=dbConParams['driver'], server=dbConParams['server'], database=dbConParams['database'])
        db.connect()
        db.execute_query(sql_queries.select_all_data_where(view_name, column_name, operator, value))
        df = db.df_query()
        db.close_connection()
        return df
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


def load_imputed_data(studyType):
    speqle = load_traces_speqle
    df = speqle(studyType, 'speqle_segmentalStrainInterp_T', None)
    tracesDF = df.tracesDF[['study_nr', 'curve_segment', 'isImputed']]
    df_isImputed_1 = tracesDF[tracesDF['isImputed'] == 1]['study_nr'].unique()
    df_isImputed_0 = tracesDF[tracesDF['isImputed'] == 0]
    unique_study_nr_0 = df_isImputed_0['study_nr'].unique()
    df_isImputed_0 = df_isImputed_0[~df_isImputed_0['study_nr'].isin(df_isImputed_1)]
    return tracesDF, df_isImputed_1, df_isImputed_0


def get_concat_18_segments(reconstructed_tracesDF):
    feature_col_name = 'concatenated_trace'
    concatenated_strain = CurveConcatenator(reconstructed_tracesDF, isReconstructed=True)
    concat_tracesDF = concatenated_strain.concatenate_traces_par()
    concat_tracesDF['concatenated_trace'] = np.array(concat_tracesDF['concatenated_trace'])
    reconstructed_tracesDF_unique = reconstructed_tracesDF.drop_duplicates(subset='study_nr')[
        ['Total_Remodeling', 'study_nr']]
    concat_tracesDF = pd.merge(concat_tracesDF,
                               reconstructed_tracesDF_unique[['Total_Remodeling', 'study_nr']],
                               on='study_nr',
                               how='left')
    return concat_tracesDF, feature_col_name


def get_echoParams_data(concat_tracesDF, reconstructed_tracesDF, is_gls_class):
    reconstructed_tracesDF_selected = (reconstructed_tracesDF.groupby('study_nr')['Total_Remodeling'].mean().
                                       astype(int).reset_index())
    unique_study_nr = reconstructed_tracesDF['study_nr'].unique()
    unique_study_nr_list = unique_study_nr.tolist()
    unique_study_nr_str = ', '.join(map(str, unique_study_nr_list))
    echo_paramDF = load_segment_table_db('echo_biomarkers_view',
                                         'study_nr', 'in', unique_study_nr_str)
    echo_paramDF = echo_paramDF[~echo_paramDF['View'].str.contains('RV')]
    if is_gls_class:
        echo_paramDF_grouped = echo_paramDF.groupby('study_nr')['Global Strain %'].mean().reset_index()
        echo_paramDF_grouped = pd.merge(echo_paramDF_grouped, reconstructed_tracesDF_selected, on='study_nr',
                                        how='left')
        feature_col_name = 'Global Strain %'
        concat_tracesDF = echo_paramDF_grouped
    else:
        pass
    return concat_tracesDF, feature_col_name


def select_data_type(knn_type, isImputed):
    global filter_by
    if knn_type == 'concat_18_segments_100' or knn_type == '100_variance_components':
        filter_by = 'PCA-full_traces-1'
    elif knn_type == 'concat_18_segments_95' or knn_type == '95_Variance_components':
        filter_by = 'PCA-full_traces-0.95'
    elif knn_type == '100_components_per_segment':
        filter_by = 'PCA-segment-1'
    elif knn_type == '95_components_per_segment':
        filter_by = 'PCA-segment-0.95'
    elif knn_type == 'gls_remod_class':
        filter_by = 'PCA-full_traces-1'
    elif knn_type == 'gls_remod_class':
        filter_by = 'PCA-echo_params-1'

    if isImputed:
        filter_by += "-1"
    else:
        filter_by += "-0"
    return filter_by


def get_pca_components(concat_tracesDF, df_isImputed_0, filter_by_type, isImputed, reconstructed_tracesDF):
    feature_col_name = 'pca_results'
    componentsDF = load_segment_table_db('pca_results_view', 'reconstructed_type'
                                         , '=', filter_by_type)
    componentsDF['pca_results'] = componentsDF['pca_results'].apply(safe_eval)
    if not isImputed:
        study_nr_df = df_isImputed_0
        componentsDF = componentsDF[componentsDF['study_nr'].isin(study_nr_df['study_nr'])]
    unique_reconstructed_traces = reconstructed_tracesDF[['study_nr', 'Total_Remodeling']].drop_duplicates(
        subset='study_nr')
    concat_tracesDF = componentsDF
    return concat_tracesDF, feature_col_name
