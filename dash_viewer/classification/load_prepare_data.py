import numpy as np
import pandas as pd
import ast
from pattern_recognition.data.database_connection import sql_queries
from pattern_recognition.utils.parameters_dics import dbConParameters, dbTableNames
from pattern_recognition.data.database_connection.db_con import DatabaseConnection as dbc
from pattern_recognition.data_preprocessing.speqle.concatenate_traces import CurveConcatenator
from pattern_recognition.data.loadTraces import LoadTraces as load_traces


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


def load_imputed_data(studyType, table_name, data_source):
    trace_loader = load_traces

    if data_source != 'GE_Philips':
        df = trace_loader(studyType, table_name, None).tracesDF
    else:
        df1 = trace_loader(studyType, table_name[0], None).tracesDF
        df2 = trace_loader(studyType, table_name[1], None).tracesDF
        if df1 is not None and df2 is not None:
            df = pd.concat([df1, df2], ignore_index=True)
        else:
            raise ValueError("Failed to load traces for GE_Philips data source")
    df = df[~df['curve_segment'].str.lower().str.contains('apex', na=False)]
    tracesDF = df[['study_nr', 'isImputed']]

    # To get the unique study_nr for the imputed and original data
    study_nr_imputed = tracesDF.loc[tracesDF['isImputed'] == True, 'study_nr'].unique()
    df_isImputed_1 = tracesDF.loc[tracesDF['study_nr'].isin(study_nr_imputed)]
    df_isImputed_0 = tracesDF.loc[~tracesDF['study_nr'].isin(study_nr_imputed)]
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


# def get_gls_data(tracesDF, data_source):
#     filtered_tracesDF = (tracesDF.groupby('study_nr')['Total_Remodeling'].mean().
#                                        astype(int).reset_index())
#     # filtered_tracesDF = filtered_tracesDF[filtered_tracesDF['data_source'] == data_source]
#
#     unique_study_nr = filtered_tracesDF['study_nr'].unique()
#     unique_study_nr_list = unique_study_nr.tolist()
#
#     gls_paramDF = load_segment_table_db('echo_biomarkers_view','study_nr', 'in', unique_study_nr_list)
#     # echo_paramDF = echo_paramDF[~echo_paramDF['View'].str.contains('RV')]
#
#     gls_paramDF_grouped = gls_paramDF.groupby('study_nr')['global_strain'].mean().reset_index()
#     gls_paramDF_grouped = pd.merge(gls_paramDF_grouped, filtered_tracesDF, on='study_nr',
#                                     how='left')
#     feature_col_name = 'global_strain'
#
#     return gls_paramDF_grouped, feature_col_name

def get_echoParams_data(tracesDF, data_source, feature_col_name):
    filtered_tracesDF = (tracesDF.groupby('study_nr')['Total_Remodeling'].mean().
                         astype(int).reset_index())

    unique_study_nr = filtered_tracesDF['study_nr'].unique()
    unique_study_nr_list = unique_study_nr.tolist()

    echo_paramDF = load_segment_table_db('echo_biomarkers_view', 'study_nr', 'in', unique_study_nr_list)
    echo_paramDF['study_nr'] = echo_paramDF['study_nr'].astype(int)
    echo_paramDF['patientID'] = echo_paramDF['patientID'].astype(int)

    if feature_col_name == 'global_strain':
        echo_paramDF = echo_paramDF.groupby('study_nr')['global_strain'].mean().reset_index()

    echo_paramDF = pd.merge(echo_paramDF, filtered_tracesDF, on='study_nr',
                                how='left')
    return echo_paramDF, feature_col_name


def select_data_type(knn_type, encoding):
    global filter_by
    if knn_type == 'concat_18_segments_100' or knn_type == '100_variance_components':
        filter_by = 'PCA-full_traces'
        n_components = 1
    elif knn_type == 'concat_18_segments_95' or knn_type == '95_Variance_components':
        filter_by = 'PCA-full_traces'
        n_components = 0.95
    elif knn_type == '100_components_per_segment':
        filter_by = 'PCA-segment'
        n_components = 1
    elif knn_type == '95_components_per_segment':
        filter_by = 'PCA-segment'
        n_components = 0.95
    elif knn_type == 'gls_remod_class':
        filter_by = 'PCA-full_traces'
        n_components = 1
    elif knn_type == 'gls_remod_class_vac':
        filter_by = 'AutoencoderDm-full_traces'
        n_components = encoding
    elif knn_type == 'echo-params_vac':
        filter_by = 'echo-params_vac'
        n_components = encoding
    elif knn_type == 'reconstructed_traces_vac':
        filter_by = 'AutoencoderDm-full_traces'
        n_components = encoding
    elif knn_type == '18_segments_vac':
        filter_by = 'AutoencoderDm-full_traces'
        n_components = encoding
    elif knn_type == 'per_segment_vac':
        filter_by = 'AutoencoderDm-per_segment'
        n_components = encoding

    filter_by += f"-{n_components}"

    return filter_by


def get_feature_components(tracesDF, data_source, filter_by_type, feature_col_name, table_name):
    componentsDF = load_segment_table_db(dbTableNames[table_name], 'reconstructed_type'
                                         , 'like', filter_by_type)
    componentsDF = componentsDF[componentsDF['data_source'] == data_source]

    componentsDF[feature_col_name] = componentsDF[feature_col_name].apply(safe_eval)

    unique_reconstructed_traces = tracesDF[['study_nr', 'Total_Remodeling']].drop_duplicates()
    # unique_reconstructed_traces = unique_reconstructed_traces[unique_reconstructed_traces['data_source'] == data_source]

    concat_tracesDF = pd.merge(unique_reconstructed_traces, componentsDF, on='study_nr', how='left')

    return concat_tracesDF, feature_col_name
