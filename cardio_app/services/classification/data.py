import json

import numpy as np
import pandas as pd
import ast
from pattern_recognition.data.database_connection import sql_queries
from pattern_recognition.utils.parameters_dics import dbConParameters, dbTableNames
from pattern_recognition.data.database_connection.db_con import DatabaseConnection as dbc
from pattern_recognition.data_preprocessing.speqle.concatenate_traces import CurveConcatenator
from pattern_recognition.data.loadTraces import LoadTraces as load_traces

from cardio_app.services.classification.engine import preprocess_data, \
    convert_echoParam_values, load_selected_features
from cardio_app.services.classification.outliers import remove_outliers_from_df, \
    remove_outliers_by_class_global


def safe_eval(val):
    try:
        return ast.literal_eval(val)
    except (ValueError, SyntaxError):
        return val


def load_queries_table_db(view_name, column_name, operator, value):
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


def get_component_options(recons_type, isImputed, table_name, conector, column_name):
    dm_params_df = load_queries_table_db(dbTableNames[table_name], 'reconstructed_type', conector,
                                         recons_type)
    # dm_params_df = dm_params_df[dm_params_df['isImputed'] == isImputed]
    components = sorted(dm_params_df[column_name].dropna().unique())
    components = [int(c) for c in components]
    options = [{'label': str(c), 'value': str(c)} for c in components]
    return options


# def get_autoenconder_component_options(recons_type, isImputed):
#     ac_params_df = load_queries_table_db(dbTableNames['calculated_dr_encoded_data_T'], 'reconstructed_type', 'in',
#                                           [f'%AutoencoderDM-{recons_type}'])
#     ac_params_df = ac_params_df[ac_params_df['isImputed'] == isImputed]
#     encodings = sorted(ac_params_df['encoding'].dropna().unique())
#     encodings = [int(c) for c in encodings]
#     options = [{'label': str(c), 'value': str(c)} for c in encodings]
#     return options

def load_data_traces(input_type, isImputed, unbalanced_type, class_type, data_source, encoding, addGLS, addFeatEcho,
                     pc_number, class_input):
    imp_feature_col_name, imp_concat_tracesDF = None, None
    orig_concat_tracesDF, imp_concat_tracesDF = None, None
    features = None

    # The reduction that produced the stored codes. The traces themselves always
    # come from the PCA reconstruction (that is what pca_results_view serves);
    # only the latent-code lookup differs, so the autoencoder key is built here
    # and the trace filtering below keeps using the PCA one.
    latent_prefix = 'AutoencoderDm' if class_input == 'autoencoder_class' else 'PCA'
    latent_filter_by_type = f'{latent_prefix}-{input_type}-{encoding}'
    filter_by_type = f'PCA-{input_type}-{encoding}'
    tracesDF = load_queries_table_db('pca_results_view', 'data_source', 'in', [data_source])
    tracesDF['strain_trace'] = tracesDF['strain_trace'].apply(lambda x: np.array(json.loads(x)))

    if isImputed and input_type:
        original_tracesDF = tracesDF[tracesDF['reconstructed_type'] == f'{filter_by_type}-0']
        imputed_tracesDF = tracesDF[tracesDF['reconstructed_type'] == f'{filter_by_type}-1']
    else:
        original_tracesDF, imputed_tracesDF = None, None
        tracesDF = tracesDF[
            tracesDF['reconstructed_type'].str.contains(
                input_type)]  ### change to original_tracesDF to simplify the code
        pass

    ## Strain Traces Classification
    if class_input in ['reconstructed_traces_vac', 'reconstructed_traces']:
        pc_number = None
        if imputed_tracesDF is None:
            orig_concat_tracesDF, orig_feature_col_name = get_concat_18_segments(tracesDF)
        else:
            orig_concat_tracesDF, orig_feature_col_name = get_concat_18_segments(original_tracesDF)
            imp_concat_tracesDF, imp_feature_col_name = get_concat_18_segments(imputed_tracesDF)

            features = imp_feature_col_name
            # imp_concat_tracesDF = remove_outliers_by_class_global(orig_concat_tracesDF, features, 'Total_Remodeling',
            #                                                           threshold=3.0, aggregation='mean')
            # features = orig_feature_col_name
            # orig_concat_tracesDF = remove_outliers_by_class_global(orig_concat_tracesDF, features, 'Total_Remodeling',
            #                                                        threshold=3.0, aggregation='mean')

    # GLS Classification
    elif class_input in ['gls_params_vac', 'gls_class']:
        features = 'global_strain'
        if imputed_tracesDF is None:
            orig_concat_tracesDF, orig_feature_col_name = get_echoParams_data(tracesDF, data_source, features)
        else:
            orig_concat_tracesDF, orig_feature_col_name = get_echoParams_data(original_tracesDF, data_source, features)
            imp_concat_tracesDF, imp_feature_col_name = get_echoParams_data(imputed_tracesDF, data_source, features)
            # imp_concat_tracesDF = remove_outliers_from_df(features, imp_concat_tracesDF, 2, 98)

        # orig_concat_tracesDF = remove_outliers_from_df(features, orig_concat_tracesDF, 2, 98)

    elif class_input in ['18_segments_vac', 'per_segment_vac']:
        table_name = 'calculated_dr_encoded_data_T'
        features = 'encoded_data'
        if imputed_tracesDF is None:
            orig_concat_tracesDF, orig_feature_col_name = get_feature_components(tracesDF, data_source,
                                                                                 filter_by_type, features,
                                                                                 table_name, None, pc_number)
        else:
            orig_concat_tracesDF, orig_feature_col_name = get_feature_components(original_tracesDF, data_source,
                                                                                 filter_by_type, features,
                                                                                 table_name, False, pc_number)
            imp_concat_tracesDF, imp_feature_col_name = get_feature_components(imputed_tracesDF, data_source,
                                                                               filter_by_type, features,
                                                                               table_name, True, pc_number)
    elif class_input in ['autoencoder_class']:
        # Same shape as the 'components' branch: one latent vector per study,
        # joined onto the labelled cohort. Only the table, the column and the
        # reconstructed_type prefix differ.
        table_name = 'calculated_dr_encoded_data_T'
        features = 'encoded_vector'

        if imputed_tracesDF is None:
            orig_concat_tracesDF, orig_feature_col_name = get_feature_components(
                tracesDF, data_source, latent_filter_by_type, features, table_name, None,
                pc_number, latent_column=features)
        else:
            orig_concat_tracesDF, orig_feature_col_name = get_feature_components(
                original_tracesDF, data_source, latent_filter_by_type, features, table_name,
                False, pc_number, latent_column=features)
            imp_concat_tracesDF, imp_feature_col_name = get_feature_components(
                imputed_tracesDF, data_source, latent_filter_by_type, features, table_name,
                True, pc_number, latent_column=features)

        features = orig_feature_col_name

    elif class_input in ['components', 'features_class']:
        table_name = 'calculated_pca_results_T'
        features = 'pca_results'

        if imputed_tracesDF is None:
            orig_concat_tracesDF, orig_feature_col_name = get_feature_components(tracesDF, data_source,
                                                                                 filter_by_type, features,
                                                                                 table_name, None, pc_number)
        else:
            orig_concat_tracesDF, orig_feature_col_name = get_feature_components(original_tracesDF, data_source,
                                                                                 filter_by_type, features,
                                                                                 table_name, False, pc_number)
            imp_concat_tracesDF, imp_feature_col_name = get_feature_components(imputed_tracesDF, data_source,
                                                                               filter_by_type, features,
                                                                               table_name, True, pc_number)
            pass
            # imp_concat_tracesDF = remove_outliers_by_class_global(orig_concat_tracesDF, features, 'Total_Remodeling',
            #                                                       threshold=3.0, aggregation='mean')

        features = orig_feature_col_name
        # orig_concat_tracesDF = remove_outliers_by_class_global(orig_concat_tracesDF, features, 'Total_Remodeling',
        #                                                        threshold=3.0, aggregation='mean')

    if class_input in ['features_class']:
        features = 'class_features'
        orig_feature_col_name = features
        if imputed_tracesDF is None:
            orig_concat_tracesDF = load_feature_params(orig_concat_tracesDF, features, original_tracesDF)
        else:
            orig_concat_tracesDF = load_feature_params(data_source, features, orig_concat_tracesDF)
            imp_concat_tracesDF = load_feature_params(data_source, features, imp_concat_tracesDF)
            pass

    col_to_keep = ['study_nr', orig_feature_col_name]
    if addGLS:
        features = 'gls'
        if imputed_tracesDF is None:
            echo_paramDF, echo_feature_col_name = get_echoParams_data(orig_concat_tracesDF, data_source, features)
            orig_concat_tracesDF = merge_df(echo_paramDF, orig_concat_tracesDF, col_to_keep)
        else:
            echo_paramDF, echo_feature_col_name = get_echoParams_data(orig_concat_tracesDF, data_source, features)
            orig_concat_tracesDF = merge_df(echo_paramDF, orig_concat_tracesDF, col_to_keep)

            echo_paramDF, echo_feature_col_name = get_echoParams_data(imp_concat_tracesDF, data_source, features)
            imp_concat_tracesDF = merge_df(echo_paramDF, imp_concat_tracesDF, col_to_keep)
            pass

    elif addFeatEcho:
        features = 'all_features'
        if imputed_tracesDF is None:
            echo_paramDF = load_feature_params(orig_concat_tracesDF, features, original_tracesDF)
            orig_concat_tracesDF = merge_df(echo_paramDF, orig_concat_tracesDF, col_to_keep)
            pass
        else:
            orig_echoParam_feat = load_feature_params(data_source, features, orig_concat_tracesDF)
            imp_echoParam_feat = load_feature_params(data_source, features, imp_concat_tracesDF)

            orig_concat_tracesDF = merge_df(orig_echoParam_feat, orig_concat_tracesDF, col_to_keep)
            imp_concat_tracesDF = merge_df(imp_echoParam_feat, imp_concat_tracesDF, col_to_keep)
            pass

    return orig_concat_tracesDF, orig_feature_col_name, imp_concat_tracesDF, imp_feature_col_name, features


def load_feature_params(data_source, features, tracesDF):
    echo_paramDF, echo_feature_col_name = get_echoParams_data(tracesDF, data_source, features)
    echoParam_tracesDF = convert_echoParam_values(echo_paramDF)
    # fit_preprocessing=False keeps missing values and raw scales: the KNN
    # imputer and the scaler are refit inside each CV fold by
    # build_estimator_pipeline() rather than once over every patient here.
    X_orig, y_orig, patient_study_info, scaler = preprocess_data(
        echoParam_tracesDF, True, fit_preprocessing=False)
    featuresDF, unique_features_list = load_selected_features()

    X_filtered = X_orig.loc[:, [col for col in X_orig.columns if col in unique_features_list]]
    echoParam_feat = X_filtered.copy()

    echoParam_feat.insert(0, 'study_nr', patient_study_info['study_nr'])
    echoParam_feat['Total_Remodeling'] = y_orig
    return echoParam_feat


def merge_df(echo_paramDF, orig_concat_tracesDF, cols_to_keep):
    orig_concat_tracesDF = orig_concat_tracesDF.loc[:, cols_to_keep]

    mergedDF = pd.merge(orig_concat_tracesDF, echo_paramDF, on='study_nr', how='inner')

    if 'Total_remodeling' in mergedDF.columns:
        mergedDF['Total_remodeling'] = mergedDF['Total_remodeling'].astype(int)

    return mergedDF


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


def get_echoParams_data(tracesDF, data_source, feature_col_name):
    if tracesDF is not None:
        unique_study_nr = tracesDF['study_nr'].unique()
        unique_study_nr_list = unique_study_nr.tolist()
        column_name = 'study_nr'
        operator = 'in'
        value = unique_study_nr_list
    else:
        column_name = None
        operator = None
        value = None

    echo_paramDF = load_queries_table_db('echo_biomarkers_view', column_name, operator, value)
    echo_paramDF['study_nr'] = echo_paramDF['study_nr'].astype(int)
    echo_paramDF = echo_paramDF.dropna(subset=['Total_Remodeling'])

    if feature_col_name == 'global_strain' or feature_col_name == 'gls':
        echo_paramDF = echo_paramDF.groupby('study_nr')[['global_strain', 'Total_Remodeling']].mean().reset_index()
        echo_paramDF = echo_paramDF.dropna(subset=['global_strain'])
    elif feature_col_name == 'class_features':
        echo_paramDF['Total_Remodeling'] = echo_paramDF['Total_Remodeling'].astype(int)
        return echo_paramDF, feature_col_name
    else:
        if tracesDF is not None:
            join_keys = ['study_nr']
            overlap_cols = [col for col in echo_paramDF.columns if col in tracesDF.columns and col not in join_keys]
            echo_paramDF_clean = echo_paramDF.drop(columns=overlap_cols)
            echo_paramDF = pd.merge(echo_paramDF_clean, tracesDF, on='study_nr', how='left')

    echo_paramDF['Total_Remodeling'] = echo_paramDF['Total_Remodeling'].astype(int)

    return echo_paramDF, feature_col_name


def select_data_type(input_type, encoding):
    global filter_by
    if input_type == 'concat_18_segments_100' or input_type == '100_variance_components':
        filter_by = 'PCA-full_traces'
        n_components = 1
    elif input_type == 'concat_18_segments_95' or input_type == '95_Variance_components':
        filter_by = 'PCA-full_traces'
        n_components = 0.95
    elif input_type == '100_components_per_segment':
        filter_by = 'PCA-segment'
        n_components = 1
    elif input_type == '95_components_per_segment':
        filter_by = 'PCA-segment'
        n_components = 0.95
    elif input_type == '1_PC_components':
        filter_by = 'PCA-full_traces'
        n_components = 1
    elif input_type == '3_PC_components':
        filter_by = 'PCA-full_traces'
        n_components = 3
    elif (input_type == 'reconstructed_traces_vac' or input_type == '18_segments_vac' or
          input_type == 'gls_params_vac' or input_type == 'echo_params-vac'):
        filter_by = 'AutoencoderDm-full_traces'
        n_components = encoding
    elif input_type == 'per_segment_vac':
        filter_by = 'AutoencoderDm-segment'
        n_components = encoding

    filter_by += f"-{n_components}"

    return filter_by


def get_feature_components(tracesDF, data_source, filter_by_type, feature_col_name, table_name, isImputed, pc_number,
                           latent_column='pca_results'):
    """Latent codes per study, joined onto the labelled cohort.

    latent_column defaults to 'pca_results' so every existing PCA caller is
    unchanged; the autoencoder table stores its codes in 'encoded_vector'.
    """
    if isImputed:
        filter_by_type = f'{filter_by_type}-1'
    elif isImputed == False:
        filter_by_type = f'{filter_by_type}-0'

    if isImputed is None:
        filter_by_type = filter_by_type

    componentsDF = load_queries_table_db(dbTableNames[table_name], 'reconstructed_type'
                                         , 'like', filter_by_type)

    if componentsDF is None or componentsDF.empty:
        raise ValueError(
            f"No rows in {table_name} matching '{filter_by_type}'. Run the reduction "
            f"for this configuration and save it before classifying on it.")

    componentsDF[latent_column] = componentsDF[latent_column].apply(lambda x: np.array(json.loads(x)))

    if pc_number is not None:
        componentsDF[latent_column] = componentsDF[latent_column].apply(
            lambda x: x[:pc_number] if len(x) >= pc_number else x)

    componentsDF = componentsDF[componentsDF['data_source'] == data_source]

    componentsDF[feature_col_name] = componentsDF[feature_col_name].apply(safe_eval)

    unique_reconstructed_traces = tracesDF[['study_nr', 'Total_Remodeling']].drop_duplicates()

    concat_tracesDF = pd.merge(unique_reconstructed_traces, componentsDF, on='study_nr', how='left')

    return concat_tracesDF, feature_col_name


def load_loadings(data_source, input_type, isImputed, n_components, table_name, n_segments=18):
    """Loadings as (n_components, 18 * n_timepoints), ready to split by segment.

    A full-trace run stores one basis fitted on all 18 segments at once, so its
    loading vector already spans them and is returned as-is.

    A segment run stores 18 separate bases, one per segment, keyed
    'segment-<name>'. Taking row [0] - which is what this did - silently picked
    one arbitrary segment and then let the caller split its 128 samples into 18
    "segments", producing a plot that looked plausible and meant nothing. The
    rows are instead stacked so that component i is every segment's own i-th
    component laid end to end, which is what the per-segment plot should show.

    Segments are ordered by name to match `segment_labels` in
    utils_clinical_relevance, which is alphabetical; the caller labels the
    split purely by position, so the two orderings have to agree.
    """
    loadingsDF = load_queries_table_db(dbTableNames[table_name], 'reconstructed_type'
                                     , 'like', input_type)

    loadingsDF = loadingsDF[loadingsDF['data_source'] == data_source]

    loadingsDF = loadingsDF[loadingsDF['n_components'].astype(int) == int(n_components)]

    if loadingsDF.empty:
        raise ValueError(
            f"No stored PCA loadings for '{input_type}' with {n_components} components "
            f"from {data_source}. Run the reduction and save it first.")

    loadingsDF = loadingsDF.sort_values(by='reconstructed_type')
    per_row = np.array(loadingsDF['components_'].apply(ast.literal_eval).tolist())

    if len(per_row) == 1:
        return per_row[0]

    if len(per_row) != n_segments:
        # Neither one whole-heart basis nor one per segment; refuse rather than
        # hand back a vector the caller will split into the wrong number of pieces.
        raise ValueError(
            f"Expected 1 or {n_segments} loading rows for '{input_type}' with "
            f"{n_components} components, found {len(per_row)} "
            f"({', '.join(loadingsDF['reconstructed_type'].astype(str)[:4])}...).")

    # (segments, components, timepoints) -> (components, segments * timepoints)
    return per_row.transpose(1, 0, 2).reshape(per_row.shape[1], -1)
