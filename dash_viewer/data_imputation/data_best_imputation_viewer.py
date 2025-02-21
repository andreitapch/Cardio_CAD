import ast
import random
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.impute import KNNImputer
import dash_bootstrap_components as dbc
from sklearn.model_selection import KFold
from dash import dcc, html, Input, Output, dash, callback

from utils.error_metrics import calculate_rrsse
from pattern_recognition.data.loadTraces import LoadTraces as load_traces
from pattern_recognition.data_preprocessing.speqle.concatenate_traces import CurveConcatenator


@callback(
    [Output('study_nr-dropdown', 'options'),
     Output('best-k-str-id', 'children'),
     Output('lowest_rrss-str-id', 'children')],
    Input('textarea-k', 'value'),
    Input('missing_traces_id', 'value'),
    Input('data-source-dropdown_bdi', 'value')
    # [State('study_nr-dropdown', 'value')]
)
def find_best_k(k_value, sample_n, data_source):
    global instance_imputation
    instance_imputation = DataImputationKBest()
    if k_value is not None and sample_n is not None and data_source is not None:
        data_source_mapping = {
            'Speqle': 'speqle_segmentalStrainInterp_T',
            'GE': 'st_segmentalStrainInterp_T',
            'Philips': 'phi_segmentalStrainInterp_T',
            'GE_Philips': ['st_segmentalStrainInterp_T', 'phi_segmentalStrainInterp_T']
        }

        if data_source in data_source_mapping:
            table_name = data_source_mapping[data_source]
        else:
            raise ValueError("Invalid data source provided")
        if data_source != 'GE_Philips':
            instance_imputation.load_data(table_name, data_source)
        else:
            instance_imputation.load_data(table_name, data_source)

        instance_imputation.find_best_k(k_value, sample_n)
        study_nr = np.sort(instance_imputation.imputed_df['study_nr'].unique())
        dropdown_options = [{'label': i, 'value': i} for i in study_nr]

        segment_word = "segment" if sample_n == 1 else "segments"
        best_K = f'Best K is for {sample_n} {segment_word}: {instance_imputation.best_k}'

        lowest_rrss = f' With lowest RRSSE:  {round(instance_imputation.lowest_rrss, 4)}'
        return dropdown_options, best_K, lowest_rrss

    else:
        # raise ValueError("Invalid data source provided")
        return dash.no_update, dash.no_update, dash.no_update


@callback(
    [Output('rrss-str-id', 'children'),
     Output('trace-graph', 'figure')],
    [Input('study_nr-dropdown', 'value'),
     Input('missing_traces_id', 'value')]
)
def update_output(study_nr, sample_n):
    if study_nr:
        test_df_nan = instance_imputation.complete_data_df.loc[
            instance_imputation.complete_data_df['study_nr'] == study_nr]
        test_df_nan.reset_index(drop=True, inplace=True)
        imputed_tracesDF = instance_imputation.trace_reconstruction(study_nr)
        if sample_n == 1:
            id_position = int(test_df_nan['id_position'][0])
            imputed_tracesDF = imputed_tracesDF.loc[(imputed_tracesDF['id_position'] == id_position)]
        else:
            id_position = test_df_nan['id_position'].astype(int).tolist()
            imputed_tracesDF = imputed_tracesDF.loc[imputed_tracesDF['id_position'].isin(id_position)]

        original_traceDF = instance_imputation.tracesDF_18.loc[
            (instance_imputation.tracesDF_18['study_nr'] == study_nr)]
        original_traceDF['id_position'] = original_traceDF['id_position'].astype(int)

        if sample_n == 1:
            original_traceDF = original_traceDF.loc[(original_traceDF['id_position'] == id_position)]
        else:
            original_traceDF = original_traceDF.loc[(original_traceDF['id_position'].isin(id_position))]

        fig = go.Figure()
        rrss_str = ''
        for i in range(len(imputed_tracesDF)):
            original_trace = original_traceDF['strain_trace'].values[i]
            original_trace = ast.literal_eval(original_trace)
            original_trace = list(map(float, original_trace))
            segment_name = imputed_tracesDF['curve_segment'].values[i]
            fig = plot_curve(fig, imputed_tracesDF.iloc[i], original_trace, segment_name, i)
            rrss = calculate_rrsse(original_trace, imputed_tracesDF['imputed_trace'].values[i])
            rrss_str += f'RRSSE for trace {i + 1} is: {round(rrss, 4)}\n'

        return rrss_str, fig

    return dash.no_update, dash.no_update


def plot_curve(fig, imputed_tracesDF, original_trace, segment_name, color_index):
    dark_colors = ['darkmagenta', 'darkred', 'darkgreen', 'darkblue', 'darkviolet', 'darkcyan']
    light_colors = ['magenta', 'lightcoral', 'lightgreen', 'lightblue', 'plum', 'lightcyan']

    imputed_trace = imputed_tracesDF['imputed_trace']
    fig.add_trace(go.Scatter(y=imputed_trace, mode='lines', name=f'Imputed {segment_name}', line=dict(color=dark_colors[color_index % len(dark_colors)], dash='dash')))
    fig.add_trace(go.Scatter(y=original_trace, mode='lines', name=f'Original {segment_name}', line=dict(color=light_colors[color_index % len(light_colors)])))
    fig.update_layout(title=f'Imputed Vs Original', title_x=0.5)
    return fig


def safe_eval(val):
    try:
        return eval(val)
    except Exception:
        return val


def replace_with_nan(strain_trace):
    strain_trace_list = safe_eval(strain_trace)
    return [np.nan] * len(strain_trace_list)


def concatenate_traces(tracesDF):
    concatenated_strain = CurveConcatenator(tracesDF, isReconstructed=False)
    concat_tracesDF = concatenated_strain.concatenate_traces_par()
    concat_tracesDF['concatenated_trace'] = np.array(concat_tracesDF['concatenated_trace'])
    return concat_tracesDF


def impute_train(X_test, X_test_nan, X_train, k, is_best_k):
    rrss_errors = []
    imputed_traces = []

    knn_impute = KNNImputer(n_neighbors=k)
    knn_impute.fit(X_train)
    for index, trace in enumerate(X_test_nan):
        trace = trace.reshape(1, -1)
        imputed_trace = knn_impute.transform(trace)
        if is_best_k:
            imputed_traces.append(imputed_traces)
        else:
            rrss = calculate_rrsse(X_test[index], imputed_trace)
            rrss_errors.append(rrss)

    return rrss_errors, imputed_traces


class DataImputationKBest:

    def __init__(self):
        self.complete_data_df = None
        self.imputed_tracesDF = None
        self.best_k, self.lowest_rrss = None, None
        self.concat_test_nan_df = None
        self.imputed_df = None
        self.imputed_trace = {}
        self.train_df, self.test_df, self.test_df_nan = None, None, None
        self.loaded_tracesDF, self.tracesDF = None, None
        self.concat_train_df, self.concat_test_df = None, None
        self.removed_rows = None
        self.tracesDF_18 = pd.DataFrame()
        self.feature_col_name = 'concatenated_trace'
        self.study_type = 'BL'

    def select_data_with_rows_per_study_nr(self, num_rows):
        counts = self.tracesDF.groupby('study_nr').size()
        study_nrs = counts[counts == num_rows].index
        filtered_df = self.tracesDF[self.tracesDF['study_nr'].isin(study_nrs)]
        return filtered_df

    def load_data(self, table_name, data_source):

        self.loaded_tracesDF = load_traces(self.study_type, table_name, 0, False)


        if data_source == 'GE_Philips':
            speqle_traces_ge = load_traces(self.study_type, table_name[0], 0, False)
            speqle_traces_ge.tracesDF['origin'] = 'GE'

            speqle_traces_philips = load_traces(self.study_type, table_name[1], 0, False)
            speqle_traces_philips.tracesDF['origin'] = 'Philips'

            self.loaded_tracesDF.tracesDF = pd.concat([speqle_traces_ge.tracesDF, speqle_traces_philips.tracesDF])
            self.loaded_tracesDF.tracesDF = self.loaded_tracesDF.tracesDF[
                ~self.loaded_tracesDF.tracesDF['segment_wall'].str.lower().str.contains('apex', na=False)]

        if data_source == 'Philips':
            self.loaded_tracesDF.tracesDF = self.loaded_tracesDF.tracesDF[
                ~self.loaded_tracesDF.tracesDF['segment_wall'].str.lower().str.contains('apex', na=False)]

        self.tracesDF = self.loaded_tracesDF.tracesDF.dropna(subset=['strain_trace'])
        self.tracesDF['id_position'] = self.tracesDF['id_position'].str.rstrip()
        self.tracesDF_18 = self.select_data_with_rows_per_study_nr(18)
        pass


    def trace_reconstruction(self, study_nr):
        start = 0
        imputed_traces = []
        imputed_trace = self.imputed_df.loc[self.imputed_df['study_nr'] == study_nr]
        imputed_trace[self.feature_col_name] = np.round(imputed_trace['imputed_trace'], decimals=6)

        curve = imputed_trace[self.feature_col_name].tolist()[0]
        curve_length = self.imputed_df['concat_curve_len'][0]
        curve_segment = self.imputed_df['concat_curve_segm'][0]
        id_segment = self.imputed_df['concat_id_segment'][0]

        for curve_len, curve_seg, id_seg in zip(curve_length, curve_segment, id_segment):
            curve_segment = curve[start:start + int(curve_len)]
            imputed_traces.append({'study_nr': study_nr,
                                   'curve_segment': curve_seg,
                                   'id_position': int(id_seg),
                                   'imputed_trace': curve_segment})
            start += int(curve_len)

        imputed_tracesDF = pd.DataFrame(imputed_traces)
        return imputed_tracesDF


    def find_best_k(self, k, sample_n):
        unique_study_nrs = self.tracesDF_18['study_nr'].unique()
        rrsse_errors_array = np.zeros((k, 5))
        kf = KFold(n_splits=5, shuffle=False)

        # Initialize the best k and lowest RRSS to track across all folds
        best_k = None
        lowest_rrss = float('inf')
        imputed_df = pd.DataFrame()  # Initialize an empty dataframe to store results

        for fold, (train_index, test_index) in enumerate(kf.split(unique_study_nrs), 1):
            train_study_nrs, test_study_nrs = unique_study_nrs[train_index], unique_study_nrs[test_index]
            self.prepare_traces(test_study_nrs, train_study_nrs, fold, sample_n)

            # Prepare the feature arrays once per fold
            X_train = np.stack(self.concat_train_df[self.feature_col_name].values)
            X_test_nan = np.stack(self.concat_test_nan_df[self.feature_col_name].values)
            X_test = np.stack(self.concat_test_df[self.feature_col_name].values)
            df_imputed = self.imputed_df.copy()

            for k_val in range(1, k + 1):
                imputedFold_df = df_imputed.copy()
                rrss_errors, imputed_traces = self.impute_train(X_test, X_test_nan, X_train, k_val)
                rrsse_errors_array[k_val - 1][fold - 1] = np.mean(rrss_errors)

                # Store imputed traces and parameters
                imputedFold_df['imputed_trace'] = imputed_traces.tolist()
                imputedFold_df['k'] = k_val
                imputedFold_df['fold'] = fold

                # Append to the final dataframe
                imputed_df = pd.concat([imputed_df, imputedFold_df], ignore_index=True)

        # Now, after all folds, find the best k
        for k_val in range(1, k + 1):
            mean_rrss = np.mean(rrsse_errors_array[k_val - 1, :])

            if mean_rrss < lowest_rrss:
                best_k = k_val
                lowest_rrss = mean_rrss

        # Final imputed dataframe with the best k
        self.best_k = best_k
        self.lowest_rrss = lowest_rrss
        mask = (imputed_df['k'] == self.best_k)
        imputed_df = imputed_df[mask]
        imputed_df.reset_index(drop=True, inplace=True)
        self.imputed_df = imputed_df



    def prepare_traces(self, test_study_nrs, train_study_nrs, fold, sample_n):
        self.train_df = self.tracesDF_18[self.tracesDF_18['study_nr'].isin(train_study_nrs)]
        self.concat_train_df = concatenate_traces(self.train_df)
        test_df = self.tracesDF_18[self.tracesDF_18['study_nr'].isin(test_study_nrs)]
        test_df_nan = test_df.copy()
        self.concatenated_with_nans(fold, sample_n, test_df, test_df_nan)

    def concatenated_with_nans(self, fold, sample_n, test_df, test_df_nan):

        if sample_n == 1:
            rows_to_replace = test_df.groupby('study_nr').sample(n=sample_n, random_state=1)
            pass
        elif sample_n == 3:
            test_df['segment_wall'] = test_df['segment_wall'].str.strip().str.lower()
            segment_wall = np.sort(test_df_nan['segment_wall'].str.lower().str.strip().unique())
            selected_segment = random.choice(segment_wall)
            rows_to_replace = test_df.loc[test_df['segment_wall'] == selected_segment]
            pass

        elif sample_n == 6:
            test_df['segment_wall'] = test_df['segment_wall'].str.strip().str.lower()
            cardiac_wall = np.sort(test_df_nan['cardiac_view'].unique())
            selected_cardiac_view = random.choice(cardiac_wall)
            rows_to_replace = test_df.loc[test_df['cardiac_view'] == selected_cardiac_view]
            pass
        else:
            rows_to_replace = None

        rows_to_replace['strain_trace'] = rows_to_replace['strain_trace'].apply(replace_with_nan)
        self.test_df_nan = rows_to_replace
        if fold == 1:
            self.complete_data_df = rows_to_replace.copy()
        else:
            self.complete_data_df = pd.concat([self.complete_data_df, rows_to_replace], ignore_index=True)
        test_df_nan.update(rows_to_replace)
        self.test_df = test_df
        self.concat_test_nan_df = concatenate_traces(test_df_nan)
        self.concat_test_df = concatenate_traces(test_df)
        self.imputed_df = self.concat_test_df.copy()
        self.imputed_df[self.feature_col_name] = np.nan
        self.imputed_df = self.imputed_df.rename(columns={self.feature_col_name: 'imputed_trace'})

    def impute_train(self, X_test, X_test_nan, X_train, k):
        rrss_errors = []
        knn_impute = KNNImputer(n_neighbors=k, weights='distance')
        knn_impute.fit(X_train)
        imputed_traces = knn_impute.transform(X_test_nan)
        for i in range(len(imputed_traces)):
            nan_indices = np.where(np.isnan(X_test_nan[i]))
            X_test_values = X_test[i][nan_indices]
            imputed_traces_values = imputed_traces[i][nan_indices]
            rrss = calculate_rrsse(X_test_values, imputed_traces_values)
            rrss_errors.append(rrss)
        return rrss_errors, imputed_traces

    def get_content(self):
        content = html.Div([

            dbc.Row([
                dbc.Col([
                    # html.Br(),
                    html.Br(),
                    dcc.Dropdown(
                        id='data-source-dropdown_bdi',
                        options=[
                            {'label': 'Speqle', 'value': 'Speqle'},
                            {'label': 'GE', 'value': 'GE'},
                            {'label': 'Philips', 'value': 'Philips'},
                            {'label': 'GE + Philips', 'value': 'GE_Philips'}
                        ],
                        placeholder='Select Data Source',
                        style={'width': '300px', 'marginLeft': '20px'}
                    ),
                    html.Br(),
                    html.Br(),
                    html.Label('Enter max K:', style={'fontWeight': 'bold', 'marginLeft': '20px'}),

                    dcc.Input(
                        id="textarea-k", type="number", debounce=True,
                        style={'width': '100px', 'marginLeft': '20px'}
                    ),

                    html.Br(),
                    html.Br(),

                    dbc.Row([
                        html.Label('Select number of missing segments:',
                                   style={'fontWeight': 'bold', 'marginLeft': '10px'}),
                        dcc.Dropdown(
                            id='missing_traces_id',
                            options=[
                                {'label': '1 Missing segment', 'value': 1},
                                # {'label': '2 Missing segment', 'value': 2},
                                {'label': '3 Missing segments', 'value': 3},
                                # {'label': '4 Missing segments', 'value': 3},
                                # {'label': '5 Missing segments', 'value': 3},
                                {'label': '6 Missing segments', 'value': 6}
                            ],
                            # value=1,
                            style={'width': '350px', 'marginLeft': '10px', 'backgroundColor': 'white'}
                        ),
                    ]),
                ], width=3),
                dbc.Col([
                    html.Br(),
                    html.Br(),
                    # html.Br(),
                    dbc.Card([
                        dbc.CardBody([
                            html.Div(id='best-k-str-id', children=' '),
                            html.Div(id='lowest_rrss-str-id', children=' '),
                        ], style={'padding': '10px', 'margin': '10px', 'width': '50%'}),
                    ]),
                ], width=5),

            ], style={'marginBottom': '20px'}),

            dbc.Card([
                dbc.Row([
                    html.Br(),
                    dbc.CardBody([
                        html.Label('Select Study Number:', style={'fontWeight': 'bold', 'marginLeft': '10px'}),
                        dcc.Dropdown(
                            id='study_nr-dropdown',
                            options=[],
                            style={'width': '200px', 'marginLeft': '10px'}
                        ),
                    ], style={'padding': '10px', 'margin': '10px', 'width': '80%'}),
                ]),

                dbc.Row([
                    html.Br(),
                    dbc.Col(
                        [
                            html.P("Segment trace", style={'fontWeight': 'bold', 'marginLeft': '10px'}),
                            dcc.Graph(id='trace-graph'),
                            html.Div(id='rrss-str-id', children=' ', style={'marginLeft': '10px'})
                        ],
                        width=6
                    ),
                ]),
            ]),
        ], style={'marginLeft': '20px', 'margin-right': '20px', 'margin-top': '20px', 'margin-bottom': '20px'})
        return content
