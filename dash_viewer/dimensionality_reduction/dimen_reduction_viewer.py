import ast
from dash import dcc, html, Input, Output, dash, State, callback, ctx
import numpy as np
import pandas as pd
import plotly.graph_objects as go

import dash_bootstrap_components as dbc
from sklearn.preprocessing import StandardScaler
from pattern_recognition.data.database_connection import sql_queries
from pattern_recognition.data.insert_to_db.insertSpeqleDB import InsertInterpTracesToDB
from pattern_recognition.data_preprocessing.speqle.concatenate_traces import CurveConcatenator
from pattern_recognition.utils.parameters_dics import dbConParameters
from pattern_recognition.data.database_connection.db_con import DatabaseConnection as dbcon
from pattern_recognition.utils.plot_dics import segment_color_dict
from pattern_recognition.data.SPEQLE_data.loadTracesSpeqle import LoadTracesSpeqle as load_traces_speqle
from dash_viewer.dimensionality_reduction.pca_viewer import calculate_pca, reconstruct_traces_pca


def safe_eval(val):
    try:
        return eval(val)
    except Exception:
        return val


@callback(
    Output('study_nr-dropdown_dr', 'options'),
    Input('isImputed-id', 'value'),
    State('study_nr-dropdown_dr', 'value'),
    State('url', 'pathname')
)
def update_dropdown(isImputed, study_nr, pathname):
    global instance_dm
    instance_dm = DimensionalityRedViewer(pathname)
    instance_dm.load_study_nr_dropdown(isImputed)
    dropdown_options = [{'label': i, 'value': i} for i in instance_dm.study_nr]
    return dropdown_options


@callback(
    [Output('pca-graph', 'figure'),
     Output('origin_trace', 'figure')],
    [Input('pca-button', 'n_clicks'),
     Input('savetoDB-button', 'n_clicks'),
     Input('study_nr-dropdown_dr', 'value')],
    [State('concat_traces_by-dropdown', 'value'),
     State('id-slider', 'value'),
     State('url', 'pathname')]
)
def update_pca_graph(pca_btn, insert_btn, study_nr, concat_traces_by, n_components, pathname):
    if "pca-button" == ctx.triggered_id and pca_btn > 0:
        fig_reconstructed, fig_original = instance_dm.trigger_callbacks(concat_traces_by, study_nr, n_components)
        return fig_reconstructed, fig_original
    # elif "savetoDB-button" == ctx.triggered_id and insert_btn > 0:
    #     handle_db_click(pathname, concat_traces_by)
    #     result = dash.no_update, dash.no_update
    #     return dash.no_update
    else:
        empty_figure = go.Figure()
        return empty_figure, empty_figure


@callback(
    Output('id-data-txt', 'children'),
    Input('isImputed-id', 'value'),
    Input('savetoDB-button', 'n_clicks'),
    [State('concat_traces_by-dropdown', 'value'),
     State('url', 'pathname')]
)
def handle_db_click(isImputed, insert_btn, concat_traces_by, data_type):
    if "savetoDB-button" == ctx.triggered_id and insert_btn > 0:
        isImputed_value = 1 if isImputed else 0
        print('Saving to DB')
        pca_result_df = pd.DataFrame()
        data_type = data_type.split('-')[-1]

        # DeleteToDB('calculated_reconstructedTraces_T')
        # DeleteToDB('calculated_pca_results_T')
        # DeleteToDB('calculated_speqle_pca_T')

        # UpdateTracesToDB('calculated_reconstructedTraces_T')
        # UpdateTracesToDB('calculated_speqle_pca_T')
        # UpdateTracesToDB('calculated_speqle_pca_T')

        recons_traces_df = instance_dm.DM_tracesDF
        # recons_traces_df = recons_traces_df.drop(columns=['index'])
        recons_traces_df['strain_trace'] = recons_traces_df['strain_trace'].apply(
            lambda x: x.tolist() if isinstance(x, np.ndarray) else x)

        recons_traces_df['study_type'] = 'BL'
        recons_traces_df['reconstructed_type'] = (data_type + '-' + concat_traces_by +
                                                  '-' + str(instance_dm.n_components) +
                                                  '-' + str(isImputed_value))

        recons_traces_df = recons_traces_df.reset_index(drop=True)

        InsertInterpTracesToDB(recons_traces_df, 'calculated_reconstructedTraces_T', True)

        # Insert values on  calculated_pca_results_T table
        unique_study_nr = instance_dm.DM_tracesDF['study_nr'].drop_duplicates()

        data = []
        for i, row in enumerate(instance_dm.pca_result):
            row_string = np.array2string(row, separator=', ', precision=16, suppress_small=True)
            data.append({'study_nr': unique_study_nr.values[i], '1': row_string})

        pca_result_df = pd.DataFrame(data)
        pca_result_df['reconstructed_type'] = (data_type + '-' + concat_traces_by +
                                               '-' + str(instance_dm.n_components) +
                                               '-' + str(isImputed_value))

        pca_result_df = pca_result_df.reset_index(drop=True)
        InsertInterpTracesToDB(pca_result_df, 'calculated_pca_results_T', False)

        components_str = np.array2string(instance_dm.pca_95_var.components_, separator=', ', precision=16,
                                         suppress_small=True)
        explained_variance_str = np.array2string(instance_dm.pca_95_var.explained_variance_, separator=', ',
                                                 precision=16,
                                                 suppress_small=True)
        explained_variance_ratio_str = np.array2string(instance_dm.pca_95_var.explained_variance_ratio_, separator=', ',
                                                       precision=16, suppress_small=True)
        mean_str = np.array2string(instance_dm.pca_95_var.mean_, separator=', ', precision=16, suppress_small=True)
        singular_values_str = np.array2string(instance_dm.pca_95_var.singular_values_, separator=', ', precision=16,
                                              suppress_small=True)

        pca_variance = {
            'components_': components_str,
            'explained_variance_': explained_variance_str,
            'explained_variance_ratio_': explained_variance_ratio_str,
            'mean_': mean_str,
            'n_components_': instance_dm.pca_95_var.n_components_,
            'n_features_': instance_dm.pca_95_var.n_features_in_,
            'singular_values_': singular_values_str
        }
        pca_var_transformed = {key: value.tolist() if isinstance(value, np.ndarray) else [value] for key, value in
                               pca_variance.items()}

        pca_var_df = pd.DataFrame(pca_var_transformed)
        pca_var_df['reconstructed_type'] = (data_type + '-' + concat_traces_by +
                                            '-' + str(instance_dm.n_components) +
                                            '-' + str(isImputed_value))
        pca_var_df = pca_var_df.reset_index(drop=True)
        InsertInterpTracesToDB(pca_var_df, 'calculated_speqle_pca_T', True)
        print('Saving completed....')
        txt_save = f' Inserting to DB completed'
        return txt_save
    return dash.no_update


def plot_curve(trace, study_nr, n_components, traceType):
    fig = go.Figure()
    title_str = ''
    trace = trace.reset_index(drop=True)
    for idx_curve, curve in enumerate(trace['strain_trace']):
        segment_name = trace['curve_segment'][idx_curve]
        segment_name = segment_name.strip()
        if traceType == 'Interpolated':
            curve = ast.literal_eval(curve)
            curve = list(map(float, curve))
            title_str = f'{traceType} Strain Curve - Study {study_nr}'
        else:
            title_str = (f'{traceType} Strain Curve - Study {study_nr} using {n_components} components of the '
                         f'explained variance')

        fig.add_trace(go.Scatter(y=curve, mode='lines', name=segment_name,
                                 line=dict(color=segment_color_dict[segment_name]), ))

    fig.update_layout(title=title_str, title_x=0.5,
                      autosize=False, height=600)
    return fig


def load_table_db(view_name, column_name, operator, value):
    try:
        dbConParams = dbConParameters
        db = dbcon(driver=dbConParams['driver'], server=dbConParams['server'], database=dbConParams['database'])
        db.connect()
        db.execute_query(sql_queries.select_all_data_where(view_name, column_name, operator, value))
        df = db.df_query()
        db.close_connection()
        return df
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


class DimensionalityRedViewer:
    def __init__(self, pathname):
        self.concat_echo_params = None
        self.df = None
        self.isImputed = None
        self.data_type = None
        self.pca_model = None
        self.studyType = 'BL'
        # self.explained_var_list = []
        self.n_components = None
        self.pathname = pathname
        self.array_curves, self.pca_result, self.pca_95_var, self.tracesDF = None, None, None, None
        self.explained_var_str, self.PCA_rec_tracesDF, self.reconstructed_tracesDF = None, None, None
        self.calculate_PCA, self.DM_tracesDF, self.array_curves, self.study_nr = None, None, None, None
        self.concat_traces_by, self.segment_tracesDF, self.concat_tracesDF, self.content = None, None, None, None

    def load_echo_params(self):
        unique_study_nr_list = self.study_nr.tolist()
        unique_study_nr_str = ', '.join(map(str, unique_study_nr_list))
        echo_params_DF = load_table_db('echo_biomarkers_view', 'study_nr', 'in', unique_study_nr_str)
        echo_paramsDF = echo_params_DF[~echo_params_DF['View'].str.contains('RV')]
        echo_paramsDF_ = echo_paramsDF.dropna(axis=1, how='all')
        study_nr = echo_paramsDF['study_nr']
        df_to_concat = echo_paramsDF.drop(['study_nr', 'View', 'Segment'], axis=1)

        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(df_to_concat)
        data_scaled_df = pd.DataFrame(data_scaled, columns=df_to_concat.columns)
        data_scaled_df = pd.concat([study_nr, data_scaled_df], axis=1)
        self.concat_echo_params = data_scaled_df
        self.concat_echo_params['study_nr'] = self.concat_echo_params['study_nr'].astype(int)

        # echo_paramsDF['concatenated'] = data_scaled_df.apply(lambda row: ' '.join(row.values.astype(str)), axis=1)
        # self.concat_echo_params = echo_paramsDF[['study_nr', 'concatenated']]
        pass

    def load_study_nr_dropdown(self, isImputed):
        self.isImputed = isImputed
        speqle = load_traces_speqle
        self.df = speqle(self.studyType, 'speqle_segmentalStrainInterp_T', self.isImputed)
        max_length = self.df.tracesDF.groupby('study_nr').size().max()
        dropna_tracesDF = self.df.tracesDF.dropna(subset=['strain_trace'])
        dropna_tracesDF = dropna_tracesDF[
            ~(dropna_tracesDF['strain_trace'].apply(lambda x: np.all(np.array(x) == 0)))]
        self.tracesDF = dropna_tracesDF.groupby('study_nr').filter(lambda x: len(x) == max_length)
        self.study_nr = self.tracesDF['study_nr'].unique()

    def load_pca_data(self):
        if self.concat_traces_by == 'full_traces' or self.concat_traces_by == 'full_traces_echo':
            self.prepare_all_traces()
        elif self.concat_traces_by == 'segment' or self.concat_traces_by == 'segment_echo':
            self.prepare_segment_traces()

    def prepare_all_traces(self):
        concatenated_strain = CurveConcatenator(self.tracesDF, isReconstructed=False)
        self.concat_tracesDF = concatenated_strain.concatenate_traces_par()
        curve = self.concat_tracesDF['concatenated_trace'].tolist()
        self.array_curves = np.array(curve)
        # if self.concat_traces_by == 'full_traces_echo':
        #     self.concat_study_traces = self.concat_tracesDF[['study_nr', 'concatenated_trace']]

    def prepare_segment_traces(self):
        self.DM_tracesDF = pd.DataFrame()
        self.tracesDF['curve_segment'] = self.tracesDF['curve_segment'].str.strip()
        unique_curve_segments = (self.tracesDF['curve_segment'].unique())

        for curve_segment in unique_curve_segments:
            self.segment_tracesDF = self.tracesDF[
                self.tracesDF['curve_segment'] == curve_segment].reset_index(drop=True)
            segment_trace = self.segment_tracesDF['strain_trace'].apply(ast.literal_eval)
            segment_trace = segment_trace.round(6)
            self.array_curves = np.array(segment_trace.tolist())

            if self.pathname == '/tab-PCA':
                if self.concat_traces_by == 'segment_echo':
                    self.load_echo_params()
                    # merged_st_echoParam_df = pd.merge(self.concat_echo_params, self.concat_study_traces, on='study_nr')
                    # merged_st_echoParam_df['concatenated'] = merged_st_echoParam_df.apply(lambda row: ' '.join(row.values.astype(str)),
                    #                                                      axis=1)
                    pass

                    self.pca_result, self.pca_95_var = calculate_pca(self.n_components, self.array_curves)
                else:
                    self.pca_result, self.pca_95_var = calculate_pca(self.n_components, self.array_curves)
            elif self.pathname == '/tab-Autoencoder':
                pass

            self.reconstruct_traces(curve_segment)
            self.DM_tracesDF = pd.concat([self.DM_tracesDF, self.PCA_rec_tracesDF], ignore_index=True)
        self.DM_tracesDF = self.DM_tracesDF.sort_values(by='study_nr').reset_index()
        self.DM_tracesDF = self.DM_tracesDF.drop(columns=['index'])
        pass

    def reconstruct_traces(self, curve_segment):
        start = 0
        pca_traces = []
        if self.pathname == '/tab-PCA':
            reconstructed_curves = reconstruct_traces_pca(self.pca_result, self.pca_95_var)
        elif self.pathname == '/tab-Autoencoder':
            pass

        rec_pca_curves = np.around(reconstructed_curves, decimals=6)

        for idx_rec, curve in enumerate(rec_pca_curves):
            if self.concat_traces_by == 'full_traces' or self.concat_traces_by == 'full_traces_echo':
                study_nr = self.study_nr[idx_rec]
                self.concatenated_traces(start, idx_rec, curve, study_nr, pca_traces)

            elif self.concat_traces_by == 'segment' or self.concat_traces_by == 'segment_echo':
                study_nr = self.segment_tracesDF['study_nr'][idx_rec]
                concatenated_dic = self.segment_traces(idx_rec, curve, study_nr, pca_traces, curve_segment)
                self.PCA_rec_tracesDF = pd.DataFrame(concatenated_dic)

    def concatenated_traces(self, start, idx_rec, curve, study_nr, pca_traces):
        self.DM_tracesDF = pd.DataFrame()
        original_curve_len = self.concat_tracesDF['concat_curve_len'][idx_rec]
        original_curve_seg = self.concat_tracesDF['concat_curve_segm'][idx_rec]
        original_id_seg = self.concat_tracesDF['concat_id_segment'][idx_rec]

        for curve_len, curve_seg, id_seg in zip(original_curve_len, original_curve_seg, original_id_seg):
            curve_segment = curve[start:start + int(curve_len)]
            pca_traces.append({'study_nr': study_nr,
                               'curve_segment': curve_seg,
                               'id_position': id_seg,
                               'strain_trace': curve_segment})
            start += int(curve_len)
        self.DM_tracesDF = pd.DataFrame(pca_traces)
        pass

    def segment_traces(self, idx_rec, curve, study_nr, pca_traces, curve_segment):
        pca_traces.append({'study_nr': study_nr,
                           'curve_segment': curve_segment,
                           'id_position': int(self.segment_tracesDF['id_position'][idx_rec]),
                           'strain_trace': curve})
        return pca_traces

    def get_content(self):
        return html.Div([
            html.H3('PCA Computation'),
            dbc.Row([
                # dbc.Col(
                html.Label('Select PCA Concatenation Type:', style={'fontWeight': 'bold'}),
                dcc.Dropdown(
                    id='concat_traces_by-dropdown',
                    options=[
                        {'label': 'Full Traces', 'value': 'full_traces'},
                        {'label': 'Segment', 'value': 'segment'}
                    ],
                    value='full_traces',
                    style={'width': '400px', 'backgroundColor': 'white'}
                ),
                html.Br(),
                html.Br(),
                dbc.Checkbox(
                    id='isImputed-id',
                    label='Include imputed data',
                    value=True,
                    style={'marginLeft': '10px'}
                ),
                html.Br(),
                html.Label('Select Study Number:', style={'fontWeight': 'bold'}),
                dcc.Dropdown(
                    id='study_nr-dropdown_dr',
                    # options=[{'label': i, 'value': i} for i in self.study_nr],

                    # value=self.study_nr[0] if self.study_nr.size > 0 else None,
                    style={'width': '200px'}
                )
            ]),
            html.Br(),
            html.P("Explained variance:"),
            dcc.Slider(
                id='id-slider',
                min=0, max=1, value=0.95,
                step=0.05
            ),

            html.Br(),

            html.Button(
                'Perform PCA', id='pca-button', style={'width': '200px'}, n_clicks=0),
            html.Br(),
            dbc.Row([
                dbc.Col(dcc.Graph(id='pca-graph'), width=6),
                dbc.Col(dcc.Graph(id='origin_trace'), width=6)
            ]), html.Br(),
            html.Button(
                'Save', id='savetoDB-button', style={'width': '200px', 'marginLeft': '10px'}, n_clicks=0),
            html.Br(),
            html.Div(id='id-data-txt', children=' ', style={'marginLeft': '10px'})

        ],
            style={'marginLeft': '20px'}
        )

    def trigger_callbacks(self, concat_traces_by, study_nr, n_components):
        self.concat_traces_by = concat_traces_by
        self.n_components = n_components
        self.load_pca_data()

        if concat_traces_by == 'full_traces' or concat_traces_by == 'full_traces_echo':
            if self.pathname == '/tab-PCA':
                self.data_type = 'PCA'
                # self.explained_var_list = calculate_n_components_var(None, self.array_curves, self.concat_traces_by)
                if concat_traces_by == 'full_traces_echo':
                    self.load_echo_params()
                    # concat_traces = self.concat_echo_params['concatenated'].apply(safe_eval)

                    # self.concat_tracesDF
                    merged_st_echoParam_df = pd.merge(self.concat_echo_params, self.concat_tracesDF, on='study_nr')

                    merged_st_echoParam_df['concatenated_all'] = merged_st_echoParam_df.apply(
                        lambda row: ' '.join([str(row['concatenated']), str(row['concatenated_trace'])]), axis=1)

                    curve = merged_st_echoParam_df['concatenated_all'].tolist()
                    array_curves = np.array(curve)
                    # array_curves = np.array(curve).astype(float)
                    self.pca_result, self.pca_95_var = calculate_pca(self.n_components, array_curves)
                    pass
                else:
                    self.pca_result, self.pca_95_var = calculate_pca(self.n_components, self.array_curves)

            elif self.pathname == '/tab-Autoencoder':
                self.data_type = 'Autoencoder'
                # autoencoder = AutoencoderViewer(self.array_curves)
                # autoencoder.calculate_autoencoder(encoding_dim=32)
                # reconstructed_curves = autoencoder.reconstruct_traces()
            self.reconstruct_traces(self.n_components)

        filtered_df = self.DM_tracesDF[self.DM_tracesDF['study_nr'] == study_nr]
        original_df = self.tracesDF[self.tracesDF['study_nr'] == study_nr]
        fig_reconstructed = plot_curve(filtered_df, study_nr, n_components, 'Reconstructed')
        fig_original = plot_curve(original_df, study_nr, n_components, 'Interpolated')
        return fig_reconstructed, fig_original
