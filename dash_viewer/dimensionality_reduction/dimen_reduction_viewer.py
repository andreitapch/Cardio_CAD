import ast
import json

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
from pattern_recognition.data.loadTraces import LoadTraces
from dash_viewer.dimensionality_reduction.AutoencoderDm import calculate_autoencoder, \
    reconstruct_traces_autoencoder
from dash_viewer.dimensionality_reduction.pca_viewer import calculate_pca, reconstruct_traces_pca

@callback(
    [Output('study_nr-dropdown_dr', 'options')],
    [Input('pca-button', 'n_clicks')],
    [State('data-source-dropdown_dr', 'value'),
     State('concat_traces_by-dropdown', 'value'),
     State('url', 'pathname'),
     State('id-slider-dm', 'value')]
)
def update_dropdown_and_load_study_nrs(pca_btn, data_source, concat_traces_by, pathname, n_components):
    dropdown_options = []
    if "pca-button" == ctx.triggered_id and pca_btn > 0:
        global instance_dm
        data_source_mapping = {
            'Speqle': 'speqle_segmentalStrainInterp_T',
            'GE': 'st_segmentalStrainInterp_T',
            'Philips': 'phi_segmentalStrainInterp_T',
            'GE_Philips': ['phi_segmentalStrainInterp_T', 'st_segmentalStrainInterp_T']
        }
        if data_source in data_source_mapping:
            table_name = data_source_mapping[data_source]
        else:
            table_name = None
            raise ValueError("Invalid data source provided")

        instance_dm = DimensionalityRedViewer(pathname)
        instance_dm.data_source = data_source
        instance_dm.table_name = table_name
        instance_dm.load_study_nr_dropdown()
        instance_dm.trigger_callbacks(concat_traces_by, n_components)
        dropdown_options = [{'label': i, 'value': i} for i in instance_dm.study_nr]

        return [dropdown_options]

    return dash.no_update



@callback(
    [Output('pca-graph', 'figure'),
     Output('origin_trace', 'figure')],
    [Input('study_nr-dropdown_dr', 'value'),
    State('url', 'pathname')]
)
def plot_traces_dm(study_nr, data_type):
    if study_nr is not None:
        data_type = data_type.split('-')[-1]
        if data_type == 'PCA':
            n_components = instance_dm.n_components_used
        else:
            n_components = instance_dm.n_components
        reconstructed_df = instance_dm.DM_tracesDF[instance_dm.DM_tracesDF['study_nr'] ==
                                                   study_nr].sort_values(by='id_position')

        original_df = instance_dm.tracesDF[instance_dm.tracesDF['study_nr'] ==
                                           study_nr].sort_values(by='id_position')

        fig_original = plot_curve(original_df, study_nr, n_components, 'Interpolated',
                                  instance_dm.pathname)
        fig_reconstructed = plot_curve(reconstructed_df, study_nr, n_components, 'Reconstructed',
                                       instance_dm.pathname)

        return fig_reconstructed, fig_original
    else:
        return go.Figure(), go.Figure()


@callback(
    Output('id-data-txt', 'children'),
    Input('savetoDB-button', 'n_clicks'),
    [State('concat_traces_by-dropdown', 'value'),
     State('url', 'pathname')]
)
def handle_db_click(insert_btn, concat_traces_by, data_type):

    if "savetoDB-button" == ctx.triggered_id and insert_btn > 0:
        print('Saving to DB')
        pathname = 1
        pca_result_df = pd.DataFrame()
        data_type = data_type.split('-')[-1]
        recons_traces_df = instance_dm.DM_tracesDF.copy()
        data_source = instance_dm.data_source
        if data_type == 'PCA':
            n_components = instance_dm.n_components_used
        else:
            n_components = instance_dm.n_components
        recons_traces_df['strain_trace'] = recons_traces_df['strain_trace'].apply(
            lambda x: x.tolist() if isinstance(x, np.ndarray) else x
        )
        recons_traces_df['isImputed'] = recons_traces_df['study_nr'].apply(
            lambda x: 1 if x in instance_dm.study_nr_imputed else (0 if x in instance_dm.study_nr_original else None)
        )
        recons_traces_df['study_type'] = 'BL'
        recons_traces_df['reconstructed_type'] = recons_traces_df.apply(
            lambda
                row: f"{data_type}-{concat_traces_by}-{n_components}-{1 if row['isImputed'] == 1 else 0}",
            axis=1
        )

        recons_traces_df['n_components'] = n_components
        recons_traces_df['data_source'] = data_source
        recons_traces_df = recons_traces_df.reset_index(drop=True)
        new_column_order = [
            'study_nr', 'curve_segment', 'id_position', 'strain_trace', 'study_type',
            'reconstructed_type', 'n_components', 'isImputed', 'data_source'
        ]
        recons_traces_df = recons_traces_df[new_column_order]
        InsertInterpTracesToDB(recons_traces_df, 'calculated_reconstructedTraces_T', True)

        unique_study_nr = instance_dm.DM_tracesDF['study_nr'].drop_duplicates()

        if data_type == 'PCA':
            recons_traces_df = recons_traces_df.drop_duplicates(subset='study_nr', keep='first')
            mapping_series_recon = recons_traces_df.set_index('study_nr')['reconstructed_type']
            mapping_series_isImp = recons_traces_df.set_index('study_nr')['isImputed']

            data = []
            for i, row in enumerate(instance_dm.pca_result):
                row_json = json.dumps(row.tolist())
                data.append({
                    'study_nr': unique_study_nr.values[i],
                    'pca_results': row_json,
                    'reconstructed_type': mapping_series_recon.get(unique_study_nr.values[i], 'Unknown'),
                    'n_components': n_components,
                    'isImputed': mapping_series_isImp.get(unique_study_nr.values[i], False),
                    'data_source': data_source
                })

            pca_result_df = pd.DataFrame(data)
            InsertInterpTracesToDB(pca_result_df, 'calculated_pca_results_T', False)

            pca_model = instance_dm.pca_var_comp

            pca_params = {
                'components_': json.dumps(pca_model.components_.tolist()),
                'explained_variance_': json.dumps(pca_model.explained_variance_.tolist()),
                'explained_variance_ratio_': json.dumps(pca_model.explained_variance_ratio_.tolist()),
                # Variance % per PC
                'mean_': json.dumps(pca_model.mean_.tolist()),
                'n_components_': pca_model.n_components_,
                'n_features_': pca_model.n_features_in_,
                'singular_values_': json.dumps(pca_model.singular_values_.tolist()),
                # Singular values (importance of PCs)
                'reconstructed_type': concat_traces_by,
                'n_components': n_components,
                'isImputed': 0.1,
                'data_source': data_source
                # 'curve_segment': json.dumps(segment_names),
                # 'id_position': json.dumps(segment_ids)
            }

            pca_var_df = pd.DataFrame([pca_params])
            InsertInterpTracesToDB(pca_var_df, 'calculated_pca_params_T', True)

        elif data_type == 'AutoencoderDm':
            recons_traces_df = recons_traces_df.drop_duplicates(subset='study_nr', keep='first')
            mapping_series_recon = recons_traces_df.set_index('study_nr')['reconstructed_type']
            mapping_series_isImp = recons_traces_df.set_index('study_nr')['isImputed']

            data = []
            for i, row in enumerate(instance_dm.encoded_data):
                row_json = json.dumps(row.tolist())
                data.append({
                    'study_nr': unique_study_nr.values[i],
                    'encoded_vector': row_json,
                    'reconstructed_type': mapping_series_recon.get(unique_study_nr.values[i], 'Unknown'),
                    'encoding': n_components,
                    'isImputed': mapping_series_isImp.get(unique_study_nr.values[i], False),
                    'data_source': data_source
                })

            autoencoder_result_df = pd.DataFrame(data)
            # InsertInterpTracesToDB(autoencoder_result_df, 'calculated_dr_encoded_data_T', True)

        print('Saving completed....')
        txt_save = f' Inserting to DB completed'
        return txt_save
    return dash.no_update


def plot_curve(trace, study_nr, n_components, traceType, pathname):
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
            if pathname == '/tab-PCA':
                title_str = (f'{traceType} Strain Curve -\n'
                             f'Study {study_nr} using {n_components} components of the explained variance')
            elif pathname == '/tab-AutoencoderDm':
                title_str = (f'{traceType} Strain Curve -\n'
                             f'Study {study_nr} using {n_components} epochs')

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


# def process_strain_traces(combined_data):
#     def convert_to_array(x):
#         if '[' in x:
#             return np.array([float(i) for i in x.strip('[]').split(',')])
#         else:
#             return np.array([float(i) for i in x.split(',')])
#
#     strain_traces = combined_data['strain_trace'].apply(convert_to_array)
#     flattened_strain_traces = np.concatenate(strain_traces.values)
#     scaler = StandardScaler()
#     scaler.fit(flattened_strain_traces.reshape(-1, 1))
#     standardized_strain_traces = strain_traces.apply(
#         lambda x: scaler.transform(x.reshape(-1, 1)).flatten()
#     )
#
#     return standardized_strain_traces


class DimensionalityRedViewer:
    def __init__(self, pathname):
        self.n_components_used = None
        self.trained_model = None
        self.encoded_data = None
        self.study_nr_imputed = None
        self.study_nr_original = None
        self.mean_model = None
        self.concat_echo_params = None
        self.df = None
        self.data_source = None
        self.table_name = None
        self.isImputed = None
        self.data_type = None
        self.pca_model = None
        self.studyType = 'BL'
        self.n_components = None
        self.pathname = pathname
        self.array_curves, self.pca_result, self.pca_var_comp, self.tracesDF = None, None, None, None
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

    def load_study_nr_dropdown(self):
        if self.data_source != 'GE_Philips':
            trace_loader = LoadTraces(self.studyType, self.table_name, self.isImputed, isComponents=True)
            self.df, self.study_nr_imputed, self.study_nr_original = (
                trace_loader.tracesDF, trace_loader.study_nr_imputed, trace_loader.study_nr_original)
        else:
            trace_loader1 = LoadTraces(self.studyType, self.table_name[0], self.isImputed, isComponents=True)
            trace_loader2 = LoadTraces(self.studyType, self.table_name[1], self.isImputed, isComponents=True)
            df1, study_nr_imputed1, study_nr_original1 = trace_loader1.tracesDF, trace_loader1.study_nr_imputed, trace_loader1.study_nr_original
            df2, study_nr_imputed2, study_nr_original2 = trace_loader2.tracesDF, trace_loader2.study_nr_imputed, trace_loader2.study_nr_original

            if df1 is not None and df2 is not None:
                self.df = pd.concat([df1, df2], ignore_index=True)
                self.study_nr_imputed = list(set(study_nr_imputed1 + study_nr_imputed2))
                self.study_nr_original = list(set(study_nr_original1 + study_nr_original2))
            else:
                raise ValueError("Failed to load traces for GE_Philips data source")

        self.df = self.df[~self.df['curve_segment'].str.lower().str.contains('apex', na=False)]
        self.df['cardiac_view'] = self.df['cardiac_view'].str.lower()
        self.df['segment_wall'] = self.df['segment_wall'].str.lower().str.strip() # .str.lower() changed on 07/08/2025
        dropna_tracesDF = self.df.dropna(subset=['strain_trace'])
        dropna_tracesDF = dropna_tracesDF[~dropna_tracesDF['strain_trace'].apply(lambda x: np.all(np.array(x) == 0))]
        self.tracesDF = dropna_tracesDF.groupby('study_nr').filter(lambda x: len(x) == 18)
        self.tracesDF = self.tracesDF.sort_values(by=['study_nr', 'id_position'])
        self.study_nr = np.sort(self.tracesDF['study_nr'].unique())

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
                    self.pca_result, self.pca_var_comp ,self.n_components_used = calculate_pca(self.n_components, self.array_curves)
                    self.mean_model = np.mean(self.array_curves, axis=0)
                else:
                    self.pca_result, self.pca_var_comp,self.n_components_used = calculate_pca(self.n_components, self.array_curves)
                    self.mean_model = np.mean(self.array_curves, axis=0)

            elif self.pathname == '/tab-AutoencoderDm':
                log_dir = "runs/autoencoder_experiment"
                self.encoded_data, self.trained_model = calculate_autoencoder(encoding_dim=self.n_components,
                                                                              array_curves=self.array_curves,
                                                                              epochs=1000,
                                                                              log_dir=log_dir)

            self.reconstruct_traces(curve_segment)
            self.DM_tracesDF = pd.concat([self.DM_tracesDF, self.PCA_rec_tracesDF], ignore_index=True)
        self.DM_tracesDF = self.DM_tracesDF.sort_values(by='study_nr').reset_index()
        self.DM_tracesDF = self.DM_tracesDF.drop(columns=['index'])
        pass

    def reconstruct_traces(self, curve_segment):
        start = 0
        pca_traces = []
        if self.pathname == '/tab-PCA':
            reconstructed_curves = reconstruct_traces_pca(self.pca_result, self.pca_var_comp)
        elif self.pathname == '/tab-AutoencoderDm':
            reconstructed_curves = reconstruct_traces_autoencoder(self.encoded_data, self.trained_model)

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
        content = html.Div([
            html.H3('Dimensionality Reduction'),
            dbc.Row([
                html.Br(),
                dcc.Dropdown(
                    id='data-source-dropdown_dr',
                    options=[
                        {'label': 'Speqle', 'value': 'Speqle'},
                        {'label': 'GE', 'value': 'GE'},
                        {'label': 'Philips', 'value': 'Philips'},
                        {'label': 'GE + Philips', 'value': 'GE_Philips'}
                    ],
                    value='GE_Philips',
                    style={'width': '300px'}
                ),
                html.Br(),
                html.Br(),

                html.Label('Select Concatenation Type:', style={'fontWeight': 'bold'}),
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

            ]),
            html.Br(),
            html.Br(),
            html.Button(
                'Perform DR', id='pca-button', style={'width': '200px'}, n_clicks=0),
            html.Br(),
            html.Label('Select Study Number:', style={'fontWeight': 'bold'}),
            dcc.Dropdown(
                id='study_nr-dropdown_dr',
                style={'width': '200px'}
            ),
            html.Br(),

            dbc.Row([
                dbc.Col(dcc.Graph(id='origin_trace'), width=6),
                dbc.Col(dcc.Graph(id='pca-graph'), width=6),
            ]), html.Br(),

            html.Button(
                'Save', id='savetoDB-button', style={'width': '200px', 'marginLeft': '10px'}, n_clicks=0),
            html.Br(),
            html.Div(id='id-data-txt', children=' ', style={'marginLeft': '10px'})

        ],
            style={'marginLeft': '20px'}
        )
        if self.pathname == '/tab-PCA':
            # content.children.insert(3, html.P("Explained variance:"))
            content.children.insert(3, html.P("Number of Components:"))
            content.children.insert(4, dcc.Slider(
                id='id-slider-dm',
                min=0,
                max=100,
                value=3,
                step=1,
                marks={
                    0: {'label': 'Optimal', 'style': {'color': '#777', 'font-weight': 'bold'}},
                    100: {'label': 'Maximum', 'style': {'color': '#0074D9', 'font-weight': 'bold'}}
                },
                tooltip={"placement": "bottom", "always_visible": True}
            ))

            # content.children.insert(4, dcc.Slider(
            #     id='id-slider-dm',
            #     min=0, max=60, value=59,
            #     step=1
            #     # min=0, max=1, value=0.95,
            #     # step=0.05
            # ))
        elif self.pathname == '/tab-AutoencoderDm':
            content.children.insert(3, html.P("Encoding Dimensions:"))
            content.children.insert(4, dcc.Slider(
                id='id-slider-dm',
                min=2, max=64, value=64,
                step=2
            ))

        return content

    def trigger_callbacks(self, concat_traces_by, n_components):
        self.concat_traces_by = concat_traces_by
        self.n_components = n_components
        self.load_pca_data()

        if concat_traces_by == 'full_traces' or concat_traces_by == 'full_traces_echo':
            if self.pathname == '/tab-PCA':
                self.data_type = 'PCA'

                if concat_traces_by == 'full_traces_echo':
                    self.load_echo_params()
                    merged_st_echoParam_df = pd.merge(self.concat_echo_params, self.concat_tracesDF, on='study_nr')

                    merged_st_echoParam_df['concatenated_all'] = merged_st_echoParam_df.apply(
                        lambda row: ' '.join([str(row['concatenated']), str(row['concatenated_trace'])]), axis=1)

                    curve = merged_st_echoParam_df['concatenated_all'].tolist()
                    array_curves = np.array(curve)

                    self.pca_result, self.pca_var_comp,self.n_components_used = calculate_pca(self.n_components, array_curves)
                    self.mean_model = np.mean(array_curves, axis=0)
                    pass
                else:
                    self.pca_result, self.pca_var_comp,self.n_components_used = calculate_pca(self.n_components, self.array_curves)
                    self.mean_model = np.mean(self.array_curves, axis=0)

            elif self.pathname == '/tab-AutoencoderDm':
                self.data_type = 'AutoencoderDm'
                log_dir = "runs/autoencoder_experiment"
                self.encoded_data, self.trained_model = calculate_autoencoder(encoding_dim=self.n_components,
                                                                              array_curves=self.array_curves,
                                                                              epochs=1000,
                                                                              log_dir=log_dir)

            self.reconstruct_traces(self.n_components)
