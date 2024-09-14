import ast
import mpld3
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.impute import KNNImputer
from utils.SmoothAHAplot.src.aha.aha import AHA
import dash_bootstrap_components as dbcomp
from dash import dcc, html, Input, Output, dash, callback, ctx
from dash_viewer.data_imputation.load_dataTables import load_data_table_db
from pattern_recognition.data_preprocessing.speqle.concatenate_traces import CurveConcatenator
from pattern_recognition.data.insert_to_db.insertSpeqleDB import InsertInterpTracesToDB, DeleteToDBWhere
from pattern_recognition.data.SPEQLE_data.loadTracesSpeqle import LoadTracesSpeqle as load_traces_speqle


@callback(
    [Output('study_nr_knn-dropdown', 'options')],
    Input('textarea-k', 'value'),
    Input('missing_traces_id', 'value')
)
def impute_traces(k_value, sample_n):
    global instance_imputation
    instance_imputation = DataImputationST()
    if k_value is not None:
        if sample_n is not None:
            instance_imputation.load_data()
            instance_imputation.missing_segments_finder(sample_n)
            study_nr = instance_imputation.to_impute_tracesDF['study_nr'].unique()
            dropdown_options = [study_nr]
            concat_tracesDF = concatenate_traces(instance_imputation.tracesDF_18)
            concat_imputedDF = concatenate_traces(instance_imputation.to_impute_tracesDF)
            instance_imputation.impute_train(concat_tracesDF, concat_imputedDF, k_value)
            instance_imputation.trace_reconstruction()
            return dropdown_options
        return dash.no_update
    else:
        return dash.no_update


@callback(
    Output('id-bullseye-plot', 'srcDoc'),
    Output('id_tracer_plot', 'figure'),
    [Input('study_nr_knn-dropdown', 'value')]
)
def plot_strain_traces(study_nr):
    img_html = go.Figure()
    fig = go.Figure()
    if study_nr:
        # fig = go.Figure()
        tracesDF = instance_imputation.imputed_tracesDF
        traces_studyDf = tracesDF[tracesDF['study_nr'] == study_nr]
        imputed_tracesDF = traces_studyDf[traces_studyDf['isImputed'] == 1]
        normal_tracesDF = traces_studyDf[traces_studyDf['isImputed'] == 0]
        fig = plot_traces(imputed_tracesDF, normal_tracesDF)
        missing_segments = imputed_tracesDF['id_position'].tolist()
        bulls_eye_fig = plot_missing_bulls_eye(missing_segments)
        img_html = f'<img src="{bulls_eye_fig}" style="width: 100%; height: auto;">'
        return img_html, fig.to_dict()
    else:
        return dash.no_update, dash.no_update


@callback(
    Output('data_text-id', 'children'),
    Input('insert-db-button', 'n_clicks')
)
def insert_into_db(insert_btn):
    if "insert-db-button" == ctx.triggered_id :
        tracesDF = instance_imputation.imputed_tracesDF
        imputed_tracesDF = tracesDF[tracesDF['isImputed'] == 1]
        if not imputed_tracesDF.empty:
            # DeleteToDBWhere('speqle_segmentalStrainInterp_T', 'isImputed', 1)
            InsertInterpTracesToDB(imputed_tracesDF, 'speqle_segmentalStrainInterp_T', True)
    return dash.no_update


def plot_missing_bulls_eye(missing_segments):
    segments_18 = ['Basal Anterior', 'Basal Anteroseptal', 'Basal Inferoseptal',
                   'Basal Inferior', 'Basal Inferolateral', 'Basal Anterolateral',
                   'Mid Anterior', 'Mid Anteroseptal', 'Mid Inferoseptal',
                   'Mid Inferior', 'Mid Inferolateral', 'Mid Anterolateral',
                   'Apical Anterior', 'Apical Anteroseptal', 'Apical Inferoseptal',
                   'Apical Inferior', 'Apical Inferolateral', 'Apical Anterolateral']
    exp_strain_data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
    strain_dict = {k: v for (k, v) in zip(segments_18, exp_strain_data)}

    aha = AHA(strain_dict, 'Strain')
    fig = aha.bullseye_smooth(missing_segments)

    return fig


def plot_traces(imputed_tracesDF, normal_tracesDF):
    fig = go.Figure()
    for idx, row in normal_tracesDF.iterrows():
        fig.add_trace(go.Scatter(y=row['imputed_trace'],
                                 mode='lines', line=dict(color='silver'), name=row['curve_segment']))

    for idx, row in imputed_tracesDF.iterrows():
        fig.add_trace(go.Scatter(y=row['imputed_trace'],
                                 mode='lines', line=dict(color='red', dash='dash'), name=row['curve_segment']))

    fig.update_layout(height=800, showlegend=True)

    return fig


def concatenate_traces(tracesDF):
    concatenated_strain = CurveConcatenator(tracesDF, isReconstructed=False)
    concat_tracesDF = concatenated_strain.concatenate_traces_par()
    concat_tracesDF['concatenated_trace'] = np.array(concat_tracesDF['concatenated_trace'])
    return concat_tracesDF


def safe_eval(val):
    try:
        return eval(val)
    except Exception:
        return val


def replace_with_nan(strain_trace):
    strain_trace_list = safe_eval(strain_trace)
    return [np.nan] * len(strain_trace_list)


def find_missing_segments(group, seg_values, cardiac_segmentsDF):
    present_segments = set(group['curve_segment'].str.strip())
    missing_segments = set(seg_values) - present_segments
    study_nr = group['study_nr'].iloc[0]
    strain_trace = replace_with_nan(group['strain_trace'].iloc[0])

    rows = []
    for segment in missing_segments:
        row = cardiac_segmentsDF.loc[cardiac_segmentsDF['segment_name'] == segment]
        if not row.empty:
            row_dict = {'study_nr': study_nr,
                        'segment_wall': row['seg_speqle'].values[0],
                        'curve_length': int(len(strain_trace)),
                        'curve_segment': segment,
                        'id_position': str(row['id_position'].values[0]),
                        'strain_trace': strain_trace,
                        'cardiac_view': row['cardiac_view'].values[0],
                        'isImputed': 1,
                        'studyType': 'BL'}
            rows.append(row_dict)

    return pd.DataFrame(rows)


class DataImputationST:

    def __init__(self):
        self.to_impute_tracesDF = pd.DataFrame()
        self.cardiac_segmentsDF = None
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

    def load_data(self):
        self.cardiac_segmentsDF = load_data_table_db('config_cardiacSegments_T', False)
        table_name = 'speqle_segmentalStrainInterp_T'
        self.loaded_tracesDF = load_traces_speqle(self.study_type, table_name, None)
        self.tracesDF = self.loaded_tracesDF.tracesDF.dropna(subset=['strain_trace'])
        self.tracesDF['id_position'] = self.tracesDF['id_position'].str.rstrip()
        self.tracesDF['curve_segment'] = self.tracesDF['curve_segment'].str.rstrip()#.str.lower()
        self.tracesDF['segment_wall'] = self.tracesDF['segment_wall'].str.rstrip()#.str.lower()
        indices = self.tracesDF[self.tracesDF['curve_segment'] == '0'].index
        self.tracesDF.drop(indices, inplace=True)
        for i in range(12, 19):
            setattr(self, f'tracesDF_{i}', self.select_data_with_rows_per_study_nr(i))

    def missing_segments_finder(self, sample_n):
        dfs_to_concat = []
        segment_names = self.cardiac_segmentsDF
        seg_list = segment_names['segment_name'].tolist()

        if sample_n == 6:
            range_start, range_end = 12, 14
        elif sample_n == 3:
            range_start, range_end = 14, 16
        elif sample_n == 1:
            range_start, range_end = 16, 18
        else:
            raise ValueError("Invalid sample_n value")

        for i in range(range_start, range_end):
        # for i in range(12, 18):
            missing_segmentsDF = getattr(self, f'tracesDF_{i}')

            missing_segments_per_study = missing_segmentsDF.groupby('study_nr').apply(
                lambda group: find_missing_segments(group, seg_list, segment_names))
            missing_segments_per_study = missing_segments_per_study.reset_index(drop=True)
            missing_segmentsDF = pd.concat([missing_segmentsDF, missing_segments_per_study], ignore_index=True)
            missing_segmentsDF = missing_segmentsDF.sort_values(by='study_nr').reset_index(drop=True)

            setattr(self, f'tracesDF_{i}', missing_segmentsDF)
            dfs_to_concat.append(missing_segmentsDF)

        self.to_impute_tracesDF = pd.concat(dfs_to_concat, ignore_index=True)

    def trace_reconstruction(self):
        imputed_tracesDF = pd.DataFrame()
        unique_study_nr = self.imputed_df['study_nr'].unique()
        for idx, study_nr in enumerate(unique_study_nr):
            start = 0
            imputed_traces = []
            studyDf = self.imputed_df[self.imputed_df['study_nr'] == study_nr]
            studyDf[self.feature_col_name] = np.round(studyDf['imputed_trace'], decimals=4)

            curve = studyDf[self.feature_col_name].tolist()[0]
            curve_length = studyDf['concat_curve_len'][idx]
            curve_segment = studyDf['concat_curve_segm'][idx]
            id_segment = studyDf['concat_id_segment'][idx]
            segments_wall = studyDf['concat_segment_wall'][idx]
            cardiac_views = studyDf['concat_cardiac_view'][idx]

            for curve_len, curve_seg, id_seg, seg_wall, cardiac_view in zip(curve_length, curve_segment, id_segment,
                                                                            segments_wall, cardiac_views):
                curve_segment = curve[start:start + int(curve_len)]
                imputed_traces.append({'study_nr': study_nr,
                                       'segment_wall': str(seg_wall),
                                       'curve_length': int(curve_len),
                                       'curve_segment': curve_seg,
                                       'id_position': int(id_seg),
                                       'imputed_trace': curve_segment,
                                       'cardiac_view': str(cardiac_view)})
                start += int(curve_len)

            imputed_tracesDF = pd.concat([imputed_tracesDF, pd.DataFrame(imputed_traces)], ignore_index=True)
            imputed_tracesDF['study_type'] = 'BL'
        self.to_impute_tracesDF['id_position'] = self.to_impute_tracesDF['id_position'].astype(int)
        to_impute_df = self.to_impute_tracesDF[['study_nr', 'id_position', 'curve_segment', 'isImputed']]
        imputed_tracesDF = pd.merge(imputed_tracesDF, to_impute_df, on=['study_nr', 'id_position', 'curve_segment'],
                                    how='left')
        self.imputed_tracesDF = imputed_tracesDF
        # return imputed_tracesDF

    def impute_train(self, concat_tracesDF, concat_imputedDF, k):
        self.imputed_df = concat_imputedDF.copy()
        self.imputed_df[self.feature_col_name] = np.nan
        self.imputed_df = self.imputed_df.rename(columns={self.feature_col_name: 'imputed_trace'})
        X_train = np.stack(concat_tracesDF[self.feature_col_name].values)
        X_test = np.stack(concat_imputedDF[self.feature_col_name].values)
        knn_impute = KNNImputer(n_neighbors=k, weights='distance')
        knn_impute.fit(X_train)
        imputed_traces = knn_impute.transform(X_test)
        self.imputed_df['imputed_trace'] = imputed_traces.tolist()

    def get_content(self):
        content = html.Div([
            dbcomp.Card([
                dbcomp.CardBody([
                    html.Br(),
                    dbcomp.Row([
                        dbcomp.Col([
                            html.Label('Select number of missing segments:',
                                       style={'fontWeight': 'bold'}),
                            dcc.Dropdown(
                                id='missing_traces_id',
                                options=[
                                    {'label': '1 or 2 Missing segments', 'value': 1},
                                    {'label': '3 or 4 Missing segments', 'value': 3},
                                    {'label': '5 or 6 Missing segments', 'value': 6}
                                ],
                                style={'width': '270px', 'backgroundColor': 'white'},
                            ),
                        ], width=3),
                        dbcomp.Col([
                            html.Label('Enter K:', style={'fontWeight': 'bold'}),
                            html.Br(),
                            dcc.Input(
                                id="textarea-k", type="number", debounce=True,
                                style={'width': '100px', }),
                        ], width=4),
                    ],style={'marginBottom': '20px'}),
                    html.Div(),
                    html.Br(),
                    html.Br(),
                    dbcomp.Row([
                        html.Label('Select Study Number:', style={'fontWeight': 'bold'}),
                        dcc.Dropdown(
                            id='study_nr_knn-dropdown',
                            options=[],
                            style={'width': '150px'}
                        ),
                    ]),
                    html.Br(),
                    html.Br(),
                    dbcomp.Row([

                        dbcomp.Col([
                            html.P("Bulls Eye Plot", style={'fontWeight': 'bold'}),
                            html.Iframe(srcDoc=None, id='id-bullseye-plot', style={'width': '100%', 'height': '700px'}),
                        ], width=6),

                        html.Br(),
                        html.Br(),
                        dbcomp.Col([
                            html.P("Imputed Traces", style={'fontWeight': 'bold'}),
                            dcc.Graph(id='id_tracer_plot'),

                        ], width=6),
                    ]),

                    dbcomp.Row([
                        dbcomp.Button("Save", id="insert-db-button", color="primary", className="mr-1",
                                      style={'padding': '10px', 'margin-bottom': '10px', 'width': '100px',
                                             'marginLeft': '10px'}),
                        html.Br(),
                        html.Div(id='data_text-id', style={'display': 'none'}),
                    ]),
                ]),
            ]),
        ], style={'margin': '20px'})
        return content
