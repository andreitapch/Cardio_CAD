from dash import dcc, html, Input, Output, callback, dash
import dash_bootstrap_components as dbc
from sqlalchemy.sql.operators import truediv

from pattern_recognition.utils.plot_dics import segment_color_dict
from pattern_recognition.data.loadTraces import LoadTraces as load_traces
import plotly.graph_objects as go

GRAPH_STYLE = {'margin-top': '50px'}
COLUMN_WIDTH = 5


@callback(
    Output('study_nr-dropdown_pat', 'options'),
    Input('data-source-dropdown_cp_pat', 'value'),
    Input('isImputed-id_pat', 'value')
)
def load_traces_pat(data_source, isImputed):
    if data_source is None:
        return dash.no_update

    global pat_traces_viewer
    study_numbers =[]
    pat_traces_viewer = PatientTracesViewer()
    isImputed = 1 if isImputed else 0

    data_source_mapping = {
        'Speqle': ('speqle_segmentalStrain_T', 'speqle_segmentalStrainInterp_T'),
        'GE': ('st_segmentalStrain_T', 'st_segmentalStrainInterp_T'),
        'Philips': ('phi_segmentalStrain_T', 'phi_segmentalStrainInterp_T'),
    }

    if data_source in data_source_mapping:
        table_orig, table_interp = data_source_mapping[data_source]

        pat_traces_viewer.speqle_traces_orig = load_traces(pat_traces_viewer.study_type, table_orig, isImputed, False)
        pat_traces_viewer.speqle_traces_interp = load_traces(pat_traces_viewer.study_type, table_interp, isImputed, False)

        study_numbers = pat_traces_viewer.speqle_traces_orig.tracesDF['study_nr'].unique()

    return [{'label': str(study_nr), 'value': study_nr} for study_nr in study_numbers]


@callback(
    [Output('orig_fig_segment_pat', 'figure'),
     Output('interp_fig_segment_pat', 'figure'),
     Output('id-figure-label_pat', 'children')],
    [Input('data-source-dropdown_cp_pat', 'value'),
     Input('study_nr-dropdown_pat', 'value')]
)
def update_plots(data_source, study_nr):
    if data_source is None or study_nr is None:
        return dash.no_update, dash.no_update, dash.no_update

    pat_traces_viewer.tracesDF_orig = pat_traces_viewer.speqle_traces_orig.tracesDF[
        pat_traces_viewer.speqle_traces_orig.tracesDF['study_nr'] == study_nr]

    pat_traces_viewer.tracesDF_interp = pat_traces_viewer.speqle_traces_interp.tracesDF[
        pat_traces_viewer.speqle_traces_interp.tracesDF['study_nr'] == study_nr]

    if pat_traces_viewer.tracesDF_orig.empty or pat_traces_viewer.tracesDF_interp.empty:
        return dash.no_update, dash.no_update, dash.no_update

    orig_fig = pat_traces_viewer.plot_trace('original', pat_traces_viewer.tracesDF_orig, 1800, 700, save_plot=False)
    interp_fig = pat_traces_viewer.plot_trace('interpolated', pat_traces_viewer.tracesDF_interp, 1800, 700, save_plot=False)
    label = f"Plot for study nr. {study_nr}"
    return orig_fig, interp_fig, label


class PatientTracesViewer:

    def __init__(self):
        self.tracesDF_orig= None
        self.tracesDF_interp = None
        self.speqle_traces_orig = None
        self.speqle_traces_interp = None
        self.es_cardiac_event_df = None
        self.study_type = 'BL'
        self.interp_selection = 'avg'

    def get_content(self):
        content = dbc.Row([
            html.Br(),
            html.Br(),
            dcc.Dropdown(
                id='data-source-dropdown_cp_pat',
                options=[
                    {'label': 'Speqle', 'value': 'Speqle'},
                    {'label': 'GE', 'value': 'GE'},
                    {'label': 'Philips', 'value': 'Philips'}
                ],
                placeholder='Select Data Source',
                style={'width': '300px', 'marginLeft': '10px'}
            ),
            html.Br(),
            dbc.Checkbox(
                id='isImputed-id_pat',
                label='Include imputed data',
                value=True,
                style={'marginLeft': '20px'}
            ),
            html.Br(),
            html.Br(),
            dbc.Row([
                html.Label('Select Study Number:', style={'fontWeight': 'bold', 'marginLeft': '10px'}),
                dcc.Dropdown(
                    id='study_nr-dropdown_pat',
                    options=[],
                    style={'width': '200px', 'marginLeft': '10px'}
                )
            ]),

            html.Br(),
            html.Br(),
            dbc.Col([
                dbc.Row([
                    html.Div(id='id-figure-label_pat', style={'marginLeft': '10px'}),
                    dbc.Col(dcc.Graph(id='orig_fig_segment_pat'), width=COLUMN_WIDTH),
                    dbc.Col(dcc.Graph(id='interp_fig_segment_pat'), width=COLUMN_WIDTH)
                ])
            ]),
        ])
        return dcc.Loading(id="loading", type="circle", children=[content])

    def plot_trace(self, trace_type, tracesDF, width, height, save_plot=False):
        fig = go.Figure()
        color_dict = segment_color_dict if trace_type == 'original' else segment_color_dict

        for i, segment in enumerate(tracesDF['curve_segment'].unique()):
            segment_traces = tracesDF[tracesDF['curve_segment'] == segment]
            for j, strain_trace in enumerate(segment_traces['strain_trace']):
                trace_values = [float(x.replace(',', '')) for x in strain_trace.split()]
                fig.add_trace(go.Scatter(
                    y=trace_values,
                    mode='lines',
                    name=f'{segment.strip()}',
                    line=dict(color=color_dict.get(segment.strip(), 'blue'))
                ))

        fig.update_layout(
            # autosize=True,
            # width=width,
            height=height,
            title=f'{trace_type.capitalize()} Strain Traces',
            xaxis_title='pts',
            yaxis_title='Strain'
        )

        if save_plot:
            fig.write_image(f"{trace_type}_strain_traces.png")

        return fig
