import pandas as pd
from dash import dcc, html, Input, Output, dash, callback
# import dash_mantine_components as dmc
import plotly.graph_objects as go
import dash_bootstrap_components as dbc
from sympy import false

from pattern_recognition.img_analysis.check_vendor_differences import analyze_vendor_differences
from pattern_recognition.utils.utils import event_loader
from pattern_recognition.utils.plot.trace_plotter import TracePlotter
from pattern_recognition.utils.parameters_dics import cardiac_events_order
from pattern_recognition.utils.plot_dics import wall_color_dict, segment_color_dict, Cardiac_view_dict
from pattern_recognition.data.loadTraces import LoadTraces as load_traces

GRAPH_STYLE = {'margin-top': '20px'}
COLUMN_WIDTH = 6


@callback(
    [Output('orig_fig_segment', 'figure'),
     Output('interp_fig_segment', 'figure'),
     Output('id-figure-label', 'children')],
    [Input('dropdown-curve-id', 'value'),
     Input('data-source-dropdown_cp', 'value'),
     Input('check-isImputed-id', 'value')]

)
def update_plots(curve_dropdown, data_source, isImputed):
    if data_source is None or curve_dropdown is None:
        return dash.no_update, dash.no_update, dash.no_update

    global traces_viewer
    traces_viewer = TracesViewer()

    data_source_mapping = {
        'Speqle': ('speqle_segmentalStrain_T', 'speqle_segmentalStrainInterp_T', 'speqle_cardiacEvent_T', None),
        'GE': ('st_segmentalStrain_T', 'st_segmentalStrainInterp_T', None, 'st_strainValuesInterp_T'),
        'Philips': ('phi_segmentalStrain_T', 'phi_segmentalStrainInterp_T', None, 'phi_strainValuesInterp_T'),
        'GE_Philips': (['st_segmentalStrain_T', 'phi_segmentalStrain_T'],
                       ['st_segmentalStrainInterp_T', 'phi_segmentalStrainInterp_T'], None, 'st_strainValuesInterp_T')
    }

    if data_source in data_source_mapping:
        orig_table, interp_table, event_table, strain_values = data_source_mapping[data_source]

        orig_fig = traces_viewer.plot_traces(orig_table, 'Original', None, isImputed, curve_dropdown, data_source,
                                             strain_values)
        interp_fig = traces_viewer.plot_traces(interp_table, 'Interpolated', event_table, isImputed, curve_dropdown,
                                               data_source, strain_values)


        figure_label = {
            'segment_wall': 'Traces by Segment',
            'curve_segment': 'Traces by Curve Segment',
            'cardiac_view': 'Traces by Cardiac View'
        }.get(curve_dropdown, None)

        return orig_fig, interp_fig, figure_label

    return dash.no_update, dash.no_update, dash.no_update


class TracesViewer:
    def __init__(self):
        self.es_cardiac_event_df = None
        self.interp_fig_segment = go.Figure()
        self.orig_fig_segment = go.Figure()
        self.interp_fig_wall = go.Figure()
        self.orig_fig_wall = go.Figure()
        self.interp_fig_cardiac_view = go.Figure()
        self.orig_fig_cardiac_view = go.Figure()
        self.study_type = 'BL'
        self.interp_selection = 'avg'
        self.cardiac_events_order = cardiac_events_order

    def get_content(self):
        content = dbc.Row(
            [
                html.Br(),
                html.Br(),
                html.Br(),
                dcc.Dropdown(
                    id='data-source-dropdown_cp',
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
                dcc.Dropdown(
                    id='dropdown-curve-id',
                    options=[
                        {'label': 'Segment Wall', 'value': 'segment_wall'},
                        {'label': 'Curve Segment', 'value': 'curve_segment'},
                        {'label': 'Cardiac View', 'value': 'cardiac_view'}
                    ],
                    style={'width': '300px', 'marginLeft': '20px'}
                ),
                html.Br(),
                html.Br(),
                dbc.Checkbox(
                    id='check-isImputed-id',
                    label='Include imputed data',
                    value=True,
                    style={'marginLeft': '20px'}
                ),
                html.Br(),

                dbc.Col(
                    [
                        dbc.Accordion(
                            [
                                dbc.AccordionItem(

                                    children=dbc.Row([
                                        html.Div(id='id-figure-label', children=' ', style={'marginLeft': '10px'}),
                                        dbc.Col([
                                            dcc.Graph(id='orig_fig_segment'),
                                        ], width=COLUMN_WIDTH),
                                        dbc.Col([
                                            dcc.Graph(id='interp_fig_segment'),
                                        ], width=COLUMN_WIDTH),

                                    ])
                                )
                            ],
                            id="accordion-always-open",
                            always_open=False,
                        ),
                        html.Div(id="accordion-contents-open-ids", className="mt-3"),
                    ],
                ),
            ]
        )
        loading_content = dcc.Loading(
            id="loading",
            type="circle",
            children=[content]
        )
        return loading_content

    def plot_traces(self, table_name, trace_type, cardiac_event_table, isImputed, plot_type, data_source, esDF):
        fig = go.Figure()
        cardiac_event_df = None
        if cardiac_event_table is not None and data_source == 'Speqle':
            cardiac_event_df = load_traces(self.study_type, cardiac_event_table, isImputed, False)
            cardiac_event_df, _ = event_loader(self.cardiac_events_order, cardiac_event_df.tracesDF,
                                               self.interp_selection)

        if data_source == 'Philips' or data_source == 'GE' or data_source == 'Speqle':
            speqle_traces = load_traces(self.study_type, table_name, isImputed, False)

        if data_source == 'GE_Philips':
            speqle_traces_ge = load_traces(self.study_type, table_name[0], isImputed, False)
            speqle_traces_ge.tracesDF['origin'] = 'GE'

            speqle_traces_philips = load_traces(self.study_type, table_name[1], isImputed, False)
            speqle_traces_philips.tracesDF['origin'] = 'Philips'

            speqle_traces = speqle_traces_ge
            speqle_traces.tracesDF = pd.concat([speqle_traces_ge.tracesDF, speqle_traces_philips.tracesDF]) #I am here

            speqle_traces.tracesDF['strain_trace'] = speqle_traces.tracesDF['strain_trace'].apply(
                lambda x: [float(value) for value in x.strip('[]').split(', ')])

            if isImputed is not None:
                analyze_vendor_differences(speqle_traces_ge, speqle_traces_philips)

        if data_source == 'Philips' or data_source == 'GE_Philips':
            speqle_traces.tracesDF = speqle_traces.tracesDF[
                ~speqle_traces.tracesDF['segment_wall'].str.contains('apex', case=False, na=False)]

        if data_source == 'GE' or data_source == 'Philips' or data_source == 'GE_Philips':
            if trace_type == 'Original':
                strain_values_df = load_traces(self.study_type, esDF, isImputed, False)
                cardiac_event_df = pd.DataFrame()
                cardiac_event_df['event_time'] = strain_values_df.tracesDF['esFrame']
                cardiac_event_df = cardiac_event_df.head(1)
                cardiac_event_df['event_name'] = 'ES'
                self.es_cardiac_event_df = cardiac_event_df
                cardiac_event_df = None
            else:
                cardiac_event_df = self.es_cardiac_event_df

        trace_plotter = TracePlotter(speqle_traces.tracesDF, cardiac_event_df, concat_traces_by='segment')

        if plot_type == 'segment_wall':
            trace_plotter.group_traces('segment_wall')
            fig = trace_plotter.plot_trace('Wall', wall_color_dict, trace_type, 1800,
                                           700, save_plot=False)
        elif plot_type == 'curve_segment':
            trace_plotter.group_traces('curve_segment')
            fig = trace_plotter.plot_trace('Segment', segment_color_dict, trace_type, 7200,
                                           700, save_plot=False)

        elif plot_type == 'cardiac_view':
            trace_plotter.group_traces('cardiac_view')
            fig = trace_plotter.plot_trace('Cardiac View', Cardiac_view_dict, trace_type,
                                           1600, 700, save_plot=False)

        return fig