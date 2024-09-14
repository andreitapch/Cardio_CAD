from dash import dcc, html, Input, Output, dash, State, callback, ctx
# import dash_mantine_components as dmc
import plotly.graph_objects as go
import dash_bootstrap_components as dbc
from pattern_recognition.utils.utils import event_loader
from pattern_recognition.utils.plot.trace_plotter import TracePlotter
from pattern_recognition.utils.parameters_dics import cardiac_events_order
from pattern_recognition.utils.plot_dics import wall_color_dict, segment_color_dict, Cardiac_view_dict
from pattern_recognition.data.SPEQLE_data.loadTracesSpeqle import LoadTracesSpeqle as load_traces_speqle

GRAPH_STYLE = {'margin-top': '-20px'}
COLUMN_WIDTH = 6


@callback(
    [Output('orig_fig_segment', 'figure'),
     Output('interp_fig_segment', 'figure'),
     Output('id-figure-label', 'children')],
    [Input('dropdown-curve-id', 'value'),
     Input('check-isImputed-id', 'value')]

)
def update_plots(curve_dropdown, isImputed):
    if curve_dropdown is not None:
        global traces_viewer
        traces_viewer = TracesViewer()
        orig_fig = traces_viewer.plot_traces('speqle_segmentalStrain_T',
                                             'Original', None, isImputed, curve_dropdown)
        interp_fig = traces_viewer.plot_traces(
            'speqle_segmentalStrainInterp_T', 'Interpolated', 'speqle_cardiacEvent_T', isImputed, curve_dropdown)

        if curve_dropdown == 'segment_wall':
            figure_label = 'Traces by Segment '
        elif curve_dropdown == 'curve_segment':
            figure_label = 'Traces by Curve Segment'
        elif curve_dropdown == 'cardiac_view':
            figure_label = 'Traces by Cardiac View'
        else:
            figure_label = None
        return orig_fig, interp_fig, figure_label
    else:
        return dash.no_update, dash.no_update, dash.no_update


class TracesViewer:
    def __init__(self):
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

    def plot_traces(self, table_name, trace_type, cardiac_event_table, isImputed, plot_type):
        fig = go.Figure()
        cardiac_event_df = None
        if cardiac_event_table is not None:
            cardiac_event_df = load_traces_speqle(self.study_type, cardiac_event_table, isImputed)
            cardiac_event_df, _ = event_loader(self.cardiac_events_order, cardiac_event_df.tracesDF,
                                               self.interp_selection)

        speqle_traces = load_traces_speqle(self.study_type, table_name, isImputed)
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
