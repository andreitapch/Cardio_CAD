import pandas as pd
from dash import dcc
from dash import html
from dash import dash_table
import plotly.graph_objects as go
from concurrent.futures import ThreadPoolExecutor
import dash_bootstrap_components as dbc_components
from pattern_recognition.data.database_connection import sql_queries
from pattern_recognition.utils.parameters_dics import dbConParameters
from pattern_recognition.data.database_connection.db_con import DatabaseConnection as dbc


def df_to_data_table(df):
    columns_to_display = [col for col in df.columns if col not in ['segment_wall', 'curve_segment', 'cardiac_view']]

    data_table = dash_table.DataTable(
        data=df.to_dict('records'),
        columns=[{'name': i, 'id': i} for i in columns_to_display],
        style_header={'textAlign': 'center'},
        filter_action='native',
        style_data_conditional=[
            {
                'if': {'column_id': 'Total RM'},
                'backgroundColor': 'lightgreen'
            },
            {
                'if': {'column_id': 'Total DRM'},
                'backgroundColor': 'lightblue'
            }
        ],
        style_data={'overflowY': 'auto', 'overflowX': 'auto', 'textAlign': 'center'}
    )
    return data_table


def create_bar_plot(df, column):
    counts = df[column].value_counts()
    colors = ['magenta', 'cyan', 'green']
    fig = go.Figure(data=go.Bar(x=counts.index, y=counts.values, marker_color=colors))
    fig.update_layout(
        title_text=column,
        xaxis_title=column,
        yaxis_title='Count',
        bargap=0.2,
        bargroupgap=0.1
    )
    return fig


def create_trace(df, column):
    counts = df[column].value_counts().sort_index()
    return go.Bar(x=counts.index, y=counts.values, name=column)


def create_grouped_bar_plot(df, type, columns):
    with ThreadPoolExecutor() as executor:
        traces = list(executor.map(lambda column: create_trace(df, column), columns))

    layout = go.Layout(
        title_text=type + ' Remodelling per image modality',
        xaxis_title='Values',
        yaxis_title='Count',
        bargap=0.2,
        bargroupgap=0.1,
        barmode='group'
    )

    fig = go.Figure(data=traces, layout=layout)

    return fig


class StrainRemodelingViewer:
    def __init__(self):
        self.merged_remodDF = None
        self.remod_dataDF = None
        self.merged_calcu_remodDF = None
        self.merged_doppler_remodDF = None
        self.calc_remod_dataDF = pd.DataFrame()
        self.doppler_remod_dataDF = pd.DataFrame()
        self.seg_strain_interpDF = None
        self.get_db_views()

    def get_db_views(self):
        db = dbc(driver=dbConParameters['driver'], server=dbConParameters['server'],
                 database=dbConParameters['database'])
        db.connect()
        db.execute_query(sql_queries.select_merged_remoded_18seg_view())
        self.remod_dataDF = db.df_query()
        db.close_connection()

        columns_to_convert = ['Study Nr', 'RM CMR', 'Total RM', 'DRM CMR', 'RM 3DEcho', 'DRM 3DEcho', 'RM 2CH',
                              'DRM 2CH', 'RM 4CH', 'DRM 4CH', 'DRM PARA', 'Total DRM']
        self.remod_dataDF[columns_to_convert] = self.remod_dataDF[columns_to_convert].astype('Int64')

    def get_content(self):
        df_rm_strain = df_to_data_table(self.remod_dataDF)
        bar_plot_grouped_calc = create_grouped_bar_plot(self.remod_dataDF, 'Calculated',
                                                        ['RM CMR', 'RM 3DEcho', 'RM 2CH', 'RM 4CH'])
        bar_plot_grouped_dop = create_grouped_bar_plot(self.remod_dataDF, 'Doppler',
                                                       ['DRM CMR', 'DRM 3DEcho', 'DRM 2CH', 'DRM 4CH'])

        plot_grouped_calc = create_bar_plot(self.remod_dataDF, 'Total RM')
        plot_grouped_dop = create_bar_plot(self.remod_dataDF, 'Total DRM')

        content = html.Div([
            dbc_components.Row(
                [
                    dbc_components.Col(
                        [
                            html.Div(id='table', children=df_rm_strain,
                                     style={'display': 'inline-block', 'overflowY': 'scroll',
                                            'height': '500px', 'margin': 'auto'}),
                        ],
                        width=12,
                    ),
                ]
            ),
            dbc_components.Row(
                [
                    dbc_components.Col(
                        [
                            dcc.Graph(id='plot_grouped_calc', figure=plot_grouped_calc),
                        ],
                        width=6,
                    ),
                    dbc_components.Col(
                        [
                            dcc.Graph(id='plot_grouped_dop', figure=plot_grouped_dop),
                        ],
                        width=6,
                    ),
                ]
            ),
            dbc_components.Row(
                [
                    dbc_components.Col(
                        [
                            dcc.Graph(id='bar_plot_grouped_calc', figure=bar_plot_grouped_calc),
                        ],
                        width=6,
                    ),
                    dbc_components.Col(
                        [
                            dcc.Graph(id='bar_plot_grouped_dop', figure=bar_plot_grouped_dop),
                        ],
                        width=6,
                    ),
                ]
            )
        ])
        loading_content = dcc.Loading(
            id="loading",
            type="circle",
            children=[content]
        )
        return loading_content
