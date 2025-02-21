import pandas as pd
from dash import dcc
from dash import html
from dash import dash_table
import plotly.graph_objects as go
from concurrent.futures import ThreadPoolExecutor
import dash_bootstrap_components as dbc_components

from dash_viewer.classification.load_prepare_data import load_segment_table_db
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


def create_grouped_bar_plot(df):
    categories = ['Speqle', 'GE', 'Philips']
    traces = []

    for category in categories:
        counts = df.groupby('remodeling')[category].sum()
        trace = go.Bar(x=counts.index, y=counts.values, name=category)
        traces.append(trace)

    layout = go.Layout(
        title_text='Remodeling per Category',
        xaxis_title='Remodeling',
        yaxis_title='Count',
        bargap=0.2,
        bargroupgap=0.1,
        barmode='group',
        height=600,  # Adjust height
        width=500,  # Adjust width
        margin=dict(l=50, r=50, t=50, b=50)  # Adjust margins
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
        db.execute_query(sql_queries.select_all_data('merged_remodeling_18seg_view'))
        self.remod_dataDF = db.df_query()
        db.close_connection()

    def get_content(self):
        df_rm_strain = df_to_data_table(self.remod_dataDF)
        bar_plot_grouped_calc = create_grouped_bar_plot(self.remod_dataDF)

        content = html.Div([
            dbc_components.Row(
                [

                    dbc_components.Col(
                        [
                            html.Div(id='table', children=df_rm_strain,
                                     style={'display': 'inline-block',
                                            'height': '500px', 'margin': 'auto'}),
                        ],
                        width=6,
                    ),
                    dbc_components.Col(
                        [
                            dcc.Graph(id='bar_plot_grouped_calc', figure=bar_plot_grouped_calc),
                        ],
                        width=5,
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