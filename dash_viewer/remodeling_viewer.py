import pandas as pd
from dash import dcc
from dash import html
from dash import dash_table
import plotly.graph_objects as go
import dash_bootstrap_components as dbc_components

from pattern_recognition.data.database_connection import sql_queries
from pattern_recognition.utils.parameters_dics import dbConParameters
from pattern_recognition.data.database_connection.db_con import DatabaseConnection as dbc


def create_bar_plot(df, column):
    counts = df[column].value_counts()
    colors = ['magenta', 'green', 'blue'] * len(counts)
    fig = go.Figure(data=go.Bar(x=counts.index, y=counts.values, marker_color=colors))
    fig.update_layout(
        title_text='Bar Plot of ' + column,
        xaxis_title=column,
        yaxis_title='Count',
        bargap=0.2,
        bargroupgap=0.1
    )
    graph = dcc.Graph(figure=fig)
    return graph


def df_to_data_table(df):
    data_table = dash_table.DataTable(
        data=df.to_dict('records'),
        columns=[{'name': i, 'id': i} for i in df.columns],
        style_header={'textAlign': 'center'},
        filter_action='native',
        style_cell={
            'minWidth': '50px', 'maxWidth': '80px',
            'whiteSpace': 'normal'
        }
    )
    return data_table


class RemodelingViewer:
    def __init__(self, studyType='BL'):
        self.studyType = studyType
        self.remod_dataDF = pd.DataFrame
        self.doppler_remod_dataDF = pd.DataFrame
        self.calc_remod_dataDF = pd.DataFrame
        self.get_db_views()

    def get_db_views(self):
        db = dbc(driver=dbConParameters['driver'], server=dbConParameters['server'],
                 database=dbConParameters['database'])
        db.connect()
        db.execute_query(sql_queries.select_total_remodeling_data())
        self.calc_remod_dataDF = db.df_query()
        db.execute_query(sql_queries.select_total_doppler_remodeling_data())
        self.doppler_remod_dataDF = db.df_query()
        db.close_connection()

    def get_content(self):
        df_calc_rm = df_to_data_table(self.calc_remod_dataDF)
        df_doppler_rm = df_to_data_table(self.doppler_remod_dataDF)
        bar_plot_doppler = create_bar_plot(self.doppler_remod_dataDF, 'Total DRM')
        bar_plot_calc = create_bar_plot(self.calc_remod_dataDF, 'Total RM')

        content = dbc_components.Row(
            [
                dbc_components.Col(
                    [
                        dbc_components.Accordion(
                            [
                                dbc_components.AccordionItem(
                                    children=[
                                        html.Div(df_doppler_rm, style={'width': '60%', 'display': 'inline-block',
                                                                       'overflowY': 'scroll', 'height': '400px'}),
                                        html.Div(bar_plot_doppler,
                                                 style={'width': '40%', 'display': 'inline-block', 'height': '400px'}),
                                    ],
                                    title="Remodeling per image Mode - Doppler",
                                ),
                                dbc_components.AccordionItem(
                                    children=[
                                        html.Div(df_calc_rm, style={'width': '60%', 'display': 'inline-block',
                                                                    'overflowY': 'scroll', 'height': '400px'}),
                                        html.Div(bar_plot_calc,
                                                 style={'width': '40%', 'display': 'inline-block', 'height': '400px'}),
                                    ],
                                    title="Remodeling per image Mode - Calculated",
                                ),
                            ],
                            id="accordion-always-open",
                            always_open=True,
                        ),
                        html.Div(id="accordion-contents-open-ids", className="mt-3"),
                    ],
                ),
            ]
        )
        return content

