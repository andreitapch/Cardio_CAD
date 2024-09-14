import ast
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.impute import KNNImputer

import dash_bootstrap_components as dbcomp
from dash import dcc, html, Input, Output, dash, State, callback, ctx

from dash_viewer.data_imputation.load_dataTables import load_data_table_db
from pattern_recognition.data.database_connection import sql_queries

from pattern_recognition.utils.parameters_dics import dbTableNames, dbConParameters

from pattern_recognition.data.database_connection.db_con import DatabaseConnection as dbc
from pattern_recognition.data_preprocessing.speqle.concatenate_traces import CurveConcatenator
from pattern_recognition.data.insert_to_db.insertSpeqleDB import InsertInterpTracesToDB, DeleteToDBWhere
from pattern_recognition.data.SPEQLE_data.loadTracesSpeqle import LoadTracesSpeqle as load_traces_speqle


@callback(
    Output('id-table-params', 'children'),
    Input('id-btn-impute', 'n_clicks'),
    Input('id-btn-insert-db', 'n_clicks')


)
def impute_params(btn_knn, btn_save):
    global instance_imputation
    instance_imputation = DataImputationParams()
    if 'id-btn-impute' == ctx.triggered_id:

        return dash.no_update
    if 'id-btn-insert-db' == ctx.triggered_id:
        return dash.no_update
    else:
        return dash.no_update


class DataImputationParams:

    def __init__(self):
        self.study_type = 'BL'


    def get_content(self):
        content = html.Div([
            dbcomp.Card([
                dbcomp.Row([
                    html.Br(),
                    dbcomp.CardBody([
                        dbcomp.Button("KNN Imputation", id="id-btn-impute", color="primary", className="mr-1",
                                      style={'padding': '10px', 'margin-bottom': '10px',
                                             'marginLeft': '10px'}),
                        html.Br(),
                        html.Div(id='id-table-params', style={'width': '60%', 'margin': 'auto'}),
                        html.Br(),
                    ], style={'padding': '10px', 'margin': '10px', 'width': '80%'}),
                ]),
                dbcomp.Row([
                    html.Br(),
                    dbcomp.Col([
                        dbcomp.Button("Save", id="id-btn-insert-db", color="primary", className="mr-1",
                                      style={'padding': '10px', 'margin-bottom': '10px',
                                             'marginLeft': '10px'}),
                    ], width=6),
                ]),
            ]),
        ], style={'marginLeft': '20px', 'margin-right': '20px', 'margin-top': '20px', 'margin-bottom': '20px'})
        return content