import dash_bootstrap_components as dbcomp
from dash import html, Input, Output, dash, callback, ctx


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