# import dash
# from dash import dcc
# from dash import Dash, dcc, html, Input, Output, State, callback
# from pattern_recognition.cad_main import main_process
#
# global speqle_data_processor
#
# class WelcomeViewer:
#     def __init__(self, app):
#         self.store = dcc.Store(id='main-process-result-store')
#         self.layout = self.get_content()
#         self.register_callbacks(app)
#         self.main_process_result = None
#
#     def get_content(self):
#         layout = html.Div([
#             html.H1('CAD'),
#             html.Br(),
#             html.Label('Interpolation for temporal alignment'),
#             dcc.Dropdown(
#                 id='interpSelection-dropdown',
#                 options=[
#                     {'label': 'Average', 'value': 'avg'},
#                     {'label': 'Maximum', 'value': 'max'},
#                 ],
#                 value='avg',
#                 style={'width': '200px'}
#             ),
#             html.Br(),
#             html.Br(),
#             html.Label('Concatenate Traces By'),
#             dcc.Dropdown(
#                 id='concatTracesBy-dropdown',
#                 options=[
#                     {'label': 'Segment', 'value': 'segment'},
#                     {'label': 'Concatenated traces', 'value': 'full_traces'},
#                 ],
#                 value='segment',
#                 style={'width': '200px'}
#             ),
#             html.Br(),
#             html.Br(),
#             html.Label('Selected Data Source'),
#             dcc.Dropdown(
#                 id='selectedSourceData-dropdown',
#                 options=[
#                     {'label': 'SPEQLE', 'value': 'SPEQLE'},
#                     {'label': 'ST', 'value': 'ST'},
#                 ],
#                 value='SPEQLE',
#                 style={'width': '200px'}
#             ),
#             html.Br(),
#             html.Br(),
#             html.Button('Submit', id='submit-button', className="btn btn-primary", n_clicks=0),
#             html.Div(id='container-button-basic'),
#             self.store,
#             dcc.Interval(id='refresh-interval', interval=1 * 1000),  # Refresh every 1 second
#         ], style={'padding-left': '20px'})
#         return layout
#
#     def register_callbacks(self, app):
#         @app.callback(
#             [Output('container-button-basic', 'children'),
#              Output('main-process-result-store', 'data')],
#             [Input('submit-button', 'n_clicks')],
#             [State('interpSelection-dropdown', 'value'),
#              State('concatTracesBy-dropdown', 'value'),
#              State('selectedSourceData-dropdown', 'value')]
#         )
#         def update_output(n_clicks, interpSelection, concatTracesBy, selectedSourceData):
#             global speqle_data_processor
#             if n_clicks > 0:
#                 speqle_data_processor = main_process(interpSelection, concatTracesBy, selectedSourceData)
#                 main_process_result = dill.dumps(speqle_data_processor)
#
#                 # main_process_result = str(speqle_data_processor)
#                 return "", main_process_result
#             else:
#                 return "", None
#
#         @app.callback(
#             Output('refresh-div', 'children'),
#             [Input('refresh-interval', 'n_intervals')],
#             [State('container-button-basic', 'children'),
#              State('main-process-result-store', 'data')]
#         )
#         def refresh_content(n, button_children, store_data):
#             if button_children == "" and store_data is not None:
#                 return dcc.Location(href="/", refresh=True)
#             return ""