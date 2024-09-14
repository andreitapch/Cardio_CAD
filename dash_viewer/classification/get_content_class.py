import dash_bootstrap_components as dash_bc
from dash import html, dcc, Output, Input, callback, State, ctx, dash


class GetComponentClass:

    def get_checkbox(self, checkbox_id, label, value, margin):
        return dash_bc.Row([
            html.Br(),
            dash_bc.Checkbox(
                id=checkbox_id,
                label=label,
                value=value,
                style={'marginLeft': margin}
            ),
        ], style={'marginLeft': margin})

    def get_input_text(self, input_id, label, value, margin, width):
        return html.Div([
            html.Label(label, style={'fontWeight': 'bold', 'marginLeft': margin}),
            dcc.Input(
                id=input_id, type="number", debounce=True,
                value=value,
                style={'width': width, 'marginLeft': margin}
            )
        ])

    def get_dropdown(self, dropdown_id, label, options, value, margin, width):
        return dash_bc.Col([
            html.Br(),
            html.Label(label, style={'fontWeight': 'bold'}),
            dcc.Dropdown(
                id=dropdown_id,
                options=options,
                value=value
            ),
        ], style={'width': width,'marginLeft': margin})

    def get_button(self, button_id, button_text, width, margin):
        return dash_bc.Row([
            (html.Button(
                button_text,
                id=button_id,
                n_clicks=0,
                style={'width': width}
            ))
        ], style={'marginLeft': margin})

    def get_accordion(self, title, label, data_plot_id, metrics_output_id, cm_plot_id, scatter_plot_id, width, margin):
        return dash_bc.Accordion([
            # dash_bc.AccordionItem(
            #     title=title,
            #     children=[
            #         dash_bc.Row([
            #             html.Label(label, style={'fontWeight': 'bold'}),
            #             dcc.Graph(id=data_plot_id)
            #         ], justify="center", style={'width': width, 'margin': margin}),
            #     ],
            # ),
            # dash_bc.AccordionItem(
            # title="Metrics:",
            # children=[
            # dash_bc.Row([
            #     dash_bc.Col([
            #         dash_bc.Row([
            #             html.Div(id=metrics_output_id, style={'width': '60%', 'margin': 'auto'})
            #         ], justify="center", style={'width': '70%', 'margin': 'auto'}),
            #     ]),
            #     dash_bc.Col([
            #         dash_bc.Row([
            #             dcc.Graph(id=cm_plot_id)
            #         ], justify="center", style={'width': '70%', 'margin': 'auto'}),
            #     ]),
            # ]),
            # dash_bc.Row([
            #     html.Label('Scatter plot:', style={'fontWeight': 'bold'}),
            #     dcc.Graph(id=scatter_plot_id)
            # ], justify="center", style={'width': '70%', 'margin': 'auto'}),
            # ]),
        ], always_open=True)

    def get_accordion_item(self, title, label, data_plot_id, width, margin):
        return dash_bc.AccordionItem(
            title=title,
            children=[
                dash_bc.Row([
                    html.Label(label, style={'fontWeight': 'bold'}),
                    dcc.Graph(id=data_plot_id)
                ], justify="center", style={'width': width, 'margin': margin}),
            ],
        ),

    # def get_content(self):
    #     return html.Div([
    #         self.get_checkbox(checkbox_id='isImputed-id', label='Include imputed data', value=True, margin='10px'),
    #         self.get_input_text(input_id='textarea-k', label='Select number of neighbors:', value=None),
    #         dash_bc.Row([
    #
    #             self.get_dropdown('id-knn_Type-dropdown', 'label', 'options', 'default_value', 'style','margin')
    #             self.get_dropdown('id-knn_distance-dropdown', options=None, value='euclidean')
    #             self.get_dropdown('id-knn_weight-dropdown', options=None, value='uniform')
    #             self.get_dropdown('id-unbalanced_Knn-dropdown', options=None, value='None')
    #
    #         ]),
    #         html.Br(),
    #         self.get_accordion()
    #         self.get_accordion_item()
    #     ])
