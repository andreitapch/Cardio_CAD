import numpy as np
import pandas as pd

import plotly.graph_objects as go
import dash_bootstrap_components as dash_bc

from dash import html, dcc, Output, Input, callback, State, ctx, dash
from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from dash_viewer.classification.load_prepare_data import select_data_type, load_imputed_data, get_concat_18_segments, \
    get_echoParams_data, get_pca_components
from dash_viewer.classification.test_classifiers import create_data_table_results, get_clinical_data, create_cm, \
    count_data, create_scatter_plot, split_data, calculate_metrics, extract_sex_data, load_data_traces, \
    classify_per_fold
from dash_viewer.dimensionality_reduction.dimen_reduction_viewer import load_table_db


@callback(
    Output('id-metrics-output_rf', 'children'),
    Output('id-cm-plot_rf', 'figure'),
    Output('id-data-count-plot_rf', 'figure'),
    Output('id-class-feature-plot_rf', 'figure'),
    Output('id-cm-male_rf', 'figure'),
    Output('id-cm-female_rf', 'figure'),
    Output('id-metrics-male_rf', 'children'),
    Output('id-metrics-female_rf', 'children'),
    Input('id-button_rf', 'n_clicks'),
    Input('isImputed-id_rf', 'value'),
    State('id-test_type-dropdown_rf', 'value'),
    State('id-unbalanced_class-dropdown_rf', 'value'),
    State('textarea-n_estimators-id', 'value'),
    State('Dropdown-max_features-id', 'value'),
    State('Dropdown-criterion-id', 'value'),
    State('textarea-max_depth-id', 'value'),

)
def update_output(btn_click, isImputed, test_type, unbalanced_type,
                  n_estimators, max_features, criterion,
                  max_depth):
    global instance_clf, param_table
    instance_clf = RFClassViewer()

    if "id-button_rf" == ctx.triggered_id and btn_click > 0:
        tracesDF, df_isImputed_1, df_isImputed_0 = load_imputed_data(studyType='BL')

        (concat_tracesDF, feature_col_name, males_df,
         females_df, over_50_df, under_50_df, reconstructed_tracesDF) = load_data_traces(test_type, isImputed,
                                                                                         df_isImputed_0)

        results, data_count_fig, scatter_fig, \
            cm_female, cm_male, metrics_male, metrics_female = classify_data_loader(n_estimators, max_features,
                                                                                    max_depth, criterion,
                                                                                    unbalanced_type,
                                                                                    reconstructed_tracesDF, test_type,
                                                                                    concat_tracesDF, feature_col_name,
                                                                                    males_df, females_df, over_50_df,
                                                                                    under_50_df)

        confusion_matrices, metrics_table = create_data_table_results(results, False)

        cm_fig = create_cm(confusion_matrices, instance_clf.clf, False)
        table_male = create_data_table_results(metrics_male, True)
        table_female = create_data_table_results(metrics_female, True)

        return metrics_table, cm_fig, data_count_fig, scatter_fig, cm_male, cm_female, table_male, table_female

    else:
        return (dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update,
                dash.no_update, dash.no_update)


def classify_data_loader(n_estimators, max_features, max_depth, criterion, unbalanced_type,
                         reconstructed_tracesDF, test_type, concat_tracesDF, feature_col_name,
                         males_df, females_df, over_50_df, under_50_df):

    results, pred_tracesDF = instance_clf.classify_data(concat_tracesDF, feature_col_name, unbalanced_type,
                                                        n_estimators, max_features, max_depth, criterion )

    cm_female, cm_male, metrics_male, metrics_female = extract_sex_data(females_df, males_df,
                                                                        pred_tracesDF, instance_clf.clf)
    cm_female_plot = create_cm(cm_female, instance_clf.clf, True)
    cm_male_plot = create_cm(cm_male, instance_clf.clf, True)

    data_count_fig = count_data(reconstructed_tracesDF)

    if not test_type == 'gls_remod_class':
        scatter_fig = create_scatter_plot(instance_clf.all_X_test, instance_clf.all_y_pred)
    else:
        scatter_fig = go.Figure()

    return results, data_count_fig, scatter_fig, cm_female_plot, cm_male_plot, metrics_male, metrics_female


class RFClassViewer:

    def __init__(self):
        self.all_y_pred = []
        self.all_X_test = []
        self.scatter_fig = go.Figure()
        self.y_pred = None
        self.y_test = None
        self.clf = None
        self.tracesDF = None
        self.df_isImputed_0 = None
        self.studyType = 'BL'
        self.pca_resultsDF = None
        self.test, self.train, self.train = None, None, None
        self.concat_array_curves = None
        self.reconstructed_tracesDF = None
        self.calc_remod_df = None

    def get_content(self):
        return html.Div([
            dash_bc.Row([
                html.Br(),
                dash_bc.Checkbox(
                    id='isImputed-id_rf',
                    label='Include imputed data',
                    value=True,
                    style={'marginLeft': '10px'}
                ),
            ], style={'marginLeft': '20px'}),

            dash_bc.Row([
                dash_bc.Col([
                    html.Br(),
                    html.Label('Classify by:', style={'fontWeight': 'bold'}),
                    dcc.Dropdown(
                        id='id-test_type-dropdown_rf',
                        options=[
                            {'label': 'Concatenating 18 segments',
                             'value': 'concat_18_segments_100'},
                            {'label': 'Concatenating 18 segments: 95% Variance',
                             'value': 'concat_18_segments_95'},
                            {'label': '100% of PCA variance components',
                             'value': '100_variance_components'},
                            {'label': '95% of PCA variance components',
                             'value': '95_Variance_components'},
                            {'label': '100% of PCA variance components per segment',
                             'value': '100_components_per_segment'},
                            {'label': '95% of PCA variance per segment concatenated',
                             'value': '95_components_per_segment'},
                            {'label': 'Classify using GLS',
                             'value': 'gls_remod_class'},
                            {'label': 'Classify using Echo Params',
                             'value': 'echo_remod_class'},

                        ],
                        value='concat_18_segments_100',
                    ),
                    html.Br(),
                    html.Button('Classify',
                                id='id-button_rf',
                                n_clicks=0,
                                style={'width': '100px'})
                ], width=3, style={'marginLeft': '20px'}),

                dash_bc.Col([
                    html.Br(),
                    html.Label('Unbalanced:', style={'fontWeight': 'bold'}),
                    dcc.Dropdown(
                        id='id-unbalanced_class-dropdown_rf',
                        options=[
                            {'label': 'None', 'value': 'None'},
                            {'label': 'SMOTE', 'value': 'SMOTE'},
                            {'label': 'ADASYN', 'value': 'ADASYN'},
                            {'label': 'SMOTEENN', 'value': 'SMOTEENN'},
                            {'label': 'SMOTETomek', 'value': 'SMOTETomek'}
                        ],
                        value='None',
                        style={'height': 'Auto', 'width': '70%'}
                    ),
                ], style={'width': '100%'}),

                dash_bc.Col([
                    dash_bc.Row([
                        html.Label('Select number of estimators:', style={'fontWeight': 'bold'}),

                        html.Br(),
                        html.Label('Estimators:', style={'fontWeight': 'bold'}),
                        html.Br(),
                        dcc.Input(
                            id="textarea-n_estimators-id",
                            type="number",
                            debounce=True,
                            value=100,
                            style={'width': '100px', 'marginLeft': '20px'}),

                        html.Br(),
                        html.Label('Max features:', style={'fontWeight': 'bold', 'marginLeft': '20px'}),
                        html.Br(),
                        dcc.Dropdown(
                            id="Dropdown-max_features-id",
                            options=[
                                {'label': 'None', 'value': 'None'},
                                {'label': 'int', 'value': 'int'},
                                {'label': 'float', 'value': 'float'},
                                {'label': 'auto', 'value': 'auto'},
                                {'label': 'sqrt', 'value': 'sqrt'},
                                {'label': 'log2', 'value': 'log2'}],
                            value='auto',
                            style={'width': '200px'}),
                        html.Br(),
                    ]),

                ], style={'width': '100%'}),

                dash_bc.Col([
                    html.Br(),
                    html.Label('Max depth:', style={'fontWeight': 'bold'}),
                    html.Br(),
                    dcc.Input(
                        id="textarea-max_depth-id",
                        type="number",
                        debounce=True,
                        value=None,
                        style={'width': '100px'}),
                    html.Br(),
                    html.Label('Criterion:', style={'fontWeight': 'bold'}),
                    dcc.Dropdown(
                        id="Dropdown-criterion-id",
                        options=[
                            {'label': 'Gini', 'value': 'gini'},
                            {'label': 'Entropy', 'value': 'entropy'}],
                        value='gini',
                        style={'width': '200px'}
                    ),
                ]),
            ], style={'width': '100%'}),

            html.Br(),
            dash_bc.Accordion([
                dash_bc.AccordionItem(
                    title="Data:",
                    children=[
                        dash_bc.Row([
                            html.Label('Data Count:', style={'fontWeight': 'bold'}),
                            dcc.Graph(id='id-data-count-plot_rf')
                        ], justify="center", style={'width': '50%', 'margin': 'auto'}),
                    ],
                ),
                dash_bc.AccordionItem(
                    title="Metrics:",
                    children=[
                        dash_bc.Row([
                            dash_bc.Col([
                                dash_bc.Row([
                                    html.Div(id='id-metrics-output_rf', style={'width': '60%', 'margin': 'auto'})
                                ], justify="center", style={'width': '70%', 'margin': 'auto'}),
                            ]),

                            dash_bc.Col([
                                dash_bc.Row([
                                    dcc.Graph(id='id-cm-plot_rf')
                                ], justify="center", style={'width': '70%', 'margin': 'auto'}),
                            ]),
                        ]),
                        dash_bc.Row([
                            html.Label('Scatter plot:', style={'fontWeight': 'bold'}),
                            dcc.Graph(id='id-class-feature-plot_rf')
                        ], justify="center", style={'width': '70%', 'margin': 'auto'}),
                    ]),
                dash_bc.AccordionItem(
                    title="Metrics Per Gender:",
                    children=[
                        dash_bc.Row([
                            dash_bc.Col([
                                html.Label('Male:', style={'fontWeight': 'bold'}),
                                dash_bc.Row([
                                    html.Div(id='id-metrics-male_rf', style={'width': '60%', 'margin': 'auto'})
                                ], justify="center", style={'width': '70%', 'margin': 'auto'}),
                                html.Br(),
                                dash_bc.Row([
                                    dcc.Graph(id='id-cm-male_rf')
                                ], justify="center", style={'width': '70%', 'margin': 'auto'}),
                            ]),

                            dash_bc.Col([
                                html.Label('Female:', style={'fontWeight': 'bold'}),
                                dash_bc.Row([
                                    html.Div(id='id-metrics-female_rf', style={'width': '60%', 'margin': 'auto'})
                                ], justify="center", style={'width': '70%', 'margin': 'auto'}),
                                html.Br(),
                                dash_bc.Row([
                                    dcc.Graph(id='id-cm-female_rf')
                                ], justify="center", style={'width': '70%', 'margin': 'auto'}),
                            ]),
                        ]),
                    ]
                ),
            ], always_open=False, )
        ])

    def classify_data(self, concat_tracesDF, feature_col_name, unbalanced_type,
                      n_estimators, max_features, max_depth, criterion):
        fold = 0
        results = []
        predictions_df = pd.DataFrame(columns=['study_nr', 'prediction'])
        self.clf = RandomForestClassifier(random_state=42, n_estimators=n_estimators, criterion=criterion,
                                          max_depth=max_depth,
                                          max_features=max_features, class_weight='balanced')

        results, pred_tracesDF = classify_per_fold(concat_tracesDF, feature_col_name, fold, predictions_df,
                                                   results, unbalanced_type, self.clf, self.all_y_pred, self.all_X_test)

        return results, pred_tracesDF
