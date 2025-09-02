import gc
import shap
import numpy as np
import pandas as pd
import xgboost as xgb
import lime.lime_tabular
from dash import html, dcc, Output, Input, ctx
import dash_bootstrap_components as dash_bc
from dash import callback, State, dash, dash_table

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.feature_selection import f_classif, mutual_info_classif, RFE

from dash_viewer.classification.load_prepare_data import get_echoParams_data
from dash_viewer.classification.machine_learning.stat_analysis import analyze_selected_features
from dash_viewer.classification.machine_learning.utils_classifier import convert_echoParam_values, preprocess_data
from large_language_models.MedicalLLMInterpreter import MedicalLLMInterpreter
from pattern_recognition.utils.parameters_dics import dbTableNames, dbConParameters
from pattern_recognition.data.database_connection.db_con import DatabaseConnection as dbc

gc.enable()

import random
random.seed(42)
np.random.seed(42)


@callback(
    Output('id-feature-selection-table', 'children'),
    Output('id-feature-importance-plot', 'figure'),
    Output('id-correlation-plot', 'figure'),
    Output('id-llm-interpretation-div', 'children'),
    Input('id-best-params-btn-echo', 'n_clicks'),
    Input('id-feature-slider', 'value'),
    State('id-data-source-dropdown_bc-echo', 'value'),
    Input('id-thresholdn_bc-echo', 'value')

)
def update_feature_selection_table(n_clicks, k_feature_sel, data_source, threshold):
    if "id-best-params-btn-echo" == ctx.triggered_id and  n_clicks > 0:
        echo_paramDF, feature_col_name = get_echoParams_data(None, data_source, None)
        echo_paramDF = convert_echoParam_values(echo_paramDF)
        X, y, patient_study_info,scaler = preprocess_data(echo_paramDF, True)

        feature_scores, selected_features = compute_feature_importance(X, y, k_feature_sel, threshold)

        feature_importance_fig, correlation_heatmap_fig = analyze_selected_features(X, y, feature_scores, selected_features)
        selected_feat_table = create_feature_table(feature_scores)

        #LLM
        # llm_interpreter = MedicalLLMInterpreter(model_dir="path_to_cache")
        # feature_dict = {"Weighted Selection": selected_features}
        # llm_analysis_div = llm_interpreter.interpret_results(feature_dict)

        if selected_features:
            feature_scores_df = feature_scores.reset_index()
            feature_scores_df = feature_scores_df.rename(columns={'index': 'Feature'})
            feature_scores_df = feature_scores_df[['Feature', 'Weighted Consensus Score']].copy()
            table_name = dbTableNames['calculated_features_remodeling_T']
            insert_into_db(table_name, feature_scores_df)

        return selected_feat_table, feature_importance_fig.to_dict(), correlation_heatmap_fig.to_dict(), dash.no_update#, llm_analysis_div


    return dash.no_update, dash.no_update, dash.no_update, dash.no_update


def create_feature_table(top_features):
    # data = top_features.reset_index()
    data = top_features.reset_index().sort_values(by='Weighted Consensus Score', ascending=False)

    columns_to_display = ['index', 'Weighted Consensus Score']
    data = data[columns_to_display]
    data_records = data.to_dict("records")

    selected_feat_table = dash_table.DataTable(
        data=data_records,
        columns=[{"name": col, "id": col} for col in columns_to_display],
        style_table={'marginBottom': '20px'},
        style_cell={'textAlign': 'left',
                    'whiteSpace': 'normal',
                    'overflow': 'hidden',
                    'textOverflow': 'ellipsis',
                    'maxWidth': 'none'},
        style_data_conditional=[
            {
                'textAlign': 'left',
                'fontWeight': 'bold'
            }
        ],
    )
    return selected_feat_table


def insert_into_db(table_name, feature_scores):
    try:
        db = dbc(driver=dbConParameters['driver'], server=dbConParameters['server'],
                 database=dbConParameters['database'])
        db.connect()
        db.cursor.execute(f"DBCC CHECKIDENT ('{table_name}', RESEED, 0);")
        db.delete_query(table_name)
        for index, row in feature_scores.iterrows():
            feature = row['Feature']
            score = row['Weighted Consensus Score']
            db.insert_values(table_name, [feature, float(score)])
        db.conn.commit()

        print("Feature scores successfully inserted into database.")
    except Exception as e:
        print(f"Database operation error: {e}")
    finally:
        db.close_connection()


def compute_feature_importance(X, y, k, threshold):

    feature_scores = pd.DataFrame(index=X.columns)

    feature_scores['F-statistic'], _ = f_classif(X, y)
    feature_scores['Mutual Information'] = mutual_info_classif(X, y)

    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)
    feature_scores['Random Forest'] = rf.feature_importances_

    lr = LogisticRegression(max_iter=500, solver='liblinear')
    rfe = RFE(lr, n_features_to_select=1)
    rfe.fit(X, y)
    feature_scores['RFE Importance'] = 1 / (rfe.ranking_ + 1e-10)

    shap_values, _ = compute_shap_values(X, y)

    feature_scores['SHAP'] = np.abs(shap_values.values).mean(axis=0)

    for col in ['F-statistic', 'Mutual Information', 'Random Forest', 'SHAP']:
        feature_scores[col] = (feature_scores[col] - feature_scores[col].min()) / \
                              (feature_scores[col].max() - feature_scores[col].min() + 1e-10)

    feature_scores['Weighted Consensus Score'] = (
            0.3 * feature_scores['SHAP'] +
            0.25 * feature_scores['Random Forest'] +
            0.2 * feature_scores['Mutual Information'] +
            0.15 * feature_scores['F-statistic'] +
            0.1 * feature_scores['RFE Importance']
    )

    feature_scores = feature_scores.sort_values(by='Weighted Consensus Score', ascending=False).round(2)

    if threshold is not None:
        feature_scores_filtered = feature_scores[feature_scores['Weighted Consensus Score'] >= threshold]
    else :
        feature_scores_filtered = feature_scores

    selected_features = feature_scores_filtered.head(k).index.tolist() if k and k < len(feature_scores) \
        else feature_scores_filtered.index.tolist()

    return feature_scores_filtered, selected_features


def compute_shap_values(X, y):
    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss")
    model.fit(X, y)
    explainer = shap.Explainer(model, seed=42)
    shap_values = explainer(X)

    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    shap.plots.beeswarm(shap_values)
    return shap_values, X.columns


class BestEchoParamsViewer:

    def __init__(self):
        self.studyType = 'BL'
        self.table_name = None

    def get_content(self):
        return html.Div([
            dash_bc.Row([
                dash_bc.Col([
                    html.Br(),
                    html.Label('Select Data Source:', style={'fontWeight': 'bold', 'marginLeft': '20px'}),
                    dcc.Dropdown(
                        id='id-data-source-dropdown_bc-echo',
                        options=[
                            {'label': 'Speqle', 'value': 'Speqle'},
                            {'label': 'GE', 'value': 'GE'},
                            {'label': 'Philips', 'value': 'Philips'},
                            {'label': 'GE + Philips', 'value': 'GE_Philips'}
                        ],
                        value='GE_Philips',
                        style={'width': '300px', 'marginLeft': '10px'}
                    ),
                ], width=3),
                dash_bc.Col([
                    html.Br(),
                    html.Label('Select Features to Keep (K):', style={'fontWeight': 'bold', 'marginLeft': '20px'}),
                    dcc.Slider(
                        id='id-feature-slider',
                        min=1,
                        max=20,
                        step=1,
                        value=10,
                        marks={i: f"{i}" for i in range(1, 21)},
                        tooltip={"placement": "bottom", "always_visible": True},
                    ),
                ], width=5),
                dash_bc.Col([
                    html.Br(),
                    html.Label('Select Threshold:', style={'fontWeight': 'bold', 'marginLeft': '20px'}),
                    dcc.Input(
                        id='id-thresholdn_bc-echo',
                        type='number',
                        value=None,
                        style={'width': '100px', 'marginLeft': '10px'}
                    ),

                ])
            ]),
            html.Br(),
            dash_bc.Row([
                dash_bc.Col([
                    html.Br(),
                    html.Label('Find the best echo parameters:', style={'fontWeight': 'bold', 'marginLeft': '20px'}),
                    dash_bc.Button('Generate', id='id-best-params-btn-echo', n_clicks=0,
                                   style={'marginLeft': '30px', 'width': '120px'})
                ], width=3),
            ]),
            dash_bc.Row([
                html.Br(),
                html.Div(id='id-feature-selection-table',
                         style={'marginLeft': '10px'}),
            ]),
            dash_bc.Row([
                html.Div([
                    html.H4("Feature Selection and Correlation Analysis"),
                    dcc.Graph(id='id-feature-importance-plot'),
                    html.Br(),
                    dcc.Graph(id='id-correlation-plot'),
                    html.Br(),
                ])
            ]),
            dash_bc.Row([
                html.Div(id='id-llm-interpretation-div',
                         style={'marginLeft': '10px', 'marginRight': '10px'}),
            ]),
        ])
