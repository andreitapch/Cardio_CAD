import gc
import shap
import numpy as np
import pandas as pd
import xgboost as xgb

from dash import html, dcc, Output, Input
import dash_bootstrap_components as dash_bc
from dash import callback, State, dash, dash_table

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.feature_selection import f_classif, mutual_info_classif, RFE

from dash_viewer.classification.machine_learning.stat_analysis import analyze_selected_features
from dash_viewer.classification.machine_learning.utils_classifier import convert_echoParam_values, load_data_traces, \
    preprocess_data
from pattern_recognition.utils.parameters_dics import dbTableNames, dbConParameters
from pattern_recognition.data.database_connection.db_con import DatabaseConnection as dbc

gc.enable()


@callback(
    Output('id-feature-selection-table', 'children'),
    Output('id-feature-importance-plot', 'figure'),
    Output('id-correlation-plot', 'figure'),
    Input('id-best-params-btn-echo', 'n_clicks'),
    Input('id-feature-slider', 'value'),
    State('id-data-source-dropdown_bc-echo', 'value')
)
def update_feature_selection_table(n_clicks, k_feature_sel, data_source):
    if n_clicks > 0:
        echo_paramDF, feature_col_name = load_data_traces(data_source)
        echo_paramDF = convert_echoParam_values(echo_paramDF)
        X, y, patient_study_info = preprocess_data(echo_paramDF)
        feature_scores, selected_features = compute_feature_importance(X, y, k_feature_sel)
        correlation_matrix, feature_importance_fig, correlation_heatmap_fig = analyze_selected_features(feature_scores)
        selected_feat_table = create_feature_table(feature_scores)

        if selected_features:
            insert_into_db(selected_features)

        # return feature_table, html.Div([shap_text, shap_plot]), feature_importance_plot.to_dict(), correlation_heatmap.to_dict(), selected_feat_table
        return selected_feat_table, feature_importance_fig.to_dict(), correlation_heatmap_fig.to_dict()

    return dash.no_update, dash.no_update, dash.no_update


def create_feature_table(top_features):
    data = top_features.reset_index()
    columns_to_display = ['index', 'Consensus Score']
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
                'if': {'state': 'header'},
                'textAlign': 'left',
                'fontWeight': 'bold'
            }
        ],
    )
    return selected_feat_table


def insert_into_db(feature_scores):
    try:
        db = dbc(driver=dbConParameters['driver'], server=dbConParameters['server'],
                 database=dbConParameters['database'])
        db.connect()
        table_name = dbTableNames['calculated_features_remodeling_T']
        db.cursor.execute(f"DBCC CHECKIDENT ('{table_name}', RESEED, 0);")
        db.delete_query(table_name)

        with db.cursor as cursor:
            for method, features in feature_scores.items():
                for feature, score in features:
                    cursor.execute(f"INSERT INTO {table_name} (method, feature, score) VALUES (?, ?, ?)",
                                   (method, feature, float(score)))
            db.conn.commit()
        print("Feature scores successfully inserted into database.")
    except Exception as e:
        print(f"Database operation error: {e}")
    finally:
        db.close_connection()


def compute_feature_importance(X, y, k=None):
    threshold = 0.5
    feature_scores = pd.DataFrame(index=X.columns)

    # ANOVA F-statistic
    f_values, _ = f_classif(X, y)
    feature_scores['F-statistic'] = f_values

    # Mutual Information
    mi_scores = mutual_info_classif(X, y)
    feature_scores['Mutual Information'] = mi_scores

    # Random Forest Importance
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)
    feature_scores['Random Forest'] = rf.feature_importances_

    # Recursive Feature Elimination (RFE) with Logistic Regression
    lr = LogisticRegression(max_iter=500, solver='liblinear')
    rfe = RFE(lr, n_features_to_select=1)
    rfe.fit(X, y)
    feature_scores['RFE Ranking'] = rfe.ranking_

    # Call compute_shap_values to get SHAP feature importance and plot
    shap_values, shap_features = compute_shap_values(X, y)
    feature_scores['SHAP'] = np.abs(shap_values).mean(axis=0)

    # Normalize all importance scores (excluding RFE ranking)
    for col in ['F-statistic', 'Mutual Information', 'Random Forest', 'SHAP']:
        feature_scores[col] = (feature_scores[col] - feature_scores[col].min()) / \
                              (feature_scores[col].max() - feature_scores[col].min() + 1e-10)  # Min-max normalization

    # Convert RFE ranking to importance (higher rank = less important)
    feature_scores['RFE Importance'] = 1 / (feature_scores['RFE Ranking'] + 1e-10)
    feature_scores.drop(columns=['RFE Ranking'], inplace=True)

    # Aggregate scores (weighted sum of normalized values)
    feature_scores['Consensus Score'] = feature_scores.sum(axis=1)
    feature_scores = feature_scores.sort_values(by='Consensus Score', ascending=False)
    feature_scores = feature_scores.round(2)

    threshold = 0.5  # Define your threshold
    feature_scores_filtered = feature_scores[feature_scores['Consensus Score'] >= threshold]

    # Select top k features if specified
    if k < len(feature_scores):
        selected_features = feature_scores_filtered.head(k).index.tolist()
    else:
        selected_features = feature_scores_filtered.index.tolist()

    return feature_scores, selected_features


def compute_shap_values(X, y):
    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss")
    model.fit(X, y)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    if isinstance(shap_values, list):
        shap_values = shap_values[1]

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
                # dash_bc.Col([
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
            # ]),
        ])
