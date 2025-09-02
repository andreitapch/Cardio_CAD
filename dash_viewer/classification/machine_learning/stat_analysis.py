import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns


def analyze_selected_features(X, y, feature_scores, selected_features):
    scaler = MinMaxScaler()
    df_normalized = pd.DataFrame(
        scaler.fit_transform(feature_scores),
        columns=feature_scores.columns,
        index=feature_scores.index
    )
    feature_importance_fig = plot_feature_importance_by_class(X, y, selected_features)
    feature_selection_comparison_fig = plot_feature_importance_comparison(feature_scores)

    # return feature_importance_fig
    return feature_selection_comparison_fig, feature_importance_fig


def plot_feature_importance_by_class(X, y, selected_features):
    X_selected = X[selected_features]
    df = X_selected.copy()
    df['Class'] = y.values
    df_grouped = df.groupby('Class').mean().T

    fig = go.Figure()
    for cls in df_grouped.columns:
        fig.add_trace(
            go.Bar(
                x=df_grouped.index,
                y=df_grouped[cls],
                name=f'Remodeling {cls}'
            )
        )
    fig.update_layout(
        title='Average Feature Values by Remodeling Class',
        xaxis_title='Selected Features',
        yaxis_title='Average Feature Value',
        barmode='group'
    )
    return fig


def plot_feature_importance_comparison(feature_scores):

    importance_cols = ['F-statistic', 'Mutual Information', 'Random Forest', 'RFE Importance', 'SHAP', 'LIME']

    feature_scores = feature_scores.reset_index()
    feature_scores.rename(columns={'index': 'Feature'}, inplace=True)

    feature_scores_melted = feature_scores.melt(id_vars='Feature', var_name='Method', value_name='Importance Score')

    sorted_features = feature_scores.sort_values(by='SHAP', ascending=False)['Feature']

    fig = px.bar(feature_scores_melted,
                 x='Feature',
                 y='Importance Score',
                 color='Method',
                 category_orders={"Feature": sorted_features.tolist()},
                 title="Comparison of Feature Importance Across Methods",
                 labels={'Feature': 'Features', 'Importance Score': 'Importance', 'Method': 'Method'})

    fig.update_layout(xaxis_tickangle=-45)
    fig.update_layout(legend=dict(title="Method", orientation="h", y=1, yanchor="bottom", x=0.5, xanchor="center"))

    return fig

