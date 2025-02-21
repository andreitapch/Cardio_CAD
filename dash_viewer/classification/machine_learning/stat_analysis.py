import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler


def analyze_selected_features(selected_features):
    scaler = MinMaxScaler()
    df_normalized = pd.DataFrame(
        scaler.fit_transform(selected_features),
        columns=selected_features.columns,
        index=selected_features.index
    )

    selected_features["Selection Count"] = (selected_features > 0).sum(axis=1)
    correlation_matrix = df_normalized.corr(method="spearman")

    feature_importance_fig = create_feature_importance_plot(df_normalized)
    correlation_heatmap_fig = create_correlation_heatmap(correlation_matrix)

    return correlation_matrix, feature_importance_fig, correlation_heatmap_fig


def create_feature_importance_plot(df_normalized):
    fig = go.Figure()

    for column in df_normalized.columns:
        fig.add_trace(go.Bar(
            x=df_normalized.index,
            y=df_normalized[column],
            name=column
        ))

    fig.update_layout(
        title="Feature Importance Across Selection Methods (Normalized)",
        xaxis_title="Features",
        yaxis_title="Normalized Importance Score",
        barmode="group"
    )

    return fig

def create_correlation_heatmap(correlation_matrix):
    fig = px.imshow(
        correlation_matrix,
        color_continuous_scale="RdBu",  # Use a valid colorscale, e.g., 'RdBu'
        title="Spearman Correlation Between Feature Selection Methods"
    )

    fig.update_layout(
        xaxis_title="Selection Methods",
        yaxis_title="Selection Methods"
    )

    return fig

