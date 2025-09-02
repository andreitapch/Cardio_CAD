import io
import os

import shap
import base64
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier

from pattern_recognition.utils.parameters_dics import parameters


def compute_shap_timepoints(model, X_val, feature_names):
    try:
        X_df = pd.DataFrame(X_val, columns=feature_names)
        object_cols = X_df.select_dtypes(include=['object']).columns.tolist()
        if object_cols:
            print(f"[DEBUG] Object-type columns found: {object_cols}")
        X_df = X_df.astype(float)
        explainer = shap.Explainer(model, X_df)
        shap_values = explainer(X_df)
        return shap_values, X_df

    except Exception as e:
        print(f"[ERROR] SHAP computation failed: {e}")
        return None, None



def compute_shap_reduced(model, X_val, feature_names):
    try:
        X_df = pd.DataFrame(X_val, columns=feature_names)
        X_df = X_df.astype(float)
        explainer = shap.Explainer(model, X_df)
        shap_values = explainer(X_df)
        print("SHAP shape:", shap_values.values.shape)
        print("Classes in model:", model.classes_)  # Should be [0, 1]

        return shap_values, X_df

    except Exception as e:
        print(f"[ERROR] SHAP computation failed: {e}")
        return None, None



def is_shap_supported_model(model):
    shap_supported_models = (
        XGBClassifier,
        RandomForestClassifier,
        GradientBoostingClassifier,
        LGBMClassifier,
    )

    if isinstance(model, Pipeline):
        model = model.steps[-1][1]

    return isinstance(model, shap_supported_models)


def shap_beeswarm_base64(shap_values, class_input):
    if shap_values is None:
        return None

    # if len(shap_values.shape) == 3:
    #     shap_values = shap_values[:, :, 1]
    shap_values = reduce_shap_to_class_1(shap_values)

    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
    top_indices = np.argsort(mean_abs_shap)[-10:]

    top_shap = shap_values[:, top_indices]
    top_shap.feature_names = [shap_values.feature_names[i] for i in top_indices]
    top_shap.data = shap_values.data[:, top_indices]

    is_time_series = class_input in ['reconstructed_traces', 'reconstructed_traces_vac']
    title = "SHAP Time Step Importance" if is_time_series else "SHAP Feature Importance"

    fig = plt.figure()
    shap.plots.beeswarm(top_shap, show=False)
    plt.title(title)

    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)

    encoded = base64.b64encode(buf.read()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


def select_important_features(shap_values, feature_names, top_n=10):
    try:
        if not hasattr(shap_values, 'values'):
            raise ValueError("Expected a shap.Explanation object.")

        shap_values = reduce_shap_to_class_1(shap_values)
        shap_vals = shap_values.values

        abs_shap = np.abs(shap_vals)
        mean_abs_shap = abs_shap.mean(axis=0)

        mean_abs_shap = np.ravel(mean_abs_shap)
        feature_names = np.ravel(feature_names)

        if len(mean_abs_shap) != len(feature_names):
            raise ValueError(f"Shape mismatch: {len(mean_abs_shap)} SHAP vs {len(feature_names)} features")

        shap_importance_df = pd.DataFrame({
            'feature': feature_names,
            'mean_abs_shap': mean_abs_shap
        }).sort_values(by='mean_abs_shap', ascending=False)

        top_features_df = shap_importance_df.head(top_n)
        important_features = top_features_df['feature'].tolist()

        return important_features, top_features_df

    except Exception as e:
        print(f"[ERROR] SHAP feature selection failed: {e}")
        return None, None

def reduce_shap_to_class_1(shap_values):
    """Convert SHAP Explanation from 3D to 2D (binary class 1), safer copy."""
    if hasattr(shap_values, "values") and shap_values.values.ndim == 3:
        shap_values = shap_values[:, :, 1]
    return shap_values


def save_shap_beeswarm_svg(shap_values, class_input, file_name='shap_beeswarm.svg'):
    """
    Save SHAP beeswarm plot as an SVG file to a defined folder.
    """
    dir_path = parameters['save_plot_dir']
    os.makedirs(dir_path, exist_ok=True)

    full_path = os.path.join(dir_path, file_name)

    shap_values = reduce_shap_to_class_1(shap_values)

    is_time_series = class_input in ['reconstructed_traces', 'reconstructed_traces_vac']
    title = "SHAP Time Step Importance" if is_time_series else "SHAP Feature Importance"

    plt.figure()
    shap.plots.beeswarm(shap_values, show=False)
    plt.title(title)
    plt.savefig(full_path, format='svg', bbox_inches='tight')
    plt.close()