import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier


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

    # The estimator may sit inside an imblearn pipeline, which is not an
    # sklearn Pipeline subclass, and LogisticRegression nests a second pipeline
    # inside that. Unwrap until the final estimator is reached.
    seen = 0
    while hasattr(model, 'steps') and model.steps and seen < 10:
        model = model.steps[-1][1]
        seen += 1

    return isinstance(model, shap_supported_models)


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


def top_features(shap_values, n=10):
    shap_values = reduce_shap_to_class_1(shap_values)
    order = np.argsort(np.abs(shap_values.values).mean(axis=0))[-n:]
    top = shap_values[:, order]
    top.feature_names = [shap_values.feature_names[i] for i in order]
    top.data = shap_values.data[:, order]
    return top
