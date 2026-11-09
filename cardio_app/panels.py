from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from cardio_app import state
from cardio_app.plots import classification as plots
from cardio_app.services.classification import clinical
from cardio_app.services.classification.engine import shap_unavailable_reason


def long_frame(sections):
    rows = [{"section": section, "field": field, "value": value}
            for section, record in sections for field, value in (record or {}).items()]
    return pd.DataFrame(rows, columns=["section", "field", "value"])


def csv_download(frame, name, label="Download CSV"):
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    st.download_button(label, frame.to_csv(index=False).encode("utf-8"), f"{name}_{stamp}.csv", "text/csv")


def classification_result(page, result):
    config = result["config"]
    if not result["best_params"]:
        st.error("The classification run produced no result. See the job log.")
        return
    st.caption(" | ".join(f"{key}: {value}" for key, value in config.items() if key != "kind"))
    st.write(f"Metrics are pooled over held-out outer folds ({int(result['cm'].sum()) if result['cm'] is not None else 0} patients, "
             f"each predicted once by a model that did not train on them). The inner PR-AUC is the selection score, not a performance estimate.")
    params = pd.DataFrame([result["best_params"]])
    params["Inner PR-AUC (selection)"] = round(result["inner_score"], 4) if result["inner_score"] is not None else None
    st.dataframe(params, hide_index=True)
    metrics = pd.DataFrame([result["metrics"]]).round(4)
    st.dataframe(metrics, hide_index=True)
    csv_download(long_frame([("run_config", config), ("metrics", result["metrics"]), ("best_params", result["best_params"])]),
                 f"classification_{config['class_type']}")
    cm_col, shap_col = st.columns((1, 2))
    if result["cm"] is not None:
        cm_col.pyplot(plots.confusion_matrix(result["cm"], "Confusion matrix"))
    reason = shap_unavailable_reason(config["class_type"], None)
    if reason:
        shap_col.info(reason)
    elif result["shap_values"] is None or not result["important_features"]:
        shap_col.warning("SHAP produced no ranking. See the job log.")
    else:
        figure = plots.shap_beeswarm(result["shap_values"], "SHAP importance, pooled over held-out folds")
        shap_col.pyplot(figure)
        plt.close(figure)
        shap_col.write("Top features: " + ", ".join(result["important_features"]))
    loadings = result.get("loadings")
    pca_features = [feature for feature in result.get("pca_features", []) if "pca_results_" in feature]
    if loadings is not None and pca_features:
        components = [int(feature.split("_")[-1]) for feature in pca_features]
        component = st.selectbox("Principal component", components, key=state.key(page, "component"))
        curves, summary = clinical.loading_segments(loadings[component])
        st.pyplot(plots.loading_segments(curves, component))
        st.dataframe(summary, hide_index=True)
