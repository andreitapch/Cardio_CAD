import pandas as pd
import streamlit as st

from cardio_app import panels, state, ui
from cardio_app.data import classification as store
from cardio_app.jobs import classify as jobs
from cardio_app.plots import classification as plots
from cardio_app.services.classification import comparison
from cardio_app.services.classification.engine import default_scaling_for
from cardio_app.services.classification.model_params import CLASSIFIER_LABELS, COMPARISON_DEFAULT_MODELS

PAGE = "compare_classifiers"
CLASS_INPUTS = {"Reconstructed traces": "reconstructed_traces", "Components": "components", "Features": "features_class", "GLS": "gls_class"}
INPUT_TYPES = {"Full traces": "full_traces", "Segments": "segment"}
SCALING = {"Auto (off for raw traces)": "auto", "Always": "on", "Never": "off"}
IMBALANCE = {option["label"]: option["value"] for option in comparison.IMBALANCE_OPTIONS}
REDUCERS = [option["value"] for option in comparison.REDUCER_OPTIONS]
TABLE_COLUMNS = ["rank", "model", "sampler", "PR-AUC", "PR-AUC_CI_low", "PR-AUC_CI_high", "lift_over_baseline", "AUC",
                 "BalancedAccuracy", "MCC", "Recall", "Precision", "Specificity", "Brier", "threshold", "n_fits", "tied_with_best", "error"]

st.title("Compare classifiers")
row = st.columns(4)
data_source = row[0].selectbox("Data source", ui.load(store.data_sources), key=state.key(PAGE, "source"))
class_input = CLASS_INPUTS[row[1].selectbox("Classify by", list(CLASS_INPUTS), key=state.key(PAGE, "class_input"))]
input_type = INPUT_TYPES[row[2].selectbox("Input", list(INPUT_TYPES), key=state.key(PAGE, "input_type"))]
n_components = row[3].selectbox("Components", ui.load(store.component_options, class_input, input_type), key=state.key(PAGE, "components"))
row = st.columns(4)
scaling = SCALING[row[0].selectbox("Standardise", list(SCALING), key=state.key(PAGE, "scaling"))]
imbalance = IMBALANCE[row[1].selectbox("Imbalance", list(IMBALANCE), index=len(IMBALANCE) - 1, key=state.key(PAGE, "imbalance"))]
reducer = row[2].selectbox("Reduce inside CV", REDUCERS, key=state.key(PAGE, "reducer"))
reducer_components = row[3].number_input("Reducer components", min_value=2, value=comparison.DEFAULT_REDUCER_COMPONENTS, key=state.key(PAGE, "reducer_k"))
row = st.columns(4)
n_repeats = row[0].number_input("Repeats", min_value=1, value=comparison.DEFAULT_REPEATS_UI, key=state.key(PAGE, "repeats"))
budget = row[1].number_input("Search budget", min_value=1, value=comparison.DEFAULT_SEARCH_BUDGET, key=state.key(PAGE, "budget"))
beta = row[2].number_input("Beta (threshold F-score)", min_value=0.1, value=2.0, step=0.1, key=state.key(PAGE, "beta"))
imputed = row[3].checkbox("Include imputed data", value=True, key=state.key(PAGE, "imputed"))
add_gls = row[3].checkbox("Add GLS", key=state.key(PAGE, "gls"))
add_echo = row[3].checkbox("Add selected echo features", key=state.key(PAGE, "echo"))
models = st.multiselect("Models", list(CLASSIFIER_LABELS), default=COMPARISON_DEFAULT_MODELS, format_func=CLASSIFIER_LABELS.get, key=state.key(PAGE, "models"))
total, per_model = comparison.estimate_cost(len(models), n_repeats, budget, imbalance == "search")
st.caption(f"About {total:,} model fits ({per_model:,} per model). This can run for hours; the table fills in as each model finishes.")

job = state.load(PAGE, "job")
if st.button("Compare", type="primary", disabled=(job is not None and not job.finished()) or not models or n_components is None):
    search = imbalance == "search"
    settings = {"data_source": data_source, "isImputed": imputed, "class_input": class_input, "input_type": input_type,
                "n_components": str(n_components), "addGLS": add_gls, "addFeatEcho": add_echo,
                "scale_features": {"on": True, "off": False}.get(scaling, default_scaling_for(class_input)),
                "scaling_mode": scaling, "unbalanced_type": "None" if search else imbalance, "loader_unbalanced": "None" if search else imbalance,
                "search_sampler": search, "n_repeats": int(n_repeats), "search_budget": int(budget), "beta": float(beta),
                "reducer": None if reducer == "none" else reducer, "reducer_components": int(reducer_components)}
    ui.launch(PAGE, jobs.start(f"Comparing {len(models)} models", {"kind": "sweep", "models": models, "settings": settings}))

job = state.load(PAGE, "job")
if job is None:
    st.stop()
ui.watch(PAGE, job.meta["output"])
if not job.meta["output"].is_file():
    st.stop()
snapshot = jobs.load(job.meta["output"])
results = snapshot["results"]
if snapshot["fatal_error"]:
    st.error(snapshot["fatal_error"])
    st.stop()
config = comparison.build_run_config(snapshot, partial=not snapshot["finished"])
rows = comparison.build_report_rows(results)
st.write(comparison.build_verdict(rows, config))
if config["pr_auc_baseline"] is not None:
    st.caption(f"PR-AUC baseline (prevalence): {config['pr_auc_baseline']:.3f}. Read every PR-AUC as a multiple of this.")
st.dataframe(pd.DataFrame(rows)[TABLE_COLUMNS], hide_index=True)
save_col, csv_col = st.columns(2)
if save_col.button("Save report"):
    paths, error = comparison.write_comparison_report(snapshot, partial=not snapshot["finished"])
    save_col.success(f"Saved {', '.join(str(p) for p in paths.values())}" if not error else error)
with csv_col:
    panels.csv_download(comparison.build_report_frame(results), comparison.report_basename(snapshot, partial=not snapshot["finished"]))
curves = [(comparison._label_for(r["class_type"]), *comparison._pooled_arrays(r)) for r in results if not r.get("error")]
curves = [c for c in curves if c[1] is not None]
if curves:
    pr_col, roc_col = st.columns(2)
    pr_col.pyplot(plots.pr_curves(curves, config["pr_auc_baseline"]))
    roc_col.pyplot(plots.roc_curves(curves))
scored = [r for r in results if not r.get("error") and r.get("cm") is not None]
if scored:
    chosen = st.selectbox("Confusion matrix", [r["class_type"] for r in scored], format_func=CLASSIFIER_LABELS.get, key=state.key(PAGE, "cm_model"))
    result = next(r for r in scored if r["class_type"] == chosen)
    st.pyplot(plots.confusion_matrix(result["cm"], f"{CLASSIFIER_LABELS[chosen]} at threshold {result.get('threshold'):.3f}"))
