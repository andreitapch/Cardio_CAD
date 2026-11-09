import streamlit as st

from cardio_app import state, ui
from cardio_app.data import traces
from cardio_app.plots import imputation as plots
from cardio_app.services import imputation

PAGE = "best_k"

st.title("Best K for KNN imputation")
source_col, k_col, mode_col = st.columns(3)
source = source_col.selectbox("Data source", [*traces.SOURCES, traces.COMBINED], key=state.key(PAGE, "source"))
max_k = k_col.number_input("Max K", min_value=1, max_value=50, value=10, key=state.key(PAGE, "max_k"))
mode = imputation.MASK_MODES[mode_col.selectbox("Simulated missing data", list(imputation.MASK_MODES), key=state.key(PAGE, "mode"))]

if st.button("Search", type="primary"):
    frame = ui.load(traces.cohort, source, 1, "original only")
    with st.spinner(f"Cross-validating K = 1..{max_k} on {frame['study_nr'].nunique()} studies"):
        state.store(PAGE, "run", ui.load(imputation.best_k, frame, int(max_k), mode))

run = state.load(PAGE, "run")
if run is None:
    st.stop()
st.write(f"Best K: {run.best_k}, lowest mean RRSSE {run.lowest:.4f}")
curve_col, table_col = st.columns((2, 1))
curve_col.pyplot(plots.k_curve(run.errors, run.best_k))
table_col.dataframe(run.errors, hide_index=True)
study = st.selectbox("Study number", sorted(run.rows["study_nr"].unique()), key=state.key(PAGE, "study"))
rows = run.rows[run.rows["study_nr"] == study]
st.pyplot(plots.imputed_vs_original(rows, f"Imputed vs original, study {study}, K = {run.best_k}"))
st.dataframe(rows[["label", "fold", "rrsse"]], hide_index=True)
