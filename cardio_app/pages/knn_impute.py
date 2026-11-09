import streamlit as st

from cardio_app import state, ui
from cardio_app.data import traces
from cardio_app.plots import bullseye
from cardio_app.plots import imputation as plots
from cardio_app.services import imputation
from cardio_app.services import traces as service

PAGE = "knn_impute"

st.title("KNN imputation of missing segments")
source_col, k_col, band_col = st.columns(3)
source = source_col.selectbox("Data source", list(traces.SOURCES), key=state.key(PAGE, "source"))
k = k_col.number_input("K", min_value=1, max_value=50, value=5, key=state.key(PAGE, "k"))
band = imputation.MISSING_BANDS[band_col.selectbox("Missing segments per study", list(imputation.MISSING_BANDS), key=state.key(PAGE, "band"))]

if st.button("Impute", type="primary"):
    rows = ui.load(traces.rows, source, "original only")
    complete = service.complete(rows)
    incomplete = imputation.candidates(rows, band)
    if complete.empty or incomplete.empty:
        st.warning("No complete training studies or no studies in the selected band")
        st.stop()
    with st.spinner(f"Imputing {incomplete['study_nr'].nunique()} studies from {complete['study_nr'].nunique()} complete ones"):
        filled = imputation.with_missing_rows(incomplete, ui.load(traces.segments), source)
        state.store(PAGE, "run", (source, ui.load(imputation.impute, complete, filled, int(k))))

stored = state.load(PAGE, "run")
if stored is None:
    st.stop()
run_source, run = stored
imputed = run.rows[run.rows["isImputed"] == 1]
st.write(f"{run_source}: {len(run.studies)} studies, {len(imputed)} imputed segments")
study = st.selectbox("Study number", run.studies, key=state.key(PAGE, "study"))
rows = run.rows[run.rows["study_nr"] == study]
labels = dict(zip(rows["position"], rows["curve_segment"]))
missing = set(rows.loc[rows["isImputed"] == 1, "position"])
eye_col, trace_col = st.columns((1, 2))
eye_col.pyplot(bullseye.bullseye(labels, missing, f"Imputed segments, study {study}"))
trace_col.pyplot(plots.study_traces(rows, f"Strain traces, study {study}"))

confirm = st.checkbox(f"Replace imputed rows of these {len(run.studies)} studies in {run_source}", key=state.key(PAGE, "confirm"))
if st.button("Save to database", disabled=not confirm):
    count = ui.load(traces.save_imputed, run_source, imputed)
    st.success(f"Wrote {len(imputed)} imputed segments ({count} rows affected)")
