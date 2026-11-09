import streamlit as st

from cardio_app import state, ui
from cardio_app.data import echo, traces
from cardio_app.data import reduction as store
from cardio_app.jobs import reduce as job_reduce
from cardio_app.plots import reduction as plots
from cardio_app.plots import traces as trace_plots
from cardio_app.services import reduction
from cardio_app.services import traces as service

PAGE = "reduce"

st.title("Dimensionality reduction")
source_col, studies_col, mode_col, method_col = st.columns(4)
source = source_col.selectbox("Data source", [*traces.SOURCES, traces.COMBINED], index=3, key=state.key(PAGE, "source"))
studies = studies_col.selectbox("Studies", traces.STUDY_FILTERS, key=state.key(PAGE, "studies"))
mode = reduction.MODES[mode_col.selectbox("Traces", list(reduction.MODES), key=state.key(PAGE, "mode"))]
method = reduction.METHODS[method_col.selectbox("Method", list(reduction.METHODS), key=state.key(PAGE, "method"))]
k_col, extra_col, run_col = st.columns(3)
if method == "PCA":
    k = k_col.number_input("Components (0 = 95% variance, 100 = all)", min_value=0, max_value=100, value=3, key=state.key(PAGE, "pca_k"))
    with_echo = extra_col.checkbox("Add echo biomarkers", key=state.key(PAGE, "echo"))
    epochs = 0
else:
    k = k_col.number_input("Latent dimensions", min_value=2, max_value=64, value=8, step=2, key=state.key(PAGE, "ae_k"))
    epochs = extra_col.number_input("Epochs", min_value=10, max_value=5000, value=1000, step=10, key=state.key(PAGE, "epochs"))
    with_echo = False

job = state.load(PAGE, "job")
busy = job is not None and not job.finished()
if run_col.button("Run", type="primary", disabled=busy):
    frame = ui.load(traces.cohort, source, 1, studies)
    items = reduction.blocks(frame, mode, ui.load(echo.biomarkers) if with_echo else None)
    state.store(PAGE, "context", {"source": source, "mode": mode, "method": method, "echo": with_echo, "items": items, "frame": frame})
    ui.launch(PAGE, job_reduce.start(f"{method} on {source}, {mode}", items, [method], int(k), int(epochs)))

done = ui.watch(PAGE)
context = state.load(PAGE, "context")
if done is None or context is None:
    st.stop()
result = state.load(PAGE, "result") or state.store(PAGE, "result", reduction.load_result(done.meta["output"])[context["method"]])
items, frame = context["items"], context["frame"]
st.dataframe(reduction.summary_table(result).set_index("block"))
if result.history:
    st.pyplot(plots.loss_curves(result.history, "Autoencoder validation loss"))
reconstructed = reduction.reconstructed_frame(items, result)
study = st.selectbox("Study number", sorted(reconstructed["study_nr"].unique()), key=state.key(PAGE, "study"))
left, right = st.columns(2)
left.pyplot(trace_plots.segment_lines(service.study(frame, study), f"Interpolated traces, study {study}"))
right.pyplot(trace_plots.segment_lines(service.study(reconstructed, study), f"Reconstructed with {context['method']}, study {study}"))

concat = context["mode"] + ("_echo" if context["echo"] else "")
imputed = set(frame.loc[frame.get("isImputed", 0) == 1, "study_nr"]) if "isImputed" in frame else set()
confirm = st.checkbox(f"Replace {context['method']}-{concat} results for {context['source']} in the database", key=state.key(PAGE, "confirm"))
if st.button("Save to database", disabled=not confirm):
    count = ui.load(store.save, context["method"], concat, context["source"], items, result, imputed)
    st.success(f"Saved {len(reconstructed)} reconstructed segments ({count} rows affected)")
