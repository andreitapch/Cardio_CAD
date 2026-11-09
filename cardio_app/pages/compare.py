import pandas as pd
import streamlit as st

from cardio_app import state, ui
from cardio_app.data import traces
from cardio_app.jobs import reduce as job_reduce
from cardio_app.plots import reduction as plots
from cardio_app.services import reduction

PAGE = "compare"
COLUMNS = {"method": "Method", "k": "Latent dim", "features": "Features", "n": "n", "params": "Params", "mse": "MSE",
           "r2": "R2 in-sample", "train_r2": "R2 train split", "holdout_r2": "R2 held-out", "cum_var": "Cum. variance"}

st.title("PCA vs autoencoder")
source_col, studies_col, mode_col, k_col, epochs_col = st.columns(5)
source = source_col.selectbox("Data source", [*traces.SOURCES, traces.COMBINED], index=3, key=state.key(PAGE, "source"))
studies = studies_col.selectbox("Studies", traces.STUDY_FILTERS, key=state.key(PAGE, "studies"))
mode = reduction.MODES[mode_col.selectbox("Traces", list(reduction.MODES), key=state.key(PAGE, "mode"))]
k = k_col.number_input("Latent dimensions", min_value=2, max_value=64, value=8, key=state.key(PAGE, "k"))
epochs = epochs_col.number_input("Epochs", min_value=10, max_value=5000, value=1000, step=10, key=state.key(PAGE, "epochs"))

job = state.load(PAGE, "job")
if st.button("Compare", type="primary", disabled=job is not None and not job.finished()):
    frame = ui.load(traces.cohort, source, 1, studies)
    items = reduction.blocks(frame, mode)
    state.store(PAGE, "context", {"items": items, "frame": frame, "mode": mode})
    ui.launch(PAGE, job_reduce.start(f"PCA and autoencoder on {source}, {mode}", items, list(reduction.METHODS.values()), int(k), int(epochs)))

done = ui.watch(PAGE)
context = state.load(PAGE, "context")
if done is None or context is None:
    st.stop()
results = state.load(PAGE, "result") or state.store(PAGE, "result", reduction.load_result(done.meta["output"]))
items, frame = context["items"], context["frame"]
pca_result, ae_result = results["PCA"], results["AutoencoderDm"]
rows = [reduction.baseline(items[0].x) if len(items) == 1 else reduction.mean_metrics(reduction.Result("base", metrics={b.name: reduction.baseline(b.x) for b in items}), "Mean curve"),
        reduction.mean_metrics(pca_result, "PCA"), reduction.mean_metrics(ae_result, "Autoencoder")]
st.dataframe(pd.DataFrame(rows)[list(COLUMNS)].rename(columns=COLUMNS), hide_index=True)
st.caption("R2 = 1 - MSE / Var: 0 is the mean-curve baseline, 1 is perfect reconstruction. Held-out R2 is the number to report.")
st.pyplot(plots.loss_curves(ae_result.history, "Autoencoder validation loss"))
frames = {"original": frame, "PCA": reduction.reconstructed_frame(items, pca_result),
          "Autoencoder": reduction.reconstructed_frame(items, ae_result)}
study = st.selectbox("Study number", sorted(frame["study_nr"].unique()), key=state.key(PAGE, "study"))
st.write(f"Study {study}: MSE PCA {reduction.study_mse(frame, frames['PCA'], study):.4f}, "
         f"autoencoder {reduction.study_mse(frame, frames['Autoencoder'], study):.4f}")
st.pyplot(plots.comparison_grid(frames, study, f"Original, PCA and autoencoder reconstructions, study {study}"))
