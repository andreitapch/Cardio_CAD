import streamlit as st

from cardio_app import panels, state, ui
from cardio_app.data import classification as store
from cardio_app.jobs import classify as jobs
from cardio_app.services.classification.engine import SAMPLER_CHOICES
from cardio_app.services.classification.model_params import CLASSIFIER_LABELS

PAGE = "ste_echo"
INPUT_TYPES = {"Reconstructed traces": "reconstructed_traces_vac", "18 segments encoded": "18_segments_vac",
               "Per segment encoded": "per_segment_vac"}

st.title("Classify with strain and echo features")
row = st.columns(5)
data_source = row[0].selectbox("Data source", ui.load(store.data_sources), key=state.key(PAGE, "source"))
input_type = INPUT_TYPES[row[1].selectbox("Strain input", list(INPUT_TYPES), key=state.key(PAGE, "input_type"))]
class_type = row[2].selectbox("Model", list(CLASSIFIER_LABELS), format_func=CLASSIFIER_LABELS.get, key=state.key(PAGE, "model"))
unbalanced = row[3].selectbox("Imbalance", SAMPLER_CHOICES, key=state.key(PAGE, "imbalance"))
encoding = row[4].number_input("Encoding dimensions", min_value=2, max_value=100, value=32, key=state.key(PAGE, "encoding"))
imputed = st.checkbox("Include imputed data", key=state.key(PAGE, "imputed"))

job = state.load(PAGE, "job")
if st.button("Classify", type="primary", disabled=job is not None and not job.finished()):
    config = {"kind": "ste_echo", "data_source": data_source, "imputed": imputed, "input_type": input_type,
              "class_type": class_type, "unbalanced_type": unbalanced, "encoding": int(encoding), "scaling_mode": "auto"}
    ui.launch(PAGE, jobs.start(f"{CLASSIFIER_LABELS[class_type]} on {input_type} + echo", config))

done = ui.watch(PAGE)
if done is None:
    st.stop()
result = state.load(PAGE, "result") or state.store(PAGE, "result", jobs.load(done.meta["output"]))
panels.classification_result(PAGE, result)
