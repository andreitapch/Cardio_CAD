import streamlit as st

from cardio_app import panels, state, ui
from cardio_app.data import classification as store
from cardio_app.jobs import classify as jobs
from cardio_app.services.classification.engine import DEFAULT_REDUCER_COMPONENTS, SAMPLER_CHOICES
from cardio_app.services.classification.model_params import CLASSIFIER_LABELS

PAGE = "classify"
CLASS_INPUTS = {"Reconstructed traces": "reconstructed_traces", "Components": "components", "Features": "features_class",
                "GLS": "gls_class", "Autoencoder encodings": "autoencoder_class"}
INPUT_TYPES = {"Full traces": "full_traces", "Segments": "segment"}
SCALING = {"Auto (off for raw traces)": "auto", "Always": "on", "Never": "off"}
REDUCERS = ["none", "pca", "umap"]

st.title("Classify remodeling")
sources = ui.load(store.data_sources)
row = st.columns(4)
data_source = row[0].selectbox("Data source", sources, key=state.key(PAGE, "source"))
class_input = CLASS_INPUTS[row[1].selectbox("Classify by", list(CLASS_INPUTS), key=state.key(PAGE, "class_input"))]
input_type = INPUT_TYPES[row[2].selectbox("Input", list(INPUT_TYPES), key=state.key(PAGE, "input_type"))]
components = ui.load(store.component_options, class_input, input_type)
n_components = row[3].selectbox("Components", components, key=state.key(PAGE, "components"))
row = st.columns(4)
class_type = row[0].selectbox("Model", list(CLASSIFIER_LABELS), index=2, format_func=CLASSIFIER_LABELS.get, key=state.key(PAGE, "model"))
unbalanced = row[1].selectbox("Imbalance", SAMPLER_CHOICES, key=state.key(PAGE, "imbalance"))
scaling = SCALING[row[2].selectbox("Standardise", list(SCALING), key=state.key(PAGE, "scaling"))]
reducer = row[3].selectbox("Reduce inside CV", REDUCERS, key=state.key(PAGE, "reducer"))
row = st.columns(4)
reducer_components = row[0].number_input("Reducer components", min_value=2, value=DEFAULT_REDUCER_COMPONENTS, key=state.key(PAGE, "reducer_k"))
imputed = row[1].checkbox("Include imputed data", value=True, key=state.key(PAGE, "imputed"))
add_gls = row[2].checkbox("Add GLS", key=state.key(PAGE, "gls"))
add_echo = row[3].checkbox("Add selected echo features", key=state.key(PAGE, "echo"))

job = state.load(PAGE, "job")
if st.button("Classify", type="primary", disabled=(job is not None and not job.finished()) or n_components is None):
    config = {"kind": "classify", "data_source": data_source, "imputed": imputed, "class_input": class_input,
              "input_type": input_type, "n_components": str(n_components), "class_type": class_type,
              "unbalanced_type": unbalanced, "add_gls": add_gls, "add_echo": add_echo, "scaling_mode": scaling,
              "reducer": reducer, "reducer_components": int(reducer_components)}
    ui.launch(PAGE, jobs.start(f"{CLASSIFIER_LABELS[class_type]} on {class_input}", config))

done = ui.watch(PAGE)
if done is None:
    st.stop()
result = state.load(PAGE, "result") or state.store(PAGE, "result", jobs.load(done.meta["output"]))
panels.classification_result(PAGE, result)
