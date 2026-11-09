import streamlit as st

from cardio_app import panels, state, ui
from cardio_app.data import classification as store
from cardio_app.plots import classification as plots
from cardio_app.services.classification import feature_selection as service
from cardio_app.services.classification.data import get_echoParams_data
from cardio_app.services.classification.engine import convert_echoParam_values, preprocess_data, resolve_exam_groups

PAGE = "echo_features"

st.title("Echo feature selection")
row = st.columns(4)
data_source = row[0].selectbox("Data source", ui.load(store.data_sources), key=state.key(PAGE, "source"))
k = row[1].slider("Features to keep", 1, 20, 10, key=state.key(PAGE, "k"))
threshold = row[2].number_input("Minimum consensus score", min_value=0.0, max_value=1.0, value=0.0, step=0.05, key=state.key(PAGE, "threshold"))
min_frequency = row[3].number_input("Minimum selection frequency", min_value=0.0, max_value=1.0,
                                    value=service.DEFAULT_MIN_SELECTION_FREQUENCY, step=0.1, key=state.key(PAGE, "frequency"))

if st.button("Rank features", type="primary"):
    with st.spinner("Scoring features fold by fold"):
        echo, _ = ui.load(get_echoParams_data, None, data_source, None)
        echo = convert_echoParam_values(echo)
        x, y, _, _ = preprocess_data(echo, True, fit_preprocessing=False)
        scores, selected = service.compute_feature_importance_cv(x, y, int(k), threshold or None, groups=resolve_exam_groups(echo),
                                                                 min_frequency=min_frequency)
        state.store(PAGE, "run", {"source": data_source, "x": x, "y": y, "scores": scores, "selected": selected})

run = state.load(PAGE, "run")
if run is None:
    st.stop()
scores, selected = run["scores"], run["selected"]
st.caption(service.LEAKAGE_NOTE)
if not selected:
    st.warning("No feature reached the selection-frequency, threshold and k criteria; loosen them and run again.")
    st.stop()
table = scores.reset_index().rename(columns={"index": "Feature"})
table.insert(1, "Selected", table["Feature"].isin(selected))
st.dataframe(table, hide_index=True)
panels.csv_download(table, f"feature_selection_{run['source']}")
st.pyplot(plots.feature_importance_comparison(scores.loc[selected], service.MEASURE_COLUMNS + ["Weighted Consensus Score"]))
st.pyplot(plots.feature_means_by_class(service.display_matrix(run["x"], selected), run["y"], selected))
confirm = st.checkbox(f"Replace the stored feature list with these {len(selected)} features", key=state.key(PAGE, "confirm"))
if st.button("Save to database", disabled=not confirm):
    ui.load(store.save_selected_features, table[table["Selected"]])
    st.success(f"Saved {len(selected)} features")
