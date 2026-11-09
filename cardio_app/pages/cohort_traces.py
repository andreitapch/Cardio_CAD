import streamlit as st

from cardio_app import ui
from cardio_app.data import traces
from cardio_app.plots import traces as plots
from cardio_app.services import traces as service

GROUPS = {"Segment wall": "segment_wall", "Curve segment": "curve_segment", "Cardiac view": "cardiac_view"}

st.title("Cohort strain traces")
source_col, group_col, studies_col = st.columns(3)
source = source_col.selectbox("Data source", [*traces.SOURCES, traces.COMBINED], key="cohort.source")
by = GROUPS[group_col.selectbox("Group by", list(GROUPS), key="cohort.group")]
studies = studies_col.selectbox("Studies", traces.STUDY_FILTERS, key="cohort.studies")

for interpolated, label in ((0, "Original"), (1, "Interpolated")):
    frame = ui.load(traces.cohort, source, interpolated, studies)
    if frame.empty:
        st.warning(f"No complete {label.lower()} studies for {source}")
        continue
    events = ui.load(traces.events, source) if interpolated else []
    st.pyplot(plots.envelope_grid(service.grouped(frame, by), by, events, f"{label} strain traces by {by.replace('_', ' ')}"))
