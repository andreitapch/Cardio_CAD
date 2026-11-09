import streamlit as st

from cardio_app import ui
from cardio_app.data import traces
from cardio_app.plots import traces as plots
from cardio_app.services import traces as service

st.title("Patient strain traces")
source_col, studies_col, study_col = st.columns(3)
source = source_col.selectbox("Data source", list(traces.SOURCES), key="patient.source")
studies = studies_col.selectbox("Studies", traces.STUDY_FILTERS, key="patient.studies")
original = ui.load(traces.cohort, source, 0, studies)
interpolated = ui.load(traces.cohort, source, 1, studies)
available = sorted(set(original["study_nr"]) & set(interpolated["study_nr"]))
if not available:
    st.warning(f"No study has both original and interpolated traces for {source}")
    st.stop()
study_nr = study_col.selectbox("Study number", available, key="patient.study")

left, right = st.columns(2)
left.pyplot(plots.segment_lines(service.study(original, study_nr), f"Original traces, study {study_nr}"))
right.pyplot(plots.segment_lines(service.study(interpolated, study_nr), f"Interpolated traces, study {study_nr}"))
