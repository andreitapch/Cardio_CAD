import streamlit as st

from cardio_app.data import db
from pattern_recognition.data.database_connection import sql_queries

HIDDEN_COLUMNS = ["segment_wall", "curve_segment", "cardiac_view"]


@st.cache_data(ttl=600)
def calculated():
    return db.read(sql_queries.select_total_remodeling_data())


@st.cache_data(ttl=600)
def doppler():
    return db.read(sql_queries.select_total_doppler_remodeling_data())


@st.cache_data(ttl=600)
def with_strain_traces():
    frame = db.read(sql_queries.select_all_data("merged_remodeling_18seg_view"))
    return frame.drop(columns=HIDDEN_COLUMNS, errors="ignore")
