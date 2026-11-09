import streamlit as st
from sklearn.preprocessing import StandardScaler

from cardio_app.data import db


@st.cache_data(ttl=600)
def biomarkers():
    frame = db.read("SELECT * FROM echo_biomarkers_view")
    numeric = frame.select_dtypes("number").drop(columns=["study_nr"], errors="ignore")
    numeric = numeric.dropna(axis=1, how="all")
    scaled = StandardScaler().fit_transform(numeric.fillna(numeric.mean()))
    return frame[["study_nr"]].assign(**dict(zip(numeric.columns, scaled.T))).groupby("study_nr").mean()
