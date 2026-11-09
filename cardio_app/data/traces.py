import pandas as pd
import streamlit as st

from cardio_app import settings
from cardio_app.data import db
from cardio_app.services import traces
from pattern_recognition.utils.parameters_dics import cardiac_events_order

SOURCES = {
    "Speqle": ("speqle_segmentalStrain", "speqle_segmentalStrainInterp"),
    "GE": ("ste_segmentalStrain", "ste_segmentalStrainInterp"),
    "Philips": ("phi_segmentalStrain", "phi_segmentalStrainInterp"),
}
COMBINED = "GE + Philips"
SOURCE_KEYS = {COMBINED: "GE_Philips"}
ES_TABLES = {"GE": "ste_strainValuesInterp", "Philips": "phi_strainValuesInterp"}
STUDY_FILTERS = ("all", "original only", "imputed only")


@st.cache_data(ttl=600)
def table(name):
    frame = db.read(f"SELECT * FROM {name}")
    frame["studyType"] = frame["studyType"].str.strip()
    return frame[frame["studyType"] == settings.STUDY_TYPE].sort_values("study_nr")


def filtered(name, studies):
    frame = table(name)
    if studies == "all" or "isImputed" not in frame:
        return frame
    imputed = frame.loc[frame["isImputed"] == 1, "study_nr"].unique()
    keep = frame["study_nr"].isin(imputed)
    return frame[keep if studies == "imputed only" else ~keep]


@st.cache_data(ttl=600)
def cohort(source, interpolated, studies):
    names = [SOURCES[s][interpolated] for s in (("GE", "Philips") if source == COMBINED else (source,))]
    frames = [filtered(name, studies).assign(origin=name) for name in names]
    return traces.clean(pd.concat(frames, ignore_index=True))


@st.cache_data(ttl=600)
def rows(source, studies):
    return traces.parse_rows(filtered(SOURCES[source][1], studies))


@st.cache_data(ttl=600)
def segments():
    return db.read("SELECT id_position, segment_name, cardiac_wall, seg_speqle, cardiac_view FROM config_cardiacSegments")


def format_trace(trace):
    return ", ".join(f"{value:.6f}" for value in trace)


def save_imputed(source, frame):
    name = SOURCES[source][1]
    keys = [(int(row.study_nr), str(row.id_position)) for row in frame.itertuples()]
    values = [(int(row.study_nr), row.segment_wall, int(row.curve_length), row.curve_segment, str(row.id_position),
               format_trace(row.trace), row.cardiac_view, settings.STUDY_TYPE) for row in frame.itertuples()]
    count = db.execute_many([
        (f"DELETE FROM {name} WHERE isImputed = 1 AND study_nr = ? AND id_position = ?", keys),
        (f"INSERT INTO {name} (study_nr, segment_wall, curve_length, curve_segment, id_position, strain_trace, "
         f"cardiac_view, studyType, isImputed) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1)", values)])
    table.clear()
    cohort.clear()
    rows.clear()
    return count


@st.cache_data(ttl=600)
def events(source):
    if source == "Speqle":
        return traces.event_marks(table("speqle_cardiacEvent"), cardiac_events_order)
    name = ES_TABLES.get("GE" if source == COMBINED else source)
    frame = table(name)
    return [(float(frame["esFrame"].mode().iloc[0]), "ES")] if len(frame) else []
