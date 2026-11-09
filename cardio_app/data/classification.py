import streamlit as st

from cardio_app.data import db

FEATURES_TABLE = "calculated_features_remodeling"


@st.cache_data(ttl=300)
def data_sources():
    return sorted(db.read("SELECT DISTINCT data_source FROM pca_results_view")["data_source"].dropna())


@st.cache_data(ttl=300)
def component_options(class_input, input_type):
    if class_input == "autoencoder_class":
        table, column, pattern = "calculated_dr_encoded_data", "encoding", f"AutoencoderDm-{input_type}%"
    else:
        table, column, pattern = "calculated_pca_params", "n_components_", f"%{input_type}%"
    frame = db.read(f"SELECT DISTINCT {column} AS value FROM {table} WHERE reconstructed_type LIKE ?", (pattern,))
    return sorted(int(value) for value in frame["value"].dropna())


def save_selected_features(frame):
    rows = [(feature, float(score)) for feature, score in zip(frame["Feature"], frame["Weighted Consensus Score"])]
    return db.execute_many([(f"DELETE FROM {FEATURES_TABLE}", None),
                            (f"DBCC CHECKIDENT ('{FEATURES_TABLE}', RESEED, 0)", None),
                            (f"INSERT INTO {FEATURES_TABLE} VALUES (?, ?)", rows)])
