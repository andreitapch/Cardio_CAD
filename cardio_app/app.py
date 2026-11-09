import os
import sys
from pathlib import Path

os.environ.setdefault("MKL_THREADING_LAYER", "TBB")
REPO = str(Path(__file__).resolve().parents[1])
if REPO not in sys.path:
    sys.path.insert(0, REPO)
try:
    import torch
except Exception:
    pass

import streamlit as st

PAGES = {
    "Data": [st.Page("pages/remodeling.py", title="Remodeling"),
             st.Page("pages/cohort_traces.py", title="Cohort traces"),
             st.Page("pages/patient_traces.py", title="Patient traces")],
    "Imputation": [st.Page("pages/best_k.py", title="Best K"),
                   st.Page("pages/knn_impute.py", title="KNN impute")],
    "Reduction": [st.Page("pages/reduce.py", title="Reduce"),
                  st.Page("pages/compare.py", title="Compare")],
    "Classification": [st.Page("pages/classify.py", title="Classify"),
                       st.Page("pages/echo_features.py", title="Echo features"),
                       st.Page("pages/ste_echo.py", title="Strain and echo"),
                       st.Page("pages/compare_classifiers.py", title="Compare classifiers")],
    "Motion": [st.Page("pages/motion.py", title="Motion viewer")],
}

st.set_page_config(page_title="Cardio CAD", layout="wide")
st.navigation(PAGES).run()
