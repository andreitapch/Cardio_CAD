import streamlit as st

from cardio_app import ui
from cardio_app.data import remodeling
from cardio_app.plots import bars

st.title("Remodeling")
for label, loader, column in (("Doppler", remodeling.doppler, "Total DRM"),
                              ("Calculated", remodeling.calculated, "Total RM")):
    frame = ui.load(loader)
    with st.expander(f"Remodeling per image mode: {label}", expanded=True):
        table, chart = st.columns((3, 2))
        table.dataframe(frame, height=400, hide_index=True)
        chart.pyplot(bars.counts(frame[column], column))

merged = ui.load(remodeling.with_strain_traces)
with st.expander("Remodeling with strain traces", expanded=True):
    table, chart = st.columns((3, 2))
    table.dataframe(merged, height=500, hide_index=True)
    chart.pyplot(bars.grouped_sums(merged, "remodeling", ["Speqle", "GE", "Philips"], "Remodeling per category"))
