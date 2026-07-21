"""
Streamlit dashboard for the AI4ALL Group 04C turbofan RUL project.

Run with: streamlit run app.py
"""

import pandas as pd
import streamlit as st

from model import FEATURE_COLS, predict_rul

DATA_PATH = "nasa_cmapss_FD001_scaled.csv"

st.set_page_config(page_title="Turbofan RUL Dashboard", layout="wide")


@st.cache_data
def load_data() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


df = load_data()

with st.sidebar:
    st.title("Turbofan Engine RUL Dashboard")
    st.markdown(
        "**AI4ALL Group 04C** — predicting a turbofan engine's Remaining "
        "Useful Life (RUL) from NASA C-MAPSS sensor telemetry, to support "
        "predictive maintenance instead of fixed replacement schedules."
    )
    st.divider()
    engine_ids = sorted(df["unit_number"].unique().tolist())
    selected_engine = st.selectbox("Select engine", engine_ids)

engine_df = df[df["unit_number"] == selected_engine].sort_values("time_in_cycles")
min_cycle = int(engine_df["time_in_cycles"].min())
max_cycle = int(engine_df["time_in_cycles"].max())

st.header(f"Engine {selected_engine}")
st.caption(f"{len(engine_df)} recorded cycles ({min_cycle}-{max_cycle}).")

selected_cycle = st.slider(
    "Cycle", min_value=min_cycle, max_value=max_cycle, value=max_cycle
)
current_row = engine_df.loc[engine_df["time_in_cycles"] == selected_cycle].iloc[0]
predicted_rul = predict_rul(current_row[FEATURE_COLS].to_frame().T)[0]

col1, col2 = st.columns(2)
col1.metric("Current cycle", selected_cycle)
col2.metric("Predicted RUL (cycles)", f"{predicted_rul:.0f}")
st.caption(
    "Prediction is a placeholder heuristic (see model.py), not a trained "
    "model's output."
)

st.subheader("Sensor readings")
sensor_cols = [c for c in engine_df.columns if c.startswith("sensor_")]
chosen_sensors = st.multiselect(
    "Sensors to plot", sensor_cols, default=sensor_cols[:4]
)
if chosen_sensors:
    st.line_chart(engine_df.set_index("time_in_cycles")[chosen_sensors])
st.caption("Sensor values are z-score scaled, not raw physical units.")
