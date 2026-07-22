"""
RUL Prediction Dashboard — Team 04C
AI4ALL Ignite SU26 | NASA C-MAPSS Turbofan

Combines the engine-browser view with CSV-upload and manual-input
prediction into one app, all backed by the same model (see model.py).

Run with: streamlit run app.py
Requires: streamlit, numpy, pandas, scikit-learn, joblib
"""

import os

import pandas as pd
import streamlit as st

from model import FEATURE_COLS, RUL_CAP, load_model, predict_rul

DATA_PATH = "nasa_cmapss_FD001_scaled.csv"

st.set_page_config(page_title="Turbofan RUL Predictor", page_icon="✈️", layout="wide")


@st.cache_resource
def get_model():
    # Prefer a pre-trained pickle (fast). On a fresh deployment the .pkl isn't
    # in the repo (it's gitignored and >100MB), so train it once from the
    # committed CSV — cached for the life of the process. Import sklearn lazily
    # and tolerate failure so an environment issue degrades to the placeholder
    # heuristic instead of crashing the whole app.
    try:
        model = load_model()
        if model is None and os.path.exists(DATA_PATH):
            from save_model import train_full_model
            with st.spinner("First run: training the model (~10s)…"):
                model = train_full_model(DATA_PATH)
        return model
    except Exception as exc:  # noqa: BLE001 - surface, don't crash
        st.warning(f"Model unavailable ({type(exc).__name__}); using placeholder.")
        return None


@st.cache_data
def load_dataset():
    if not os.path.exists(DATA_PATH):
        return None
    data = pd.read_csv(DATA_PATH)
    # Standardize the id columns so the rest of the app can use unit/cycle.
    data = data.rename(columns={"unit_number": "unit", "time_in_cycles": "cycle"})
    # The CSV ships raw RUL; apply the same 125-cycle cap the model trains on.
    data["RUL"] = data["RUL"].clip(upper=RUL_CAP)
    return data


model = get_model()
dataset = load_dataset()

with st.sidebar:
    st.title("✈️ Turbofan RUL Predictor")
    st.caption("Team 04C · AI4ALL Ignite SU26 · NASA C-MAPSS FD001")
    st.markdown(
        "Predicting a turbofan engine's Remaining Useful Life (RUL) from "
        "sensor telemetry, to support predictive maintenance instead of "
        "fixed replacement schedules."
    )
    if model is None:
        st.warning(
            f"No model and no `{DATA_PATH}` to train from — showing "
            "placeholder predictions."
        )
    st.divider()
    mode = st.radio("Mode", ["Browse engine", "Upload CSV", "Manual input"])

# ── Browse engine ────────────────────────────────────────────────────────────
if mode == "Browse engine":
    st.header("Browse a training-set engine")
    if dataset is None:
        st.error(f"`{DATA_PATH}` not found in the repo root.")
    else:
        engine_ids = sorted(dataset["unit"].unique().tolist())
        selected_engine = st.selectbox("Select engine", engine_ids)

        engine_df = dataset[dataset["unit"] == selected_engine].sort_values("cycle")
        min_cycle = int(engine_df["cycle"].min())
        max_cycle = int(engine_df["cycle"].max())
        st.caption(f"{len(engine_df)} recorded cycles ({min_cycle}-{max_cycle}).")

        selected_cycle = st.slider(
            "Cycle", min_value=min_cycle, max_value=max_cycle, value=max_cycle
        )
        current_row = engine_df.loc[engine_df["cycle"] == selected_cycle].iloc[[0]]
        predicted_rul = predict_rul(current_row, model)[0]

        col1, col2 = st.columns(2)
        col1.metric("Current cycle", selected_cycle)
        col2.metric("Predicted RUL (cycles)", f"{predicted_rul:.0f}")

        st.subheader("Sensor readings")
        sensor_cols = [c for c in FEATURE_COLS if c.startswith("sensor_")]
        chosen_sensors = st.multiselect(
            "Sensors to plot", sensor_cols, default=sensor_cols[:4]
        )
        if chosen_sensors:
            st.line_chart(engine_df.set_index("cycle")[chosen_sensors])
        st.caption("Sensor values are z-score scaled, not raw physical units.")

# ── Upload CSV ───────────────────────────────────────────────────────────────
elif mode == "Upload CSV":
    st.header("Upload sensor data for live predictions")
    st.caption(
        "CSV must be pre-scaled (z-score standardized) with columns: "
        + ", ".join(FEATURE_COLS)
    )
    uploaded = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded:
        upload_df = pd.read_csv(uploaded)
        st.write("Preview:", upload_df.head())

        missing = [c for c in FEATURE_COLS if c not in upload_df.columns]
        if missing:
            st.error(f"Missing columns: {missing}")
        else:
            upload_df["Predicted RUL"] = predict_rul(upload_df, model)
            st.success("Predictions complete!")
            id_cols = [c for c in ("unit", "cycle") if c in upload_df.columns]
            st.dataframe(upload_df[id_cols + ["Predicted RUL"]] if id_cols else upload_df)
            csv_out = upload_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download Predictions CSV", csv_out, "rul_predictions.csv", "text/csv"
            )

# ── Manual input ─────────────────────────────────────────────────────────────
else:
    st.header("Manual sensor input")
    st.caption("Enter a single engine's sensor readings (z-score standardized).")

    col1, col2 = st.columns(2)
    inputs = {}
    for i, feat in enumerate(FEATURE_COLS):
        col = col1 if i % 2 == 0 else col2
        inputs[feat] = col.number_input(feat, value=0.0, format="%.4f")

    if st.button("Predict RUL", type="primary"):
        row = pd.DataFrame([inputs])
        pred = predict_rul(row, model)[0]
        st.metric("Predicted Remaining Useful Life", f"{pred:.1f} cycles")
        if pred < 30:
            st.error("⚠️ Engine approaching end of life — maintenance recommended soon.")
        elif pred < 60:
            st.warning("\U0001f536 Moderate degradation detected — schedule inspection.")
        else:
            st.success("✅ Engine health looks good.")

st.divider()
st.caption(
    "Model: tuned Random Forest (RMSE 18.01, MAE 13.33, R² 0.81 on a grouped "
    "FD001 hold-out). RUL capped at 125 cycles."
)
