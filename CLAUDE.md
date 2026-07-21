# AI4ALL-Group-04C

## Project

"Predicting Jet Engine Failure Through Machine Learning" — an AI4ALL course group project.
Team: Erronn Bridgewater, Tom Chatto, Gabe Meredith, Nish Methuku, Alejandro Hernandez, William Mahunda, Hunter Ngo.

Goal: predict a turbofan engine's Remaining Useful Life (RUL) — how many flight cycles remain
before failure — from sensor telemetry, to support predictive maintenance instead of fixed
replacement schedules.

Research question: can ML models predict early engine failure to support predictive
maintenance and reduce unexpected downtime?

Full background (problem motivation, bias/mitigation notes, citations) lives in the team's
project deck, not duplicated here.

## Dataset

`nasa_cmapss_FD001_scaled.csv` — NASA C-MAPSS FD001 turbofan degradation set (Prognostics CoE,
NASA Ames). 20,631 rows, 100 engines (`unit_number` 1-100).

Columns:
- `unit_number` — engine ID
- `time_in_cycles` — cycle count since that engine's start of run
- `operational_setting_1` — operating condition
- `sensor_2, sensor_3, sensor_4, sensor_6, sensor_7, sensor_8, sensor_9, sensor_11, sensor_12, sensor_13, sensor_14, sensor_15, sensor_17, sensor_20, sensor_21` — sensor channels (temperature, pressure, vibration, fan speed, etc.)
- `RUL` — ground-truth remaining useful life in cycles (label)

All sensor/operational-setting columns are already z-score scaled — values are not in raw
physical units. Keep that in mind in any UI copy ("scaled sensor reading", not "°F"/"psi").

## Models

Two models are planned per the team deck:
- **Logistic Regression** — interpretable baseline; classifies "will this engine fail within
  W cycles?"
- **Random Forest** — captures nonlinear degradation and sensor interactions

`logistic_regression_base.py` is the existing baseline script: it frames the same problem two
ways (logistic classification vs. linear regression on capped RUL) on an identical
engine-grouped train/test split, and reports both native metrics and a shared classification
scoreboard, plus a custom asymmetric C-MAPSS scoring function.

No LSTM model exists in this repo. The dashboard's RUL prediction is currently backed by a
stub function in `model.py` (`predict_rul`) — replace that function's internals with a real
trained model when one is ready; nothing else in the dashboard should need to change.

## Dashboard

`app.py` — a Streamlit app: sidebar project description, dropdown to pick an engine, sensor
readings and predicted RUL for that engine, plus a CSV-upload flow for live predictions on
uploaded sensor data.

Run it with:
```
pip install -r requirements.txt
streamlit run app.py
```

## Conventions

- Do not add a `Co-Authored-By: Claude` trailer to commits in this repo.
- Keep commits scoped and descriptive — this is a team repo other members read the history of.
