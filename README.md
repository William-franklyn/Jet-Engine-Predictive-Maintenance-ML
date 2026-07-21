# AI4ALL Group 04C — Predicting Jet Engine Failure

Predicting a turbofan engine's Remaining Useful Life (RUL) — how many flight cycles remain
before failure — from sensor telemetry, to support predictive maintenance instead of fixed
replacement schedules.

## Dataset

`nasa_cmapss_FD001_scaled.csv` — NASA C-MAPSS FD001 turbofan degradation dataset (Prognostics
CoE, NASA Ames): [kaggle.com/datasets/behrad3d/nasa-cmaps](https://www.kaggle.com/datasets/behrad3d/nasa-cmaps).
100 engines, sensor and operational-setting readings already z-score scaled, with ground-truth
RUL per row.

## Setup

```
pip install -r requirements.txt
```

## Baseline model

```
python logistic_regression_base.py
```

Compares a logistic regression classifier ("will this engine fail within W cycles?") against
a linear regression on capped RUL, evaluated on an identical engine-grouped train/test split.

## Dashboard

```
streamlit run app.py
```

Lets you pick an engine, inspect its sensor readings and predicted RUL across cycles, and
upload a CSV of sensor readings for a live prediction. The RUL prediction is currently backed
by a placeholder heuristic in `model.py` — see the `TODO` there for swapping in a trained
model (Random Forest / Logistic Regression per the team's plan).
