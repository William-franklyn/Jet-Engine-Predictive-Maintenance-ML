"""
Train and save the Random Forest RUL model for the Streamlit dashboard.

Trains on nasa_cmapss_FD001_scaled.csv (the committed, labeled FD001 dataset).
Evaluates on an engine-grouped hold-out split so no engine leaks across
train/test, then refits on all 100 engines and saves the deployable model.

The dashboard (app.py) also imports `train_full_model()` from here so a fresh
deployment can build the model on startup when no .pkl is present.

Usage:  python save_model.py
Output: random_forest_model.pkl
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

from model import FEATURE_COLS, MODEL_PATH, RUL_CAP

DATA_PATH = "nasa_cmapss_FD001_scaled.csv"
ID_COL = "unit_number"


def cmapss_score(y_true, y_pred):
    """Asymmetric PHM score (lower is better). Late predictions — engine
    predicted to last LONGER than it does — are the dangerous ones and are
    penalized harder (/10) than early predictions (/13)."""
    d = y_pred - y_true
    return np.sum(np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1))


def build_model():
    # Tuned on a grouped hold-out. Kept lean on purpose: n=200 / leaf=10
    # matches the deeper model's accuracy (RMSE ~18.0) at ~1/4 the memory and
    # ~1/2 the train time — this is what lets the deployed app train on
    # startup within free-tier limits.
    return RandomForestRegressor(
        n_estimators=200,
        min_samples_leaf=10,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1,
    )


def load_xy(data_path=DATA_PATH):
    """Load features/labels/groups from the committed dataset.

    RUL ships raw (uncapped) in the CSV; we apply the piecewise-linear cap at
    RUL_CAP so the model isn't forced to fit long, flat early-life stretches.
    """
    df = pd.read_csv(data_path)
    X = df[FEATURE_COLS].values
    y = np.minimum(df["RUL"].values, RUL_CAP)
    groups = df[ID_COL].values
    return X, y, groups


def train_full_model(data_path=DATA_PATH):
    """Fit build_model() on ALL rows and return it (no split, no file I/O).

    This is what app.py calls to build the model on the fly when the .pkl is
    absent (e.g. a fresh cloud deployment).
    """
    X, y, _ = load_xy(data_path)
    model = build_model()
    model.fit(X, y)
    return model


def main():
    X, y, groups = load_xy()

    # Engine-grouped hold-out: test engines are entirely unseen in training.
    gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
    tr, te = next(gss.split(X, y, groups))

    print(f"Evaluating on a grouped hold-out "
          f"({np.unique(groups[tr]).size} train / "
          f"{np.unique(groups[te]).size} test engines)...")
    model = build_model()
    model.fit(X[tr], y[tr])
    pred = np.clip(model.predict(X[te]), 0, RUL_CAP)

    rmse = np.sqrt(mean_squared_error(y[te], pred))
    mae = mean_absolute_error(y[te], pred)
    r2 = r2_score(y[te], pred)
    score = cmapss_score(y[te], pred)

    print("\n=== Random Forest — held-out performance ===")
    print(f"RMSE          : {rmse:.2f} cycles")
    print(f"MAE           : {mae:.2f} cycles")
    print(f"R^2           : {r2:.3f}")
    print(f"C-MAPSS score : {score:.1f}  (lower is better)")

    importances = sorted(
        zip(FEATURE_COLS, model.feature_importances_),
        key=lambda t: t[1], reverse=True,
    )
    print("\nTop features:")
    for name, imp in importances[:6]:
        print(f"  {name:22s} {imp:.3f}")

    # Refit on ALL engines for the deployed model — the dashboard should use
    # every row we have, not just the 75% training split.
    print("\nRefitting on all 100 engines and saving...")
    final = build_model()
    final.fit(X, y)
    joblib.dump(final, MODEL_PATH)
    print(f"Saved: {MODEL_PATH}")


if __name__ == "__main__":
    main()
