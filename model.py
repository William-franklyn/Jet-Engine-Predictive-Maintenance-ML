"""
Placeholder RUL prediction — NOT a trained model.

TODO: replace the body of predict_rul() with inference from a trained model
(Random Forest / Logistic Regression per the team's plan, or an LSTM if the
team moves to one). The dashboard only ever calls this function, so
swapping in a real model is a one-function change.
"""

import numpy as np
import pandas as pd

FEATURE_COLS = [
    "operational_setting_1",
    "sensor_2", "sensor_3", "sensor_4", "sensor_6", "sensor_7", "sensor_8",
    "sensor_9", "sensor_11", "sensor_12", "sensor_13", "sensor_14",
    "sensor_15", "sensor_17", "sensor_20", "sensor_21",
]

STUB_RUL_CAP = 125


def predict_rul(rows: pd.DataFrame) -> np.ndarray:
    """Heuristic stand-in for a trained model.

    Maps the average magnitude of the (already z-score scaled) sensor
    readings to a predicted RUL: readings further from baseline (0) are
    treated as more degraded and given a lower predicted RUL. This is only
    meant to give the dashboard something plausible to show before a real
    model is trained.
    """
    present_cols = [c for c in FEATURE_COLS if c in rows.columns]
    magnitude = rows[present_cols].abs().mean(axis=1)
    predicted = STUB_RUL_CAP * np.exp(-magnitude)
    return np.clip(predicted, 0, STUB_RUL_CAP).round(1).to_numpy()
