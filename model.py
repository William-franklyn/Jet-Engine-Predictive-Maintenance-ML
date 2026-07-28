"""
RUL prediction.

Loads the trained Random Forest (random_forest_model.pkl, produced by
save_model.py) if it exists. Falls back to a placeholder heuristic — NOT a
trained model — so the dashboard still shows something before the pickle
exists. Run `python save_model.py` to train and save the real model on
nasa_cmapss_FD001_scaled.csv (committed to the repo).
"""

import os

import joblib
import numpy as np
import pandas as pd

MODEL_PATH = "random_forest_model.pkl"
LSTM_MODEL_PATH = "lstm_rul_model.pt"

# The 16 model inputs, matching the columns of nasa_cmapss_FD001_scaled.csv
# (the single committed, labeled FD001 dataset). All are z-score scaled.
FEATURE_COLS = [
    "operational_setting_1",
    "sensor_2", "sensor_3", "sensor_4", "sensor_6", "sensor_7", "sensor_8",
    "sensor_9", "sensor_11", "sensor_12", "sensor_13", "sensor_14",
    "sensor_15", "sensor_17", "sensor_20", "sensor_21",
]

RUL_CAP = 125


def load_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return None


def predict_rul(rows: pd.DataFrame, model=None) -> np.ndarray:
    """Predict RUL for rows containing FEATURE_COLS.

    Uses `model.predict()` when a trained model is passed in; otherwise
    falls back to a heuristic that maps average scaled-sensor magnitude to
    a predicted RUL (further from baseline = more degraded = lower RUL).
    """
    X = rows[FEATURE_COLS]
    if model is not None:
        return np.clip(model.predict(X.values), 0, RUL_CAP).round(1)
    magnitude = X.abs().mean(axis=1)
    return np.clip(RUL_CAP * np.exp(-magnitude), 0, RUL_CAP).round(1).to_numpy()


# ── LSTM (sequence model) ────────────────────────────────────────────────────
# The LSTM is the most accurate model we have (see docs/reproduction-log.md),
# but it cannot predict from a single row: it needs SEQ_LEN cycles of history.
# So the dashboard routes per prediction — LSTM when the history exists, Random
# Forest when it doesn't — and always tells the user which one answered.

LSTM_SEQ_LEN = 30


def load_lstm():
    """Load the deployed LSTM, or None if torch/the checkpoint isn't available.

    torch is an optional dependency: a deployment without it should fall back
    to the Random Forest rather than fail to start.
    """
    if not os.path.exists(LSTM_MODEL_PATH):
        return None
    try:
        import torch
        from lstm_model import LSTMRegressor

        blob = torch.load(LSTM_MODEL_PATH, map_location="cpu", weights_only=False)
        model = LSTMRegressor(n_features=blob["n_features"])
        model.load_state_dict(blob["state_dict"])
        model.eval()
        return model
    except Exception:  # noqa: BLE001 - optional dependency, degrade quietly
        return None


def predict_rul_sequence(history: pd.DataFrame, lstm, seq_len=LSTM_SEQ_LEN):
    """Predict RUL at the LAST row of `history` using the preceding cycles.

    `history` must be one engine's rows, sorted by cycle, ending at the cycle
    being predicted. Shorter-than-seq_len histories are left-padded with the
    earliest available reading — the same rule used during training.

    Returns None if the LSTM is unavailable, so callers can fall back.
    """
    if lstm is None or history.empty:
        return None
    import torch

    feats = history[FEATURE_COLS].to_numpy(dtype=np.float32)[-seq_len:]
    if len(feats) < seq_len:
        pad = np.repeat(feats[:1], seq_len - len(feats), axis=0)
        feats = np.concatenate([pad, feats], axis=0)

    with torch.no_grad():
        pred = lstm(torch.from_numpy(feats[None, ...])).item()
    return float(np.clip(pred, 0, RUL_CAP))
