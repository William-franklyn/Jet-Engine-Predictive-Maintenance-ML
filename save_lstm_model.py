"""
Train and save the deployable LSTM for the dashboard.

Distinct from `compare_models.py`, which trains an LSTM on 60 engines only so
that 25 stay unseen for honest evaluation. That checkpoint is for *measuring*.
This one is for *serving*: it trains on all 100 engines (85 fit + 15 held out
purely for early stopping) so the deployed model uses every row we have.

Usage:  python save_lstm_model.py
Output: lstm_rul_model.pt  (~220 KB, committed — cheap to ship, costly to retrain)
"""

import numpy as np
import pandas as pd
import torch

from lstm_model import SEED, SEQ_LEN, make_windows, train_lstm
from model import LSTM_MODEL_PATH

DATA_PATH = "nasa_cmapss_FD001_scaled.csv"
ID_COL = "unit_number"
VAL_FRACTION = 0.15


def main():
    df = pd.read_csv(DATA_PATH)
    engines = np.sort(df[ID_COL].unique())

    # Early stopping needs a validation set, so hold out a slice of engines —
    # but unlike the evaluation script, nothing is reserved for testing here.
    # Every engine contributes to the model that actually ships.
    shuffled = np.random.default_rng(SEED).permutation(engines)
    n_val = max(1, int(len(shuffled) * VAL_FRACTION))
    val_engines = np.sort(shuffled[:n_val])
    fit_engines = np.sort(shuffled[n_val:])

    print(f"Training deployment LSTM on all {len(engines)} engines "
          f"({len(fit_engines)} fit + {len(val_engines)} early-stopping val)")

    X_fit, y_fit, _, _ = make_windows(df, fit_engines)
    X_val, y_val, _, _ = make_windows(df, val_engines)
    print(f"  {len(y_fit):,} training windows / {len(y_val):,} validation windows")

    model, history = train_lstm(X_fit, y_fit, X_val, y_val)

    torch.save({
        "state_dict": model.state_dict(),
        "seq_len": SEQ_LEN,
        "n_features": X_fit.shape[2],
    }, LSTM_MODEL_PATH)

    hist = pd.DataFrame(history)
    i = hist["val_rmse"].idxmin()
    print(f"\nBest checkpoint: epoch {int(hist.loc[i, 'epoch'])} "
          f"(val RMSE {hist.loc[i, 'val_rmse']:.2f})")
    print(f"Saved: {LSTM_MODEL_PATH}")
    print("\nNote: this val RMSE is NOT a headline metric — these engines were "
          "used\nfor early stopping. Quote compare_models.py's held-out numbers "
          "instead.")


if __name__ == "__main__":
    main()
