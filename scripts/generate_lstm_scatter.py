"""
Predicted-vs-actual scatter for the LSTM, matching scripts/generate_figures.py.

Reuses that script's plot style so the LSTM chart is a drop-in replacement for
the Random Forest one (docs/images/pred_vs_actual.png), which still carries the
superseded RMSE 18.0 figures.

Loads the EVALUATION checkpoint written by compare_models.py — trained on 60
engines so 25 stay unseen — and scores exactly the points that produced the
published RMSE 16.59. No retraining.

Usage: python scripts/generate_lstm_scatter.py
Output: docs/images/lstm_pred_vs_actual.png
"""

import os
import sys

import numpy as np
import pandas as pd
import torch
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from lstm_model import SEED, SEQ_LEN, LSTMRegressor, make_windows, predict  # noqa: E402
from model import RUL_CAP  # noqa: E402

DATA = os.path.join(ROOT, "nasa_cmapss_FD001_scaled.csv")
CKPT = os.path.join(ROOT, "lstm_model.pt")
OUT = os.path.join(ROOT, "docs", "images")

SURFACE, INK, INK2, MUTED, GRID, BASE = (
    "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7")
BLUE, AQUA = "#2a78d6", "#1baf7a"

mpl.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"],
    "text.color": INK, "axes.labelcolor": INK2, "axes.titlecolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.edgecolor": BASE, "axes.linewidth": 1.0, "figure.dpi": 140,
})


def style(ax, title, xlabel, ylabel):
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12, loc="left")
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)


def main():
    if not os.path.exists(CKPT):
        sys.exit("missing lstm_model.pt — run `python compare_models.py` first")

    df = pd.read_csv(DATA)
    groups = df["unit_number"].values
    _, te = next(GroupShuffleSplit(
        n_splits=1, test_size=0.25, random_state=SEED).split(df, groups=groups))
    test_engines = np.unique(groups[te])

    # Same scoring points as the published comparison: test cycles >= SEQ_LEN.
    Xw, y_true, _, _ = make_windows(df, test_engines, min_cycle=SEQ_LEN)

    model = LSTMRegressor(n_features=Xw.shape[2])
    model.load_state_dict(torch.load(CKPT, map_location="cpu"))
    model.eval()
    pred = predict(model, Xw)

    rmse = np.sqrt(mean_squared_error(y_true, pred))
    mae = mean_absolute_error(y_true, pred)
    r2 = r2_score(y_true, pred)
    print(f"LSTM on hold-out: RMSE {rmse:.2f}  MAE {mae:.2f}  R2 {r2:.3f} "
          f"({len(y_true):,} points, {len(test_engines)} unseen engines)")

    os.makedirs(OUT, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.4, 6))
    ax.scatter(y_true, pred, s=9, color=BLUE, alpha=0.18, edgecolors="none",
               zorder=3)
    ax.plot([0, RUL_CAP], [0, RUL_CAP], color=MUTED, linewidth=1.5,
            linestyle="--", zorder=4)
    ax.text(0.04, 0.96, f"RMSE  {rmse:.1f}\nMAE   {mae:.1f}\nR²      {r2:.2f}",
            transform=ax.transAxes, va="top", ha="left", fontsize=10,
            color=INK, family="monospace",
            bbox=dict(boxstyle="round,pad=0.5", fc=SURFACE, ec=BASE))
    style(ax, "LSTM — predicted vs. actual RUL (held-out engines)",
          "actual RUL (cycles)", "predicted RUL (cycles)")
    ax.set_xlim(-3, 130)
    ax.set_ylim(-3, 130)
    fig.tight_layout()
    path = os.path.join(OUT, "lstm_pred_vs_actual.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print("wrote", os.path.relpath(path, ROOT))


if __name__ == "__main__":
    main()
