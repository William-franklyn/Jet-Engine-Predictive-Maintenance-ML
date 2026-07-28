"""
Generate the data-visualization figures used in README.md.

Reproducible: reads nasa_cmapss_FD001_scaled.csv, trains the same grouped-split
Random Forest as save_model.py, and writes six PNGs to docs/images/.

Usage: python scripts/generate_figures.py
Requires: matplotlib (pip install matplotlib) in addition to the app deps.
"""

import os

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# repo root regardless of where the script is invoked from
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "docs", "images")
os.makedirs(OUT, exist_ok=True)

FEATURE_COLS = [
    "operational_setting_1",
    "sensor_2", "sensor_3", "sensor_4", "sensor_6", "sensor_7", "sensor_8",
    "sensor_9", "sensor_11", "sensor_12", "sensor_13", "sensor_14",
    "sensor_15", "sensor_17", "sensor_20", "sensor_21",
]
RUL_CAP = 125

# ── validated dataviz palette (light surface, readable on GitHub light+dark) ──
SURFACE, INK, INK2, MUTED, GRID, BASE = (
    "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7")
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
RED, GOOD = "#d03b3b", "#0ca30c"

mpl.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"],
    "text.color": INK, "axes.labelcolor": INK2, "axes.titlecolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.edgecolor": BASE, "axes.linewidth": 1.0,
    "figure.dpi": 140,
})


def style(ax, title, xlabel=None, ylabel=None):
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12, loc="left")
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=10)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)


def save(fig, name):
    path = os.path.join(OUT, name)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print("wrote", os.path.relpath(path, ROOT))


def main():
    df = pd.read_csv(os.path.join(ROOT, "nasa_cmapss_FD001_scaled.csv"))
    df_capped = df.copy()
    df_capped["RUL_c"] = np.minimum(df["RUL"], RUL_CAP)

    X = df[FEATURE_COLS].values
    y = df_capped["RUL_c"].values
    groups = df["unit_number"].values
    gss = GroupShuffleSplit(1, test_size=0.25, random_state=42)
    tr, te = next(gss.split(X, y, groups))
    rf = RandomForestRegressor(n_estimators=200, min_samples_leaf=10,
                               max_features="sqrt", random_state=42, n_jobs=-1)
    rf.fit(X[tr], y[tr])
    pred = np.clip(rf.predict(X[te]), 0, RUL_CAP)
    rmse = np.sqrt(mean_squared_error(y[te], pred))
    mae = mean_absolute_error(y[te], pred)
    r2 = r2_score(y[te], pred)

    # 1 ── RUL distribution + piecewise-linear cap ───────────────────────────
    fig, ax = plt.subplots(figsize=(8, 4.4))
    ax.hist(df["RUL"], bins=60, color=BLUE, edgecolor=SURFACE, linewidth=0.4,
            zorder=3)
    ax.axvline(RUL_CAP, color=RED, linewidth=2, linestyle="--", zorder=4)
    capped = int((df["RUL"] > RUL_CAP).mean() * 100)
    ax.text(RUL_CAP + 6, ax.get_ylim()[1] * 0.9,
            f"cap = 125 cycles\n({capped}% of rows clipped)",
            color=RED, fontsize=9, va="top")
    style(ax, "Distribution of Remaining Useful Life (raw labels)",
          "RUL — cycles until failure", "row count")
    save(fig, "rul_distribution.png")

    # 2 ── Sensor degradation as engines approach failure ────────────────────
    fig, ax = plt.subplots(figsize=(8, 4.4))
    bins = np.arange(0, 131, 5)
    mids = bins[:-1] + 2.5
    df_capped["bin"] = pd.cut(df_capped["RUL_c"], bins, labels=mids,
                              include_lowest=True)
    for sensor, color, lab in [("sensor_11", BLUE, "sensor 11"),
                               ("sensor_4", ORANGE, "sensor 4"),
                               ("sensor_12", AQUA, "sensor 12")]:
        m = df_capped.groupby("bin", observed=True)[sensor].mean()
        ax.plot(m.index.astype(float), m.values, color=color, linewidth=2.2,
                marker="o", markersize=4, label=lab, zorder=3)
    ax.axvline(0, color=MUTED, linewidth=1, linestyle=":")
    ax.invert_xaxis()  # failure (RUL=0) on the right
    style(ax, "Sensors drift systematically as failure approaches",
          "Remaining Useful Life (cycles) — failure at right",
          "mean scaled sensor reading")
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    save(fig, "sensor_degradation.png")

    # 3 ── Linear correlation of each feature with RUL (diverging) ────────────
    corr = (df_capped[FEATURE_COLS + ["RUL_c"]].corr()["RUL_c"]
            .drop("RUL_c").sort_values())
    fig, ax = plt.subplots(figsize=(8, 5.4))
    colors = [BLUE if v >= 0 else RED for v in corr.values]
    ax.barh(range(len(corr)), corr.values, color=colors, zorder=3, height=0.7)
    ax.set_yticks(range(len(corr)))
    ax.set_yticklabels([c.replace("operational_setting", "op_setting")
                        for c in corr.index], fontsize=9)
    ax.axvline(0, color=BASE, linewidth=1)
    ax.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)
    ax.grid(axis="y", visible=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_axisbelow(True)
    ax.set_title("Pearson correlation of each feature with RUL",
                 fontsize=13, fontweight="bold", pad=12, loc="left")
    ax.set_xlabel("correlation coefficient  (blue = + / red = −)", fontsize=10)
    save(fig, "rul_correlation.png")

    # 4 ── Random-Forest feature importances ─────────────────────────────────
    imp = pd.Series(rf.feature_importances_, index=FEATURE_COLS).sort_values()
    fig, ax = plt.subplots(figsize=(8, 5.4))
    ax.barh(range(len(imp)), imp.values, color=BLUE, zorder=3, height=0.7)
    ax.set_yticks(range(len(imp)))
    ax.set_yticklabels([c.replace("operational_setting", "op_setting")
                        for c in imp.index], fontsize=9)
    ax.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_axisbelow(True)
    ax.set_title("Random Forest feature importance",
                 fontsize=13, fontweight="bold", pad=12, loc="left")
    ax.set_xlabel("mean decrease in impurity", fontsize=10)
    save(fig, "feature_importance.png")

    # 5 ── Predicted vs actual on the held-out engines ───────────────────────
    fig, ax = plt.subplots(figsize=(6.4, 6))
    ax.scatter(y[te], pred, s=9, color=BLUE, alpha=0.18, edgecolors="none",
               zorder=3)
    ax.plot([0, RUL_CAP], [0, RUL_CAP], color=MUTED, linewidth=1.5,
            linestyle="--", zorder=4)
    ax.text(0.04, 0.96, f"RMSE  {rmse:.1f}\nMAE   {mae:.1f}\nR²      {r2:.2f}",
            transform=ax.transAxes, va="top", ha="left", fontsize=10,
            color=INK, family="monospace",
            bbox=dict(boxstyle="round,pad=0.5", fc=SURFACE, ec=BASE))
    style(ax, "Predicted vs. actual RUL (held-out engines)",
          "actual RUL (cycles)", "predicted RUL (cycles)")
    ax.set_xlim(-3, 130)
    ax.set_ylim(-3, 130)
    save(fig, "pred_vs_actual.png")

    # 6 ── Residual distribution ─────────────────────────────────────────────
    resid = pred - y[te]
    fig, ax = plt.subplots(figsize=(8, 4.4))
    ax.hist(resid, bins=50, color=AQUA, edgecolor=SURFACE, linewidth=0.4,
            zorder=3)
    ax.axvline(0, color=MUTED, linewidth=1.5, linestyle="--", zorder=4)
    ax.text(0.03, 0.95,
            f"mean error {resid.mean():+.1f} cyc\nstd {resid.std():.1f} cyc",
            transform=ax.transAxes, va="top", fontsize=9, color=INK2)
    style(ax, "Prediction error distribution (predicted − actual)",
          "error in cycles  (negative = conservative / early warning)",
          "row count")
    save(fig, "residuals.png")

    print(f"\nModel on hold-out: RMSE {rmse:.2f}  MAE {mae:.2f}  R2 {r2:.3f}")


if __name__ == "__main__":
    main()
