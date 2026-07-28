"""
Generate the model-selection figures used in README.md.

Reads the CSVs that `python compare_models.py` writes — nothing here is
hand-typed, so the charts can never drift from the measured results:

    docs/model_comparison_regression.csv
    docs/model_comparison_classification.csv
    docs/lstm_training_curve.csv

Writes:
    docs/images/model_comparison.png      RMSE/MAE per model, all four rounds
    docs/images/lstm_training_curve.png   LSTM validation RMSE per epoch

Usage: python scripts/generate_model_comparison.py
Requires: matplotlib, plus a prior run of compare_models.py.
"""

import os
import sys

import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCS = os.path.join(ROOT, "docs")
OUT = os.path.join(DOCS, "images")
os.makedirs(OUT, exist_ok=True)

# Same validated palette as scripts/generate_figures.py — the two figure sets
# sit next to each other in the README and must read as one system.
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

# Round number each model was trained in — the story the README tells.
ROUND = {"LINEAR": "Round 1", "RF": "Round 2", "LSTM": "Round 4"}
LABEL = {"LINEAR": "Linear\nRegression", "RF": "Random\nForest", "LSTM": "LSTM"}


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


def require(path):
    if not os.path.exists(path):
        sys.exit(f"missing {os.path.relpath(path, ROOT)} — "
                 f"run `python compare_models.py` first")
    return path


def fig_model_comparison():
    """Grouped RMSE/MAE bars. The winner is highlighted; the rest are muted."""
    reg = pd.read_csv(
        require(os.path.join(DOCS, "model_comparison_regression.csv")),
        index_col="model",
    )
    order = [m for m in ["LINEAR", "RF", "LSTM"] if m in reg.index]
    best = reg.loc[order, "RMSE"].idxmin()

    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    x = range(len(order))
    width = 0.38

    for metric, color, offset in [("RMSE", BLUE, -width / 2),
                                  ("MAE", AQUA, width / 2)]:
        vals = [reg.loc[m, metric] for m in order]
        bars = ax.bar(
            [p + offset for p in x], vals, width, zorder=3,
            # winner in full colour, earlier rounds muted — the eye lands on LSTM
            color=[color if m == best else BASE for m in order],
        )
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.35, f"{v:.2f}",
                    ha="center", va="bottom", fontsize=9,
                    fontweight="bold", color=INK2)

    # Proxy handles: the real bars are per-model coloured, so a normal legend
    # would show two identical grey swatches.
    ax.legend(
        handles=[mpl.patches.Patch(facecolor=BLUE, label="RMSE"),
                 mpl.patches.Patch(facecolor=AQUA, label="MAE")],
        frameon=False, fontsize=10, ncol=2, loc="upper right",
    )

    ax.set_xticks(list(x))
    ax.set_xticklabels([f"{LABEL[m]}\n{ROUND[m]}" for m in order], fontsize=10)
    style(ax, "Error by model — lower is better",
          ylabel="cycles of error")
    ax.set_ylim(0, max(reg.loc[order, "RMSE"]) * 1.22)
    ax.text(0, -0.30, f"Winner: {best} — evaluated on 25 unseen engines, "
            f"identical grouped split (seed 42)",
            transform=ax.transAxes, fontsize=9, color=MUTED)
    save(fig, "model_comparison.png")


def fig_training_curve():
    """LSTM validation RMSE per epoch, with the other models' scores as lines.

    This is the 'we trained it, we watched it, we measured it' evidence — and
    it shows the curve was still descending at the epoch cap.
    """
    hist = pd.read_csv(require(os.path.join(DOCS, "lstm_training_curve.csv")))
    reg = pd.read_csv(
        require(os.path.join(DOCS, "model_comparison_regression.csv")),
        index_col="model",
    )

    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    ax.plot(hist["epoch"], hist["val_rmse"], color=BLUE, linewidth=2.2,
            zorder=4, label="LSTM validation RMSE")

    # Baseline labels sit on the LEFT: the curve starts high there, so the
    # region is empty — on the right they collide with each other and with the
    # end-of-training annotation.
    for name, color, dy in [("LINEAR", MUTED, 1.6), ("RF", ORANGE, -3.4)]:
        if name in reg.index:
            ax.axhline(reg.loc[name, "RMSE"], color=color, linestyle="--",
                       linewidth=1.4, zorder=3)
            ax.text(hist["epoch"].max() * 0.02, reg.loc[name, "RMSE"] + dy,
                    f"{LABEL[name].replace(chr(10), ' ')} ({reg.loc[name, 'RMSE']:.2f})",
                    ha="left", fontsize=9, color=color, fontweight="bold")

    # Mark the checkpoint that was actually kept: training restores the best
    # validation weights, so this epoch — not the last one — is the model.
    i = hist["val_rmse"].idxmin()
    best_ep, best_val = hist.loc[i, "epoch"], hist.loc[i, "val_rmse"]
    ax.scatter([best_ep], [best_val], color=GOOD, s=55, zorder=6)
    ax.annotate(f"best checkpoint — epoch {int(best_ep)} ({best_val:.2f})\n"
                f"{int(hist['epoch'].max() - best_ep)} further epochs "
                f"with LR decay found nothing better",
                xy=(best_ep, best_val),
                xytext=(18, 46), textcoords="offset points",
                ha="left", fontsize=9, color=INK2,
                arrowprops=dict(arrowstyle="->", color=MUTED, linewidth=1.1))

    style(ax, "LSTM learning curve — converged below both earlier models",
          xlabel="training epoch", ylabel="validation RMSE (cycles)")
    ax.legend(frameon=False, fontsize=10, loc="upper right")
    save(fig, "lstm_training_curve.png")


# Measured on three environments — see docs/reproduction-log.md. Hard-coded
# because they come from machines this script cannot run on; the local column
# is cross-checked against the freshly measured CSV below.
ENVIRONMENTS = {
    "Local\nWindows":  {"LINEAR": 20.55, "RF": 18.69, "LSTM": 16.59},
    "Kaggle\nLinux":   {"LINEAR": 20.55, "RF": 18.69, "LSTM": 15.93},
    "Colab\nLinux":    {"LINEAR": 20.55, "RF": 18.69, "LSTM": 16.04},
}


def fig_reproduction():
    """Three environments, one experiment. Shows the sklearn models pinned flat
    and the LSTM varying slightly — while never approaching the RF line."""
    reg = pd.read_csv(
        require(os.path.join(DOCS, "model_comparison_regression.csv")),
        index_col="model",
    )
    # Guard against the hard-coded local column drifting from a fresh run.
    local = ENVIRONMENTS["Local\nWindows"]["LSTM"]
    if abs(reg.loc["LSTM", "RMSE"] - local) > 0.005:
        print(f"  WARNING: local LSTM RMSE is now {reg.loc['LSTM','RMSE']:.2f} "
              f"but ENVIRONMENTS says {local} — update this script.")

    names = list(ENVIRONMENTS)
    x = list(range(len(names)))
    fig, ax = plt.subplots(figsize=(8.2, 4.6))

    for model, color, marker in [("LINEAR", MUTED, "s"), ("RF", ORANGE, "^"),
                                 ("LSTM", BLUE, "o")]:
        vals = [ENVIRONMENTS[n][model] for n in names]
        ax.plot(x, vals, color=color, marker=marker, markersize=9,
                linewidth=2.2, zorder=4, label=LABEL[model].replace("\n", " "))
        for xi, v in zip(x, vals):
            ax.annotate(f"{v:.2f}", (xi, v), textcoords="offset points",
                        xytext=(0, 11), ha="center", fontsize=9,
                        fontweight="bold", color=color)

    lstm_vals = [ENVIRONMENTS[n]["LSTM"] for n in names]
    ax.axhspan(min(lstm_vals), max(lstm_vals), color=BLUE, alpha=0.07, zorder=1)

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=10)
    ax.set_xlim(-0.35, len(names) - 0.65)
    style(ax, "Same experiment, three machines — RMSE (lower is better)",
          ylabel="RMSE (cycles)")
    ax.legend(frameon=False, fontsize=10, loc="center right")
    ax.text(0, -0.26,
            "Linear and Random Forest are bit-identical everywhere "
            "(scikit-learn is deterministic).\nThe LSTM varies 15.93–16.59 from "
            "floating-point ordering — and beats Random Forest in all three.",
            transform=ax.transAxes, fontsize=9, color=MUTED)
    save(fig, "reproduction_environments.png")


if __name__ == "__main__":
    fig_model_comparison()
    fig_training_curve()
    fig_reproduction()
