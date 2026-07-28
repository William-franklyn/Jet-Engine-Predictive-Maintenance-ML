"""
Data audit for nasa_cmapss_FD001_scaled.csv — run BEFORE trusting any model.

Checks the dataset on four axes and prints a pass/fail report:

  1. INTEGRITY      missing values, duplicates, dtypes, cycle continuity
  2. LABELS         is RUL internally consistent with the cycle counts?
  3. PREPROCESSING  was the scaler fit on training data only? (leakage check)
  4. BIAS           what this data does and does not represent

Also writes docs/images/data_audit.png — engine-lifetime and RUL distributions
with the train/test split overlaid, so the split's representativeness is visible
rather than assumed.

Usage: python scripts/audit_data.py
Requires: matplotlib (for the figure); the checks alone need only pandas/numpy.
"""

import os
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "nasa_cmapss_FD001_scaled.csv")
OUT = os.path.join(ROOT, "docs", "images")

SEED, W, RUL_CAP, SEQ_LEN = 42, 30, 125, 30
ID_COL, CYCLE_COL = "unit_number", "time_in_cycles"

PASS, FAIL, WARN = "  [PASS]", "  [FAIL]", "  [WARN]"
findings = []


def check(ok, label, detail="", warn_only=False):
    tag = PASS if ok else (WARN if warn_only else FAIL)
    print(f"{tag} {label}" + (f" — {detail}" if detail else ""))
    if not ok:
        findings.append((label, detail, warn_only))
    return ok


def main():
    df = pd.read_csv(DATA)
    feats = [c for c in df.columns if c.startswith(("sensor_", "operational"))]

    print("=" * 72)
    print("DATA AUDIT — nasa_cmapss_FD001_scaled.csv")
    print("=" * 72)
    print(f"{len(df):,} rows · {df[ID_COL].nunique()} engines · "
          f"{len(feats)} features\n")

    # ── 1. Integrity ────────────────────────────────────────────────────────
    print("1. INTEGRITY")
    check(df.isna().sum().sum() == 0, "No missing values",
          f"{int(df.isna().sum().sum())} found" if df.isna().sum().sum() else "")
    check(df.duplicated().sum() == 0, "No duplicate rows",
          f"{int(df.duplicated().sum())} found")
    non_numeric = [c for c in df.columns
                   if not np.issubdtype(df[c].dtype, np.number)]
    check(not non_numeric, "All columns numeric", str(non_numeric))
    dead = [c for c in feats if df[c].std() < 0.01]
    check(not dead, "No constant/dead feature columns", str(dead))

    gaps = sum(
        1 for _, g in df.groupby(ID_COL)
        if g[CYCLE_COL].min() != 1
        or not (g.sort_values(CYCLE_COL)[CYCLE_COL].diff().dropna() == 1).all()
    )
    check(gaps == 0, "Every engine's cycles run 1..N with no gaps",
          f"{gaps} engines affected")

    # ── 2. Labels ───────────────────────────────────────────────────────────
    print("\n2. LABELS")
    bad = sum(
        1 for _, g in df.groupby(ID_COL)
        if not (g.sort_values(CYCLE_COL)["RUL"].values
                == (g[CYCLE_COL].max() - g.sort_values(CYCLE_COL)[CYCLE_COL]).values).all()
    )
    check(bad == 0, "RUL == max_cycle - current_cycle for every engine",
          f"{bad} engines inconsistent")
    check(df["RUL"].min() == 0, "Every engine runs to failure (min RUL = 0)",
          f"min RUL is {df['RUL'].min()}")

    pos_rate = (df["RUL"] <= W).mean()
    print(f"        Class balance for 'fails within {W} cycles': "
          f"{pos_rate:.1%} of all rows — imbalanced, so report F1/PR-AUC, "
          f"not accuracy.")
    print(f"        (compare_models.py reports 17.9% because it scores only "
          f"test cycles >= {SEQ_LEN}.\n         Dropping early-life rows removes "
          f"negatives only, so the rate rises while the\n         positive count "
          f"— exactly {W + 1} rows per engine — is unchanged. Both are correct.)")

    # ── 3. Preprocessing / leakage ──────────────────────────────────────────
    print("\n3. PREPROCESSING")
    groups = df[ID_COL].values
    tr, te = next(GroupShuffleSplit(
        n_splits=1, test_size=0.25, random_state=SEED).split(df, groups=groups))

    global_mean = df[feats].values.mean()
    train_mean = df.iloc[tr][feats].values.mean()

    # If the scaler had been fit on TRAIN only, the train subset would be the
    # one centred at exactly 0. It's the full file that is — so the scaler saw
    # the test engines too.
    fit_on_all = abs(global_mean) < abs(train_mean)
    check(not fit_on_all,
          "Scaler fit on training data only",
          f"scaler was fit on ALL 100 engines "
          f"(global mean {global_mean:+.8f} vs train mean {train_mean:+.5f}) "
          f"— mild preprocessing leakage; see README Limitations",
          warn_only=True)

    extreme = (np.abs(df[feats].values) > 6).mean()
    print(f"        Readings beyond 6 sd: {extreme:.3%} "
          f"(max |z| = {np.abs(df[feats].values).max():.2f}) — these are "
          f"near-failure engines, i.e. signal, not error. Not removed.")

    # ── 4. Split representativeness ─────────────────────────────────────────
    print("\n4. SPLIT REPRESENTATIVENESS")
    life = df.groupby(ID_COL)[CYCLE_COL].max()
    tr_life = life[np.unique(groups[tr])]
    te_life = life[np.unique(groups[te])]
    print(f"        Train engines: n={len(tr_life)}, "
          f"lifetime mean {tr_life.mean():.0f}, median {tr_life.median():.0f}")
    print(f"        Test  engines: n={len(te_life)}, "
          f"lifetime mean {te_life.mean():.0f}, median {te_life.median():.0f}")
    drift = abs(tr_life.mean() - te_life.mean()) / tr_life.mean()
    check(drift < 0.10,
          "Test engines have a representative lifetime distribution",
          f"means differ by {drift:.1%}", warn_only=True)
    check(life.min() >= SEQ_LEN,
          f"Every engine is longer than the LSTM's {SEQ_LEN}-cycle window",
          f"shortest engine is {life.min()} cycles")

    # ── 5. Bias / representativeness (not fixable by cleaning) ──────────────
    print("\n5. BIAS & REPRESENTATIVENESS  (inherent to the dataset)")
    print("        These are limits of FD001 itself. No amount of cleaning")
    print("        removes them; they bound what the model may be used for.")
    for line in [
        "SIMULATED, not real engines — C-MAPSS is a NASA simulation. Real "
        "telemetry carries sensor drift, maintenance events and varied duty "
        "cycles this data does not.",
        "ONE operating condition — FD001 is the simplest subset (sea level). "
        "FD002/FD004 add regimes the model has never seen.",
        "ONE fault mode — HPC degradation only. The model has no exposure to "
        "fan or bearing failures and would be blind to them.",
        "NO CENSORED ENGINES — all 100 run to failure. Real fleets retire "
        "healthy engines on schedule; those observations are absent, so the "
        "model has never seen an engine that was fine-but-withdrawn.",
        f"IMBALANCED alarm class — only {pos_rate:.1%} of rows are within "
        f"{W} cycles of failure.",
        "ASYMMETRIC COST — a late prediction risks an in-flight failure; an "
        "early one wastes money. Reported via the C-MAPSS score.",
    ]:
        print(f"          • {line}")

    # ── figure ──────────────────────────────────────────────────────────────
    try:
        make_figure(df, life, tr_life, te_life)
    except ImportError:
        print("\n(matplotlib not installed — skipped the figure)")

    # ── verdict ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    hard = [f for f in findings if not f[2]]
    warns = [f for f in findings if f[2]]
    if hard:
        print(f"VERDICT: {len(hard)} INTEGRITY FAILURE(S) — do not train on this.")
        for label, detail, _ in hard:
            print(f"  - {label}: {detail}")
        sys.exit(1)
    print("VERDICT: dataset is clean — no missing values, no duplicates, no "
          "dead\ncolumns, no cycle gaps, and labels are internally consistent.")
    if warns:
        print(f"\n{len(warns)} disclosed caveat(s), documented in the README:")
        for label, detail, _ in warns:
            print(f"  - {label}: {detail}")
    print("=" * 72)


def make_figure(df, life, tr_life, te_life):
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    SURFACE, INK, INK2, MUTED, GRID, BASE = (
        "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7")
    BLUE, ORANGE = "#2a78d6", "#eb6834"
    mpl.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE, "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"],
        "text.color": INK, "axes.labelcolor": INK2, "axes.titlecolor": INK,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.edgecolor": BASE, "figure.dpi": 140,
    })
    os.makedirs(OUT, exist_ok=True)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))

    bins = np.linspace(life.min(), life.max(), 22)
    a1.hist(tr_life, bins=bins, color=BLUE, alpha=0.75,
            label=f"Train ({len(tr_life)})", zorder=3)
    a1.hist(te_life, bins=bins, color=ORANGE, alpha=0.75,
            label=f"Test ({len(te_life)})", zorder=3)
    a1.set_title("Engine lifetimes — train vs test", fontsize=12,
                 fontweight="bold", loc="left", pad=10)
    a1.set_xlabel("total cycles before failure")
    a1.set_ylabel("engines")
    a1.legend(frameon=False, fontsize=9)

    a2.hist(df["RUL"], bins=40, color=BLUE, zorder=3)
    a2.axvline(RUL_CAP, color=ORANGE, linestyle="--", linewidth=1.6, zorder=4)
    a2.text(RUL_CAP + 6, a2.get_ylim()[1] * 0.86, f"cap = {RUL_CAP}",
            color=ORANGE, fontsize=9, fontweight="bold")
    a2.set_title("RUL distribution (uncapped)", fontsize=12,
                 fontweight="bold", loc="left", pad=10)
    a2.set_xlabel("remaining useful life (cycles)")
    a2.set_ylabel("rows")

    for ax in (a1, a2):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)

    fig.tight_layout()
    path = os.path.join(OUT, "data_audit.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"\nwrote {os.path.relpath(path, ROOT)}")


if __name__ == "__main__":
    main()
