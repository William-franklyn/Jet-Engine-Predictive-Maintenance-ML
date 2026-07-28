"""
Head-to-head model comparison on NASA C-MAPSS FD001.

Answers the team's question: does an LSTM actually beat what we already have?

Four models, one fair fight:
  LINEAR    regression on capped RUL, one cycle at a time
  LOGISTIC  classification "fails within W cycles?", one cycle at a time
  RF        Random Forest regression (the dashboard's current model)
  LSTM      regression over a SEQ_LEN-cycle window of history

Fairness rules this script enforces:
  1. ONE engine-grouped split (GroupShuffleSplit, test_size=0.25, seed 42) —
     identical to logistic_regression_base.py and save_model.py, so these
     numbers line up with what the team already reported. No engine appears
     in both train and test.
  2. ONE evaluation set of prediction points. The LSTM needs SEQ_LEN cycles
     of history, so it can't score cycles 1-29. Rather than let it skip the
     hard early cycles the others were charged for, EVERY model is scored on
     the same (engine, cycle) points: test cycles >= SEQ_LEN.
  3. ONE scoreboard. Regression models report RMSE/MAE/R^2/C-MAPSS; then
     every model's output is converted to the same "fails within W cycles"
     binary flag so logistic can be compared against the regressors.

Usage:  python compare_models.py
"""

import os

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    average_precision_score, f1_score, mean_absolute_error,
    mean_squared_error, r2_score, roc_auc_score,
)

from lstm_model import SEED, SEQ_LEN, make_windows, predict, train_lstm
from model import FEATURE_COLS, RUL_CAP
from save_model import build_model, cmapss_score

DATA_PATH = "nasa_cmapss_FD001_scaled.csv"
ID_COL = "unit_number"
CYCLE_COL = "time_in_cycles"

W = 30              # classification horizon: "fails within W cycles"
VAL_FRACTION = 0.2  # share of TRAIN engines held out for LSTM early stopping


def split_engines(df):
    """Engine-grouped train/test split, then carve a validation set out of train.

    The validation engines are used only for LSTM early stopping. They stay in
    the training data for the sklearn models, which have no such knob — that
    gives the LSTM slightly LESS training data than its competitors, which is
    the conservative direction for the claim "the LSTM wins".
    """
    groups = df[ID_COL].values
    gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=SEED)
    tr_idx, te_idx = next(gss.split(df, groups=groups))

    train_engines = np.unique(groups[tr_idx])
    test_engines = np.unique(groups[te_idx])

    rng = np.random.default_rng(SEED)
    shuffled = rng.permutation(train_engines)
    n_val = max(1, int(len(shuffled) * VAL_FRACTION))
    val_engines = np.sort(shuffled[:n_val])
    fit_engines = np.sort(shuffled[n_val:])

    return train_engines, fit_engines, val_engines, test_engines


def flat_frame(df, engines):
    """All rows for the given engines, as (X, y_reg, y_cls)."""
    sub = df[df[ID_COL].isin(engines)]
    X = sub[FEATURE_COLS].to_numpy()
    y_reg = np.minimum(sub["RUL"].to_numpy(), RUL_CAP)
    y_cls = (sub["RUL"].to_numpy() <= W).astype(int)
    return X, y_reg, y_cls


def eval_points(df, engines):
    """The shared evaluation set: test rows at cycle >= SEQ_LEN.

    Returns the flat feature matrix, the LSTM's window tensor, and the shared
    targets — all aligned row-for-row, so index i is the same (engine, cycle)
    for every model.
    """
    Xw, y_reg, units, cycles = make_windows(df, engines, min_cycle=SEQ_LEN)

    # Pull the matching single-row features in the exact same order.
    indexed = df.set_index([ID_COL, CYCLE_COL]).sort_index()
    keys = list(zip(units, cycles))
    rows = indexed.loc[keys]
    X_flat = rows[FEATURE_COLS].to_numpy()
    y_cls = (rows["RUL"].to_numpy() <= W).astype(int)

    assert len(X_flat) == len(Xw) == len(y_reg) == len(y_cls)
    return X_flat, Xw, y_reg, y_cls


def regression_row(name, y_true, y_pred):
    return {
        "model": name,
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "MAE": mean_absolute_error(y_true, y_pred),
        "R2": r2_score(y_true, y_pred),
        "CMAPSS": cmapss_score(y_true, y_pred),
    }


def classification_row(name, y_cls, flag, rank):
    return {
        "model": name,
        "Accuracy": (flag == y_cls).mean(),
        "F1": f1_score(y_cls, flag),
        "ROC-AUC": roc_auc_score(y_cls, rank),
        "PR-AUC": average_precision_score(y_cls, rank),
    }


def main():
    df = pd.read_csv(DATA_PATH)
    train_engines, fit_engines, val_engines, test_engines = split_engines(df)

    print(f"Data   : {len(df):,} rows, {df[ID_COL].nunique()} engines")
    print(f"Split  : {len(train_engines)} train "
          f"({len(fit_engines)} fit + {len(val_engines)} val, LSTM only) / "
          f"{len(test_engines)} test engines")

    X_tr, y_reg_tr, y_cls_tr = flat_frame(df, train_engines)
    X_te, Xw_te, y_reg_te, y_cls_te = eval_points(df, test_engines)
    print(f"Scoring: {len(y_reg_te):,} shared prediction points "
          f"(test cycles >= {SEQ_LEN}), {y_cls_te.mean():.1%} within {W} cycles of failure\n")

    preds = {}   # model -> predicted RUL on the shared points

    # ---- Linear regression -------------------------------------------------
    print("Training LINEAR regression...")
    lin = LinearRegression().fit(X_tr, y_reg_tr)
    preds["LINEAR"] = np.clip(lin.predict(X_te), 0, RUL_CAP)

    # ---- Logistic regression (classifier — no RUL output) ------------------
    print("Training LOGISTIC regression...")
    log = LogisticRegression(class_weight="balanced", max_iter=1000)
    log.fit(X_tr, y_cls_tr)
    p_log = log.predict_proba(X_te)[:, 1]

    # ---- Random Forest (the dashboard's current model) ---------------------
    print("Training RANDOM FOREST...")
    rf = build_model().fit(X_tr, y_reg_tr)
    preds["RF"] = np.clip(rf.predict(X_te), 0, RUL_CAP)

    # ---- LSTM --------------------------------------------------------------
    print(f"Training LSTM (window={SEQ_LEN} cycles)...")
    Xw_fit, y_fit, _, _ = make_windows(df, fit_engines)
    Xw_val, y_val, _, _ = make_windows(df, val_engines)
    print(f"  {len(y_fit):,} training windows / {len(y_val):,} validation windows")
    lstm, history = train_lstm(Xw_fit, y_fit, Xw_val, y_val)
    preds["LSTM"] = predict(lstm, Xw_te)

    # ---- Scoreboard 1: regression -----------------------------------------
    order = ["LINEAR", "RF", "LSTM"]
    reg_table = pd.DataFrame(
        [regression_row(n, y_reg_te, preds[n]) for n in order]
    ).set_index("model")

    print("\n" + "=" * 68)
    print("REGRESSION — predicting RUL in cycles (lower RMSE/MAE/C-MAPSS = better)")
    print("=" * 68)
    print(reg_table.to_string(
        float_format=lambda v: f"{v:8.2f}",
        formatters={"R2": lambda v: f"{v:8.3f}"},
    ))
    print("\n(LOGISTIC is a classifier — it outputs a probability, not a cycle "
          "count,\n so it has no row here. It competes on the scoreboard below.)")

    # ---- Scoreboard 2: shared classification -------------------------------
    # Every model reduced to the same question: "will this engine fail within
    # W cycles?" For the regressors, that's predicted RUL <= W. For ranking
    # metrics, -RUL is the score (lower predicted RUL = more likely failing).
    cls_rows = [classification_row("LOGISTIC", y_cls_te,
                                   (p_log >= 0.5).astype(int), p_log)]
    for n in order:
        cls_rows.append(classification_row(
            f"{n}->flag", y_cls_te, (preds[n] <= W).astype(int), -preds[n]
        ))
    cls_table = pd.DataFrame(cls_rows).set_index("model")

    print("\n" + "=" * 68)
    print(f"CLASSIFICATION — 'fails within {W} cycles?' (higher = better)")
    print("=" * 68)
    print(cls_table.to_string(float_format=lambda v: f"{v:9.3f}"))

    # ---- Verdict -----------------------------------------------------------
    best_rmse = reg_table["RMSE"].idxmin()
    best_f1 = cls_table["F1"].idxmax()
    lstm_vs_rf = reg_table.loc["RF", "RMSE"] - reg_table.loc["LSTM", "RMSE"]
    lstm_vs_lin = reg_table.loc["LINEAR", "RMSE"] - reg_table.loc["LSTM", "RMSE"]

    print("\n" + "=" * 68)
    print("VERDICT")
    print("=" * 68)
    print(f"Best RMSE : {best_rmse}")
    print(f"Best F1   : {best_f1}")
    print(f"LSTM vs LINEAR : {lstm_vs_lin:+.2f} RMSE cycles "
          f"({'LSTM better' if lstm_vs_lin > 0 else 'LINEAR better'})")
    print(f"LSTM vs RF     : {lstm_vs_rf:+.2f} RMSE cycles "
          f"({'LSTM better' if lstm_vs_rf > 0 else 'RF better'})")

    # ---- Persist results so the README figures are reproducible ------------
    os.makedirs("docs", exist_ok=True)
    reg_table.to_csv("docs/model_comparison_regression.csv")
    cls_table.to_csv("docs/model_comparison_classification.csv")
    pd.DataFrame(history).to_csv("docs/lstm_training_curve.csv", index=False)

    torch.save(lstm.state_dict(), "lstm_model.pt")
    print("\nSaved LSTM weights : lstm_model.pt")
    print("Saved results      : docs/model_comparison_*.csv, "
          "docs/lstm_training_curve.csv")
    print("Regenerate figures : python scripts/generate_model_comparison.py")


if __name__ == "__main__":
    main()
