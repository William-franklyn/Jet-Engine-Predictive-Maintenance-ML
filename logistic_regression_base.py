"""
Baseline comparison on NASA C-MAPSS FD001: logistic vs linear regression.

Two framings of the same degradation problem:
  - LOGISTIC  -> classification: "is this engine within its last W cycles?"
  - LINEAR    -> regression:     "how many cycles of life remain (RUL)?"

They're bridged two ways so they're actually comparable:
  1. each reported on its native metric (F1/AUC vs RMSE/MAE),
  2. linear's predicted RUL is thresholded at W to emit the SAME binary
     flag as logistic, putting both on one classification scoreboard.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    f1_score, roc_auc_score, average_precision_score,
    mean_squared_error, mean_absolute_error,
)

df = pd.read_csv("nasa_cmapss_FD001_scaled.csv")

W = 30          # classification horizon (fail within W cycles)
RUL_CAP = 125   # piecewise-linear RUL cap: the landmine fix for regression

feature_cols = [c for c in df.columns
                if c.startswith("sensor_") or c.startswith("operational_setting")]
X = df[feature_cols].values
groups = df["unit_number"].values

# two targets, one per framing
y_cls = (df["RUL"] <= W).astype(int).values         # logistic target
y_reg = np.minimum(df["RUL"].values, RUL_CAP)       # linear target (CAPPED)

# identical engine-level split for both models -> fair comparison
gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
tr, te = next(gss.split(X, y_cls, groups))


def cmapss_score(y_true, y_pred):
    """Asymmetric PHM score. d>0 = predicted engine lasts LONGER than it
    does (late/dangerous, penalized harder, /10). d<0 = early (gentler, /13)."""
    d = y_pred - y_true
    return np.sum(np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1))


# ---- Model 1: LOGISTIC  ----
log = LogisticRegression(class_weight="balanced", max_iter=1000)
log.fit(X[tr], y_cls[tr])
p_log = log.predict_proba(X[te])[:, 1]
flag_log = (p_log >= 0.5).astype(int)

# ---- Model 2: LINEAR (native: regression on capped RUL) ----
lin = LinearRegression()
lin.fit(X[tr], y_reg[tr])
rul_pred = np.clip(lin.predict(X[te]), 0, RUL_CAP)   # no negative RUL
flag_lin = (rul_pred <= W).astype(int)               # bridge -> same binary flag

# ---- Native metrics ----
print("=== LINEAR regression ===")
print(f"RMSE          : {np.sqrt(mean_squared_error(y_reg[te], rul_pred)):.2f} cycles")
print(f"MAE           : {mean_absolute_error(y_reg[te], rul_pred):.2f} cycles")
print(f"C-MAPSS score : {cmapss_score(y_reg[te], rul_pred):.1f}  (lower is better)")

# ---- Shared classification scoreboard (both models, same labels) ----
print("\n=== SHARED SCOREBOARD (fail-within-30 flag) ===")
rows = [
    ("LOGISTIC",     flag_log, p_log),
    ("LINEAR->flag", flag_lin, -rul_pred),   # lower RUL = more likely positive
]
for name, flag, rank in rows:
    print(f"{name:13s} F1: {f1_score(y_cls[te], flag):.3f}   "
          f"ROC-AUC: {roc_auc_score(y_cls[te], rank):.3f}   "
          f"PR-AUC: {average_precision_score(y_cls[te], rank):.3f}")