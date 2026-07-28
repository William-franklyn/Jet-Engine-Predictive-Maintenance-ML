"""
Self-contained reproduction of the four-round model comparison — built to be
pasted into a single Kaggle / Colab notebook cell.

Everything the repo splits across model.py, lstm_model.py, save_model.py and
compare_models.py is inlined here, with the same constants and the same seed,
so a teammate can verify the README's numbers without cloning anything.

Requirements on Kaggle: turn **Internet ON** (the script pulls the CSV from
GitHub). torch / sklearn / pandas are preinstalled.

Deliberately CPU-only — no .to(device) calls. GPU kernels are not
bit-reproducible across machines, and reproducibility matters more here than
the ~2 minutes a GPU would save.

Runtime: ~10-15 min.

Verified: reproduces the README exactly — Linear 20.55 / RF 18.69 / LSTM 16.59,
best LSTM checkpoint at epoch 56 (val RMSE 15.71), early stop at epoch 81.

Windows note: on some Windows setups this segfaults mid-training — a torch/
OpenMP threading fault, not a bug in this code (it dies at a different epoch
each run). Cap the threads and it completes with identical numbers:

    set OMP_NUM_THREADS=2
    python kaggle_reproduce.py

Kaggle and Colab run Linux and don't need this.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    average_precision_score, f1_score, mean_absolute_error,
    mean_squared_error, r2_score, roc_auc_score,
)

# ── config (identical to the repo) ───────────────────────────────────────────
DATA_URL = ("https://raw.githubusercontent.com/William-franklyn/"
            "Jet-Engine-Predictive-Maintenance-ML/main/"
            "nasa_cmapss_FD001_scaled.csv")

SEED = 42
RUL_CAP = 125          # piecewise-linear RUL cap
W = 30                 # classification horizon: "fails within W cycles"
SEQ_LEN = 30           # cycles of history per LSTM window
VAL_FRACTION = 0.2     # share of TRAIN engines held out for LSTM early stopping

HIDDEN_SIZE, NUM_LAYERS, DROPOUT = 64, 2, 0.2
EPOCHS, BATCH_SIZE, LEARNING_RATE = 300, 256, 1e-3
PATIENCE, LR_FACTOR, LR_PATIENCE, MIN_LR = 25, 0.5, 8, 1e-5

ID_COL, CYCLE_COL = "unit_number", "time_in_cycles"
FEATURE_COLS = [
    "operational_setting_1",
    "sensor_2", "sensor_3", "sensor_4", "sensor_6", "sensor_7", "sensor_8",
    "sensor_9", "sensor_11", "sensor_12", "sensor_13", "sensor_14",
    "sensor_15", "sensor_17", "sensor_20", "sensor_21",
]

# What the README claims — printed back at the end for a direct check.
EXPECTED = {
    "LINEAR": (20.55, 15.89, 0.758),
    "RF":     (18.69, 13.82, 0.800),
    "LSTM":   (16.59, 12.20, 0.842),
}


def cmapss_score(y_true, y_pred):
    """Asymmetric PHM score (lower is better). Late predictions — engine
    predicted to last LONGER than it does — are penalized harder (/10) than
    early ones (/13), because late is the dangerous direction."""
    d = y_pred - y_true
    return np.sum(np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1))


# ── sequence windowing ───────────────────────────────────────────────────────
def make_windows(df, engine_ids, seq_len=SEQ_LEN, min_cycle=None):
    """Sliding windows: rows [t-seq_len+1 .. t] -> capped RUL at cycle t."""
    Xs, ys, us, cs = [], [], [], []
    for unit in engine_ids:
        eng = df[df[ID_COL] == unit].sort_values(CYCLE_COL)
        feats = eng[FEATURE_COLS].to_numpy(dtype=np.float32)
        rul = np.minimum(eng["RUL"].to_numpy(dtype=np.float32), RUL_CAP)
        cycles = eng[CYCLE_COL].to_numpy()
        for i in range(len(eng)):
            if min_cycle is not None and cycles[i] < min_cycle:
                continue
            start = i - seq_len + 1
            if start < 0:                       # left-pad short engines
                pad = np.repeat(feats[:1], -start, axis=0)
                window = np.concatenate([pad, feats[: i + 1]], axis=0)
            else:
                window = feats[start : i + 1]
            Xs.append(window); ys.append(rul[i]); us.append(unit); cs.append(cycles[i])
    return (np.stack(Xs).astype(np.float32),
            np.asarray(ys, dtype=np.float32),
            np.asarray(us), np.asarray(cs))


class LSTMRegressor(nn.Module):
    """2-layer LSTM -> dropout -> linear head. Only the final timestep's hidden
    state feeds the head: it has seen the whole window."""

    def __init__(self, n_features):
        super().__init__()
        self.lstm = nn.LSTM(n_features, HIDDEN_SIZE, NUM_LAYERS,
                            dropout=DROPOUT, batch_first=True)
        self.dropout = nn.Dropout(DROPOUT)
        self.head = nn.Linear(HIDDEN_SIZE, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(self.dropout(out[:, -1, :])).squeeze(-1)


def train_lstm(X_tr, y_tr, X_val, y_val):
    """MSE + Adam, LR decay on plateau, early stopping on validation RMSE.

    Restores BEST-validation weights, not final-epoch weights — this is why
    the 60-epoch and 300-epoch runs give identical test metrics.
    """
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    model = LSTMRegressor(X_tr.shape[2])
    opt = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, factor=LR_FACTOR, patience=LR_PATIENCE, min_lr=MIN_LR)
    loss_fn = nn.MSELoss()

    loader = DataLoader(
        TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr)),
        batch_size=BATCH_SIZE, shuffle=True)
    Xv, yv = torch.from_numpy(X_val), torch.from_numpy(y_val)

    best_rmse, best_state, stale, history = float("inf"), None, 0, []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total = 0.0
        for xb, yb in loader:
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            total += loss.item() * len(xb)

        model.eval()
        with torch.no_grad():
            val_rmse = torch.sqrt(
                loss_fn(torch.clamp(model(Xv), 0, RUL_CAP), yv)).item()

        sched.step(val_rmse)
        history.append({"epoch": epoch, "val_rmse": val_rmse,
                        "lr": opt.param_groups[0]["lr"]})

        if epoch % 10 == 0 or epoch == 1:
            print(f"  epoch {epoch:3d}  train MSE {total/len(y_tr):7.2f}"
                  f"   val RMSE {val_rmse:6.2f}"
                  f"   lr {opt.param_groups[0]['lr']:.1e}")

        if val_rmse < best_rmse - 1e-4:
            best_rmse, stale = val_rmse, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= PATIENCE:
                print(f"  early stop at epoch {epoch} "
                      f"(best val RMSE {best_rmse:.2f})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, pd.DataFrame(history)


def lstm_predict(model, X, batch_size=1024):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            out.append(model(torch.from_numpy(X[i:i + batch_size])).numpy())
    return np.clip(np.concatenate(out), 0, RUL_CAP)


# ── the comparison ───────────────────────────────────────────────────────────
def main():
    df = pd.read_csv(DATA_URL)
    groups = df[ID_COL].values

    # One engine-grouped split, shared by every model.
    tr_idx, te_idx = next(GroupShuffleSplit(
        n_splits=1, test_size=0.25, random_state=SEED).split(df, groups=groups))
    train_engines = np.unique(groups[tr_idx])
    test_engines = np.unique(groups[te_idx])

    # Carve a validation set out of TRAIN — used only for LSTM early stopping.
    shuffled = np.random.default_rng(SEED).permutation(train_engines)
    n_val = max(1, int(len(shuffled) * VAL_FRACTION))
    val_engines, fit_engines = np.sort(shuffled[:n_val]), np.sort(shuffled[n_val:])

    print(f"Data   : {len(df):,} rows, {df[ID_COL].nunique()} engines")
    print(f"Split  : {len(train_engines)} train ({len(fit_engines)} fit + "
          f"{len(val_engines)} val, LSTM only) / {len(test_engines)} test\n")

    # Flat training rows (all 75 train engines) for the sklearn models.
    sub = df[df[ID_COL].isin(train_engines)]
    X_tr = sub[FEATURE_COLS].to_numpy()
    y_reg_tr = np.minimum(sub["RUL"].to_numpy(), RUL_CAP)
    y_cls_tr = (sub["RUL"].to_numpy() <= W).astype(int)

    # Shared evaluation points: test cycles >= SEQ_LEN, aligned across models.
    Xw_te, y_reg_te, units, cycles = make_windows(df, test_engines, min_cycle=SEQ_LEN)
    rows = df.set_index([ID_COL, CYCLE_COL]).sort_index().loc[list(zip(units, cycles))]
    X_te = rows[FEATURE_COLS].to_numpy()
    y_cls_te = (rows["RUL"].to_numpy() <= W).astype(int)
    print(f"Scoring: {len(y_reg_te):,} shared points (test cycles >= {SEQ_LEN}), "
          f"{y_cls_te.mean():.1%} within {W} cycles of failure\n")

    preds = {}

    print("Training LINEAR regression...")
    preds["LINEAR"] = np.clip(
        LinearRegression().fit(X_tr, y_reg_tr).predict(X_te), 0, RUL_CAP)

    print("Training LOGISTIC regression...")
    log = LogisticRegression(class_weight="balanced", max_iter=1000)
    p_log = log.fit(X_tr, y_cls_tr).predict_proba(X_te)[:, 1]

    print("Training RANDOM FOREST...")
    rf = RandomForestRegressor(n_estimators=200, min_samples_leaf=10,
                               max_features="sqrt", random_state=SEED, n_jobs=-1)
    preds["RF"] = np.clip(rf.fit(X_tr, y_reg_tr).predict(X_te), 0, RUL_CAP)

    print(f"Training LSTM (window={SEQ_LEN} cycles)...")
    Xw_fit, y_fit, _, _ = make_windows(df, fit_engines)
    Xw_val, y_val, _, _ = make_windows(df, val_engines)
    print(f"  {len(y_fit):,} training windows / {len(y_val):,} validation windows")
    lstm, history = train_lstm(Xw_fit, y_fit, Xw_val, y_val)
    preds["LSTM"] = lstm_predict(lstm, Xw_te)

    # ── scoreboards ─────────────────────────────────────────────────────────
    order = ["LINEAR", "RF", "LSTM"]
    reg = pd.DataFrame([{
        "model": n,
        "RMSE": np.sqrt(mean_squared_error(y_reg_te, preds[n])),
        "MAE": mean_absolute_error(y_reg_te, preds[n]),
        "R2": r2_score(y_reg_te, preds[n]),
        "CMAPSS": cmapss_score(y_reg_te, preds[n]),
    } for n in order]).set_index("model")

    print("\n" + "=" * 68)
    print("REGRESSION — predicting RUL in cycles (lower RMSE/MAE/CMAPSS better)")
    print("=" * 68)
    print(reg.to_string(float_format=lambda v: f"{v:8.2f}"))

    cls_rows = [{"model": "LOGISTIC", "Accuracy": ((p_log >= 0.5).astype(int) == y_cls_te).mean(),
                 "F1": f1_score(y_cls_te, (p_log >= 0.5).astype(int)),
                 "ROC-AUC": roc_auc_score(y_cls_te, p_log),
                 "PR-AUC": average_precision_score(y_cls_te, p_log)}]
    for n in order:
        flag = (preds[n] <= W).astype(int)
        cls_rows.append({"model": f"{n}->flag",
                         "Accuracy": (flag == y_cls_te).mean(),
                         "F1": f1_score(y_cls_te, flag),
                         "ROC-AUC": roc_auc_score(y_cls_te, -preds[n]),
                         "PR-AUC": average_precision_score(y_cls_te, -preds[n])})
    cls = pd.DataFrame(cls_rows).set_index("model")

    print("\n" + "=" * 68)
    print(f"CLASSIFICATION — 'fails within {W} cycles?' (higher is better)")
    print("=" * 68)
    print(cls.to_string(float_format=lambda v: f"{v:9.3f}"))

    # ── did we reproduce the README? ────────────────────────────────────────
    i = history["val_rmse"].idxmin()
    print("\n" + "=" * 68)
    print("REPRODUCTION CHECK  (repo values in brackets)")
    print("=" * 68)
    for n in order:
        e_rmse, e_mae, e_r2 = EXPECTED[n]
        got = reg.loc[n]
        ok = "OK " if abs(got["RMSE"] - e_rmse) < 0.05 else "DIFF"
        print(f"{ok} {n:7s} RMSE {got['RMSE']:6.2f} [{e_rmse}]   "
              f"MAE {got['MAE']:6.2f} [{e_mae}]   R2 {got['R2']:.3f} [{e_r2}]")
    print(f"\nBest LSTM checkpoint: epoch {int(history.loc[i,'epoch'])} "
          f"(val RMSE {history.loc[i,'val_rmse']:.2f})   [repo: epoch 56, 15.71]")
    print(f"Stopped after {len(history)} epochs                        "
          f"[repo: 81]")
    print("\nSmall differences in the LSTM row are expected on a different "
          "torch version;\nthe sklearn rows should match exactly.")


# The guard matters even in a notebook: RandomForest uses n_jobs=-1, and
# joblib's workers import the parent module. Without it, an unguarded main()
# can be re-executed inside every worker.
if __name__ == "__main__":
    main()
