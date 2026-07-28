# AI4ALL-Group-04C

## Project

"Predicting Jet Engine Failure Through Machine Learning" — an AI4ALL course group project.
Team: Erronn Bridgewater, Tom Chatto, Gabe Meredith, Nish Methuku, Alejandro Hernandez, William Mahunda, Hunter Ngo.

Goal: predict a turbofan engine's Remaining Useful Life (RUL) — how many flight cycles remain
before failure — from sensor telemetry, to support predictive maintenance instead of fixed
replacement schedules.

Research question: can ML models predict early engine failure to support predictive
maintenance and reduce unexpected downtime?

Full background (problem motivation, bias/mitigation notes, citations) lives in the team's
project deck, not duplicated here.

## Datasets

`nasa_cmapss_FD001_scaled.csv` (committed) is the **single source of truth** — everything
(baseline, trainer, and all three dashboard modes) reads it. 20,631 rows, 100 engines.
Columns: `unit_number`, `time_in_cycles`, `operational_setting_1`,
`sensor_2/3/4/6/7/8/9/11/12/13/14/15/17/20/21`, `RUL`. That gives the model 16 features
(`operational_setting_1` + those 15 sensors); they're listed as `FEATURE_COLS` in `model.py`.

Note `RUL` in the file is **raw/uncapped** (0–361). Both the trainer and the dashboard apply
the piecewise-linear cap at 125 cycles themselves, so don't assume the on-disk RUL is capped.

Historical note: a second, differently-preprocessed `train_FD001_scaled.csv` (schema `unit`,
`cycle`, `op_setting_1`, `op_setting_2`, 14 sensors — no `sensor_6`) circulated on the team
earlier and the code used to target it. We deliberately consolidated onto the committed file
so the project trains and runs with no external file dependency; the schemas are **not**
interchangeable. If the team later standardizes on that other file, realign `FEATURE_COLS`
and the loaders in `save_model.py`/`app.py`.

All sensor/operational-setting columns are already z-score scaled — values are not in raw
physical units. Keep that in mind in any UI copy ("scaled sensor reading", not "°F"/"psi").

## Models

Two models were planned per the team deck; an LSTM was added later at the team's request:
- **Logistic Regression** — interpretable baseline; classifies "will this engine fail within
  W cycles?"
- **Random Forest** — captures nonlinear degradation and sensor interactions

`logistic_regression_base.py` is the existing baseline script: it frames the same problem two
ways (logistic classification vs. linear regression on capped RUL) on an identical
engine-grouped train/test split, and reports both native metrics and a shared classification
scoreboard, plus a custom asymmetric C-MAPSS scoring function.

`save_model.py` trains the real `RandomForestRegressor` for the dashboard on
`nasa_cmapss_FD001_scaled.csv`, and saves it to `random_forest_model.pkl` (not committed —
generate it locally with `python save_model.py`). It caps RUL at 125, evaluates on an
engine-grouped hold-out (RMSE ≈ 18.0, MAE ≈ 13.3, R² ≈ 0.81), prints feature importances,
then refits on all 100 engines before saving. The RF is tuned and kept deliberately lean
(`n_estimators=200`, `min_samples_leaf=10`, `max_features="sqrt"`) — same accuracy as a
heavier model at ~1/4 the size (~28MB pkl) and ~9s to train. On this data/feature set tree
models top out around RMSE 18; rolling-window features and gradient boosting were tried and
didn't beat it.

`save_model.py` also exposes `train_full_model()`, which `app.py` calls to build the model
in-process when no `.pkl` is present. This is what keeps the **deployed** app working: the
pickle is gitignored (and >100MB uncompressed would exceed GitHub's limit anyway), so on a
fresh Streamlit Cloud deploy the app trains once from the committed CSV (cached via
`@st.cache_resource`) instead of shipping a binary. Run `python save_model.py` locally only
when you want the faster pre-baked pickle.

`lstm_model.py` adds a **PyTorch LSTM** regressor — the sequence model the team asked about.
Where the other three models see one cycle at a time (a single row of 16 scaled readings), the
LSTM reads a 30-cycle sliding window per engine, so it can use the degradation *trend* rather
than a snapshot. 2-layer LSTM, hidden 64, dropout 0.2, MSE + Adam, early stopping on a
validation set carved from the training engines.

`compare_models.py` is the head-to-head benchmark (Linear / Logistic / Random Forest / LSTM).
It enforces the fairness rules that make the numbers quotable: one engine-grouped split
(same seed 42 as the other scripts), one shared set of evaluation points (test cycles >= 30,
since the LSTM can't score cycles 1-29 and shouldn't get to skip the hard early ones), and one
shared classification scoreboard so the classifier can be compared against the regressors.
Held-out results: LSTM RMSE 16.59 / MAE 12.20 / R² 0.842, beating RF (18.69) and Linear
(20.55); LSTM F1 0.906 vs Logistic 0.828. Full run takes ~12 min on CPU.

**The LSTM is converged — do not describe it as under-trained.** An early draft did, because
the 60-epoch run ended at val RMSE 16.09 and looked like it was still descending. A 300-epoch
rerun with `ReduceLROnPlateau` (documented as Round 5) early-stopped at epoch 81 and produced
*identical* test metrics: the best checkpoint is **epoch 56**, already inside the original run,
and `train_lstm()` restores best-validation weights rather than final-epoch weights. The
apparent downward slope at epoch 60 was noise around a minimum already reached. Further gains
need a different architecture/window length, not more epochs.

`compare_models.py` writes its measured results to `docs/model_comparison_regression.csv`,
`docs/model_comparison_classification.csv` and `docs/lstm_training_curve.csv`.
`scripts/generate_model_comparison.py` reads **only** those CSVs to build the README's
model-selection figures — no numbers are hand-typed, so the charts can't drift from the run.
If you retrain, regenerate the figures rather than editing the README tables by hand.

The README frames the work as **four rounds of model selection** (1: linear/logistic,
2: tuned RF, 3: rolling-window features + gradient boosting — a kept negative result,
4: LSTM). That narrative is real, not retrofitted; keep it accurate if models change.

**torch is now IN `requirements.txt`** (this reversed an earlier decision) because the deployed
dashboard serves LSTM predictions. The first line pins the CPU-only wheel via
`--extra-index-url https://download.pytorch.org/whl/cpu` — do **not** remove it; default PyPI
torch pulls ~2GB of CUDA the app never uses and will break the free-tier deploy.
`requirements-lstm.txt` is now just a back-compat shim that includes `requirements.txt`.

**Two models ship, routed per prediction.** The LSTM can't score a single row — it needs
`LSTM_SEQ_LEN` (30) cycles of history. So `app.py` calls `predict_rul_sequence()` when history
exists (Browse mode; Upload CSV *with* `unit`+`cycle` columns) and falls back to
`predict_rul()` (Random Forest) otherwise (Manual input; CSVs without those columns). The UI
always names the model that answered — don't "simplify" that away, it's the honest disclosure
that the headline accuracy doesn't apply to every prediction.

`save_lstm_model.py` trains the **deployed** LSTM on all 100 engines (85 fit + 15 for early
stopping) and writes `lstm_rul_model.pt` (~220KB, committed — unlike the 28MB RF pickle it's
cheap to ship and costly to retrain). This is distinct from `compare_models.py`'s checkpoint,
which trains on 60 engines so 25 stay unseen for honest evaluation. Never quote the deployment
model's validation RMSE as a headline metric.

Note the README quotes RF at both **18.0** RMSE (save_model.py, all test cycles) and **18.69**
(compare_models.py, test cycles >= 30 only, where the LSTM can also compete). Same model and
split, different evaluation subset — both are stated explicitly so the numbers don't look
inconsistent.

**Reproducibility, verified — read `docs/reproduction-log.md` before touching any metric.**
The comparison was re-run on Kaggle and Colab. Linear/Logistic/RF come back **bit-identical**
everywhere (sklearn is deterministic given a seed) — quote those as exact. The **LSTM is not
bit-reproducible across machines**: RMSE ranged 15.93–16.59 because torch's CPU reductions
depend on thread count and instruction width, so each environment picks a different best epoch
(56/62/60). Hence the README reports the LSTM as **16.19 ± 0.35 (n=3 environments)**, not a
single figure. Two earlier claims were corrected after this: the "~41% safer C-MAPSS" figure
(actually 28–41% depending on machine) and treating "epoch 56" as a property of the model
rather than of one laptop. Don't reintroduce either.

**Data audit — `scripts/audit_data.py`.** Run it before trusting any model result; it exits
non-zero on integrity failure. Current verdict: the dataset is genuinely clean (0 missing, 0
duplicates, no dead columns, no cycle gaps, and `RUL == max_cycle - cycle` for all 100 engines).
Two things it surfaced that must not be "tidied away":

- **The scaler was fit on all 100 engines**, not just the 75 training ones — the global mean is
  exactly 0 while the train-subset mean is -0.00023. That's mild preprocessing leakage. It
  leaks no labels and hits all four models equally, so the comparison is unaffected. Fixing it
  needs raw FD001 (we only hold the scaled CSV) and would invalidate every published number, so
  it is **disclosed, not fixed**. Don't silently "correct" it.
- **Class balance is 15.0% of all rows** but **17.9%** in the Results section. Both are right:
  compare_models.py scores only test cycles >= 30, and dropping early-life rows removes
  negatives only (positives are fixed at 31 rows/engine). The audit prints this reconciliation
  so it never reads as a contradiction.

Outliers (0.148% beyond 6 sd) are deliberately **kept** — they are near-failure engines, i.e.
the signal itself.

`kaggle_reproduce.py` is a single self-contained file (no repo imports) for teammates to verify
the numbers in Kaggle/Colab; it prints an OK/DIFF check against the published values. Keep its
inlined constants in sync if the real modules change.

The dashboard still serves Random Forest. `model.py`'s `predict_rul()` loads
`random_forest_model.pkl` when present; when it isn't, it falls back to a placeholder
heuristic (clearly marked, not a trained model) so the dashboard still shows something. That
function is the only place a real/updated model needs to be wired in.

## Dashboard

`app.py` — a Streamlit app with three sidebar-selectable modes, all backed by the same
`predict_rul()`:
- **Browse engine** — pick an engine from `nasa_cmapss_FD001_scaled.csv`, scrub cycles, see
  sensor charts and predicted RUL. (On load the app renames `unit_number`/`time_in_cycles` to
  `unit`/`cycle` internally.)
- **Upload CSV** — upload sensor readings, get predicted RUL per row, download results.
- **Manual input** — hand-enter one engine's sensor readings for a single prediction.

This is a merge of two dashboards that existed briefly in parallel (one engine-browser-first,
one upload/manual-input-first) — there is only one dashboard entry point now, `app.py`.

Run it with:
```
pip install -r requirements.txt
streamlit run app.py
```

## Conventions

- Do not add a `Co-Authored-By: Claude` trailer to commits in this repo.
- Keep commits scoped and descriptive — this is a team repo other members read the history of.
