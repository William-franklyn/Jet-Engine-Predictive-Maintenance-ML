# Reproduction Log

The model comparison in [README.md](../README.md) was re-run on **three independent
environments** — different operating systems, CPUs and library versions. This file records
what matched, what didn't, and why.

Reproduce it yourself: [`kaggle_reproduce.py`](../kaggle_reproduce.py), or open the
**[live Colab notebook](https://colab.research.google.com/drive/1qzrg2iQnuTor42PjmQoMWagofw3uq6xi?usp=sharing)**
and press *Runtime → Run all* (~12 min, CPU).

---

## Summary

| | Local<br>Windows 10 | Kaggle<br>Linux | Colab<br>Linux |
|---|---:|---:|---:|
| **Linear RMSE** | 20.55 | 20.55 | 20.55 |
| **Random Forest RMSE** | 18.69 | 18.69 | 18.69 |
| **LSTM RMSE** | 16.59 | 15.93 | 16.04 |
| LSTM MAE | 12.20 | 11.21 | 11.36 |
| LSTM R² | 0.842 | 0.854 | 0.853 |
| LSTM C-MAPSS | 25,629 | 31,167 | 29,400 |
| LSTM F1 | 0.906 | 0.904 | 0.906 |
| Best LSTM epoch | 56 | 62 | 60 |
| Total epochs run | 81 | 87 | 85 |

Every environment reported the same split — `75 train (60 fit + 15 val) / 25 test` and
`4,337 shared scoring points, 17.9% positive`.

---

## What reproduced exactly

**Linear Regression and Random Forest are bit-identical across all three environments.**
`20.55` and `18.69`, every run, to the decimal. scikit-learn is fully deterministic given
`random_state=42`, so this confirms the dataset, the engine-grouped split and the evaluation
pipeline are identical everywhere. Logistic Regression likewise: F1 `0.828`, ROC-AUC `0.989`,
PR-AUC `0.954` on all three.

## What did not reproduce

**The LSTM does not reproduce to the decimal across machines.** Values ranged 15.93–16.59.

The cause is floating-point operation ordering. torch's CPU kernels reduce sums in an order
that depends on thread count, vector-instruction width and library version, so tiny
differences appear early and compound. You can watch it begin in the logs below: at epoch 10
the three runs report train MSE `4449.10`, `4449.37` and `4448.89`. By epoch 40 the validation
curves have visibly separated, and each run settles on a different best checkpoint
(epoch 56 / 62 / 60).

This is a known property of neural network training, not a defect in this code. Seeding
(`torch.manual_seed(42)`) makes a run repeatable **on the same machine**; it cannot make
floating-point arithmetic identical **across** machines.

## Reported values

Because of the above, the LSTM is reported as a mean across environments rather than a single
figure:

> **RMSE 16.19 ± 0.35** · **MAE 11.59 ± 0.53** · **R² 0.850 ± 0.007** · **F1 0.905 ± 0.001**
> *(n = 3 environments, seed 42)*

The linear and tree models are reported as exact values, because they are.

## Does the conclusion survive?

**Yes, on every environment.** The ordering **LSTM < Random Forest < Linear** held in all three
runs, and the margin never came close to closing:

| | LSTM beats RF by |
|---|---:|
| Local | 2.10 cycles |
| Kaggle | 2.76 cycles |
| Colab | 2.65 cycles |

The *worst* LSTM run still beat the Random Forest by 2.10 cycles. F1 was the most stable metric
of all — 0.904 to 0.906, essentially machine-independent.

Note that **C-MAPSS is the most variable metric** (25,629–31,167, a ~10% spread) because its
exponential penalty amplifies small differences in the tail of the error distribution. The LSTM
beat the Random Forest's 43,450 on all three runs, but by between 28% and 41% depending on the
environment — so the README states "roughly a third better" rather than a fixed percentage.

---

## Raw output

### 1. Local — Windows 10, Python 3.13.7, torch 2.9.1+cpu

```
Data   : 20,631 rows, 100 engines
Split  : 75 train (60 fit + 15 val, LSTM only) / 25 test engines
Scoring: 4,337 shared prediction points (test cycles >= 30), 17.9% within 30 cycles of failure

  epoch   1  train MSE 8722.95   val RMSE  92.59   lr 1.0e-03
  epoch  10  train MSE 4449.10   val RMSE  67.32   lr 1.0e-03
  epoch  20  train MSE 1815.96   val RMSE  42.99   lr 1.0e-03
  epoch  30  train MSE  617.96   val RMSE  25.33   lr 1.0e-03
  epoch  40  train MSE  259.16   val RMSE  18.53   lr 1.0e-03
  epoch  50  train MSE  136.43   val RMSE  16.57   lr 1.0e-03
  epoch  60  train MSE   84.38   val RMSE  16.09   lr 1.0e-03
  epoch  70  train MSE   62.76   val RMSE  17.04   lr 5.0e-04
  epoch  80  train MSE   56.30   val RMSE  16.66   lr 2.5e-04
  early stop at epoch 81 (best val RMSE 15.71)

           RMSE      MAE       R2   CMAPSS
LINEAR    20.55    15.89    0.758 33565.41
RF        18.69    13.82    0.800 43449.73
LSTM      16.59    12.20    0.842 25629.32

              Accuracy        F1   ROC-AUC    PR-AUC
LOGISTIC         0.928     0.828     0.989     0.954
LINEAR->flag     0.923     0.734     0.985     0.938
RF->flag         0.953     0.857     0.988     0.953
LSTM->flag       0.967     0.906     0.992     0.973
```

### 2. Kaggle — Linux, Python 3.12

```
Data   : 20,631 rows, 100 engines
Split  : 75 train (60 fit + 15 val, LSTM only) / 25 test
Scoring: 4,337 shared points (test cycles >= 30), 17.9% within 30 cycles of failure

  epoch   1  train MSE 8722.95   val RMSE  92.59   lr 1.0e-03
  epoch  10  train MSE 4449.37   val RMSE  67.32   lr 1.0e-03
  epoch  20  train MSE 1807.12   val RMSE  42.83   lr 1.0e-03
  epoch  30  train MSE  612.33   val RMSE  25.34   lr 1.0e-03
  epoch  40  train MSE  261.30   val RMSE  18.22   lr 1.0e-03
  epoch  50  train MSE  138.52   val RMSE  18.09   lr 1.0e-03
  epoch  60  train MSE   85.91   val RMSE  15.72   lr 1.0e-03
  epoch  70  train MSE   66.99   val RMSE  16.23   lr 1.0e-03
  epoch  80  train MSE   57.48   val RMSE  17.92   lr 2.5e-04
  early stop at epoch 87 (best val RMSE 15.54)

           RMSE      MAE       R2   CMAPSS
LINEAR    20.55    15.89     0.76 33565.41
RF        18.69    13.82     0.80 43449.73
LSTM      15.93    11.21     0.85 31167.15

              Accuracy        F1   ROC-AUC    PR-AUC
LOGISTIC         0.928     0.828     0.989     0.954
LINEAR->flag     0.923     0.734     0.985     0.938
RF->flag         0.953     0.857     0.988     0.953
LSTM->flag       0.966     0.904     0.992     0.972

REPRODUCTION CHECK  (repo values in brackets)
OK  LINEAR  RMSE  20.55 [20.55]   MAE  15.89 [15.89]   R2 0.758 [0.758]
OK  RF      RMSE  18.69 [18.69]   MAE  13.82 [13.82]   R2 0.800 [0.8]
DIFF LSTM   RMSE  15.93 [16.59]   MAE  11.21 [12.2]    R2 0.854 [0.842]

Best LSTM checkpoint: epoch 62 (val RMSE 15.54)   [repo: epoch 56, 15.71]
Stopped after 87 epochs   [repo: 81]
```

### 3. Google Colab — Linux

```
Data   : 20,631 rows, 100 engines
Split  : 75 train (60 fit + 15 val, LSTM only) / 25 test
Scoring: 4,337 shared points (test cycles >= 30), 17.9% within 30 cycles of failure

  epoch   1  train MSE 8722.95   val RMSE  92.59   lr 1.0e-03
  epoch  10  train MSE 4448.89   val RMSE  67.32   lr 1.0e-03
  epoch  20  train MSE 1810.18   val RMSE  42.90   lr 1.0e-03
  epoch  30  train MSE  616.44   val RMSE  25.52   lr 1.0e-03
  epoch  40  train MSE  265.96   val RMSE  17.85   lr 1.0e-03
  epoch  50  train MSE  134.43   val RMSE  17.00   lr 1.0e-03
  epoch  60  train MSE   84.03   val RMSE  15.57   lr 1.0e-03
  epoch  70  train MSE   59.84   val RMSE  16.31   lr 5.0e-04
  epoch  80  train MSE   53.29   val RMSE  16.97   lr 2.5e-04
  early stop at epoch 85 (best val RMSE 15.57)

           RMSE      MAE       R2   CMAPSS
LINEAR    20.55    15.89     0.76 33565.41
RF        18.69    13.82     0.80 43449.73
LSTM      16.04    11.36     0.85 29400.36

              Accuracy        F1   ROC-AUC    PR-AUC
LOGISTIC         0.928     0.828     0.989     0.954
LINEAR->flag     0.923     0.734     0.985     0.938
RF->flag         0.953     0.857     0.988     0.953
LSTM->flag       0.967     0.906     0.992     0.972

REPRODUCTION CHECK  (repo values in brackets)
OK  LINEAR  RMSE  20.55 [20.55]   MAE  15.89 [15.89]   R2 0.758 [0.758]
OK  RF      RMSE  18.69 [18.69]   MAE  13.82 [13.82]   R2 0.800 [0.8]
DIFF LSTM   RMSE  16.04 [16.59]   MAE  11.36 [12.2]    R2 0.853 [0.842]

Best LSTM checkpoint: epoch 60 (val RMSE 15.57)   [repo: epoch 56, 15.71]
Stopped after 85 epochs   [repo: 81]
```

---

## Known issue: Windows segfault

On some Windows setups `kaggle_reproduce.py` and `compare_models.py` segfault partway through
LSTM training — at a *different* epoch each run, which is the signature of a torch/OpenMP
threading fault rather than a bug in the code. Capping threads fixes it, with identical results:

```bash
set OMP_NUM_THREADS=2
python kaggle_reproduce.py
```

Linux (Kaggle, Colab) is unaffected.
