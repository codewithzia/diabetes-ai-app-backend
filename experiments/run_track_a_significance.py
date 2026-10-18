"""Experiment B supplement - significance check on the improved
dev/val adaptive trajectory (V0 vs V4).

The improved protocol shows only a ~+0.001 F1 change V0->V4 on the
67,834-row validation holdout. This script quantifies whether that
delta is distinguishable from noise:

* paired bootstrap (2,000 resamples, seed 42) over validation rows for
  delta-F1, delta-recall and delta-accuracy 95% CIs;
* McNemar's test on discordant predictions (V0 vs V4).

Both models are refitted exactly as in run_track_a_adaptive.py
(improved_devval protocol): balanced LR, liblinear, max_iter=1000,
seed 42; V0 = stratified 80% of the dev partition, V4 = dev initial +
all four feedback quarters (i.e. the full 271,332-row dev partition).
Validation rows are never trained on.

Run:  python experiments/run_track_a_significance.py
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.model_selection import train_test_split

RS = 42
THRESHOLD = 0.50
B = 2000
THESIS = Path(r"D:\MSc\BRFSS_Diabetes_Thesis")
ML = THESIS / "data" / "processed" / "ml"
OUT = Path(__file__).resolve().parent / "results"
OUT.mkdir(parents=True, exist_ok=True)

MODEL_PARAMS = dict(class_weight="balanced", solver="liblinear",
                    max_iter=1000, random_state=RS, C=1.0)


def fit(X_fit, y_fit):
    m = LogisticRegression(**MODEL_PARAMS)
    m.fit(X_fit, y_fit)
    return m


def main():
    X_train = sparse.load_npz(ML / "X_train_selected.npz").tocsr()
    y_train = pd.read_csv(ML / "y_train.csv").iloc[:, 0].astype(int)

    # Reproduce the improved_devval splits exactly (same calls/seed).
    idx = np.arange(len(y_train))
    dev_idx, val_idx = train_test_split(
        idx, test_size=0.20, stratify=y_train, random_state=RS)
    dev_idx, val_idx = np.sort(dev_idx), np.sort(val_idx)

    dev_pos = np.arange(len(dev_idx))
    init_pos, pool_pos = train_test_split(
        dev_pos, test_size=0.20,
        stratify=y_train.iloc[dev_idx], random_state=RS)
    initial_idx = dev_idx[init_pos]
    pool_idx = dev_idx[pool_pos]
    rng = np.random.default_rng(RS)
    quarters = np.array_split(rng.permutation(pool_idx), 4)

    X_val, y_val = X_train[val_idx], y_train.iloc[val_idx].to_numpy()

    print("Fitting V0 (dev-initial) and V4 (dev-initial + 4 quarters)...")
    m0 = fit(X_train[initial_idx], y_train.iloc[initial_idx])
    v4_idx = np.concatenate([initial_idx] + list(quarters))
    m4 = fit(X_train[v4_idx], y_train.iloc[v4_idx])
    assert len(v4_idx) == len(dev_idx)

    p0 = (m0.predict_proba(X_val)[:, 1] >= THRESHOLD).astype(int)
    p4 = (m4.predict_proba(X_val)[:, 1] >= THRESHOLD).astype(int)

    f1_0, f1_4 = f1_score(y_val, p0), f1_score(y_val, p4)
    print(f"V0 val F1 = {f1_0:.6f}   V4 val F1 = {f1_4:.6f}   "
          f"delta = {f1_4 - f1_0:+.6f}")

    # Paired bootstrap
    rngb = np.random.default_rng(RS)
    n = len(y_val)
    d_f1, d_acc, d_rec = [], [], []
    for _ in range(B):
        b = rngb.integers(0, n, n)
        d_f1.append(f1_score(y_val[b], p4[b])
                    - f1_score(y_val[b], p0[b]))
        d_acc.append(accuracy_score(y_val[b], p4[b])
                     - accuracy_score(y_val[b], p0[b]))
        d_rec.append(recall_score(y_val[b], p4[b])
                     - recall_score(y_val[b], p0[b]))
    ci_f1 = np.percentile(d_f1, [2.5, 97.5])
    ci_acc = np.percentile(d_acc, [2.5, 97.5])
    ci_rec = np.percentile(d_rec, [2.5, 97.5])
    print(f"bootstrap 95% CI  dF1 = [{ci_f1[0]:+.5f}, {ci_f1[1]:+.5f}]  "
          f"dAcc = [{ci_acc[0]:+.5f}, {ci_acc[1]:+.5f}]  "
          f"dRec = [{ci_rec[0]:+.5f}, {ci_rec[1]:+.5f}]")

    # McNemar (discordant pairs)
    b_cnt = int(np.sum((p0 == y_val) & (p4 != y_val)))
    c_cnt = int(np.sum((p0 != y_val) & (p4 == y_val)))
    from math import exp
    chi2 = ((abs(b_cnt - c_cnt) - 1) ** 2) / max(b_cnt + c_cnt, 1)
    # normal approx p-value for chi2(df=1) via erfc on sqrt
    from math import erfc, sqrt
    pval = erfc(sqrt(chi2 / 2.0))
    print(f"McNemar: b={b_cnt} c={c_cnt} chi2_cc={chi2:.3f} p={pval:.4g}")

    summary = {
        "experiment": "V0 vs V4 significance check (improved_devval)",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "threshold": THRESHOLD,
        "model_params": MODEL_PARAMS,
        "v0_training_rows": int(len(initial_idx)),
        "v4_training_rows": int(len(v4_idx)),
        "eval": "stratified 20% validation holdout of training partition",
        "eval_rows": int(n),
        "v0_f1": float(f1_0), "v4_f1": float(f1_4),
        "delta_f1": float(f1_4 - f1_0),
        "bootstrap_B": B,
        "ci95_delta_f1": [float(ci_f1[0]), float(ci_f1[1])],
        "ci95_delta_accuracy": [float(ci_acc[0]), float(ci_acc[1])],
        "ci95_delta_recall": [float(ci_rec[0]), float(ci_rec[1])],
        "mcnemar": {"b": b_cnt, "c": c_cnt,
                    "chi2_cc": float(chi2), "p": float(pval)},
        "leakage_check": "validation rows disjoint from both models' "
                         "training data (dev-only fitting)",
        "interpretation": (
            "Delta is not statistically distinguishable from zero if "
            "CIs contain 0 / McNemar p > 0.05."),
    }
    (OUT / "track_a_significance_v0v4.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nSaved: {OUT / 'track_a_significance_v0v4.json'}")


if __name__ == "__main__":
    main()
