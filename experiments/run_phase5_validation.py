"""Phase 5.2 - Track 1: NB09-faithful human-feedback-guided adaptive
learning validation.

Reproduces the Notebook 09 cell-8/9/16/17 protocol EXACTLY:
- base model trained from scratch on the full 339,166-row training set
  (class_weight=None, as in the recorded NB09 run),
- V0 = first contiguous 80%; V1-V4 add contiguous quarters of the
  trailing 20% (NB09 cell 17; NOT notebook 08's stratified/shuffled batches),
- each version retrained from scratch via clone(base),
- every version evaluated on the complete frozen 84,792-row test set,
- at BOTH thresholds: 0.38 (deployed production) and 0.50 (NB09 comparison).

Read-only against the thesis tree. Writes ONLY to experiments/results/.
Does NOT touch the backend registry, database, or deployed model.

Run:  python experiments/run_phase5_validation.py
"""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

RS = 42
ML = Path(r"D:\MSc\BRFSS_Diabetes_Thesis\data\processed\ml")
OUT = Path(__file__).resolve().parent / "results"
OUT.mkdir(parents=True, exist_ok=True)

THRESHOLDS = (0.38, 0.50)  # 0.38 deployed; 0.50 for NB09 comparability

# Expected NB09 recorded results at threshold 0.50 (from notebook outputs)
NB09_RECORDED_050 = {
    "V0": {"training_size": 271332, "accuracy": 0.8695, "precision": 0.5626,
           "recall": 0.1793, "f1": 0.2719, "roc_auc": 0.8265, "pr_auc": 0.4229},
    "V4": {"training_size": 339166, "accuracy": 0.8696, "precision": 0.5632,
           "recall": 0.1802, "f1": 0.2731, "roc_auc": 0.8265, "pr_auc": 0.4229},
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


ARTIFACTS = [
    ML / "X_train_selected.npz", ML / "y_train.csv",
    ML / "X_test_selected.npz", ML / "y_test.csv",
    ML / "selected_feature_names.csv",
    ML / "brfss_preprocessor_final.joblib",
    Path(r"D:\MSc\BRFSS_Diabetes_Thesis\results\ml\rlhf_adaptive_logistic_regression_final.joblib"),
    Path(r"D:\MSc\BRFSS_Diabetes_Thesis\notebooks\08_adaptive_learning.ipynb"),
    Path(r"D:\MSc\BRFSS_Diabetes_Thesis\notebooks\09_rlhf_adaptive_learning.ipynb"),
]
hashes_before = {p: sha256(p) for p in ARTIFACTS}

# ---------------------------------------------------------------------------
# 1. Load + verify datasets
# ---------------------------------------------------------------------------
X_train = sparse.load_npz(ML / "X_train_selected.npz")
y_train = pd.read_csv(ML / "y_train.csv").iloc[:, 0]
X_test = sparse.load_npz(ML / "X_test_selected.npz")
y_test = pd.read_csv(ML / "y_test.csv").iloc[:, 0].astype(int)

print("=" * 80)
print("PHASE 5.2 - TRACK 1: NB09-FAITHFUL VALIDATION")
print("=" * 80)
print(f"X_train: {X_train.shape}  y_train: {y_train.shape}")
print(f"X_test : {X_test.shape}  y_test : {y_test.shape}")

assert X_train.shape == (339166, 49), f"unexpected X_train shape {X_train.shape}"
assert X_test.shape == (84792, 49), f"unexpected X_test shape {X_test.shape}"
assert len(y_test) == 84792
assert np.isfinite(X_train.data).all() and np.isfinite(X_test.data).all()
print("Dataset verification: 339,166 train / 84,792 test / 49 features  [OK]")


def evaluate(model, X, y, threshold):
    p = model.predict_proba(X)[:, 1]
    yhat = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yhat).ravel()
    return {
        "accuracy": accuracy_score(y, yhat),
        "precision": precision_score(y, yhat, zero_division=0),
        "recall": recall_score(y, yhat, zero_division=0),
        "f1": f1_score(y, yhat, zero_division=0),
        "roc_auc": roc_auc_score(y, p),
        "pr_auc": average_precision_score(y, p),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


# ---------------------------------------------------------------------------
# 2. NB09 base model (recorded NB09 run path: trained from scratch)
# ---------------------------------------------------------------------------
base = LogisticRegression(max_iter=2000, class_weight=None,
                          solver="liblinear", random_state=RS)
print("\nTraining NB09 base model on full 339,166-row training set...")
base.fit(X_train, y_train)
print("Base model trained:", base)

# ---------------------------------------------------------------------------
# 3. NB09 cell-17 version sequence (contiguous trailing-20% quarters)
# ---------------------------------------------------------------------------
n = len(y_train)
initial_idx = np.arange(int(n * 0.80))
remaining = np.arange(int(n * 0.80), n)
batches = list(np.array_split(remaining, 4))

# Guard: feedback quarters must not overlap the initial set, and the test
# set is never touched by training (separate artifact).
assert len(set(initial_idx) & set(np.concatenate(batches))) == 0

X_cur = X_train[initial_idx]
y_cur = y_train.iloc[initial_idx].copy()

results = []
versions = ["V0", "V1", "V2", "V3", "V4"]
for i, version in enumerate(versions):
    if i > 0:
        b = batches[i - 1]
        X_cur = sparse.vstack([X_cur, X_train[b]])
        y_cur = pd.concat([y_cur, y_train.iloc[b]], ignore_index=True)

    model = clone(base)
    model.fit(X_cur, y_cur)

    for thr in THRESHOLDS:
        results.append({
            "version": version, "threshold": thr,
            "training_size": int(len(y_cur)),
            "feedback_quarters_added": i,
            **evaluate(model, X_test, y_test, thr),
        })
    print(f"{version}: trained on {len(y_cur):,} rows; evaluated on "
          f"{len(y_test):,} test rows at thresholds {THRESHOLDS}")

trajectory = pd.DataFrame(results)

# ---------------------------------------------------------------------------
# 4. Output tables
# ---------------------------------------------------------------------------
for thr in THRESHOLDS:
    print("\n" + "=" * 80)
    print(f"V0 -> V4 RESULTS AT THRESHOLD {thr} "
          f"({'deployed production' if thr == 0.38 else 'NB09 comparison'})")
    print("=" * 80)
    print(trajectory[trajectory["threshold"] == thr].round(4).to_string(index=False))

    v0 = trajectory[(trajectory["threshold"] == thr)
                    & (trajectory["version"] == "V0")].iloc[0]
    v4 = trajectory[(trajectory["threshold"] == thr)
                    & (trajectory["version"] == "V4")].iloc[0]
    metric_cols = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]
    cmp_df = pd.DataFrame({
        "metric": metric_cols,
        "V0": [v0[m] for m in metric_cols],
        "V4": [v4[m] for m in metric_cols],
    })
    cmp_df["delta"] = cmp_df["V4"] - cmp_df["V0"]
    print(f"\nV0 vs V4 (threshold {thr}):")
    print(cmp_df.round(6).to_string(index=False))

# Compare with recorded NB09 outputs at 0.50
print("\n" + "=" * 80)
print("REPRODUCED VS RECORDED NB09 (threshold 0.50)")
print("=" * 80)
for version in ("V0", "V4"):
    row = trajectory[(trajectory["threshold"] == 0.50)
                     & (trajectory["version"] == version)].iloc[0]
    rec = NB09_RECORDED_050[version]
    print(f"{version}: training_size reproduced={row['training_size']} "
          f"recorded={rec['training_size']}")
    for m in ("accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"):
        diff = row[m] - rec[m]
        print(f"   {m:10s} reproduced={row[m]:.4f}  recorded={rec[m]:.4f}  "
              f"diff={diff:+.4f}")

# ---------------------------------------------------------------------------
# 5. Save results + provenance (experiments/results/ only)
# ---------------------------------------------------------------------------
trajectory_path = OUT / "phase5_version_trajectory.csv"
trajectory.round(6).to_csv(trajectory_path, index=False)

provenance = {
    "experiment": "Phase 5.2 Track 1 - human-feedback-guided adaptive "
                  "learning validation (NB09-faithful)",
    "created_at": datetime.now(timezone.utc).isoformat(),
    "random_state": RS,
    "base_model": {
        "estimator": "LogisticRegression",
        "max_iter": 2000, "class_weight": None,
        "solver": "liblinear", "random_state": RS,
        "trained_on": "full 339,166-row training set (matches recorded "
                      "NB09 run, which found no saved model)",
    },
    "version_sequence": "NB09 cell 17: V0 = first contiguous 80%; "
                        "V1-V4 add contiguous quarters of trailing 20%; "
                        "retrained from scratch via sklearn clone",
    "evaluation": {
        "dataset": "frozen thesis test set X_test_selected/y_test",
        "rows": int(len(y_test)),
        "thresholds": list(THRESHOLDS),
        "learnt_from_test_set": False,
    },
    "label_source": "true training labels as verified_label "
                    "(agree/disagree and reward never used as labels)",
    "outputs": [str(trajectory_path)],
    "artifact_sha256_before": {p.name: h for p, h in hashes_before.items()},
}
meta_path = OUT / "phase5_provenance.json"
meta_path.write_text(json.dumps(provenance, indent=2), encoding="utf-8")

# ---------------------------------------------------------------------------
# 6. Post-run artifact integrity check
# ---------------------------------------------------------------------------
changed = [p.name for p in ARTIFACTS
           if sha256(p) != hashes_before[p]]
print("\n" + "=" * 80)
print("THESIS ARTIFACT INTEGRITY")
print("=" * 80)
if changed:
    print("MODIFIED (unexpected!):", changed)
else:
    print(f"All {len(ARTIFACTS)} thesis artifacts byte-identical after run.")

print(f"\nSaved: {trajectory_path}")
print(f"Saved: {meta_path}")
print("\nPhase 5.2 Track 1 complete. Numbers reported raw; interpretation "
      "is deferred to the thesis analysis chapter.")
