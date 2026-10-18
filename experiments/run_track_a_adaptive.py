"""Experiment B - Controlled adaptive trajectory (simulated feedback).

Runs two clearly-separated protocols over the same frozen artifacts:

1. "historical_track_a" - exact NB08 reproduction:
   stratified 80/20 of the FULL training set (rs=42), initial = 80%,
   4 shuffled feedback quarters (rng 42), balanced LR (liblinear,
   max_iter=1000, C=1.0), each version evaluated on the FROZEN TEST
   set at threshold 0.50.
   LIMITATION: the test set is reused for every version - historical
   reproduction only, NOT an unbiased final evaluation.

2. "improved_devval" - leakage-safe protocol:
   dev/validation split of the training partition (rs=42, same as
   NB06/06b). Within dev: stratified 80% initial + 20% feedback pool
   split into 4 shuffled quarters. Each version is evaluated on the
   VALIDATION set at 0.50. Only the final version is evaluated once on
   the frozen test set.

Feedback is RETROSPECTIVE/SIMULATED: labels are true training labels
treated as verified_label; agree/disagree and reward are not used.
This is verified-label supervised retraining, NOT reinforcement
learning.

Read-only against the thesis tree. Writes only to experiments/results/.

Run:  python experiments/run_track_a_adaptive.py
"""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split

RS = 42
THRESHOLD = 0.50
N_BATCHES = 4
THESIS = Path(r"D:\MSc\BRFSS_Diabetes_Thesis")
ML = THESIS / "data" / "processed" / "ml"
OUT = Path(__file__).resolve().parent / "results"
OUT.mkdir(parents=True, exist_ok=True)

FILES = {
    "X_train": ML / "X_train_selected.npz",
    "y_train": ML / "y_train.csv",
    "X_test": ML / "X_test_selected.npz",
    "y_test": ML / "y_test.csv",
    "nb08_results": THESIS / "results" / "ml" / "adaptive_learning_results.csv",
}

MODEL_PARAMS = dict(
    class_weight="balanced", solver="liblinear",
    max_iter=1000, random_state=RS, C=1.0,
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def evaluate(model, X, y):
    p = model.predict_proba(X)[:, 1]
    yhat = (p >= THRESHOLD).astype(int)
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


def run_trajectory(name, initial_idx, quarters, eval_sets, X, y, rows):
    """Cumulative retraining: V0 = initial, Vi adds quarter i."""
    train_idx = list(initial_idx)
    model = None
    for version in range(len(quarters) + 1):
        if version > 0:
            train_idx = train_idx + list(quarters[version - 1])
        X_fit, y_fit = X[train_idx], y.iloc[train_idx]
        model = LogisticRegression(**MODEL_PARAMS)
        model.fit(X_fit, y_fit)
        for eval_label, X_e, y_e, is_final_only in eval_sets:
            if is_final_only and version != len(quarters):
                continue
            m = evaluate(model, X_e, y_e)
            rows.append({
                "protocol": name,
                "version": f"V{version}",
                "training_rows": int(len(y_fit)),
                "n_positive": int(np.sum(y_fit)),
                "n_negative": int(len(y_fit) - np.sum(y_fit)),
                "model_params": str(MODEL_PARAMS),
                "threshold": THRESHOLD,
                "eval_dataset": eval_label,
                "eval_rows": int(len(y_e)),
                **m,
            })
            print(f"  {name} V{version} on {eval_label}: "
                  f"acc={m['accuracy']:.4f} rec={m['recall']:.4f} "
                  f"f1={m['f1']:.4f} roc={m['roc_auc']:.4f}")
    return model


def main():
    hashes = {k: sha256(v) for k, v in FILES.items()}

    X_train = sparse.load_npz(FILES["X_train"]).tocsr()
    y_train = pd.read_csv(FILES["y_train"]).iloc[:, 0].astype(int)
    X_test = sparse.load_npz(FILES["X_test"]).tocsr()
    y_test = pd.read_csv(FILES["y_test"]).iloc[:, 0].astype(int)
    assert X_train.shape == (339166, 49) and X_test.shape == (84792, 49)

    rows = []

    # ==================== Protocol 1: historical Track A ====================
    print("=== Protocol 1: historical_track_a (NB08 reproduction, "
          "test-set reuse limitation) ===")
    idx = np.arange(len(y_train))
    initial_idx, pool_idx = train_test_split(
        idx, test_size=0.20, stratify=y_train, random_state=RS)
    rng = np.random.default_rng(RS)
    quarters = np.array_split(
        rng.permutation(pool_idx), N_BATCHES)
    run_trajectory(
        "historical_track_a",
        initial_idx, quarters,
        [("frozen_test_84792_REUSED (historical)", X_test, y_test, False)],
        X_train, y_train, rows,
    )

    # ==================== Protocol 2: improved dev/val ====================
    print("=== Protocol 2: improved_devval (leakage-safe validation) ===")
    dev_idx, val_idx = train_test_split(
        idx, test_size=0.20, stratify=y_train, random_state=RS)
    dev_idx, val_idx = np.sort(dev_idx), np.sort(val_idx)
    # Within dev: stratified 80% initial + 20% feedback pool
    dev_pos = np.arange(len(dev_idx))
    init_pos, pool_pos = train_test_split(
        dev_pos, test_size=0.20,
        stratify=y_train.iloc[dev_idx], random_state=RS)
    initial_idx2 = dev_idx[init_pos]
    pool_idx2 = dev_idx[pool_pos]
    rng2 = np.random.default_rng(RS)
    quarters2 = np.array_split(
        rng2.permutation(pool_idx2), N_BATCHES)
    run_trajectory(
        "improved_devval",
        initial_idx2, quarters2,
        [("train_validation_67834", X_train[val_idx], y_train.iloc[val_idx], False),
         ("frozen_test_84792_FINAL_ONLY", X_test, y_test, True)],
        X_train, y_train, rows,
    )

    out = pd.DataFrame(rows)
    out.round(6).to_csv(OUT / "track_a_adaptive_trajectory.csv", index=False)

    # Compare historical Track A vs recorded NB08 results
    nb08 = pd.read_csv(FILES["nb08_results"])
    hist = out[out.protocol == "historical_track_a"].reset_index(drop=True)
    print("\nHistorical Track A vs recorded NB08 results:")
    diffs = []
    for i in range(len(nb08)):
        for k in ("accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"):
            diffs.append(float(hist.loc[i, k] - nb08.loc[i, k]))
    max_abs = max(abs(d) for d in diffs)
    print(f"max abs diff across V0-V4 metrics = {max_abs:.6f} "
          f"({'MATCH' if max_abs < 1e-3 else 'INVESTIGATE'})")

    provenance = {
        "experiment": "Experiment B - controlled adaptive trajectory",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "random_state": RS,
        "threshold": THRESHOLD,
        "batches": N_BATCHES,
        "model_params": MODEL_PARAMS,
        "feedback": "simulated retrospective; true training labels as "
                    "verified_label; rewards/agree-disagree unused",
        "protocols": {
            "historical_track_a": "NB08 exact reproduction; test set "
                                  "reused per version (limitation flagged)",
            "improved_devval": "dev/val holdout for trajectory; frozen "
                               "test evaluated once on final version",
        },
        "nb08_max_abs_diff": max_abs,
        "artifact_sha256": hashes,
        "outputs": [str(OUT / "track_a_adaptive_trajectory.csv")],
    }
    (OUT / "track_a_adaptive_provenance.json").write_text(
        json.dumps(provenance, indent=2), encoding="utf-8")
    print(f"\nSaved: {OUT / 'track_a_adaptive_trajectory.csv'}")


if __name__ == "__main__":
    main()
