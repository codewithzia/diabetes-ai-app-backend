"""Experiment A - Track A-aligned baseline verification.

Evaluates the deployed Track A reference artifact
(adaptive_logistic_regression_final.joblib, balanced LR) on:
  1. the frozen thesis test set (84,792) at threshold 0.50
     -> must equal NB08's recorded V4 row (adaptive_learning_results.csv);
  2. the stratified validation split of the training partition (67,834)
     -> the improved-protocol headline evaluation.

The legacy NB09 artifact (class_weight=None) is evaluated alongside at
the same records/threshold for the paper's side-by-side comparison.

Read-only against the thesis tree. Writes only to experiments/results/.

Run:  python experiments/run_track_a_baseline.py
"""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics import (
    accuracy_score, average_precision_score, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split

RS = 42
THRESHOLD = 0.50
THESIS = Path(r"D:\MSc\BRFSS_Diabetes_Thesis")
ML = THESIS / "data" / "processed" / "ml"
OUT = Path(__file__).resolve().parent / "results"
OUT.mkdir(parents=True, exist_ok=True)

ARTIFACTS = {
    "track_a_model": THESIS / "results" / "ml" / "adaptive_logistic_regression_final.joblib",
    "nb09_model": THESIS / "results" / "ml" / "rlhf_adaptive_logistic_regression_final.joblib",
    "preprocessor": ML / "brfss_preprocessor_final.joblib",
    "selected_features": ML / "selected_feature_names.csv",
    "X_train": ML / "X_train_selected.npz",
    "y_train": ML / "y_train.csv",
    "X_test": ML / "X_test_selected.npz",
    "y_test": ML / "y_test.csv",
    "nb08_results": THESIS / "results" / "ml" / "adaptive_learning_results.csv",
    "nb06_final_test": THESIS / "results" / "ml" / "final_test_results.csv",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


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


def main():
    hashes = {k: sha256(v) for k, v in ARTIFACTS.items()}

    track_a = joblib.load(ARTIFACTS["track_a_model"])
    nb09 = joblib.load(ARTIFACTS["nb09_model"])
    print("Track A params:", {k: track_a.get_params()[k] for k in
                             ("class_weight", "solver", "max_iter", "random_state")})
    assert track_a.get_params()["class_weight"] == "balanced"
    assert track_a.n_features_in_ == 49

    X_test = sparse.load_npz(ARTIFACTS["X_test"]).tocsr()
    y_test = pd.read_csv(ARTIFACTS["y_test"]).iloc[:, 0].astype(int)
    X_train = sparse.load_npz(ARTIFACTS["X_train"]).tocsr()
    y_train = pd.read_csv(ARTIFACTS["y_train"]).iloc[:, 0].astype(int)
    assert X_test.shape == (84792, 49) and X_train.shape == (339166, 49)

    # Stratified dev/validation split of the training partition (NB06-faithful).
    idx = np.arange(len(y_train))
    dev_idx, val_idx = train_test_split(
        idx, test_size=0.20, stratify=y_train, random_state=RS)
    X_val, y_val = X_train[np.sort(val_idx)], y_train.iloc[np.sort(val_idx)]

    rows = []
    for name, model, params_label in (
        ("track_a_reference", track_a, "balanced, liblinear, max_iter=1000"),
        ("nb09_legacy", nb09, "class_weight=None, liblinear, max_iter=2000"),
    ):
        for ds_label, X, y in (("frozen_test_84792", X_test, y_test),
                               ("train_validation_67834", X_val, y_val)):
            m = evaluate(model, X, y, THRESHOLD)
            rows.append({
                "model": name, "model_params": params_label,
                "eval_dataset": ds_label, "eval_rows": int(len(y)),
                "positive_rows": int(np.sum(y)),
                "threshold": THRESHOLD, **m})
            print(f"{name} on {ds_label}: acc={m['accuracy']:.4f} "
                  f"rec={m['recall']:.4f} f1={m['f1']:.4f} "
                  f"roc={m['roc_auc']:.4f} pr={m['pr_auc']:.4f}")

    out = pd.DataFrame(rows)
    out.round(6).to_csv(OUT / "track_a_baseline_metrics.csv", index=False)

    # Verify against recorded NB08 V4 row (historical test-set protocol)
    nb08 = pd.read_csv(ARTIFACTS["nb08_results"])
    rec = nb08.iloc[-1]
    mine = out[(out.model == "track_a_reference")
               & (out.eval_dataset == "frozen_test_84792")].iloc[0]
    print("\nRecorded NB08 V4 (@0.50) vs regenerated:")
    diffs = {}
    for k in ("accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"):
        d = float(mine[k] - rec[k])
        diffs[k] = d
        print(f"  {k:10s} recorded={rec[k]:.6f} regenerated={mine[k]:.6f} "
              f"diff={d:+.6f}")
    max_abs = max(abs(v) for v in diffs.values())
    print(f"max abs diff = {max_abs:.6f} "
          f"({'MATCH' if max_abs < 1e-4 else 'INVESTIGATE'})")

    provenance = {
        "experiment": "Experiment A - Track A-aligned baseline verification",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "random_state": RS,
        "threshold": THRESHOLD,
        "track_a_model_params": {k: track_a.get_params()[k] for k in
                                 ("class_weight", "solver", "max_iter",
                                  "random_state", "C")},
        "nb08_recorded_v4_diffs": diffs,
        "nb08_reproduced_within_1e-4": bool(max_abs < 1e-4),
        "artifact_sha256": {k: v for k, v in hashes.items()},
        "outputs": [str(OUT / "track_a_baseline_metrics.csv")],
    }
    (OUT / "track_a_baseline_provenance.json").write_text(
        json.dumps(provenance, indent=2), encoding="utf-8")
    print(f"\nSaved: {OUT / 'track_a_baseline_metrics.csv'}")
    print(f"Saved: {OUT / 'track_a_baseline_provenance.json'}")


if __name__ == "__main__":
    main()
