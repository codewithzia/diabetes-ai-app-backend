"""Final held-out reproducibility evaluation — frozen V4 thesis model.

Runs inference ONCE on the held-out test partition with the exact
artifacts and contract of the active V4 model:

    leakage_safe_preprocessor.joblib          (18 + 18 flags -> 85)
    final_selected_features.csv               (85 -> 50, ordered)
    leakage_safe_logistic_regression.joblib   (50 features, V4)
    X_test_raw.csv / y_test.csv               (84,792 rows, 1 = diabetes)

NOTE: this test set was previously INSPECTED for manual candidate
comparisons (never used for fitting, selection, threshold tuning or
acceptance), so this is a REPRODUCIBILITY evaluation of model
performance on the held-out test set, not an untouched-test claim.

Fails CLOSED: any contract deviation (artifact hash, row count, label
mapping, pipeline widths/order, threshold) aborts BEFORE inference.
Writes exactly one new JSON result file; never overwrites existing
artifacts, never trains, never touches a database or the registry.
"""

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score,
)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import (  # noqa: E402
    MODEL_PATH, PREPROCESSOR_PATH, PREDICTION_THRESHOLD,
    SELECTED_FEATURES_PATH, THESIS_TEST_X_PATH, THESIS_TEST_Y_PATH,
    THESIS_DIR,
)
from app.constants import FRONTEND_TO_BRFSS  # noqa: E402

OUT_PATH = (THESIS_DIR / "results" / "ml" / "results_leakage_safe"
            / "V4_held_out_test_evaluation.json")

# Pinned artifact hashes (recorded 2026-10-11, verified pre-release).
EXPECTED_MODEL_SHA256 = (
    "682c4ed5eb3e80b9d6934fb591ae4e70a1c3d40e8612abfdc5b17d195a9788a7"
)
EXPECTED_PREPROCESSOR_SHA256 = (
    "bc36b36cd7f63497f2ae3e9a2f95b671c92b0601fa62871b04a649bbf4d2b3c7"
)
EXPECTED_SELECTED_SHA256 = (
    "a91075cbb0b235d9a5e9b64db217b666b6b85f22a159731a910aef730db142b0"
)
EXPECTED_TEST_ROWS = 84792          # experiment_metadata.json
EXPECTED_THRESHOLD = 0.63
RAW_COLUMNS = list(FRONTEND_TO_BRFSS.values())


def fail(message: str):
    sys.exit(f"ABORT (fail-closed, no inference run): {message}")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# 1. Fail-closed contract checks (before inference)
# ---------------------------------------------------------------------------
for path in (MODEL_PATH, PREPROCESSOR_PATH, SELECTED_FEATURES_PATH,
             THESIS_TEST_X_PATH, THESIS_TEST_Y_PATH):
    if not path.exists():
        fail(f"missing input: {path.name}")
if OUT_PATH.exists():
    fail(f"result file already exists: {OUT_PATH.name} (never overwrite)")

model_sha = sha256(MODEL_PATH)
preproc_sha = sha256(PREPROCESSOR_PATH)
selected_sha = sha256(SELECTED_FEATURES_PATH)
if model_sha != EXPECTED_MODEL_SHA256:
    fail("active V4 model artifact hash mismatch")
if preproc_sha != EXPECTED_PREPROCESSOR_SHA256:
    fail("preprocessor artifact hash mismatch")
if selected_sha != EXPECTED_SELECTED_SHA256:
    fail("selected-features file hash mismatch")
if float(PREDICTION_THRESHOLD) != EXPECTED_THRESHOLD:
    fail(f"threshold {PREDICTION_THRESHOLD} != {EXPECTED_THRESHOLD}")

preprocessor = joblib.load(PREPROCESSOR_PATH)
model = joblib.load(MODEL_PATH)
if not isinstance(model, LogisticRegression):
    fail(f"unexpected model type {type(model).__name__}")
if model.n_features_in_ != 50:
    fail(f"model expects {model.n_features_in_} features, not 50")

expected_inputs = RAW_COLUMNS + [f"{c}__missing" for c in RAW_COLUMNS]
if list(preprocessor.feature_names_in_) != expected_inputs:
    fail("preprocessor input column contract mismatch")

selected = pd.read_csv(SELECTED_FEATURES_PATH)
selected_indices = selected["selected_feature_index"].astype(int).to_numpy()
selected_names = selected["selected_feature_name"].astype(str).tolist()
if len(selected_indices) != 50 or len(set(selected_names)) != 50:
    fail("selected-features file must contain 50 unique entries")

X_test_raw = pd.read_csv(THESIS_TEST_X_PATH)
y_test = pd.read_csv(THESIS_TEST_Y_PATH).iloc[:, 0].to_numpy(dtype=int)
if len(X_test_raw) != EXPECTED_TEST_ROWS or len(y_test) != EXPECTED_TEST_ROWS:
    fail(f"test partition size {len(X_test_raw)} != {EXPECTED_TEST_ROWS}")
if set(np.unique(y_test).tolist()) - {0, 1}:
    fail("test labels must be binary {0, 1}")
for col in RAW_COLUMNS:
    if col not in X_test_raw.columns:
        fail(f"test partition missing raw column {col}")

# ---------------------------------------------------------------------------
# 2. Transform (transform-only; nothing is fitted here)
# ---------------------------------------------------------------------------
frame = X_test_raw[RAW_COLUMNS].copy()
for col in RAW_COLUMNS:
    frame[f"{col}__missing"] = frame[col].isna().astype(np.int8)
frame = frame[expected_inputs]

processed = preprocessor.transform(frame)
if processed.shape != (EXPECTED_TEST_ROWS, 85):
    fail(f"preprocessed width {processed.shape[1]} != 85")
X_test = processed[:, selected_indices]
if float(np.max(selected_indices)) >= 85:
    fail("selected index out of bounds")
if not np.isfinite(np.asarray(X_test.data if hasattr(X_test, "data")
                             else X_test)).all():
    fail("non-finite values after preprocessing")

# ---------------------------------------------------------------------------
# 3. Inference (single pass, fixed threshold 0.63)
# ---------------------------------------------------------------------------
probabilities = model.predict_proba(X_test)[:, 1]
predictions = (probabilities >= EXPECTED_THRESHOLD).astype(int)

tn, fp, fn, tp = confusion_matrix(y_test, predictions, labels=[0, 1]).ravel()
result = {
    "evaluation_type": (
        "reproducibility evaluation on the held-out test set; the set "
        "was previously inspected for manual candidate comparisons but "
        "never used for fitting, feature selection, threshold tuning, "
        "model selection, or candidate acceptance"
    ),
    "model_version": "V4",
    "model_artifact": MODEL_PATH.name,
    "model_sha256": model_sha,
    "threshold": EXPECTED_THRESHOLD,
    "threshold_origin": "leakage-safe validation split (max F1)",
    "samples": int(len(y_test)),
    "positive_class_prevalence": float(np.mean(y_test)),
    "positive_count": int(y_test.sum()),
    "negative_count": int((1 - y_test).sum()),
    "metrics": {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "pr_auc": float(average_precision_score(y_test, probabilities)),
    },
    "confusion_matrix": {"tn": int(tn), "fp": int(fp),
                         "fn": int(fn), "tp": int(tp)},
    "prediction_counts": {"predicted_positive": int(predictions.sum()),
                          "predicted_negative": int((1 - predictions).sum())},
    "dependencies": {"scikit-learn": sklearn.__version__,
                     "numpy": np.__version__,
                     "pandas": pd.__version__,
                     "joblib": joblib.__version__},
    "inputs": {
        "preprocessor_sha256": preproc_sha,
        "selected_features_sha256": selected_sha,
        "test_x": THESIS_TEST_X_PATH.name,
        "test_y": THESIS_TEST_Y_PATH.name,
    },
    "evaluated_at": datetime.now(timezone.utc).isoformat(),
}

OUT_PATH.write_text(json.dumps(result, indent=2), encoding="utf-8")
print(json.dumps(result, indent=2))
print(f"\nSaved: {OUT_PATH}")
