"""Human-feedback-guided adaptive learning service.

Implements incremental candidate retraining for the thesis diabetes
model following the methodology of notebooks 08/09
("human-feedback-guided adaptive learning"):

  clone(base Logistic Regression) -> fit on cumulative data
  -> evaluate with the thesis metric set on the frozen thesis test set

Hard methodological rules enforced here
---------------------------------------
* ``feedback`` (agree/disagree) and ``reward`` are reward signals ONLY.
  They are never used as training labels. Supervised labels come
  exclusively from ``feedback.verified_label`` (trusted/verified source).
* The thesis baseline artefacts (preprocessor, 49 selected features,
  training/test splits, V0-V4 models, threshold 0.38) are read-only and
  never modified. Candidate artefacts live under
  ``config.ADAPTIVE_MODELS_DIR`` inside the backend.
* Feedback rows are marked ``adaptive_processed = 1`` only AFTER a
  candidate model has been successfully trained, evaluated and
  persisted. Failed runs leave them unprocessed.
* ``activate_model`` never runs in TEST MODE and requires an explicit
  ``approved_by``. No automatic acceptance policy is invented: the
  comparison result is exposed and a human approves activation.

This is NOT deep RLHF; it is human-feedback-guided incremental
retraining of the existing Logistic Regression approach.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from ..config import (
    ADAPTIVE_MODELS_DIR,
    ADAPTIVE_REGISTRY_PATH,
    APP_MODE,
    MODEL_PATH,
    THESIS_TEST_X_PATH,
    THESIS_TEST_Y_PATH,
)
from ..constants import MODEL_INFO, OPTIMIZED_THRESHOLD
from ..database import (
    get_predictions_by_ids,
    get_verified_feedback,
    mark_feedback_processed,
)
from .model_runtime import (
    MODEL,
    PREPROCESSOR,
    SELECTED_INDICES,
    build_model_input,
)

# Evaluation defaults: the frozen thesis test set (read-only). A sample
# cap keeps candidate evaluation fast during development/testing.
DEFAULT_EVAL_SAMPLE_SIZE = 5000
RANDOM_STATE = 42  # matches the thesis notebooks


# ---------------------------------------------------------------------------
# 1. Eligible feedback
# ---------------------------------------------------------------------------

def get_eligible_feedback() -> list[dict]:
    """Verified, unprocessed feedback rows usable as supervised labels.

    agree/disagree and reward are NOT used here except mean_reward,
    which is recorded as descriptive metadata.
    """
    return get_verified_feedback(only_unprocessed=True)


# ---------------------------------------------------------------------------
# 2. Candidate dataset
# ---------------------------------------------------------------------------

def build_candidate_dataset(feedback_rows: list[dict]) -> dict:
    """Join verified feedback with its predictions and rebuild features.

    Recovers the original frontend features from ``predictions.features``
    and applies the SAME preprocessing + selected-feature transformation
    used by the production pipeline (35 -> 84 -> 49), via model_runtime.
    """
    prediction_ids = [row["prediction_id"] for row in feedback_rows]
    predictions = get_predictions_by_ids(prediction_ids)

    frames, labels, used_feedback_ids, skipped = [], [], [], []
    for row in feedback_rows:
        prediction = predictions.get(row["prediction_id"])
        if prediction is None:
            skipped.append({
                "feedback_id": row["feedback_id"],
                "reason": "prediction not found",
            })
            continue

        raw_features = json.loads(prediction["features"])
        frames.append(build_model_input(raw_features))
        labels.append(int(row["verified_label"]))
        used_feedback_ids.append(row["feedback_id"])

    if not frames:
        return {
            "X": None, "y": None,
            "feedback_ids": [], "skipped": skipped,
            "feedback_sample_count": 0, "verified_feedback_count": 0,
            "mean_reward": None,
        }

    model_input = pd.concat(frames, ignore_index=True)
    processed = PREPROCESSOR.transform(model_input)      # same as prod
    X = sparse.csr_matrix(processed[:, SELECTED_INDICES])  # same 49 feats
    y = np.asarray(labels, dtype=int)

    rewards = [
        row["reward"] for row in feedback_rows
        if row["feedback_id"] in set(used_feedback_ids)
        and row["reward"] is not None
    ]

    return {
        "X": X,
        "y": y,
        "feedback_ids": used_feedback_ids,
        "skipped": skipped,
        "feedback_sample_count": len(feedback_rows),
        "verified_feedback_count": len(used_feedback_ids),
        # Reward is descriptive metadata only, never a training label.
        "mean_reward": float(np.mean(rewards)) if rewards else None,
    }


# ---------------------------------------------------------------------------
# 3. Candidate training (thesis Logistic Regression, cloned from baseline)
# ---------------------------------------------------------------------------

def train_candidate_model(
    X_new,
    y_new,
    baseline_sample_size: int | None = None,
):
    """Train a candidate by cloning the thesis Logistic Regression.

    Hyperparameters are inherited from the loaded thesis model via
    ``sklearn.base.clone`` (max_iter=2000, solver='liblinear',
    random_state=42). If ``baseline_sample_size`` is set, a deterministic
    sample of the frozen thesis TRAINING split (read-only) is prepended,
    mirroring the cumulative-data approach of notebook 09.
    """
    if X_new is None or len(y_new) == 0:
        raise ValueError("No verified feedback data to train on.")
    if len(np.unique(y_new)) < 2:
        raise ValueError(
            "Candidate training requires both classes in verified_label."
        )

    X_train, y_train = X_new, y_new

    if baseline_sample_size:
        train_X_path = THESIS_TEST_X_PATH.with_name("X_train_selected.npz")
        train_y_path = THESIS_TEST_Y_PATH.with_name("y_train.csv")
        X_base = sparse.load_npz(train_X_path)
        y_base = pd.read_csv(train_y_path).iloc[:, 0].to_numpy(dtype=int)
        rng = np.random.default_rng(RANDOM_STATE)
        idx = rng.choice(
            X_base.shape[0],
            size=min(baseline_sample_size, X_base.shape[0]),
            replace=False,
        )
        X_train = sparse.vstack([X_base[idx], X_train]).tocsr()
        y_train = np.concatenate([y_base[idx], y_train])

    candidate = clone(MODEL)
    candidate.fit(X_train, y_train)

    return {
        "model": candidate,
        "training_sample_count": int(len(y_train)),
    }


# ---------------------------------------------------------------------------
# 4. Evaluation (identical metric set to notebook 09 evaluate_model)
# ---------------------------------------------------------------------------

def evaluate_candidate_model(model, X, y, threshold: float = OPTIMIZED_THRESHOLD):
    y_probability = model.predict_proba(X)[:, 1]
    y_prediction = (y_probability >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y, y_prediction).ravel()

    return {
        "accuracy": float(accuracy_score(y, y_prediction)),
        "precision": float(precision_score(y, y_prediction, zero_division=0)),
        "recall": float(recall_score(y, y_prediction, zero_division=0)),
        "f1": float(f1_score(y, y_prediction, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, y_probability)),
        "pr_auc": float(average_precision_score(y, y_probability)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def load_thesis_test_set(sample_size: int | None = DEFAULT_EVAL_SAMPLE_SIZE):
    """Frozen thesis test split (READ-ONLY) for candidate evaluation."""
    X = sparse.load_npz(THESIS_TEST_X_PATH)
    y = pd.read_csv(THESIS_TEST_Y_PATH).iloc[:, 0].to_numpy(dtype=int)
    if sample_size and sample_size < X.shape[0]:
        rng = np.random.default_rng(RANDOM_STATE)
        idx = np.sort(rng.choice(X.shape[0], size=sample_size, replace=False))
        X, y = X[idx], y[idx]
    return X, y


# ---------------------------------------------------------------------------
# 5. Model comparison (exposed; no automatic acceptance policy)
# ---------------------------------------------------------------------------

def compare_models(candidate_model, X, y, threshold: float = OPTIMIZED_THRESHOLD):
    """Side-by-side evaluation of candidate vs active (thesis) model."""
    candidate_metrics = evaluate_candidate_model(candidate_model, X, y, threshold)
    active_metrics = evaluate_candidate_model(MODEL, X, y, threshold)

    metric_keys = ("accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc")
    return {
        "active_version": MODEL_INFO["current_version"],
        "candidate_metrics": candidate_metrics,
        "active_metrics": active_metrics,
        "delta": {
            key: round(candidate_metrics[key] - active_metrics[key], 6)
            for key in metric_keys
        },
        "threshold": threshold,
        "evaluation_samples": int(len(y)),
        # No acceptance decision is made here; activation is a human call.
        "acceptance_criteria": "not defined - manual approval required",
    }


# ---------------------------------------------------------------------------
# 6. Versioned candidate persistence + registry
# ---------------------------------------------------------------------------

def _load_registry() -> dict:
    if ADAPTIVE_REGISTRY_PATH.exists():
        return json.loads(ADAPTIVE_REGISTRY_PATH.read_text(encoding="utf-8"))
    return {
        "active_version": MODEL_INFO["current_version"],
        "active_model_path": str(MODEL_PATH),
        "versions": {},
        "note": "V0-V4 are frozen thesis artefacts; adaptive candidates "
                "are VA-#### versions stored under backend/models/adaptive.",
    }


def _save_registry(registry: dict) -> None:
    ADAPTIVE_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    ADAPTIVE_REGISTRY_PATH.write_text(
        json.dumps(registry, indent=2), encoding="utf-8"
    )


def _next_version(registry: dict) -> str:
    numbers = [
        int(v.removeprefix("VA-"))
        for v in list(registry["versions"]) + _existing_version_dirs()
        if v.startswith("VA-") and v.removeprefix("VA-").isdigit()
    ]
    return f"VA-{max(numbers, default=0) + 1:04d}"


def _existing_version_dirs() -> list[str]:
    if not ADAPTIVE_MODELS_DIR.exists():
        return []
    return [p.name for p in ADAPTIVE_MODELS_DIR.iterdir() if p.is_dir()]


def create_model_version(
    candidate_model,
    comparison: dict,
    dataset: dict,
    threshold: float = OPTIMIZED_THRESHOLD,
    training_sample_count: int = 0,
) -> dict:
    """Persist a candidate artefact + metadata and gate feedback.

    Only after everything is written are the source feedback rows marked
    ``adaptive_processed = 1``.
    """
    registry = _load_registry()
    version = _next_version(registry)

    version_dir = ADAPTIVE_MODELS_DIR / version
    version_dir.mkdir(parents=True, exist_ok=False)
    joblib.dump(candidate_model, version_dir / "model.joblib")

    metrics = comparison["candidate_metrics"]
    metadata = {
        "version": version,
        "parent_version": comparison["active_version"],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "training_sample_count": training_sample_count,
        "feedback_sample_count": dataset["feedback_sample_count"],
        "verified_feedback_count": dataset["verified_feedback_count"],
        "feedback_ids": dataset["feedback_ids"],
        "skipped_feedback": dataset["skipped"],
        "accuracy": metrics["accuracy"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "roc_auc": metrics["roc_auc"],
        "pr_auc": metrics["pr_auc"],
        "confusion_matrix": {
            "tn": metrics["tn"], "fp": metrics["fp"],
            "fn": metrics["fn"], "tp": metrics["tp"],
        },
        "threshold": threshold,
        "mean_reward": dataset["mean_reward"],
        "evaluation": {
            "dataset": "thesis frozen test split (read-only)",
            "evaluation_samples": comparison["evaluation_samples"],
            "comparison": comparison,
        },
        "label_source": "feedback.verified_label only; agree/disagree "
                        "and reward were never used as training labels",
        "base_model": "LogisticRegression cloned from thesis model "
                      "(liblinear, max_iter=2000, random_state=42)",
        "status": "candidate",
    }
    (version_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    registry["versions"][version] = {
        "path": str(version_dir),
        "status": "candidate",
        "created_at": metadata["created_at"],
    }
    _save_registry(registry)

    # Feedback is marked processed ONLY now, after the artefacts exist.
    processed = mark_feedback_processed(dataset["feedback_ids"])
    metadata["feedback_marked_processed"] = processed

    return metadata


# ---------------------------------------------------------------------------
# 7. Activation (manual approval required; blocked in TEST MODE)
# ---------------------------------------------------------------------------

def activate_model(version: str, approved_by: str) -> dict:
    """Activate a candidate after explicit human approval.

    Acceptance criteria are intentionally NOT hard-coded: the caller must
    review the comparison stored in the candidate metadata and approve
    explicitly. Blocked in TEST MODE so test runs can never affect the
    production model path.
    """
    if APP_MODE == "test":
        raise RuntimeError(
            "Model activation is disabled in TEST MODE."
        )
    if not approved_by or not approved_by.strip():
        raise ValueError(
            "Activation requires an explicit approved_by reviewer."
        )

    registry = _load_registry()
    entry = registry["versions"].get(version)
    if entry is None:
        raise ValueError(f"Unknown model version: {version}")

    model_path = Path(entry["path"]) / "model.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"Candidate artefact missing: {model_path}")

    previous = registry.get("active_version")
    if previous in registry["versions"]:
        registry["versions"][previous]["status"] = "archived"

    entry["status"] = "active"
    entry["activated_at"] = datetime.now(timezone.utc).isoformat()
    entry["approved_by"] = approved_by
    registry["active_version"] = version
    registry["active_model_path"] = str(model_path)
    _save_registry(registry)

    return {
        "status": "activated",
        "version": version,
        "previous_version": previous,
        "approved_by": approved_by,
        "note": "Active model path updated in the registry; it takes "
                "effect on the next backend process start (the runtime "
                "loads the model once at import time).",
    }


# ---------------------------------------------------------------------------
# 8. Orchestration
# ---------------------------------------------------------------------------

def run_adaptive_retraining(
    baseline_sample_size: int | None = None,
    eval_sample_size: int | None = DEFAULT_EVAL_SAMPLE_SIZE,
    threshold: float = OPTIMIZED_THRESHOLD,
) -> dict:
    """Full candidate cycle (no activation):

    eligible verified feedback -> candidate dataset -> cloned LR training
    -> thesis-metric evaluation on the frozen test set -> comparison ->
    versioned candidate artefact. Returns the comparison + metadata.
    """
    feedback_rows = get_eligible_feedback()
    if not feedback_rows:
        return {
            "status": "skipped",
            "reason": "No verified, unprocessed feedback available.",
        }

    dataset = build_candidate_dataset(feedback_rows)
    if dataset["verified_feedback_count"] == 0:
        return {
            "status": "skipped",
            "reason": "Verified feedback could not be joined to predictions.",
            "skipped": dataset["skipped"],
        }

    trained = train_candidate_model(
        dataset["X"], dataset["y"], baseline_sample_size
    )

    X_eval, y_eval = load_thesis_test_set(eval_sample_size)
    comparison = compare_models(
        trained["model"], X_eval, y_eval, threshold=threshold
    )

    metadata = create_model_version(
        trained["model"],
        comparison,
        dataset,
        threshold=threshold,
        training_sample_count=trained["training_sample_count"],
    )

    return {
        "status": "candidate_created",
        "version": metadata["version"],
        "comparison": comparison,
        "metadata": metadata,
        "activation": "Manual approval required via activate_model(); "
                      "no automatic acceptance policy is applied.",
    }
