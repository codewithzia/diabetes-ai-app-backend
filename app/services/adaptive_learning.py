"""Human-feedback-guided adaptive learning service.

Implements incremental candidate retraining for the thesis diabetes
model following the methodology of notebooks 08/09
("human-feedback-guided adaptive learning"):

  clone(base Logistic Regression) -> fit on cumulative data
  -> evaluate with the thesis metric set on a deterministic validation
  split of the training partition (the frozen test set stays isolated)

Hard methodological rules enforced here
---------------------------------------
* ``feedback`` (agree/disagree) and ``reward`` are reward signals ONLY.
  They are never used as training labels. Supervised labels come
  exclusively from ``feedback.verified_label`` (trusted/verified source).
* The thesis baseline artefacts (preprocessor, 50 selected features,
  training/test splits, V0-V4 models, configured threshold) are read-only
  and never modified. Candidate artefacts live under
  ``config.ADAPTIVE_MODELS_DIR`` inside the backend.
* Feedback rows are marked ``adaptive_processed = 1`` only AFTER a
  candidate model has been successfully trained, evaluated and
  persisted. Failed runs leave them unprocessed.
* ``activate_model`` never runs in TEST MODE and requires an explicit
  ``approved_by``. Acceptance criteria are pre-registered (see
  ``ACCEPTANCE_CRITERIA``) but never auto-applied: the comparison result
  is exposed and a human approves activation.

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
    ADAPTIVE_BASELINE_SAMPLE_SIZE,
    ADAPTIVE_MIN_VERIFIED_PER_CLASS,
    ADAPTIVE_MIN_VERIFIED_SAMPLES,
    ADAPTIVE_MODELS_DIR,
    ADAPTIVE_REGISTRY_PATH,
    ADAPTIVE_VALIDATION_FRACTION,
    APP_MODE,
    MODEL_PATH,
    THESIS_TEST_X_PATH,
    THESIS_TEST_Y_PATH,
    THESIS_TRAIN_X_PATH,
    THESIS_TRAIN_Y_PATH,
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
    add_missingness_flags,
    build_model_input,
)

# Evaluation defaults: the deterministic VALIDATION split of the thesis
# training partition (never the frozen test set). A sample cap keeps
# candidate evaluation fast during development/testing.
DEFAULT_EVAL_SAMPLE_SIZE = 5000
RANDOM_STATE = 42  # matches the thesis notebooks

# Pre-registered candidate acceptance criteria (defined BEFORE any future
# candidate result is examined). They are reported with every comparison
# and reviewed by a human; activation always remains manual.
ACCEPTANCE_CRITERIA = {
    "rule": (
        "candidate passes quality gates AND "
        "roc_auc >= active_roc_auc - 0.005 AND "
        "pr_auc >= active_pr_auc - 0.005"
    ),
    "roc_auc_tolerance": 0.005,
    "pr_auc_tolerance": 0.005,
    "decision": "manual approval required; criteria are pre-registered, "
                "never auto-applied",
}


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
    used by the production pipeline (36 -> 85 -> 50), via model_runtime.
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
    X = sparse.csr_matrix(processed[:, SELECTED_INDICES])  # same 50 feats
    y = np.asarray(labels, dtype=int)
    class_counts = {
        int(cls): int(count)
        for cls, count in zip(*np.unique(y, return_counts=True))
    }

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
        "verified_class_counts": class_counts,
        # Reward is descriptive metadata only, never a training label.
        "mean_reward": float(np.mean(rewards)) if rewards else None,
    }


# ---------------------------------------------------------------------------
# 3. Candidate training (thesis Logistic Regression, cloned from baseline)
# ---------------------------------------------------------------------------

def train_candidate_model(
    X_new,
    y_new,
    baseline_sample_size: int | None = ADAPTIVE_BASELINE_SAMPLE_SIZE,
):
    """Train a candidate by cloning the thesis Logistic Regression.

    Hyperparameters are inherited from the loaded thesis model via
    ``sklearn.base.clone`` (max_iter=2000, solver='liblinear',
    random_state=42). A deterministic sample of the thesis TRAINING
    partition (fit side, read-only — validation and test rows are never
    used) is prepended via ``load_baseline_sample``, mirroring the
    cumulative-data approach of notebook 09. Pass ``0``/``None`` only for
    deliberate experiments; production runs should always include the
    baseline sample so a small verified-feedback batch cannot produce a
    degenerate model.
    """
    if X_new is None or len(y_new) == 0:
        raise ValueError("No verified feedback data to train on.")
    if len(np.unique(y_new)) < 2:
        raise ValueError(
            "Candidate training requires both classes in verified_label."
        )

    X_train, y_train = X_new, y_new
    baseline_used = 0

    if baseline_sample_size:
        X_base, y_base = load_baseline_sample(baseline_sample_size)
        baseline_used = X_base.shape[0]
        X_train = sparse.vstack([X_base, sparse.csr_matrix(X_train)]).tocsr()
        y_train = np.concatenate([y_base, np.asarray(y_train, dtype=int)])

    candidate = clone(MODEL)
    candidate.fit(X_train, y_train)

    if baseline_used and X_train.shape[0] != baseline_used + len(y_new):
        raise RuntimeError("Baseline/feedback row mismatch after merge.")

    return {
        "model": candidate,
        "training_sample_count": int(len(y_train)),
        "baseline_sample_count": int(baseline_used),
        "fit_converged": bool(
            np.all(np.asarray(getattr(candidate, "n_iter_", [0])) > 0)
        ),
    }


# ---------------------------------------------------------------------------
# 3b. Thesis training-partition loaders (fit / validation isolation)
# ---------------------------------------------------------------------------

def _train_fit_validation_indices():
    """Deterministic stratified split of the thesis TRAINING partition.

    Mirrors the notebook-05b split (validation_size=0.2,
    random_state=42): validation rows are used ONLY for candidate
    evaluation; the fit complement is the only source of baseline
    training samples. The frozen TEST partition is never touched.
    """
    y = pd.read_csv(THESIS_TRAIN_Y_PATH).iloc[:, 0].to_numpy(dtype=int)
    rng = np.random.default_rng(RANDOM_STATE)
    validation = []
    for cls in (0, 1):
        cls_idx = np.where(y == cls)[0]
        count = int(round(len(cls_idx) * ADAPTIVE_VALIDATION_FRACTION))
        validation.append(rng.choice(cls_idx, size=count, replace=False))
    validation_idx = np.sort(np.concatenate(validation))
    mask = np.ones(len(y), dtype=bool)
    mask[validation_idx] = False
    return np.flatnonzero(mask), validation_idx, y


def _load_partition(frame_indices, y, sample_size):
    raw = pd.read_csv(THESIS_TRAIN_X_PATH)
    frame = add_missingness_flags(raw.iloc[frame_indices])
    processed = PREPROCESSOR.transform(frame)
    if not sparse.issparse(processed):
        processed = sparse.csr_matrix(processed)
    if processed.shape[1] != 85:
        raise RuntimeError(
            f"Baseline/validation preprocessing produced "
            f"{processed.shape[1]} features; expected 85."
        )
    X = processed[:, SELECTED_INDICES].tocsr()
    if X.shape[1] != MODEL.n_features_in_:
        raise RuntimeError(
            f"Selected baseline/validation width {X.shape[1]} does not "
            f"match the active model ({MODEL.n_features_in_})."
        )
    y_part = y[np.asarray(frame_indices)]
    if sample_size and sample_size < X.shape[0]:
        rng = np.random.default_rng(RANDOM_STATE)
        idx = np.sort(rng.choice(X.shape[0], size=sample_size, replace=False))
        X, y_part = X[idx], y_part[idx]
    return X, y_part


def load_baseline_sample(sample_size: int):
    """Deterministic sample from the thesis training FIT partition."""
    fit_idx, _, y = _train_fit_validation_indices()
    return _load_partition(fit_idx, y, sample_size)


def load_validation_set(sample_size: int | None = DEFAULT_EVAL_SAMPLE_SIZE):
    """Deterministic stratified VALIDATION split of the training partition.

    Used for all candidate evaluation. The frozen final test set stays
    fully isolated and is never loaded here.
    """
    _, validation_idx, y = _train_fit_validation_indices()
    return _load_partition(validation_idx, y, sample_size)


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
        "predicted_positive": int(y_prediction.sum()),
        "predicted_negative": int((1 - y_prediction).sum()),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def assess_candidate_quality(candidate, X_evaluation) -> dict:
    """Degeneracy/validity gates checked BEFORE a candidate is persisted.

    A candidate is degenerate when it cannot produce a meaningful ranking
    (constant probabilities or an all-zero coefficient vector after fit).
    Non-zero coefficients alone are NOT the acceptance criterion — final
    decisions use validation performance plus the pre-registered
    acceptance criteria (manual approval).
    """
    checks = {}
    try:
        probabilities = candidate.predict_proba(X_evaluation)[:, 1]
        checks["fit_completed"] = True
    except Exception:
        return {
            "valid": False, "degenerate": True,
            "reasons": ["predict_proba raised an exception"],
            "checks": {"fit_completed": False},
        }

    checks["has_both_classes"] = (
        set(np.asarray(candidate.classes_).tolist()) == {0, 1}
    )
    checks["converged"] = bool(
        np.all(np.asarray(getattr(candidate, "n_iter_", [0])) > 0)
    )
    checks["finite_probabilities"] = bool(np.isfinite(probabilities).all())
    checks["probabilities_in_range"] = bool(
        probabilities.size == 0
        or (probabilities.min() >= 0.0 and probabilities.max() <= 1.0)
    )
    checks["non_constant_probabilities"] = bool(
        probabilities.size > 0 and float(np.std(probabilities)) > 1e-9
    )
    checks["non_zero_coefficients"] = bool(
        np.count_nonzero(candidate.coef_) > 0
    )

    reasons = []
    if not checks["has_both_classes"]:
        reasons.append("model does not expose both classes {0, 1}")
    if not checks["finite_probabilities"] or not checks["probabilities_in_range"]:
        reasons.append("probabilities are not finite within [0, 1]")
    if not checks["non_constant_probabilities"]:
        reasons.append("constant probability output (no ranking ability)")
    if not checks["non_zero_coefficients"]:
        reasons.append("all-zero coefficient vector after fit")
    if not checks["converged"]:
        # Recorded as a warning: solvers may legitimately report n_iter_=0
        # only in degenerate fits, but converged=False never by itself
        # invalidates a candidate.
        reasons.append("warning: solver did not report iterations (n_iter_=0)")

    hard_fail = not (
        checks["has_both_classes"]
        and checks["finite_probabilities"]
        and checks["probabilities_in_range"]
        and checks["non_constant_probabilities"]
        and checks["non_zero_coefficients"]
    )
    return {
        "valid": not hard_fail,
        "degenerate": (
            not checks["non_constant_probabilities"]
            or not checks["non_zero_coefficients"]
        ),
        "reasons": reasons,
        "probability_stats": {
            "min": float(probabilities.min()) if probabilities.size else None,
            "mean": float(probabilities.mean()) if probabilities.size else None,
            "max": float(probabilities.max()) if probabilities.size else None,
        },
        "checks": checks,
    }


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
        # Pre-registered criteria; activation always remains manual.
        "acceptance_criteria": ACCEPTANCE_CRITERIA["rule"],
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
    baseline_sample_count: int = 0,
    quality: dict | None = None,
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
        "verified_class_counts": dataset["verified_class_counts"],
        "baseline_sample_count": baseline_sample_count,
        "quality_gates": quality,
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
        "prediction_counts": {
            "predicted_positive": metrics["predicted_positive"],
            "predicted_negative": metrics["predicted_negative"],
        },
        "threshold": threshold,
        "mean_reward": dataset["mean_reward"],
        "evaluation": {
            "dataset": "deterministic validation split of the thesis "
                       "training partition (read-only); the frozen test "
                       "set is fully isolated and never used",
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

    # Only registry entries with status exactly "candidate" are eligible.
    # Rejected, archived, active, invalid or status-less versions can
    # never be (re)activated through the approval workflow.
    status = entry.get("status")
    if status != "candidate":
        raise ValueError(
            f"Activation rejected: '{version}' has registry status "
            f"'{status}'; only versions with status 'candidate' may be "
            "activated after explicit review."
        )

    model_path = Path(entry["path"]) / "model.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"Candidate artefact missing: {model_path}")

    # Fail-closed quality-gate verification: the candidate's metadata
    # must exist, parse, and contain well-formed quality gates that
    # explicitly PASSED (valid is True, no degenerate marking). Missing
    # or malformed records are treated as unknown risk and blocked.
    metadata_path = Path(entry["path"]) / "metadata.json"
    gates = None
    if metadata_path.exists():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            gates = metadata.get("quality_gates")
        except (OSError, json.JSONDecodeError):
            gates = None
    if not isinstance(gates, dict):
        raise ValueError(
            f"Activation rejected: '{version}' has no well-formed "
            "quality-gate record; candidates created before quality "
            "gating must be re-validated through a new adaptive cycle."
        )
    if gates.get("degenerate") is True:
        raise ValueError(
            f"Activation rejected: '{version}' is marked degenerate "
            f"({'; '.join(gates.get('reasons', [])) or 'no reasons recorded'})."
        )
    if gates.get("valid") is not True:
        raise ValueError(
            f"Activation rejected: quality gates for '{version}' did "
            f"not explicitly pass "
            f"({'; '.join(gates.get('reasons', [])) or 'no reasons recorded'})."
        )

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
# 7b. Registry status / manual rejection
# ---------------------------------------------------------------------------

def get_registry_status() -> dict:
    """Sanitised registry view for API responses (no filesystem paths)."""
    registry = _load_registry()

    candidates = sorted(
        (v for v in registry["versions"] if v.startswith("VA-")),
    )
    latest = candidates[-1] if candidates else None
    latest_status = (
        registry["versions"][latest]["status"] if latest else None
    )

    eligible = len(get_eligible_feedback())

    return {
        "active_version": registry.get("active_version",
                                       MODEL_INFO["current_version"]),
        "latest_candidate": latest,
        "candidate_status": latest_status,
        "adaptive_enabled": APP_MODE != "test",
        "pending_verified_feedback": eligible,
        "candidate_versions": {
            v: registry["versions"][v]["status"] for v in candidates
        },
    }


def reject_model(version: str, rejected_by: str, reason: str = "") -> dict:
    """Manually reject a candidate (metadata only; never affects the
    active model). Kept manual pending formal thesis acceptance rules."""
    if not rejected_by or not rejected_by.strip():
        raise ValueError("Rejection requires an explicit rejected_by.")

    registry = _load_registry()
    entry = registry["versions"].get(version)
    if entry is None:
        raise ValueError(f"Unknown model version: {version}")
    if entry["status"] == "active":
        raise ValueError("Cannot reject the active model.")

    entry["status"] = "rejected"
    entry["rejected_by"] = rejected_by
    entry["rejected_at"] = datetime.now(timezone.utc).isoformat()
    if reason:
        entry["rejection_reason"] = reason
    _save_registry(registry)

    metadata_path = Path(entry["path"]) / "metadata.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["status"] = "rejected"
        metadata_path.write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )

    return {"status": "rejected", "version": version,
            "rejected_by": rejected_by}


# ---------------------------------------------------------------------------
# 8. Orchestration
# ---------------------------------------------------------------------------

def run_adaptive_retraining(
    baseline_sample_size: int | None = ADAPTIVE_BASELINE_SAMPLE_SIZE,
    eval_sample_size: int | None = DEFAULT_EVAL_SAMPLE_SIZE,
    threshold: float = OPTIMIZED_THRESHOLD,
) -> dict:
    """Full candidate cycle (no activation):

    eligible verified feedback -> minimum-data safeguards -> candidate
    dataset -> cloned LR training on baseline+feedback -> quality gates
    -> thesis-metric evaluation on the VALIDATION split (never the frozen
    test set) -> comparison -> versioned candidate artefact.
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

    # Minimum-data safeguards (configurable engineering floors, see config).
    class_counts = dataset["verified_class_counts"]
    if dataset["verified_feedback_count"] < ADAPTIVE_MIN_VERIFIED_SAMPLES:
        return {
            "status": "skipped",
            "reason": (
                f"Insufficient verified samples "
                f"({dataset['verified_feedback_count']} < "
                f"{ADAPTIVE_MIN_VERIFIED_SAMPLES})."
            ),
            "verified_feedback_count": dataset["verified_feedback_count"],
            "verified_class_counts": class_counts,
            "required_min_samples": ADAPTIVE_MIN_VERIFIED_SAMPLES,
            "required_min_per_class": ADAPTIVE_MIN_VERIFIED_PER_CLASS,
        }
    if any(count < ADAPTIVE_MIN_VERIFIED_PER_CLASS
           for count in class_counts.values()) or len(class_counts) < 2:
        return {
            "status": "skipped",
            "reason": (
                f"Insufficient verified samples per class "
                f"({class_counts}; need >= {ADAPTIVE_MIN_VERIFIED_PER_CLASS} "
                f"for BOTH classes)."
            ),
            "verified_feedback_count": dataset["verified_feedback_count"],
            "verified_class_counts": class_counts,
            "required_min_samples": ADAPTIVE_MIN_VERIFIED_SAMPLES,
            "required_min_per_class": ADAPTIVE_MIN_VERIFIED_PER_CLASS,
        }

    trained = train_candidate_model(
        dataset["X"], dataset["y"], baseline_sample_size
    )

    X_eval, y_eval = load_validation_set(eval_sample_size)

    # Quality gates BEFORE persistence: a degenerate candidate is marked
    # invalid, never persisted, never activated, and feedback stays
    # unprocessed (consistent with failed-run semantics).
    quality = assess_candidate_quality(trained["model"], X_eval)
    if not quality["valid"]:
        return {
            "status": "candidate_invalid",
            "reason": "Candidate failed quality gates; not persisted.",
            "quality": quality,
            "verified_feedback_count": dataset["verified_feedback_count"],
            "verified_class_counts": class_counts,
            "fit_converged": trained["fit_converged"],
            "evaluation_dataset": "validation split (test set isolated)",
            "activation": "No candidate was created; the active model "
                          "is unchanged.",
        }

    comparison = compare_models(
        trained["model"], X_eval, y_eval, threshold=threshold
    )

    metadata = create_model_version(
        trained["model"],
        comparison,
        dataset,
        threshold=threshold,
        training_sample_count=trained["training_sample_count"],
        baseline_sample_count=trained["baseline_sample_count"],
        quality=quality,
    )

    return {
        "status": "candidate_created",
        "version": metadata["version"],
        "comparison": comparison,
        "metadata": metadata,
        "activation": "Manual approval required via activate_model(); "
                      "no automatic acceptance policy is applied.",
    }
