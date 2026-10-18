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
  training/test splits, V0-V4 models, Track A threshold 0.50) are
  read-only and never modified. Candidate artefacts live under
  ``config.ADAPTIVE_MODELS_DIR`` inside the backend.
* Baseline training data is MANDATORY: every candidate is fitted on a
  deterministic sample of the development partition plus verified
  feedback. Feedback-only training is not permitted.
* Minimum verified-feedback totals and per-class floors are enforced
  BEFORE any fitting or persistence.
* Quality gates (both classes, no convergence warning, finite/non-zero
  coefficients, finite in-range non-constant probabilities, feature-count
  match) run before persistence and are enforced fail-closed by
  ``activate_model``.
* Candidate evaluation uses the stratified validation split of the
  training partition by default (leakage-safe); the frozen test set is
  only used when eval_source="test" is requested explicitly (legacy).
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
import os
import threading
import warnings
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from ..config import (
    ADAPTIVE_EVAL_SOURCE,
    ADAPTIVE_MODELS_DIR,
    ADAPTIVE_RANDOM_STATE,
    ADAPTIVE_REGISTRY_PATH,
    APP_MODE,
    DEFAULT_BASELINE_SAMPLE_SIZE,
    MIN_VERIFIED_FEEDBACK_PER_CLASS,
    MIN_VERIFIED_FEEDBACK_TOTAL,
    MODEL_PATH,
    THESIS_TEST_X_PATH,
    THESIS_TEST_Y_PATH,
    THESIS_TRAIN_X_PATH,
    THESIS_TRAIN_Y_PATH,
    VALIDATION_SPLIT_SIZE,
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

# Evaluation defaults. ``validation`` (default) evaluates candidates on a
# deterministic stratified 20% holdout carved from the TRAINING partition
# (the same split used by notebooks 06/06b); ``test`` is the legacy
# frozen-test behaviour retained only for historical reproduction.
# A sample cap keeps candidate evaluation fast during development/testing.
DEFAULT_EVAL_SAMPLE_SIZE = 5000
RANDOM_STATE = 42  # matches the thesis notebooks

_CYCLE_THREAD_LOCK = threading.Lock()
_DEV_VAL_CACHE = None


@contextmanager
def _acquire_cycle_lock():
    """Thread- and process-safe guard against concurrent candidate cycles."""
    if not _CYCLE_THREAD_LOCK.acquire(blocking=False):
        yield False
        return
    acquired = False
    fd = None
    lock_path = ADAPTIVE_MODELS_DIR / ".cycle.lock"
    try:
        ADAPTIVE_MODELS_DIR.mkdir(parents=True, exist_ok=True)
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, f"pid={os.getpid()}".encode())
            acquired = True
        except FileExistsError:
            acquired = False
        yield acquired
    finally:
        if fd is not None:
            os.close(fd)
            if acquired:
                try:
                    os.unlink(lock_path)
                except OSError:
                    pass
        _CYCLE_THREAD_LOCK.release()


def _dev_validation_indices() -> tuple[np.ndarray, np.ndarray]:
    """Deterministic stratified dev/validation split of the TRAINING set.

    Uses the exact same call as notebooks 06/06b/08
    (``train_test_split(test_size=0.20, stratify=y, random_state=42)``),
    so the validation indices equal NB06's validation set and NB08's
    simulated-feedback pool. Candidate baseline sampling draws only from
    the dev partition, so evaluation rows never enter candidate training.
    """
    global _DEV_VAL_CACHE
    if _DEV_VAL_CACHE is None:
        y = pd.read_csv(THESIS_TRAIN_Y_PATH).iloc[:, 0].to_numpy(dtype=int)
        idx = np.arange(len(y))
        dev_idx, val_idx = train_test_split(
            idx,
            test_size=VALIDATION_SPLIT_SIZE,
            stratify=y,
            random_state=ADAPTIVE_RANDOM_STATE,
        )
        _DEV_VAL_CACHE = (np.sort(dev_idx), np.sort(val_idx))
    return _DEV_VAL_CACHE


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

def _run_quality_gates(candidate, X_fit, y_fit, fit_warnings) -> dict:
    """Structural/behavioural sanity checks on a fitted candidate.

    A candidate that fails any gate must not be persisted or activated
    (activation enforces ``quality_gates.all_passed`` fail-closed).
    """
    gates = {}

    classes = np.unique(y_fit)
    gates["both_classes_present"] = len(classes) == 2

    gates["no_convergence_warning"] = not any(
        issubclass(w.category, ConvergenceWarning) for w in fit_warnings
    )
    n_iter = getattr(candidate, "n_iter_", None)
    gates["n_iter"] = (
        [int(n) for n in np.atleast_1d(n_iter)]
        if n_iter is not None else None
    )

    coef = getattr(candidate, "coef_", None)
    gates["coefficients_finite"] = (
        coef is not None and bool(np.isfinite(coef).all())
    )
    gates["coefficients_nonzero"] = (
        coef is not None and bool((np.abs(coef) > 1e-8).any())
    )

    n = X_fit.shape[0]
    X_probe = X_fit[:5000] if n > 5000 else X_fit
    try:
        proba = candidate.predict_proba(X_probe)[:, 1]
        gates["probabilities_finite"] = bool(np.isfinite(proba).all())
        gates["probabilities_in_range"] = bool(
            ((proba >= 0.0) & (proba <= 1.0)).all()
        )
        gates["probabilities_not_constant"] = bool(
            proba.min() < proba.max()
        )
    except Exception:
        gates["probabilities_finite"] = False
        gates["probabilities_in_range"] = False
        gates["probabilities_not_constant"] = False

    gates["n_features_match"] = (
        getattr(candidate, "n_features_in_", None)
        == len(SELECTED_INDICES)
    )

    gates["all_passed"] = all(
        v for k, v in gates.items() if k != "n_iter"
    )
    return gates


def train_candidate_model(
    X_new,
    y_new,
    baseline_sample_size: int | None = None,
):
    """Train a candidate by cloning the active Track A Logistic Regression.

    Hyperparameters are inherited via ``sklearn.base.clone``
    (class_weight='balanced', solver='liblinear', max_iter=1000,
    random_state=42).

    Baseline data is MANDATORY: ``baseline_sample_size`` rows are drawn
    deterministically (seed 42) from the development partition of the
    thesis TRAINING set — i.e. excluding the validation rows used for
    candidate evaluation — and prepended to the verified feedback rows.
    """
    if baseline_sample_size is None or baseline_sample_size <= 0:
        raise ValueError(
            "Baseline training data is mandatory for candidates: "
            "baseline_sample_size must be a positive integer."
        )
    if X_new is None or len(y_new) == 0:
        raise ValueError("No verified feedback data to train on.")
    if len(np.unique(y_new)) < 2:
        raise ValueError(
            "Candidate training requires both classes in verified_label."
        )

    dev_idx, _ = _dev_validation_indices()
    X_base = sparse.load_npz(THESIS_TRAIN_X_PATH)
    y_base = pd.read_csv(THESIS_TRAIN_Y_PATH).iloc[:, 0].to_numpy(dtype=int)

    rng = np.random.default_rng(RANDOM_STATE)
    take = min(baseline_sample_size, len(dev_idx))
    sampled = dev_idx[rng.choice(len(dev_idx), size=take, replace=False)]

    X_train = sparse.vstack([X_base[sampled], X_new]).tocsr()
    y_train = np.concatenate([y_base[sampled], y_new])

    candidate = clone(MODEL)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        candidate.fit(X_train, y_train)

    gates = _run_quality_gates(candidate, X_train, y_train, caught)

    return {
        "model": candidate,
        "training_sample_count": int(len(y_train)),
        "baseline_rows": int(take),
        "fit_warnings": [str(w.message) for w in caught],
        "quality_gates": gates,
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
    """Frozen thesis test split (READ-ONLY) — legacy eval_source='test'."""
    X = sparse.load_npz(THESIS_TEST_X_PATH)
    y = pd.read_csv(THESIS_TEST_Y_PATH).iloc[:, 0].to_numpy(dtype=int)
    if sample_size and sample_size < X.shape[0]:
        rng = np.random.default_rng(RANDOM_STATE)
        idx = np.sort(rng.choice(X.shape[0], size=sample_size, replace=False))
        X, y = X[idx], y[idx]
    return X, y


def load_validation_set(sample_size: int | None = None):
    """Stratified validation holdout carved from the TRAINING partition.

    Deterministic (same seed/split as notebooks 06/06b), read-only, and
    disjoint from the dev rows used for candidate training.
    """
    _, val_idx = _dev_validation_indices()
    X = sparse.load_npz(THESIS_TRAIN_X_PATH)[val_idx]
    y = pd.read_csv(THESIS_TRAIN_Y_PATH).iloc[:, 0].to_numpy(dtype=int)[val_idx]
    if sample_size and sample_size < X.shape[0]:
        rng = np.random.default_rng(RANDOM_STATE)
        sel = np.sort(rng.choice(X.shape[0], size=sample_size, replace=False))
        X, y = X[sel], y[sel]
    return X, y


def load_eval_set(eval_source: str, sample_size: int | None):
    """Resolve the candidate evaluation dataset for the given source."""
    if eval_source == "test":
        return load_thesis_test_set(sample_size)
    if eval_source == "validation":
        return load_validation_set(sample_size)
    raise ValueError(f"Unknown eval_source: {eval_source!r}")


EVAL_SOURCE_LABELS = {
    "validation": (
        "stratified 20% validation holdout of the thesis TRAINING "
        "partition (leakage-safe; test set untouched)"
    ),
    "test": (
        "frozen thesis test split, 84,792 rows (legacy; repeated-use "
        "limitation — not an unbiased final evaluation)"
    ),
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
    quality_gates: dict | None = None,
    training_provenance: dict | None = None,
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

    params = candidate_model.get_params()
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
        "quality_gates": quality_gates,
        "model_parameters": {
            "class_weight": params.get("class_weight"),
            "solver": params.get("solver"),
            "max_iter": params.get("max_iter"),
            "random_state": params.get("random_state"),
            "C": params.get("C"),
        },
        "training_provenance": training_provenance,
        "evaluation": {
            "dataset": EVAL_SOURCE_LABELS.get(
                comparison.get("eval_source", "test"),
                "unknown evaluation dataset",
            ),
            "eval_source": comparison.get("eval_source"),
            "evaluation_samples": comparison["evaluation_samples"],
            "comparison": comparison,
        },
        "label_source": "feedback.verified_label only; agree/disagree "
                        "and reward were never used as training labels",
        "base_model": "LogisticRegression cloned from the active Track A "
                      "model (class_weight='balanced', liblinear, "
                      "max_iter=1000, random_state=42)",
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

    # Fail-closed activation: only a persisted candidate whose recorded
    # quality gates explicitly passed may be activated.
    if entry.get("status") != "candidate":
        raise ValueError(
            f"Version {version} has status '{entry.get('status')}'; "
            "only versions with status 'candidate' may be activated."
        )

    metadata_path = Path(entry["path"]) / "metadata.json"
    if not metadata_path.exists():
        raise RuntimeError(
            f"Candidate metadata missing for {version}; "
            "activation requires recorded quality gates."
        )
    stored_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    stored_gates = stored_metadata.get("quality_gates") or {}
    if stored_gates.get("all_passed") is not True:
        raise RuntimeError(
            f"Candidate {version} has no passing quality-gate record; "
            "activation fails closed."
        )

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
    baseline_sample_size: int | None = None,
    eval_sample_size: int | None = DEFAULT_EVAL_SAMPLE_SIZE,
    threshold: float = OPTIMIZED_THRESHOLD,
    min_feedback_total: int = MIN_VERIFIED_FEEDBACK_TOTAL,
    min_feedback_per_class: int = MIN_VERIFIED_FEEDBACK_PER_CLASS,
    eval_source: str = ADAPTIVE_EVAL_SOURCE,
) -> dict:
    """Full candidate cycle (no activation):

    eligible verified feedback (minimum floors enforced)
    -> candidate dataset -> baseline + feedback training (mandatory)
    -> quality gates -> evaluation -> comparison
    -> versioned candidate artefact. Returns the comparison + metadata.

    A cycle-level lock prevents concurrent candidate creation. Feedback
    rows are marked processed only after successful persistence.
    """
    if baseline_sample_size is None:
        baseline_sample_size = DEFAULT_BASELINE_SAMPLE_SIZE
    if baseline_sample_size <= 0:
        raise ValueError(
            "baseline_sample_size must be > 0: baseline training data "
            "is mandatory for candidate training."
        )

    with _acquire_cycle_lock() as acquired:
        if not acquired:
            return {
                "status": "busy",
                "reason": "Another adaptive-learning cycle is already "
                          "in progress.",
            }

        feedback_rows = get_eligible_feedback()
        if not feedback_rows:
            return {
                "status": "skipped",
                "reason": "No verified, unprocessed feedback available.",
            }
        if len(feedback_rows) < min_feedback_total:
            return {
                "status": "skipped",
                "reason": (
                    f"Insufficient verified feedback: "
                    f"{len(feedback_rows)} rows < minimum "
                    f"{min_feedback_total}."
                ),
            }

        dataset = build_candidate_dataset(feedback_rows)
        if dataset["verified_feedback_count"] == 0:
            return {
                "status": "skipped",
                "reason": "Verified feedback could not be joined to "
                          "predictions.",
                "skipped": dataset["skipped"],
            }
        if dataset["verified_feedback_count"] < min_feedback_total:
            return {
                "status": "skipped",
                "reason": (
                    f"Insufficient usable verified feedback: "
                    f"{dataset['verified_feedback_count']} joined rows "
                    f"< minimum {min_feedback_total}."
                ),
            }

        class_counts = {
            int(c): int((dataset["y"] == c).sum())
            for c in np.unique(dataset["y"])
        }
        if len(class_counts) < 2 or (
            min(class_counts.values()) < min_feedback_per_class
        ):
            return {
                "status": "skipped",
                "reason": (
                    f"Per-class minimum not met: {class_counts} "
                    f"(minimum {min_feedback_per_class} per class)."
                ),
            }

        trained = train_candidate_model(
            dataset["X"], dataset["y"], baseline_sample_size
        )

        gates = trained["quality_gates"]
        if not gates["all_passed"]:
            return {
                "status": "rejected",
                "reason": "Candidate failed quality gates; nothing was "
                          "persisted and feedback stays unprocessed.",
                "quality_gates": gates,
                "fit_warnings": trained["fit_warnings"],
            }

        X_eval, y_eval = load_eval_set(eval_source, eval_sample_size)
        comparison = compare_models(
            trained["model"], X_eval, y_eval, threshold=threshold
        )
        comparison["eval_source"] = eval_source

        provenance = {
            "baseline_sample_size": baseline_sample_size,
            "baseline_rows_used": trained["baseline_rows"],
            "baseline_source": (
                "deterministic sample (seed 42) of the dev partition of "
                "X_train_selected (NB06-compatible split)"
            ),
            "feedback_rows_used": dataset["verified_feedback_count"],
            "feedback_class_counts": {
                int(c): int((dataset["y"] == c).sum())
                for c in np.unique(dataset["y"])
            },
        }

        metadata = create_model_version(
            trained["model"],
            comparison,
            dataset,
            threshold=threshold,
            training_sample_count=trained["training_sample_count"],
            quality_gates=gates,
            training_provenance=provenance,
        )

        return {
            "status": "candidate_created",
            "version": metadata["version"],
            "comparison": comparison,
            "metadata": metadata,
            "activation": "Manual approval required via activate_model(); "
                          "no automatic acceptance policy is applied.",
        }
