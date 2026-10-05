"""Adaptive learning endpoints.

Endpoints
---------
GET  /api/adaptive/metrics  - static thesis metrics (unchanged)
POST /api/adaptive/update   - guarded human-feedback-guided incremental
                              retraining cycle (candidate only)
GET  /api/adaptive/status   - model-version registry status (sanitised)

Design rules
------------
- All adaptive logic lives in services/adaptive_learning.py.
- TEST MODE blocks the update cycle entirely (existing protection).
- Candidate ACTIVATION stays manual: it is an internal service
  operation (activate_model), not exposed as a public route because this
  backend has no admin/auth mechanism yet.
"""

from flask import Blueprint, current_app, jsonify

from ..config import APP_MODE
from ..constants import MODEL_INFO, PERFORMANCE_METRICS, FEEDBACK_METRICS
from ..database import count_feedback, count_predictions
from ..services import adaptive_learning

adaptive_bp = Blueprint("adaptive", __name__)


@adaptive_bp.route("/api/adaptive/metrics", methods=["GET"])
def adaptive_metrics():
    """Get adaptive learning metrics across model versions."""
    return jsonify({
        "performance_metrics": PERFORMANCE_METRICS,
        "feedback_metrics": FEEDBACK_METRICS,
        "total_feedback": count_feedback() + MODEL_INFO["feedback_observations"],
        "total_predictions": count_predictions() + 50000,
        "current_version": MODEL_INFO["current_version"],
        "last_update": MODEL_INFO["last_update"],
    })


@adaptive_bp.route("/api/adaptive/update", methods=["POST"])
def adaptive_update():
    """Run one human-feedback-guided incremental retraining cycle.

    Creates, evaluates and persists a CANDIDATE model version only.
    The active model is never replaced automatically.
    """
    if APP_MODE == "test":
        return jsonify({
            "status": "blocked",
            "mode": "test",
            "message": (
                "Adaptive model updates are disabled "
                "in TEST MODE."
            ),
        }), 403

    try:
        result = adaptive_learning.run_adaptive_retraining()
    except Exception:
        current_app.logger.exception("Adaptive retraining failed")
        return jsonify({
            "status": "error",
            "message": "Adaptive retraining failed. The active model "
                       "was not modified.",
        }), 500

    if result["status"] == "skipped":
        return jsonify({
            "status": "skipped",
            "mode": APP_MODE,
            "reason": result["reason"],
            "active_version": MODEL_INFO["current_version"],
        })

    comparison = result["comparison"]
    return jsonify({
        "status": "candidate_created",
        "mode": APP_MODE,
        "version": result["version"],
        "active_version": comparison["active_version"],
        "candidate_status": "awaiting_approval",
        "evaluation": {
            "threshold": comparison["threshold"],
            "evaluation_samples": comparison["evaluation_samples"],
            "candidate_metrics": comparison["candidate_metrics"],
            "active_metrics": comparison["active_metrics"],
            "delta": comparison["delta"],
            "acceptance_criteria": comparison["acceptance_criteria"],
        },
        "feedback_sample_count": result["metadata"]["feedback_sample_count"],
        "verified_feedback_count":
            result["metadata"]["verified_feedback_count"],
        "activation": result["activation"],
    })


@adaptive_bp.route("/api/adaptive/status", methods=["GET"])
def adaptive_status():
    """Model-version registry status (no filesystem paths exposed)."""
    status = adaptive_learning.get_registry_status()
    status["mode"] = APP_MODE
    return jsonify(status)
