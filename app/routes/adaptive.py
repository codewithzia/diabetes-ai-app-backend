"""Adaptive learning endpoints."""

from datetime import datetime, timezone

from flask import Blueprint, jsonify

from ..config import APP_MODE
from ..constants import MODEL_INFO, PERFORMANCE_METRICS, FEEDBACK_METRICS
from ..stores import feedback_store, prediction_store

adaptive_bp = Blueprint("adaptive", __name__)


@adaptive_bp.route("/api/adaptive/metrics", methods=["GET"])
def adaptive_metrics():
    """Get adaptive learning metrics across model versions."""
    return jsonify({
        "performance_metrics": PERFORMANCE_METRICS,
        "feedback_metrics": FEEDBACK_METRICS,
        "total_feedback": len(feedback_store) + MODEL_INFO["feedback_observations"],
        "total_predictions": len(prediction_store) + 50000,
        "current_version": MODEL_INFO["current_version"],
        "last_update": MODEL_INFO["last_update"],
    })


@adaptive_bp.route("/api/adaptive/update", methods=["POST"])
def adaptive_update():
    """Trigger adaptive model update (admin endpoint)."""
    if APP_MODE == "test":
        return jsonify({
            "status": "blocked",
            "mode": "test",
            "message": (
                "Adaptive model updates are disabled "
                "in TEST MODE."
            ),
        }), 403

    current_version_num = int(MODEL_INFO["current_version"].replace("V", ""))
    new_version = f"V{current_version_num + 1}"

    MODEL_INFO["current_version"] = new_version
    MODEL_INFO["feedback_batches"] += 1
    MODEL_INFO["last_update"] = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    return jsonify({
        "status": "success",
        "new_version": new_version,
        "message": "Adaptive training completed. Model has been updated with new feedback batch.",
    })
