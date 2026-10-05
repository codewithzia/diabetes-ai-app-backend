"""Read-only prediction history endpoints.

Expose submitted predictions and their feedback status for the
reviewer-facing history UI. No filesystem paths, secrets or internal
implementation details are returned. These endpoints never trigger
adaptive retraining.
"""

from flask import Blueprint, jsonify

from ..database import count_feedback, count_predictions, get_prediction_detail, list_predictions

history_bp = Blueprint("history", __name__)


@history_bp.route("/api/predictions", methods=["GET"])
def prediction_history():
    """List submitted predictions (newest first) with feedback status."""
    return jsonify({
        "count_total": count_predictions(),
        "feedback_total": count_feedback(),
        "items": list_predictions(),
    })


@history_bp.route("/api/predictions/<prediction_id>", methods=["GET"])
def prediction_detail(prediction_id: str):
    """Full detail for one submitted prediction."""
    detail = get_prediction_detail(prediction_id)
    if detail is None:
        return jsonify({"error": "Prediction not found."}), 404
    return jsonify(detail)
