"""Read-only prediction history endpoints.

Expose submitted predictions and their feedback status for the
reviewer-facing history UI. No filesystem paths, secrets or internal
implementation details are returned. These endpoints never trigger
adaptive retraining.
"""

from flask import Blueprint, jsonify, request

from ..config import APP_MODE
from ..database import (
    count_feedback,
    count_predictions,
    delete_demo_predictions,
    get_prediction_detail,
    list_predictions,
)

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


@history_bp.route("/api/predictions/clear-demo", methods=["POST"])
def clear_demo_predictions():
    """Safely remove explicitly identified demo/test predictions.

    Hard safety rules:
    - Only available in TEST MODE. In production mode this route is
      disabled entirely, so production/research records can never be
      affected.
    - Requires an explicit list of prediction_ids; it never deletes
      records it was not told about.
    - Only touches the mode-specific demo database — never thesis
      datasets, model artifacts, or the V4 model.
    """
    if APP_MODE != "test":
        return jsonify({
            "error": "Demo cleanup is disabled in production mode.",
            "mode": APP_MODE,
        }), 403

    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "JSON request body is required."}), 400

    prediction_ids = data.get("prediction_ids")
    if not isinstance(prediction_ids, list) or not prediction_ids \
            or not all(isinstance(p, str) for p in prediction_ids):
        return jsonify({
            "error": "prediction_ids must be a non-empty list of strings."
        }), 400

    result = delete_demo_predictions(prediction_ids)
    return jsonify({
        "status": "ok",
        "mode": APP_MODE,
        **result,
    })
