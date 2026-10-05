"""Human feedback collection endpoint."""

from datetime import datetime, timezone

from flask import Blueprint, jsonify, request

from ..constants import MODEL_INFO
from ..database import count_feedback, save_feedback

feedback_bp = Blueprint("feedback", __name__)


@feedback_bp.route("/api/feedback", methods=["POST"])
def feedback():
    """Collect user feedback and generate reward signal."""
    data = request.get_json()
    if not data:
        return jsonify({"error": "Missing request body"}), 400

    prediction_id = data.get("prediction_id", "unknown")
    feedback_type = data.get("feedback", "agree")
    helpfulness = data.get("helpfulness", "neutral")
    comment = data.get("comment", "")

    reward = 1 if feedback_type == "agree" else -1

    feedback_id = f"FB-{count_feedback() + 123:06d}"

    feedback_record = {
        "feedback_id": feedback_id,
        "prediction_id": prediction_id,
        "feedback": feedback_type,
        "helpfulness": helpfulness,
        "comment": comment,
        "reward": reward,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_version": MODEL_INFO["current_version"],
        # verified_label is deliberately NOT set here: user agree/disagree
        # is a reward signal, not a verified clinical label.
        "verified_label": None,
        "adaptive_processed": False,
    }
    save_feedback(feedback_record)

    return jsonify({
        "feedback_id": feedback_id,
        "reward": reward,
        "status": "recorded",
    })
