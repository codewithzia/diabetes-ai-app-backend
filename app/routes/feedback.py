"""Human feedback collection endpoint."""

from datetime import datetime, timezone

from flask import Blueprint, jsonify, request

from ..config import REVIEWER_API_KEY
from ..constants import MODEL_INFO
from ..database import next_feedback_id, save_feedback, set_verified_label

feedback_bp = Blueprint("feedback", __name__)


def _reviewer_authorized():
    """Gate for reviewer-only routes.

    Requires ``REVIEWER_API_KEY`` to be configured in the environment and
    the request to carry it in the ``X-Reviewer-Key`` header. Fails
    closed: if no key is configured, the endpoint is disabled.
    """
    if not REVIEWER_API_KEY:
        return None  # unconfigured -> disabled
    return request.headers.get("X-Reviewer-Key") == REVIEWER_API_KEY


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

    feedback_id = next_feedback_id()

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


# Outcomes a Doctor/Reviewer may attest. "unable_to_verify" deliberately
# does NOT create a supervised training label.
_VERIFY_OUTCOMES = {
    "diabetes": 1,
    "no_diabetes": 0,
    "unable_to_verify": None,
}


@feedback_bp.route("/api/feedback/<feedback_id>/verify", methods=["POST"])
def verify_feedback(feedback_id: str):
    """Record a verified clinical outcome for a feedback record.

    DOCTOR/REVIEWER-ONLY WORKFLOW: requires the ``X-Reviewer-Key``
    header to match the configured ``REVIEWER_API_KEY`` environment
    variable. If no key is configured the endpoint is disabled
    (fail-closed).

    Agree/disagree feedback is NEVER a label. Only an explicit verified
    outcome submitted through this route creates a supervised label via
    database.set_verified_label — the single trusted write path.
    It does NOT trigger adaptive retraining; verified labels merely
    become eligible for the controlled candidate pipeline.
    """
    authorized = _reviewer_authorized()
    if authorized is None:
        return jsonify({
            "error": "Reviewer verification endpoint is disabled: "
                     "REVIEWER_API_KEY is not configured."
        }), 503
    if not authorized:
        return jsonify({
            "error": "Reviewer authorization required "
                     "(X-Reviewer-Key header)."
        }), 401

    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "JSON request body is required."}), 400

    outcome = data.get("outcome")
    if outcome not in _VERIFY_OUTCOMES:
        return jsonify({
            "error": "Invalid outcome.",
            "allowed": sorted(_VERIFY_OUTCOMES.keys()),
        }), 400

    label = _VERIFY_OUTCOMES[outcome]
    updated = set_verified_label(feedback_id, label)
    if updated == 0:
        return jsonify({"error": "Feedback record not found."}), 404

    if label is None:
        # "Unable to Verify": any existing label is cleared and no
        # training label is created.
        return jsonify({
            "feedback_id": feedback_id,
            "outcome": outcome,
            "verified_label": None,
            "status": "no_label_created",
            "eligible_for_adaptive": False,
        })

    return jsonify({
        "feedback_id": feedback_id,
        "outcome": outcome,
        "verified_label": label,
        "status": "verified_label_recorded",
        "eligible_for_adaptive": True,
        "message": (
            "Verified outcome recorded. This feedback is now eligible "
            "for the controlled adaptive-learning pipeline. No retraining "
            "has been triggered."
        ),
    })
