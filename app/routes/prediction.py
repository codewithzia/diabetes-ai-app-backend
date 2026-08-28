"""Prediction endpoints (real and test mode)."""

from flask import Blueprint, current_app, jsonify, request

from ..config import APP_MODE
from ..services.prediction import run_prediction_pipeline
from ..services.test_data import TEST_PATIENTS

prediction_bp = Blueprint("prediction", __name__)


@prediction_bp.route("/api/test/cases", methods=["GET"])
def get_test_cases():
    if APP_MODE != "test":
        return jsonify({
            "error": "Test endpoints are disabled "
                     "in production mode."
        }), 403

    cases = [
        {
            "test_case_id": test_id,
            "description": patient["description"],
        }
        for test_id, patient in TEST_PATIENTS.items()
    ]

    return jsonify({
        "mode": APP_MODE,
        "count": len(cases),
        "cases": cases,
    })


@prediction_bp.route("/api/test/predict", methods=["POST"])
def test_predict():
    if APP_MODE != "test":
        return jsonify({
            "error": "Test prediction is disabled "
                     "in production mode."
        }), 403

    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "JSON request body is required."}), 400

    test_case_id = data.get("test_case_id")
    if not test_case_id:
        return jsonify({"error": "test_case_id is required."}), 400

    if test_case_id not in TEST_PATIENTS:
        return jsonify({
            "error": "Unknown test case.",
            "available_cases": list(TEST_PATIENTS.keys()),
        }), 400

    # Use the SAME prediction function as /api/predict.
    # Do NOT create a second prediction algorithm here.
    result = run_prediction_pipeline(TEST_PATIENTS[test_case_id]["features"])
    result["test_case_id"] = test_case_id
    result["mode"] = "test"

    return jsonify(result)


@prediction_bp.route("/api/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "JSON request body is required."}), 400

    features = data.get("features", data)

    try:
        result = run_prediction_pipeline(features)
        result["mode"] = APP_MODE
        return jsonify(result)
    except ValueError as error:
        return jsonify({"error": str(error)}), 400
    except Exception:
        current_app.logger.exception("Prediction error")
        return jsonify({"error": "Prediction failed."}), 500
