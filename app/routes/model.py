"""Model metadata endpoints."""

from flask import Blueprint, jsonify

from ..constants import MODEL_INFO, MODEL_VERSIONS, MODEL_COMPARISON

model_bp = Blueprint("model", __name__)


@model_bp.route("/api/model/status", methods=["GET"])
def model_status():
    """Get current model status and metadata."""
    return jsonify(MODEL_INFO)


@model_bp.route("/api/model/versions", methods=["GET"])
def model_versions():
    """Get all model versions."""
    return jsonify(MODEL_VERSIONS)


@model_bp.route("/api/model/comparison", methods=["GET"])
def model_comparison():
    """Get model comparison table."""
    return jsonify(MODEL_COMPARISON)
