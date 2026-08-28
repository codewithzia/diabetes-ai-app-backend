"""Health check endpoint."""

from datetime import datetime, timezone

from flask import Blueprint, jsonify

from ..config import APP_MODE

health_bp = Blueprint("health", __name__)


@health_bp.route("/api/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "mode": APP_MODE,
        "test_mode": APP_MODE == "test",
        "adaptive_updates_enabled": APP_MODE == "production",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })
