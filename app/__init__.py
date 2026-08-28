"""
Adaptive AI System for Diabetes Prediction Using Human Feedback
Flask Backend API — Research Prototype

This backend provides REST API endpoints for:
- Diabetes risk prediction (Logistic Regression)
- Explainable AI (feature contributions)
- Human feedback collection with reward signal
- Adaptive model management and monitoring
- Model comparison and versioning

Dataset: BRFSS 2021
Features: 49 selected processed features
Model: Logistic Regression with optimized threshold
"""

from flask import Flask
from flask_cors import CORS

from .config import APP_MODE, CORS_ORIGINS
from .routes.health import health_bp
from .routes.prediction import prediction_bp
from .routes.feedback import feedback_bp
from .routes.model import model_bp
from .routes.adaptive import adaptive_bp


def create_app() -> Flask:
    """Application factory."""
    app = Flask(__name__)
    CORS(app, origins=CORS_ORIGINS)

    app.register_blueprint(health_bp)
    app.register_blueprint(prediction_bp)
    app.register_blueprint(feedback_bp)
    app.register_blueprint(model_bp)
    app.register_blueprint(adaptive_bp)

    print("=" * 80)
    print(f"APPLICATION MODE: {APP_MODE.upper()}")
    print("=" * 80)

    return app
