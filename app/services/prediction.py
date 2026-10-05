"""Prediction pipeline: risk scoring and explanations from the trained model."""

import uuid

from ..constants import MODEL_INFO, OPTIMIZED_THRESHOLD
from ..database import save_prediction
from .model_runtime import (
    build_model_input,
    generate_explanations,
    predict_probability,
)


def get_risk_category(probability: float) -> str:
    if probability >= 0.7:
        return "Higher Risk"
    if probability >= 0.4:
        return "Moderate Risk"
    return "Lower Risk"


def run_prediction_pipeline(features: dict) -> dict:
    """
    Common prediction pipeline used by:

        /api/predict
        /api/test/predict

    This is deliberately shared so that TEST MODE and
    REAL MODE cannot accidentally use different prediction
    logic.
    """
    model_input = build_model_input(features)
    probability = predict_probability(model_input)
    prediction = 1 if probability >= OPTIMIZED_THRESHOLD else 0

    prediction_id = f"P-{uuid.uuid4().hex[:12].upper()}"

    result = {
        "prediction": prediction,
        "probability": round(probability, 4),
        "risk_category": get_risk_category(probability),
        "model_version": MODEL_INFO["current_version"],
        "threshold": OPTIMIZED_THRESHOLD,
        "prediction_id": prediction_id,
        "explanations": generate_explanations(model_input, features),
    }

    save_prediction(
        prediction_id,
        features,
        result,
        model_version=MODEL_INFO["current_version"],
    )

    return result
