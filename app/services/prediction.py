"""Prediction pipeline: feature mapping, risk scoring, and explanations."""

import math

from ..constants import (
    FEATURE_MAPPINGS,
    FEATURE_DISPLAY_NAMES,
    FEATURE_VALUE_DISPLAY,
    FEATURE_EXPLANATIONS,
    MODEL_WEIGHTS,
    OPTIMIZED_THRESHOLD,
)


def map_features(raw_features: dict) -> dict:
    """Map human-readable feature values to numeric codes."""
    return {
        key: FEATURE_MAPPINGS[key].get(value, 0)
        if key in FEATURE_MAPPINGS else 0
        for key, value in raw_features.items()
    }


def compute_risk_probability(features: dict) -> float:
    """
    Compute a simulated risk probability.
    In production, this would use the trained Logistic Regression model.
    """
    score = 0.0
    for feature, weight in MODEL_WEIGHTS.items():
        if feature not in features:
            continue
        value = features[feature]
        if not isinstance(value, (int, float)):
            continue
        mapping = FEATURE_MAPPINGS.get(feature)
        max_code = max(mapping.values()) if mapping else 1
        score += weight * (value / max_code if max_code else value)

    probability = 1 / (1 + math.exp(-score))
    return min(max(probability, 0.01), 0.99)


def get_risk_category(probability: float) -> str:
    if probability >= 0.7:
        return "Higher Risk"
    if probability >= 0.4:
        return "Moderate Risk"
    return "Lower Risk"


def generate_explanations(features: dict) -> list:
    """Generate explanation factors based on model weights and feature values."""
    explanations = []

    for feature, weight in MODEL_WEIGHTS.items():
        if feature not in features:
            continue

        value = features[feature]
        contribution = weight * (1 if isinstance(value, (int, float)) else 1)

        display_name = FEATURE_DISPLAY_NAMES.get(feature, feature)
        if feature in FEATURE_VALUE_DISPLAY and value in FEATURE_VALUE_DISPLAY[feature]:
            value_label = FEATURE_VALUE_DISPLAY[feature][value]
            display_name = f"{display_name}: {value_label}"

        direction = "increase" if contribution > 0 else "decrease"
        explanation_text = FEATURE_EXPLANATIONS.get(feature, f"Contribution of {feature} to the prediction.")

        explanations.append({
            "feature": feature,
            "display_name": display_name,
            "direction": direction,
            "contribution": round(abs(contribution) * 100, 2),
            "explanation": explanation_text,
        })

    return explanations


def run_prediction_pipeline(features: dict) -> dict:
    """
    Common prediction pipeline used by:

        /api/predict
        /api/test/predict

    This is deliberately shared so that TEST MODE and
    REAL MODE cannot accidentally use different prediction
    logic.
    """
    mapped_features = map_features(features)
    probability = compute_risk_probability(mapped_features)
    prediction = 1 if probability >= OPTIMIZED_THRESHOLD else 0

    return {
        "prediction": prediction,
        "probability": round(probability, 4),
        "risk_category": get_risk_category(probability),
        "explanations": generate_explanations(mapped_features),
    }
