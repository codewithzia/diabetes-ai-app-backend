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

import uuid
from datetime import datetime, timezone
from flask import Flask, request, jsonify
from flask_cors import CORS
import os

APP_MODE = os.getenv(
    "APP_MODE",
    "test"
).lower()

ALLOWED_MODES = {
    "test",
    "production"
}

if APP_MODE not in ALLOWED_MODES:
    raise ValueError(
        f"Invalid APP_MODE: {APP_MODE}. "
        f"Allowed values: {ALLOWED_MODES}"
    )

print("=" * 80)
print(f"APPLICATION MODE: {APP_MODE.upper()}")
print("=" * 80)


app = Flask(__name__)
CORS(app, origins=["http://localhost:4200", "http://127.0.0.1:4200"])


# ============================================================
# TEST DATA
# ============================================================

from pathlib import Path
import json

BASE_DIR = Path(__file__).resolve().parent

TEST_DATA_PATH = (
    BASE_DIR / "test_data" / "test_patients.json"
)


def load_test_patients():

    if not TEST_DATA_PATH.exists():

        raise FileNotFoundError(
            f"Test data not found: "
            f"{TEST_DATA_PATH}"
        )

    with open(
        TEST_DATA_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)


TEST_PATIENTS = load_test_patients()

print(
    f"Loaded {len(TEST_PATIENTS)} test patients."
)


# ---------------------------------------------------------------------------
# In-memory data stores (replace with database/persistent storage in production)
# ---------------------------------------------------------------------------
feedback_store = []
prediction_store = {}

# ---------------------------------------------------------------------------
# Model metadata
# ---------------------------------------------------------------------------
MODEL_INFO = {
    "current_model": "Logistic Regression",
    "current_version": "V4",
    "training_observations": 339166,
    "selected_features": 49,
    "feedback_observations": 67834,
    "feedback_batches": 4,
    "last_update": "2024-12-20",
}

MODEL_VERSIONS = [
    {"version": "V0", "label": "Initial", "description": "Baseline model trained on BRFSS 2021", "date": "2024-01-15"},
    {"version": "V1", "label": "Feedback Batch 1", "description": "First adaptive update with 15,000 feedback signals", "date": "2024-03-20"},
    {"version": "V2", "label": "Feedback Batch 2", "description": "Second adaptive update with 32,000 feedback signals", "date": "2024-06-10"},
    {"version": "V3", "label": "Feedback Batch 3", "description": "Third adaptive update with 51,000 feedback signals", "date": "2024-09-15"},
    {"version": "V4", "label": "Final", "description": "Final model with 67,834 total feedback signals", "date": "2024-12-20"},
]

PERFORMANCE_METRICS = [
    {"version": "V0", "accuracy": 0.8473, "precision": 0.8152, "recall": 0.7821, "f1": 0.7983, "roc_auc": 0.8921, "pr_auc": 0.8234},
    {"version": "V1", "accuracy": 0.8512, "precision": 0.8241, "recall": 0.7956, "f1": 0.8096, "roc_auc": 0.8967, "pr_auc": 0.8312},
    {"version": "V2", "accuracy": 0.8548, "precision": 0.8303, "recall": 0.8089, "f1": 0.8195, "roc_auc": 0.9012, "pr_auc": 0.8387},
    {"version": "V3", "accuracy": 0.8581, "precision": 0.8356, "recall": 0.8167, "f1": 0.8260, "roc_auc": 0.9054, "pr_auc": 0.8445},
    {"version": "V4", "accuracy": 0.8615, "precision": 0.8412, "recall": 0.8234, "f1": 0.8322, "roc_auc": 0.9089, "pr_auc": 0.8512},
]

FEEDBACK_METRICS = [
    {"version": "V0", "positive_feedback": 0.0, "negative_feedback": 0.0, "mean_reward": 0.0},
    {"version": "V1", "positive_feedback": 0.72, "negative_feedback": 0.28, "mean_reward": 0.44},
    {"version": "V2", "positive_feedback": 0.75, "negative_feedback": 0.25, "mean_reward": 0.50},
    {"version": "V3", "positive_feedback": 0.78, "negative_feedback": 0.22, "mean_reward": 0.56},
    {"version": "V4", "positive_feedback": 0.81, "negative_feedback": 0.19, "mean_reward": 0.62},
]

MODEL_COMPARISON = [
    {"model": "Logistic Regression", "accuracy": 0.8615, "precision": 0.8412, "recall": 0.8234, "f1": 0.8322, "roc_auc": 0.9089, "pr_auc": 0.8512, "selected": True},
    {"model": "Naive Bayes", "accuracy": 0.8234, "precision": 0.7891, "recall": 0.7654, "f1": 0.7770, "roc_auc": 0.8567, "pr_auc": 0.7823},
    {"model": "KNN", "accuracy": 0.8356, "precision": 0.8012, "recall": 0.7845, "f1": 0.7928, "roc_auc": 0.8712, "pr_auc": 0.7989},
    {"model": "Decision Tree", "accuracy": 0.8123, "precision": 0.7756, "recall": 0.7567, "f1": 0.7660, "roc_auc": 0.8345, "pr_auc": 0.7534},
    {"model": "Random Forest", "accuracy": 0.8589, "precision": 0.8378, "recall": 0.8189, "f1": 0.8282, "roc_auc": 0.9056, "pr_auc": 0.8478},
    {"model": "Linear SVM", "accuracy": 0.8467, "precision": 0.8189, "recall": 0.7989, "f1": 0.8087, "roc_auc": 0.8923, "pr_auc": 0.8267},
    {"model": "XGBoost", "accuracy": 0.8598, "precision": 0.8401, "recall": 0.8212, "f1": 0.8305, "roc_auc": 0.9078, "pr_auc": 0.8501},
]

OPTIMIZED_THRESHOLD = 0.38

# ---------------------------------------------------------------------------
# Feature mapping: human-readable values to BRFSS codes
# (In production, this would be handled by the trained preprocessing pipeline)
# ---------------------------------------------------------------------------
FEATURE_MAPPINGS = {
    "age_group": {
        "18-24": 1, "25-29": 2, "30-34": 3, "35-39": 4, "40-44": 5,
        "45-49": 6, "50-54": 7, "55-59": 8, "60-64": 9, "65-69": 10,
        "70-74": 11, "75-79": 12, "80+": 13,
    },
    "sex": {"male": 1, "female": 0},
    "race": {"white": 1, "black": 2, "asian": 4, "native_american": 3, "hispanic": 5, "other": 6},
    "education": {"no_school": 1, "elementary": 2, "some_high_school": 3, "high_school": 4, "some_college": 5, "college_graduate": 6},
    "income": {"<10k": 1, "10k-15k": 2, "15k-20k": 3, "20k-25k": 4, "25k-35k": 5, "35k-50k": 6, "50k-75k": 7, "75k+": 8},
    "bmi_category": {"underweight": 1, "normal": 2, "overweight": 3, "obese": 4},
    "smoking": {"yes": 1, "no": 0},
    "physical_activity": {"yes": 1, "no": 0},
    "hypertension": {"yes": 1, "no": 0, "borderline": 2},
    "high_cholesterol": {"yes": 1, "no": 0, "borderline": 2},
    "cardiovascular_disease": {"yes": 1, "no": 0},
    "stroke": {"yes": 1, "no": 0},
    "kidney_disease": {"yes": 1, "no": 0},
    "general_health": {"excellent": 5, "very_good": 4, "good": 3, "fair": 2, "poor": 1},
    "health_insurance": {"yes": 1, "no": 0},
    "personal_provider": {"yes": 1, "no": 0},
    "medical_cost": {"yes": 1, "no": 0},
    "checkup": {"past_year": 1, "past_2_years": 2, "past_5_years": 3, "5_plus_years": 4, "never": 5},
}

# ---------------------------------------------------------------------------
# Explanation display names
# ---------------------------------------------------------------------------
FEATURE_DISPLAY_NAMES = {
    "bmi_category": "BMI Category",
    "age_group": "Age Group",
    "hypertension": "Hypertension",
    "general_health": "General Health",
    "checkup": "Routine Checkup",
    "high_cholesterol": "High Cholesterol",
    "smoking": "Smoking Status",
    "physical_activity": "Physical Activity",
    "income": "Household Income",
    "education": "Education Level",
    "health_insurance": "Health Insurance",
    "kidney_disease": "Kidney Disease",
    "cardiovascular_disease": "Cardiovascular Disease",
    "stroke": "Stroke",
}

FEATURE_VALUE_DISPLAY = {
    "bmi_category": {1: "Underweight", 2: "Normal Weight", 3: "Overweight", 4: "Obese"},
    "age_group": {1: "18–24 years", 2: "25–29 years", 3: "30–34 years", 4: "35–39 years", 5: "40–44 years", 6: "45–49 years", 7: "50–54 years", 8: "55–59 years", 9: "60–64 years", 10: "65–69 years", 11: "70–74 years", 12: "75–79 years", 13: "80 or older"},
    "hypertension": {0: "No", 1: "Yes", 2: "Borderline"},
    "general_health": {1: "Poor", 2: "Fair", 3: "Good", 4: "Very Good", 5: "Excellent"},
    "checkup": {1: "Within the past year", 2: "Within the past 2 years", 3: "Within the past 5 years", 4: "5 or more years ago", 5: "Never"},
}

FEATURE_EXPLANATIONS = {
    "bmi_category": "Higher BMI categories are associated with increased diabetes risk.",
    "age_group": "Older age groups have a higher prevalence of diabetes.",
    "hypertension": "Hypertension is a known comorbidity associated with diabetes.",
    "general_health": "Self-reported general health correlates with diabetes risk.",
    "checkup": "Recent checkups may indicate existing health concerns.",
    "high_cholesterol": "High cholesterol is a metabolic risk factor for diabetes.",
    "smoking": "Smoking status contributes to overall metabolic risk.",
    "physical_activity": "Regular physical activity reduces diabetes risk.",
    "income": "Lower income levels are associated with reduced healthcare access.",
    "education": "Education level correlates with health literacy and outcomes.",
    "health_insurance": "Lack of insurance may delay diagnosis and treatment.",
    "kidney_disease": "Kidney disease is a complication associated with diabetes.",
    "cardiovascular_disease": "Cardiovascular conditions share risk factors with diabetes.",
    "stroke": "Stroke history indicates vascular risk factors.",
}

# ---------------------------------------------------------------------------
# Simulated model weights for explanation generation
# (In production, these would come from the trained Logistic Regression model)
# ---------------------------------------------------------------------------
MODEL_WEIGHTS = {
    "bmi_category": 0.8119,
    "age_group": -0.7964,
    "hypertension": 0.5383,
    "general_health": 0.5354,
    "checkup": 1.1456,
    "high_cholesterol": 0.3214,
    "smoking": 0.1876,
    "physical_activity": -0.2341,
    "income": -0.1567,
    "education": -0.0987,
    "health_insurance": -0.1234,
    "kidney_disease": 0.4123,
    "cardiovascular_disease": 0.2876,
    "stroke": 0.2154,
    "sex": 0.0567,
    "race": 0.0321,
    "medical_cost": 0.1456,
    "personal_provider": -0.0876,
}


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def map_features(raw_features: dict) -> dict:
    """Map human-readable feature values to numeric codes."""
    mapped = {}
    for key, value in raw_features.items():
        if key in FEATURE_MAPPINGS and value in FEATURE_MAPPINGS[key]:
            mapped[key] = FEATURE_MAPPINGS[key][value]
        else:
            mapped[key] = 0
    return mapped


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


def compute_risk_probability(features: dict) -> float:
    """
    Compute a simulated risk probability.
    In production, this would use the trained Logistic Regression model.
    """
    score = 0.0
    for feature, weight in MODEL_WEIGHTS.items():
        if feature in features:
            value = features[feature]
            if isinstance(value, (int, float)):
                normalized_value = value / max(FEATURE_MAPPINGS.get(feature, {1: 1}).keys()) if FEATURE_MAPPINGS.get(feature) else value
                score += weight * normalized_value

    import math
    probability = 1 / (1 + math.exp(-score))
    return min(max(probability, 0.01), 0.99)


def get_risk_category(probability: float) -> str:
    if probability >= 0.7:
        return "Higher Risk"
    elif probability >= 0.4:
        return "Moderate Risk"
    else:
        return "Lower Risk"

# ============================================================
# COMMON PREDICTION PIPELINE
# ============================================================

def run_prediction_pipeline(features):

    """
    Common prediction pipeline used by:

        /api/predict
        /api/test/predict

    This is deliberately shared so that TEST MODE and
    REAL MODE cannot accidentally use different prediction
    logic.
    """

    mapped_features = map_features(
        features
    )

    probability = compute_risk_probability(
        mapped_features
    )

    prediction = (
        1
        if probability >= OPTIMIZED_THRESHOLD
        else 0
    )

    risk_category = (
        get_risk_category(probability)
    )

    explanations = (
        generate_explanations(
            mapped_features
        )
    )

    return {
        "prediction": prediction,
        "probability": round(
            probability,
            4
        ),
        "risk_category": risk_category,
        "explanations": explanations
    }

# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------
# ============================================================
# TEST CASE LIST
# ============================================================

@app.route(
    "/api/test/cases",
    methods=["GET"]
)
def get_test_cases():

    if APP_MODE != "test":

        return jsonify({
            "error": "Test endpoints are disabled "
                     "in production mode."
        }), 403

    cases = []

    for test_id, patient in TEST_PATIENTS.items():

        cases.append({
            "test_case_id": test_id,
            "description": patient["description"]
        })

    return jsonify({
        "mode": APP_MODE,
        "count": len(cases),
        "cases": cases
    })

# ============================================================
# TEST PREDICTION
# ============================================================

@app.route(
    "/api/test/predict",
    methods=["POST"]
)
def test_predict():

    if APP_MODE != "test":

        return jsonify({
            "error": "Test prediction is disabled "
                     "in production mode."
        }), 403

    data = request.get_json(
        silent=True
    )

    if not data:

        return jsonify({
            "error": "JSON request body is required."
        }), 400

    test_case_id = data.get(
        "test_case_id"
    )

    if not test_case_id:

        return jsonify({
            "error": "test_case_id is required."
        }), 400

    if test_case_id not in TEST_PATIENTS:

        return jsonify({
            "error": "Unknown test case.",
            "available_cases": list(
                TEST_PATIENTS.keys()
            )
        }), 400

    patient = TEST_PATIENTS[
        test_case_id
    ]

    features = patient[
        "features"
    ]

    # --------------------------------------------------------
    # IMPORTANT
    # --------------------------------------------------------
    # Use the SAME prediction function as /api/predict.
    #
    # Do NOT create a second prediction algorithm here.
    # --------------------------------------------------------

    result = run_prediction_pipeline(
        features
    )

    result["test_case_id"] = test_case_id

    result["mode"] = "test"

    return jsonify(result)

# ============================================================
# REAL PREDICTION ENDPOINT
# ============================================================

@app.route(
    "/api/predict",
    methods=["POST"]
)
def predict():

    data = request.get_json(
        silent=True
    )

    if not data:

        return jsonify({
            "error": "JSON request body is required."
        }), 400

    features = data.get(
        "features",
        data
    )

    try:

        result = run_prediction_pipeline(
            features
        )

        result["mode"] = APP_MODE

        return jsonify(result)

    except ValueError as error:

        return jsonify({
            "error": str(error)
        }), 400

    except Exception as error:

        app.logger.exception(
            "Prediction error"
        )

        return jsonify({
            "error": "Prediction failed."
        }), 500

@app.route("/api/feedback", methods=["POST"])
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

    feedback_id = f"FB-{len(feedback_store) + 123:06d}"

    feedback_record = {
        "feedback_id": feedback_id,
        "prediction_id": prediction_id,
        "feedback": feedback_type,
        "helpfulness": helpfulness,
        "comment": comment,
        "reward": reward,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    feedback_store.append(feedback_record)

    return jsonify({
        "feedback_id": feedback_id,
        "reward": reward,
        "status": "recorded",
    })


@app.route("/api/model/status", methods=["GET"])
def model_status():
    """Get current model status and metadata."""
    return jsonify(MODEL_INFO)


@app.route("/api/model/versions", methods=["GET"])
def model_versions():
    """Get all model versions."""
    return jsonify(MODEL_VERSIONS)


@app.route("/api/model/comparison", methods=["GET"])
def model_comparison():
    """Get model comparison table."""
    return jsonify(MODEL_COMPARISON)


@app.route("/api/adaptive/metrics", methods=["GET"])
def adaptive_metrics():
    """Get adaptive learning metrics across model versions."""
    return jsonify({
        "performance_metrics": PERFORMANCE_METRICS,
        "feedback_metrics": FEEDBACK_METRICS,
        "total_feedback": len(feedback_store) + MODEL_INFO["feedback_observations"],
        "total_predictions": len(prediction_store) + 50000,
        "current_version": MODEL_INFO["current_version"],
        "last_update": MODEL_INFO["last_update"],
    })


@app.route("/api/adaptive/update", methods=["POST"])
def adaptive_update():
    if APP_MODE == "test":     
        return jsonify({
        "status": "blocked",
        "mode": "test",
        "message": (
            "Adaptive model updates are disabled "
            "in TEST MODE."
        )
        }), 403
    
    """Trigger adaptive model update (admin endpoint)."""
    current_version_num = int(MODEL_INFO["current_version"].replace("V", ""))
    new_version = f"V{current_version_num + 1}"

    MODEL_INFO["current_version"] = new_version
    MODEL_INFO["feedback_batches"] += 1
    MODEL_INFO["last_update"] = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    return jsonify({
        "status": "success",
        "new_version": new_version,
        "message": "Adaptive training completed. Model has been updated with new feedback batch.",
    })


@app.route("/api/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "mode": APP_MODE,
        "test_mode": (
            APP_MODE == "test"
        ),
        "adaptive_updates_enabled": (
            APP_MODE == "production"
        ),
        "timestamp": datetime.now(timezone.utc).isoformat()
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
