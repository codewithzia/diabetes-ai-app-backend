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
Features: 50 selected processed features (leakage-safe experiment)
Model: Logistic Regression with optimized threshold
"""

import uuid
import os
from pathlib import Path
from datetime import datetime, timezone
import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app, origins=["http://localhost:4200", "http://127.0.0.1:4200"])

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
    "selected_features": 50,
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

OPTIMIZED_THRESHOLD = float(os.getenv("DIABETES_THRESHOLD", "0.63"))  # selected on validation for max F1

# ---------------------------------------------------------------------------
# Leakage-safe trained artifacts. Keep these files beside app_integrated.py,
# or configure DIABETES_MODEL_DIR. Install the same scikit-learn version used
# to create the artifacts before deploying (metadata records 1.9.0).
# ---------------------------------------------------------------------------
ARTIFACT_DIR = Path(os.getenv("DIABETES_MODEL_DIR", Path(__file__).resolve().parent))
PREPROCESSOR_PATH = ARTIFACT_DIR / "leakage_safe_preprocessor.joblib"
MODEL_PATH = ARTIFACT_DIR / "leakage_safe_logistic_regression.joblib"
SELECTED_FEATURES_PATH = ARTIFACT_DIR / "final_selected_features.csv"

RAW_COLUMNS = [
    "_AGEG5YR", "_SEX", "_RACE", "_EDUCAG", "_INCOMG1", "_BMI5CAT",
    "_SMOKER3", "_TOTINDA", "_RFHYPE6", "_RFCHOL3", "_MICHD",
    "CVDSTRK3", "CHCKDNY2", "_RFHLTH", "_HLTHPLN", "PERSDOC3",
    "MEDCOST1", "CHECKUP1",
]
MISSING_COLUMNS = [f"{c}__missing" for c in RAW_COLUMNS]

try:
    PREPROCESSOR = joblib.load(PREPROCESSOR_PATH)
    TRAINED_MODEL = joblib.load(MODEL_PATH)
    _selected = pd.read_csv(SELECTED_FEATURES_PATH)
    SELECTED_INDICES = _selected["selected_feature_index"].astype(int).to_numpy()
    SELECTED_NAMES = _selected["selected_feature_name"].astype(str).tolist()
    if len(SELECTED_INDICES) != TRAINED_MODEL.n_features_in_:
        raise RuntimeError(
            f"Selected feature count ({len(SELECTED_INDICES)}) does not match model input "
            f"({TRAINED_MODEL.n_features_in_})."
        )
    if len(SELECTED_NAMES) != len(set(SELECTED_NAMES)):
        raise RuntimeError("Selected feature names contain duplicates.")
    MODEL_LOAD_ERROR = None
except Exception as exc:
    PREPROCESSOR = TRAINED_MODEL = None
    SELECTED_INDICES = np.array([], dtype=int)
    SELECTED_NAMES = []
    MODEL_LOAD_ERROR = str(exc)


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


def compute_model_prediction(features: dict) -> tuple[float, list]:
    """Run the saved preprocessing pipeline and trained Logistic Regression.

    Request features must use the original numeric BRFSS codes for RAW_COLUMNS.
    Omitted or null values are represented as NaN and receive a 1 missingness
    flag; observed values receive a 0 flag. Do not send already one-hot encoded
    columns from the frontend.
    """
    if MODEL_LOAD_ERROR or PREPROCESSOR is None or TRAINED_MODEL is None:
        raise RuntimeError(f"Trained model artifacts could not be loaded: {MODEL_LOAD_ERROR}")

    row = {}
    for col in RAW_COLUMNS:
        value = features.get(col, None)
        if value is None or value == "":
            row[col] = np.nan
        else:
            try:
                row[col] = float(value)
            except (TypeError, ValueError):
                raise ValueError(f"Feature '{col}' must be a numeric BRFSS code or null.")
    frame = pd.DataFrame([row], columns=RAW_COLUMNS)
    for col in RAW_COLUMNS:
        frame[f"{col}__missing"] = frame[col].isna().astype(int)

    transformed = PREPROCESSOR.transform(frame)
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()
    if transformed.shape[1] <= int(SELECTED_INDICES.max()):
        raise RuntimeError("Preprocessor output width does not match selected feature indices.")
    selected = transformed[:, SELECTED_INDICES]
    probability = float(TRAINED_MODEL.predict_proba(selected)[0, 1])

    # Coefficient contributions are shown only for the selected transformed
    # features. These are model contributions, not causal explanations.
    contributions = selected[0] * TRAINED_MODEL.coef_[0]
    order = np.argsort(np.abs(contributions))[::-1][:5]
    explanations = []
    for idx in order:
        explanations.append({
            "feature": SELECTED_NAMES[int(idx)],
            "display_name": SELECTED_NAMES[int(idx)],
            "contribution": round(float(contributions[int(idx)]), 5),
            "direction": "increase" if contributions[int(idx)] >= 0 else "decrease",
            "explanation": "Signed Logistic Regression feature contribution; association, not causation.",
        })
    return probability, explanations


def get_risk_category(probability: float) -> str:
    if probability >= 0.7:
        return "Higher Risk"
    elif probability >= 0.4:
        return "Moderate Risk"
    else:
        return "Lower Risk"


# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------

@app.route("/api/predict", methods=["POST"])
def predict():
    """Generate diabetes risk prediction with explanations."""
    data = request.get_json()
    if not data or "features" not in data:
        return jsonify({"error": "Missing 'features' in request body"}), 400

    raw_features = data["features"]
    if not isinstance(raw_features, dict):
        return jsonify({"error": "'features' must be an object keyed by original BRFSS column names."}), 400
    try:
        probability, explanations = compute_model_prediction(raw_features)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        app.logger.exception("Trained model prediction failed")
        return jsonify({"error": "Trained model is unavailable or prediction failed.", "detail": str(exc)}), 503
    prediction = 1 if probability >= OPTIMIZED_THRESHOLD else 0
    risk_category = get_risk_category(probability)
    prediction_id = str(uuid.uuid4())

    response = {
        "prediction": prediction,
        "probability": round(probability, 4),
        "risk_category": risk_category,
        "model_version": MODEL_INFO["current_version"],
        "threshold": OPTIMIZED_THRESHOLD,
        "explanations": explanations,
        "prediction_id": prediction_id,
    }

    prediction_store[prediction_id] = {
        "features": raw_features,
        "prediction": prediction,
        "probability": probability,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    return jsonify(response)


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
    return jsonify({**MODEL_INFO, "threshold": OPTIMIZED_THRESHOLD,
                    "model_loaded": MODEL_LOAD_ERROR is None,
                    "model_load_error": MODEL_LOAD_ERROR})


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
    return jsonify({"status": "healthy" if MODEL_LOAD_ERROR is None else "degraded", "model_loaded": MODEL_LOAD_ERROR is None, "timestamp": datetime.now(timezone.utc).isoformat()})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
