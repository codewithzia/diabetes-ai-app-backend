"""Static model metadata, feature mappings, and explanation data."""

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
