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
# Feature mapping: frontend values to raw BRFSS 2021 codes, matching the
# codes the trained preprocessing pipeline was fitted on.
# ---------------------------------------------------------------------------
FEATURE_MAPPINGS = {
    # _AGEG5YR: 1-13 age groups
    "age_group": {
        "18-24": 1, "25-29": 2, "30-34": 3, "35-39": 4, "40-44": 5,
        "45-49": 6, "50-54": 7, "55-59": 8, "60-64": 9, "65-69": 10,
        "70-74": 11, "75-79": 12, "80+": 13,
    },
    # _SEX: 1 = male, 2 = female
    "sex": {"male": 1, "female": 2},
    # _RACE: 1 = White, 2 = Black, 3 = AI/AN, 4 = Asian, 6 = Other, 8 = Hispanic
    "race": {"white": 1, "black": 2, "native_american": 3, "asian": 4, "hispanic": 8, "other": 6},
    # _EDUCAG: 1 = did not graduate HS, 2 = graduated HS, 3 = some college, 4 = graduated college
    "education": {"no_school": 1, "elementary": 1, "some_high_school": 1, "high_school": 2, "some_college": 3, "college_graduate": 4},
    # _INCOMG1: 1 = <$15k, 2 = $15-25k, 3 = $25-35k, 4 = $35-50k, 5 = $50-100k
    "income": {"<10k": 1, "10k-15k": 1, "15k-20k": 2, "20k-25k": 2, "25k-35k": 3, "35k-50k": 4, "50k-75k": 5, "75k+": 5},
    # _BMI5CAT: 1 = underweight, 2 = normal, 3 = overweight, 4 = obese
    "bmi_category": {"underweight": 1, "normal": 2, "overweight": 3, "obese": 4},
    # _SMOKER3: 1 = current smoker, 4 = never smoked
    "smoking": {"yes": 1, "no": 4},
    # _TOTINDA: 1 = physically active, 2 = not active
    "physical_activity": {"yes": 1, "no": 2},
    # _RFHYPE6: 1 = no, 2 = yes (borderline is not coded as hypertension)
    "hypertension": {"yes": 2, "no": 1, "borderline": 1},
    # _RFCHOL3: 1 = no, 2 = yes (borderline is not coded as high cholesterol)
    "high_cholesterol": {"yes": 2, "no": 1, "borderline": 1},
    # _MICHD: 1 = yes, 2 = no
    "cardiovascular_disease": {"yes": 1, "no": 2},
    # CVDSTRK3: 1 = yes, 2 = no
    "stroke": {"yes": 1, "no": 2},
    # CHCKDNY2: 1 = yes, 2 = no
    "kidney_disease": {"yes": 1, "no": 2},
    # _RFHLTH: 1 = good or better, 2 = fair or poor
    "general_health": {"excellent": 1, "very_good": 1, "good": 1, "fair": 2, "poor": 2},
    # _HLTHPLN: 1 = yes, 2 = no
    "health_insurance": {"yes": 1, "no": 2},
    # PERSDOC3: 1 = yes one provider, 2 = more than one, 3 = none
    "personal_provider": {"yes": 1, "no": 3},
    # MEDCOST1: 1 = yes, 2 = no
    "medical_cost": {"yes": 1, "no": 2},
    # CHECKUP1: 1-4 time categories, 8 = never
    "checkup": {"past_year": 1, "past_2_years": 2, "past_5_years": 3, "5_plus_years": 4, "never": 8},
}

# Frontend feature key -> raw BRFSS column name (model input)
FRONTEND_TO_BRFSS = {
    "age_group": "_AGEG5YR",
    "sex": "_SEX",
    "race": "_RACE",
    "education": "_EDUCAG",
    "income": "_INCOMG1",
    "bmi_category": "_BMI5CAT",
    "smoking": "_SMOKER3",
    "physical_activity": "_TOTINDA",
    "hypertension": "_RFHYPE6",
    "high_cholesterol": "_RFCHOL3",
    "cardiovascular_disease": "_MICHD",
    "stroke": "CVDSTRK3",
    "kidney_disease": "CHCKDNY2",
    "general_health": "_RFHLTH",
    "health_insurance": "_HLTHPLN",
    "personal_provider": "PERSDOC3",
    "medical_cost": "MEDCOST1",
    "checkup": "CHECKUP1",
}

BRFSS_TO_FRONTEND = {v: k for k, v in FRONTEND_TO_BRFSS.items()}

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
    "sex": "Sex",
    "race": "Race / Ethnicity",
    "personal_provider": "Personal Healthcare Provider",
    "medical_cost": "Medical Cost Barrier",
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
    "sex": "Diabetes prevalence differs between sexes in the BRFSS population.",
    "race": "Diabetes prevalence varies across racial and ethnic groups.",
    "personal_provider": "Having a personal healthcare provider supports early detection and management.",
    "medical_cost": "Cost barriers to care are associated with delayed diagnosis and treatment.",
}


