"""Integration tests: trained leakage-safe model inside the Flask app.

Covers:
- artifact loading and dimension validation (36 -> 85 -> 50)
- missingness-flag logic identical to training
- HTTP 400 on invalid / missing request fields
- finite probabilities in [0, 1]
- endpoint predictions identical to direct model.predict_proba
- threshold affects classification only, never the probability
- frontend request/response contract compatibility
- feedback and adaptive endpoints still work
- TEST MODE restrictions enforced (no adaptive updates / retraining)

Runs against the isolated TEST database (sqlite test engine) and never
touches the production database or the active model.

Run:  APP_MODE=test DB_ENGINE=sqlite python test_model_integration.py
"""

import os

os.environ.setdefault("APP_MODE", "test")
os.environ.setdefault("DB_ENGINE", "sqlite")

import numpy as np

PASS, results = "PASS", []


def check(name, condition):
    results.append((name, bool(condition)))
    print(f"[{PASS if condition else 'FAIL'}] {name}")


from app import create_app
from app.config import PREDICTION_THRESHOLD
from app.services import model_runtime as mr
from app.services import prediction as prediction_service
from app.constants import FRONTEND_TO_BRFSS, OPTIMIZED_THRESHOLD

VALID_FEATURES = {
    "age_group": "45-49", "sex": "female", "race": "white",
    "education": "high_school", "income": "25k-35k",
    "bmi_category": "obese", "smoking": "no",
    "physical_activity": "no", "hypertension": "yes",
    "high_cholesterol": "yes", "cardiovascular_disease": "no",
    "stroke": "no", "kidney_disease": "no",
    "general_health": "fair", "health_insurance": "yes",
    "personal_provider": "yes", "medical_cost": "no",
    "checkup": "past_year",
}

# ------------------------------------------------------------------
# 1. Artifacts
# ------------------------------------------------------------------
check("artifacts loaded (preprocessor + model + selection)",
      mr.PREPROCESSOR is not None and mr.MODEL is not None)
check("preprocessor input = 18 raw + 18 missingness flags",
      len(mr.INPUT_COLUMNS) == 36)

probe = mr.build_model_input(VALID_FEATURES)
processed = mr.PREPROCESSOR.transform(probe)
check("preprocessor output has exactly 85 features", processed.shape[1] == 85)
check("selected features = 50 and match model input",
      len(mr.SELECTED_INDICES) == 50
      and len(mr.SELECTED_INDICES) == mr.MODEL.n_features_in_)
check("selected feature names are unique",
      len(set(mr.SELECTED_FEATURE_NAMES)) == 50)

# ------------------------------------------------------------------
# 2. Missingness flags (training-time logic)
# ------------------------------------------------------------------
flags = [c for c in probe.columns if c.endswith("__missing")]
check("all 18 missingness flags present", len(flags) == 18)
check("complete input -> all flags 0", probe[flags].sum().sum() == 0.0)

partial = mr.build_model_input({k: v for k, v in VALID_FEATURES.items()
                                if k != "bmi_category"})
check("omitted field -> NaN and flag=1",
      bool(np.isnan(partial["_BMI5CAT"].iloc[0]))
      and partial["_BMI5CAT__missing"].iloc[0] == 1.0)

# ------------------------------------------------------------------
# 3. Validation errors
# ------------------------------------------------------------------
try:
    mr.build_model_input({**VALID_FEATURES, "smoking": "sometimes"})
    invalid_raises = False
except ValueError:
    invalid_raises = True
check("unsupported value rejected with ValueError", invalid_raises)

# ------------------------------------------------------------------
# 4. Endpoint behaviour
# ------------------------------------------------------------------
app = create_app()
client = app.test_client()

res = client.post("/api/predict", json={})
check("missing body fields -> HTTP 400", res.status_code == 400)
res = client.post("/api/predict",
                  json={"features": {**VALID_FEATURES, "smoking": "bad"}})
check("invalid value -> HTTP 400", res.status_code == 400)

res = client.post("/api/predict", json={"features": VALID_FEATURES})
check("valid input -> HTTP 200", res.status_code == 200)
body = res.get_json()

probability = body.get("probability")
check("probability finite and within [0, 1]",
      probability is not None and np.isfinite(probability)
      and 0.0 <= probability <= 1.0)

# Direct model comparison: identical transformed input, identical output.
direct = float(mr.MODEL.predict_proba(processed[:, mr.SELECTED_INDICES])[0, 1])
check("endpoint probability == direct model.predict_proba",
      abs(probability - round(direct, 4)) < 1e-4)
check("classification uses configured threshold",
      body["prediction"] == int(direct >= OPTIMIZED_THRESHOLD)
      and body["threshold"] == PREDICTION_THRESHOLD)

# Frontend response contract.
expected_keys = {"prediction", "probability", "risk_category",
                 "model_version", "threshold", "explanations",
                 "prediction_id", "mode"}
check("response contract keys preserved", expected_keys <= set(body.keys()))
check("explanations use trained coefficients",
      isinstance(body["explanations"], list)
      and all({"feature", "display_name", "direction",
               "contribution", "explanation"} <= set(e)
              for e in body["explanations"]))

# Threshold changes flip classification but never the probability.
original_threshold = prediction_service.OPTIMIZED_THRESHOLD
try:
    prediction_service.OPTIMIZED_THRESHOLD = 0.0
    low = prediction_service.run_prediction_pipeline(VALID_FEATURES)
    prediction_service.OPTIMIZED_THRESHOLD = 1.0
    high = prediction_service.run_prediction_pipeline(VALID_FEATURES)
finally:
    prediction_service.OPTIMIZED_THRESHOLD = original_threshold
check("threshold flips classification only",
      low["prediction"] == 1 and high["prediction"] == 0
      and low["probability"] == high["probability"])

# ------------------------------------------------------------------
# 5. Feedback + adaptive endpoints + TEST MODE restrictions
# ------------------------------------------------------------------
res = client.post("/api/feedback", json={
    "prediction_id": body["prediction_id"],
    "feedback": "agree", "helpfulness": "helpful",
})
fb = res.get_json()
check("feedback endpoint works (reward = 1 for agree)",
      res.status_code == 200 and fb.get("reward") == 1
      and fb.get("status") == "recorded")

res = client.get("/api/adaptive/metrics")
check("GET /api/adaptive/metrics works", res.status_code == 200)
res = client.get("/api/adaptive/status")
check("GET /api/adaptive/status works", res.status_code == 200)

res = client.post("/api/adaptive/update", json={})
check("TEST MODE blocks adaptive updates (HTTP 403)",
      res.status_code == 403 and res.get_json().get("status") == "blocked")

res = client.get("/api/test/cases")
check("test-mode endpoints available in TEST MODE", res.status_code == 200)

# ------------------------------------------------------------------
# Summary
# ------------------------------------------------------------------
failed = [name for name, ok in results if not ok]
print("=" * 70)
print(f"{len(results) - len(failed)}/{len(results)} checks passed")
if failed:
    print("FAILED:", *failed, sep="\n  - ")
    raise SystemExit(1)
print("ALL CHECKS PASSED")
