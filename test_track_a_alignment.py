"""Track A alignment regression tests (in-process, TEST MODE).

Verifies the backend is aligned to the notebook-08 Track A reference:
balanced Logistic Regression artifact, threshold 0.50, real missingness
indicators, correct feature order, adaptive-cycle safety floors, quality
gates, reviewer authorization, and API compatibility.

Uses only the TEST database (sqlite) and self-cleaning TEMP- rows.

Run:  python test_track_a_alignment.py
"""

import os

# Environment BEFORE importing the app package.
os.environ["APP_MODE"] = "test"
os.environ["DB_ENGINE"] = "sqlite"
os.environ["REVIEWER_API_KEY"] = "test-reviewer-key"

import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy import sparse

from app import create_app
from app.config import (
    ADAPTIVE_MODELS_DIR, MODEL_PATH, SELECTED_FEATURES_PATH,
)
from app.constants import OPTIMIZED_THRESHOLD
from app.database import (
    _connect, count_feedback, count_predictions,
    save_feedback, save_prediction,
)
from app.services import adaptive_learning as al
from app.services import model_runtime as mr

results = []


def check(name, condition, detail=""):
    results.append((name, bool(condition)))
    print(f"[{'PASS' if condition else 'FAIL'}] {name}" +
          (f"  ({detail})" if detail else ""))


app = create_app()
client = app.test_client()

# Baseline row counts BEFORE any API call that persists rows.
before_p, before_f = count_predictions(), count_feedback()

VALID = {"age_group": "45-49", "sex": "female", "race": "white",
         "education": "college_graduate", "income": "25k-35k",
         "bmi_category": "obese", "smoking": "yes",
         "physical_activity": "no", "hypertension": "yes",
         "high_cholesterol": "yes", "cardiovascular_disease": "no",
         "stroke": "no", "kidney_disease": "no",
         "general_health": "poor", "health_insurance": "yes",
         "personal_provider": "yes", "medical_cost": "no",
         "checkup": "past_year"}

# ------------------------------------------------------------------
# 1. Model configuration = Track A reference artifact
# ------------------------------------------------------------------
params = mr.MODEL.get_params()
check("model is balanced Track A LR",
      params["class_weight"] == "balanced"
      and params["solver"] == "liblinear"
      and params["max_iter"] == 1000
      and params["random_state"] == 42)
check("model input width = 49 selected features",
      mr.MODEL.n_features_in_ == 49)
check("decision threshold = 0.50 (Track A)", OPTIMIZED_THRESHOLD == 0.50)
check("model artifact path is Track A (NB08) joblib",
      MODEL_PATH.name == "adaptive_logistic_regression_final.joblib")

# ------------------------------------------------------------------
# 2. Feature order + preprocessing wiring
# ------------------------------------------------------------------
sel = pd.read_csv(SELECTED_FEATURES_PATH)
check("selected features order matches CSV exactly",
      mr.SELECTED_FEATURE_NAMES == sel["selected_feature_name"].tolist()
      and (mr.SELECTED_INDICES
           == sel["feature_index_in_processed_matrix"].to_numpy()).all())
check("model input width = 35 raw columns (18 + 17 missing flags)",
      len(mr.INPUT_COLUMNS) == 35)

# ------------------------------------------------------------------
# 3. Missingness indicators (notebook-03 semantics)
# ------------------------------------------------------------------
mi_ok = mr.build_model_input(VALID)
flag_cols = [c for c in mi_ok.columns if c.endswith("_missing")]
check("fully-mapped input -> all missing flags 0",
      mi_ok[flag_cols].iloc[0].sum() == 0 and len(flag_cols) == 17)

bad = dict(VALID)
bad["smoking"] = "former_smoker"        # unrepresentable BRFSS code
del bad["checkup"]                      # absent key
mi_bad = mr.build_model_input(bad)
check("unrepresentable value -> missing flag = 1",
      mi_bad["_SMOKER3_missing"].iloc[0] == 1.0
      and np.isnan(mi_bad["_SMOKER3"].iloc[0]))
check("absent key -> missing flag = 1",
      mi_bad["CHECKUP1_missing"].iloc[0] == 1.0)
check("unmapped columns are NaN (imputer fills mode later)",
      np.isnan(mi_bad["CHECKUP1"].iloc[0]))

# ------------------------------------------------------------------
# 4. Notebook/backend parity on identical feature construction
# ------------------------------------------------------------------
# Same row expressed once via build_model_input and once directly as
# raw BRFSS codes -> identical transformed vector and probability.
raw_row = {col: np.nan for col in mr.PREPROCESSOR.feature_names_in_}
for col in flag_cols:
    raw_row[col] = 0.0
for key, code in (("age_group", "45-49"), ("sex", "female")):
    pass  # covered by VALID below
from app.constants import FEATURE_MAPPINGS, FRONTEND_TO_BRFSS
for key, column in FRONTEND_TO_BRFSS.items():
    code = FEATURE_MAPPINGS[key][VALID[key]]
    raw_row[column] = float(code)
direct = mr.PREPROCESSOR.transform(
    pd.DataFrame([raw_row], columns=list(mr.PREPROCESSOR.feature_names_in_))
)[:, mr.SELECTED_INDICES]
via_api = mr.PREPROCESSOR.transform(mi_ok)[:, mr.SELECTED_INDICES]
check("backend row == direct notebook row (exact)",
      np.allclose(np.asarray(direct.todense() if sparse.issparse(direct)
                             else direct),
                  np.asarray(via_api.todense() if sparse.issparse(via_api)
                             else via_api)))
check("identical probability",
      mr.predict_probability(mi_ok)
      == float(mr.MODEL.predict_proba(direct)[0, 1]))

# ------------------------------------------------------------------
# 5. API compatibility (unchanged response schema)
# ------------------------------------------------------------------
r = client.post("/api/predict", json={"features": VALID})
check("/api/predict 200", r.status_code == 200)
body = r.get_json()
check("response schema keys unchanged",
      all(k in body for k in ("prediction", "probability",
                              "risk_category", "model_version",
                              "threshold", "prediction_id",
                              "explanations")))
check("threshold in response = 0.50", body["threshold"] == 0.50)

# ------------------------------------------------------------------
# 5b. Invalid-input validation (reject BEFORE inference/persistence)
# ------------------------------------------------------------------
pred_before = count_predictions()

def expect_400(name, **kwargs):
    resp = client.post("/api/predict", **kwargs)
    j = resp.get_json() or {}
    ok = (resp.status_code == 400 and "error" in j)
    check(name, ok, f"status={resp.status_code}")

expect_400("missing 'features' key -> 400", json={"foo": 1})
expect_400("bare features body (no wrapper) -> 400", json=VALID)
expect_400("features: null -> 400", json={"features": None})
expect_400("features: {} -> 400", json={"features": {}})
expect_400("features: [] -> 400", json={"features": []})
expect_400("features: 'str' -> 400", json={"features": "yes"})
r = client.post("/api/predict", json=None)
check("empty body -> 400", r.status_code in (400, 415))
r = client.post("/api/predict", data="not json",
                content_type="application/json")
check("non-JSON body -> 400", r.status_code == 400)
expect_400("missing required field -> 400",
           json={"features": {k: v for k, v in VALID.items()
                              if k != "smoking"}})
expect_400("unknown field -> 400",
           json={"features": {**VALID, "bogus_field": "x"}})
expect_400("unsupported category -> 400",
           json={"features": {**VALID, "smoking": "former_smoker"}})
expect_400("invalid type (number) -> 400",
           json={"features": {**VALID, "age_group": 5}})
expect_400("invalid type (bool) -> 400",
           json={"features": {**VALID, "hypertension": True}})
expect_400("empty-string value -> 400",
           json={"features": {**VALID, "sex": ""}})
check("invalid requests created no prediction records",
      count_predictions() == pred_before)

# Legitimate missing value: explicit null -> 200 + missing flag
r = client.post("/api/predict",
                json={"features": {**VALID, "smoking": None}})
check("explicit null field -> 200 (missing semantics)",
      r.status_code == 200)
null_pred_id = r.get_json().get("prediction_id") \
    if r.status_code == 200 else None
if r.status_code == 200:
    check("null field produced missing flag in model input",
          mr.build_model_input({**VALID, "smoking": None})
          ["_SMOKER3_missing"].iloc[0] == 1.0)
    api_p = r.get_json()["probability"]
    check("probability from loaded model, not fallback",
          abs(api_p - round(mr.predict_probability(
              mr.build_model_input({**VALID, "smoking": None})), 4)) < 1e-9)

# ------------------------------------------------------------------
# 6. Reviewer authorization on /feedback/<id>/verify
# ------------------------------------------------------------------
r = client.post("/api/feedback/FB-NONE/verify", json={"outcome": "diabetes"})
check("verify without reviewer key -> 401", r.status_code == 401)
r = client.post("/api/feedback/FB-NONE/verify",
                json={"outcome": "diabetes"},
                headers={"X-Reviewer-Key": "wrong"})
check("verify with wrong key -> 401", r.status_code == 401)
r = client.post("/api/feedback/FB-NONE/verify",
                json={"outcome": "diabetes"},
                headers={"X-Reviewer-Key": "test-reviewer-key"})
check("verify with key reaches handler (404 for unknown id)",
      r.status_code == 404)

# ------------------------------------------------------------------
# 7. Adaptive cycle: minimum floors, gates, rollback
# ------------------------------------------------------------------
now = datetime.now(timezone.utc).isoformat()

# Insufficient total -> skipped, no training, nothing persisted
res = al.run_adaptive_retraining(
    baseline_sample_size=500, eval_sample_size=500,
    min_feedback_total=999, min_feedback_per_class=1)
check("cycle below min_feedback_total -> skipped",
      res.get("status") == "skipped", res.get("reason", "")[:60])

# Seed: one verified feedback row (single class) + one agree-only
save_prediction("P-TA-0001", VALID, {"prediction": 1}, model_version="V4")
save_feedback({"feedback_id": "FB-TA-0001", "prediction_id": "P-TA-0001",
               "feedback": "agree", "helpfulness": "high", "comment": "",
               "reward": 1, "timestamp": now, "model_version": "V4",
               "verified_label": 1, "adaptive_processed": False})
save_feedback({"feedback_id": "FB-TA-UNV", "prediction_id": "P-TA-0001",
               "feedback": "agree", "helpfulness": "low", "comment": "",
               "reward": 1, "timestamp": now, "model_version": "V4",
               "verified_label": None, "adaptive_processed": False})

res = al.run_adaptive_retraining(
    baseline_sample_size=500, eval_sample_size=500,
    min_feedback_total=1, min_feedback_per_class=1)
check("single-class verified set -> skipped (per-class floor)",
      res.get("status") == "skipped", res.get("reason", "")[:60])
with _connect() as c:
    unproc = c.execute(
        "SELECT adaptive_processed FROM feedback WHERE feedback_id='FB-TA-0001'"
    ).fetchone()["adaptive_processed"]
check("skipped cycle left feedback unprocessed", unproc == 0)

# Two-class verified set -> candidate with real baseline
save_prediction("P-TA-0002", VALID, {"prediction": 0}, model_version="V4")
save_feedback({"feedback_id": "FB-TA-0002", "prediction_id": "P-TA-0002",
               "feedback": "disagree", "helpfulness": "high", "comment": "",
               "reward": -1, "timestamp": now, "model_version": "V4",
               "verified_label": 0, "adaptive_processed": False})
res = al.run_adaptive_retraining(
    baseline_sample_size=500, eval_sample_size=500,
    min_feedback_total=1, min_feedback_per_class=1)
check("two-class cycle -> candidate_created",
      res.get("status") == "candidate_created", res.get("version", ""))
meta = res.get("metadata", {})
check("candidate inherits Track A params (balanced/1000)",
      meta.get("model_parameters", {}).get("class_weight") == "balanced"
      and meta.get("model_parameters", {}).get("max_iter") == 1000)
check("metadata records quality gates + all_passed",
      meta.get("quality_gates", {}).get("all_passed") is True)
check("metadata records baseline provenance (500 + 2 rows)",
      meta.get("training_provenance", {}).get("baseline_rows_used") == 500
      and meta.get("training_sample_count") == 502)
check("candidate evaluated on validation split by default",
      res["comparison"].get("eval_source") == "validation")
check("no auto-activation (registry active still V4)",
      json.loads((ADAPTIVE_MODELS_DIR / "registry.json").read_text())
      ["active_version"] == "V4")
try:
    al.activate_model(meta["version"], approved_by="tester")
    act_blocked = False
except RuntimeError:
    act_blocked = True
check("activation blocked in TEST MODE", act_blocked)

# Concurrency guard
with al._acquire_cycle_lock() as first:
    with al._acquire_cycle_lock() as second:
        pass
check("cycle lock prevents concurrent cycles",
      first is True and second is False)

# Cleanup temp rows
with _connect() as c:
    c.execute("DELETE FROM feedback WHERE feedback_id LIKE 'FB-TA-%'")
    c.execute("DELETE FROM predictions WHERE prediction_id LIKE 'P-TA-%'")
    for pid in {body.get("prediction_id"), null_pred_id} - {None}:
        c.execute("DELETE FROM predictions WHERE prediction_id = ?",
                  (pid,))
check("row counts restored",
      count_predictions() == before_p and count_feedback() == before_f)

failed = [n for n, ok in results if not ok]
print("\n" + "=" * 60)
print(f"SUMMARY: {len(results) - len(failed)}/{len(results)} checks passed")
if failed:
    print("FAILED:", failed)
    raise SystemExit(1)
print("ALL CHECKS PASSED")
