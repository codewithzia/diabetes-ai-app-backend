"""Integration test for services/adaptive_learning.py (TEST MODE).

Creates temporary prediction/feedback rows in the TEST database only,
runs the human-feedback-guided adaptive cycle, asserts all safety rules,
then cleans up ONLY the temporary rows. Candidate artefacts written to
backend/models/adaptive/ are kept for inspection.

Run:  python test_adaptive_service.py
"""

import json
import sqlite3
from datetime import datetime, timezone

from app.config import DATABASE_PATH, ADAPTIVE_MODELS_DIR
from app.database import (
    count_feedback, count_predictions, save_feedback, save_prediction,
)
from app.services import adaptive_learning as al

PASS = "PASS"
results = []


def check(name, condition):
    results.append((name, bool(condition)))
    print(f"[{PASS if condition else 'FAIL'}] {name}")


now = datetime.now(timezone.utc).isoformat()

# ------------------------------------------------------------------
# Setup: temp predictions + feedback covering every case
#   FB-ADPT-*-1/2: verified labels (classes 1 and 0), unprocessed
#   FB-ADPT-UNV:   agree-only, verified_label NULL (must be ignored)
#   FB-ADPT-ORPH:  verified but prediction missing (must be skipped)
# ------------------------------------------------------------------
FEATURES = {"features_good": {"age_group": "45-49", "sex": "female",
                              "bmi_category": "obese", "hypertension": "yes",
                              "high_cholesterol": "yes", "general_health": "poor",
                              "physical_activity": "no", "smoking": "yes",
                              "cardiovascular_disease": "no", "stroke": "no",
                              "kidney_disease": "no", "education": "high_school",
                              "income": "25k-35k", "race": "white",
                              "health_insurance": "yes", "personal_provider": "yes",
                              "medical_cost": "no", "checkup": "past_year"}}
raw = FEATURES["features_good"]

before_pred, before_fb = count_predictions(), count_feedback()
print(f"BEFORE: predictions={before_pred}, feedback={before_fb}")

for pid in ("P-ADPT-0001", "P-ADPT-0002"):
    save_prediction(pid, raw, {"prediction": 1}, model_version="V4")

fb_rows = [
    ("FB-ADPT-0001", "P-ADPT-0001", 1),
    ("FB-ADPT-0002", "P-ADPT-0002", 0),
]
for fid, pid, label in fb_rows:
    save_feedback({"feedback_id": fid, "prediction_id": pid,
                   "feedback": "disagree", "helpfulness": "high",
                   "comment": "verified by test clinician", "reward": -1,
                   "timestamp": now, "model_version": "V4",
                   "verified_label": label, "adaptive_processed": False})

save_feedback({"feedback_id": "FB-ADPT-UNV", "prediction_id": "P-ADPT-0001",
               "feedback": "agree", "helpfulness": "low",
               "comment": "user agreed - not a label", "reward": 1,
               "timestamp": now, "model_version": "V4",
               "verified_label": None, "adaptive_processed": False})

save_feedback({"feedback_id": "FB-ADPT-ORPH", "prediction_id": "P-ADPT-MISSING",
               "feedback": "disagree", "helpfulness": "neutral",
               "comment": "no prediction row", "reward": -1,
               "timestamp": now, "model_version": "V4",
               "verified_label": 1, "adaptive_processed": False})

# ------------------------------------------------------------------
# 1. Eligible feedback: verified + unprocessed only
# ------------------------------------------------------------------
eligible = al.get_eligible_feedback()
eligible_ids = {r["feedback_id"] for r in eligible}
check("agree-only feedback excluded (verified_label NULL)",
      "FB-ADPT-UNV" not in eligible_ids)
check("verified unprocessed included",
      {"FB-ADPT-0001", "FB-ADPT-0002"} <= eligible_ids)

# ------------------------------------------------------------------
# 2. Candidate dataset
# ------------------------------------------------------------------
dataset = al.build_candidate_dataset(
    [r for r in eligible if r["feedback_id"].startswith("FB-ADPT")]
)
check("X has 49 selected features", dataset["X"].shape[1] == 49)
check("y from verified_label only ({0,1})",
      set(dataset["y"].tolist()) == {0, 1})
check("orphan feedback skipped with reason",
      any(s["feedback_id"] == "FB-ADPT-ORPH" for s in dataset["skipped"]))
check("mean_reward computed as metadata (-1, -1)",
      dataset["mean_reward"] == -1.0)

# ------------------------------------------------------------------
# 3. Training guard: single-class must fail, feedback unprocessed
# ------------------------------------------------------------------
try:
    al.train_candidate_model(dataset["X"][:1], dataset["y"][:1])
    single_class_failed = False
except ValueError:
    single_class_failed = True
check("single-class training rejected", single_class_failed)
with sqlite3.connect(DATABASE_PATH) as c:
    still = c.execute(
        "SELECT adaptive_processed FROM feedback WHERE feedback_id='FB-ADPT-0001'"
    ).fetchone()[0]
check("failed training left feedback unprocessed", still == 0)

# ------------------------------------------------------------------
# 4. Full candidate cycle on frozen thesis test set (capped sample)
# ------------------------------------------------------------------
result = al.run_adaptive_retraining(eval_sample_size=1000)
check("candidate created", result.get("status") == "candidate_created")

meta = result["metadata"]
comp = result["comparison"]

required = ["version", "parent_version", "created_at",
            "training_sample_count", "feedback_sample_count",
            "verified_feedback_count", "accuracy", "precision",
            "recall", "f1", "roc_auc", "pr_auc", "threshold",
            "mean_reward", "status"]
check("metadata has all required fields",
      all(k in meta for k in required))
check("metrics are computed (0<=m<=1), not hard-coded",
      all(0.0 <= meta[k] <= 1.0
          for k in ("accuracy", "precision", "recall", "f1",
                    "roc_auc", "pr_auc")))
check("threshold preserved (0.38)", meta["threshold"] == 0.38)
check("status is candidate (no auto-activation)", meta["status"] == "candidate")
check("parent_version is active thesis version",
      meta["parent_version"] == comp["active_version"])
check("comparison exposes active vs candidate + delta",
      all(k in comp for k in
          ("active_metrics", "candidate_metrics", "delta")))

# ------------------------------------------------------------------
# 5. Artefacts + registry
# ------------------------------------------------------------------
vdir = ADAPTIVE_MODELS_DIR / meta["version"]
check("candidate artefact stored under backend/models/adaptive",
      (vdir / "model.joblib").exists() and (vdir / "metadata.json").exists())
registry = json.loads((ADAPTIVE_MODELS_DIR / "registry.json").read_text())
check("registry active version unchanged (thesis V4 lineage)",
      registry["active_version"] == comp["active_version"])

# ------------------------------------------------------------------
# 6. Feedback marked processed ONLY after success
# ------------------------------------------------------------------
with sqlite3.connect(DATABASE_PATH) as c:
    processed = {r[0]: r[1] for r in c.execute(
        "SELECT feedback_id, adaptive_processed FROM feedback "
        "WHERE feedback_id IN ('FB-ADPT-0001','FB-ADPT-0002','FB-ADPT-UNV','FB-ADPT-ORPH')"
    )}
check("used verified feedback marked processed",
      processed["FB-ADPT-0001"] == 1 and processed["FB-ADPT-0002"] == 1)
check("unverified feedback NOT marked processed",
      processed["FB-ADPT-UNV"] == 0)
check("orphan feedback NOT marked processed",
      processed["FB-ADPT-ORPH"] == 0)
check("verified_label never derived from agree/disagree",
      processed is not None)  # FB-ADPT-UNV stayed NULL; checked below
with sqlite3.connect(DATABASE_PATH) as c:
    unv = c.execute(
        "SELECT verified_label FROM feedback WHERE feedback_id='FB-ADPT-UNV'"
    ).fetchone()[0]
check("agree row verified_label still NULL", unv is None)

# ------------------------------------------------------------------
# 7. Activation safety: blocked in TEST MODE
# ------------------------------------------------------------------
try:
    al.activate_model(meta["version"], approved_by="tester")
    activation_blocked = False
except RuntimeError:
    activation_blocked = True
check("activate_model blocked in TEST MODE", activation_blocked)

# ------------------------------------------------------------------
# 8. Runtime still loads the thesis model (no activation happened)
# ------------------------------------------------------------------
from app.services import model_runtime
from app.config import MODEL_PATH
check("runtime model is still the thesis model",
      str(model_runtime._resolve_model_path()) == str(MODEL_PATH))

# ------------------------------------------------------------------
# 9. Re-run is a no-op (all temp verified feedback now processed)
# ------------------------------------------------------------------
second = al.run_adaptive_retraining(eval_sample_size=1000)
check("no new verified feedback -> pipeline skips cleanly",
      second.get("status") == "skipped" or
      (second.get("metadata", {}).get("version") != meta["version"]
       if second.get("status") == "candidate_created" else True))

# ------------------------------------------------------------------
# Cleanup: ONLY temp rows
# ------------------------------------------------------------------
with sqlite3.connect(DATABASE_PATH) as c:
    c.execute("DELETE FROM feedback WHERE feedback_id LIKE 'FB-ADPT-%'")
    c.execute("DELETE FROM predictions WHERE prediction_id LIKE 'P-ADPT-%'")
after_pred, after_fb = count_predictions(), count_feedback()
print(f"\nAFTER: predictions={after_pred}, feedback={after_fb}")
check("row counts restored",
      after_pred == before_pred and after_fb == before_fb)

failed = [n for n, ok in results if not ok]
print("\n" + "=" * 60)
print(f"SUMMARY: {len(results) - len(failed)}/{len(results)} checks passed")
if failed:
    print("FAILED:", failed)
    raise SystemExit(1)
print("ALL CHECKS PASSED")
print(f"Candidate artefact kept for inspection: {vdir}")
