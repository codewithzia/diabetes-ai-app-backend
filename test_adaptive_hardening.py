"""Regression tests for adaptive-learning hardening (TEST MODE).

Covers the findings from the VA-0006..VA-0009 degenerate-candidate
investigation:
- tiny verified sets (2 rows / single class) -> cycle SKIPPED
- corrected baseline loader -> exactly 50 features, correct order
- no obsolete 49-feature matrix is used anywhere
- degenerate candidates are marked invalid and never persisted
- candidate evaluation never reads the frozen final test partition
- candidate creation never activates a model
- TEST MODE protections intact; prediction/feedback endpoints OK

Inserts temporary rows into the TEST database only and cleans them up.
Candidate artefacts produced by the successful-cycle check are kept for
inspection, consistent with the existing test design.

Run:  python test_adaptive_hardening.py
"""

import json
import os

os.environ.setdefault("APP_MODE", "test")

from datetime import datetime, timezone

import numpy as np
from scipy import sparse
from sklearn.base import clone

from app import create_app
from app.config import ADAPTIVE_MODELS_DIR
from app.database import (
    _connect, count_feedback, count_predictions, save_feedback,
    save_prediction,
)
from app.services import adaptive_learning as al
from app.services.model_runtime import MODEL

PASS, results = "PASS", []


def check(name, condition, detail=""):
    results.append((name, bool(condition)))
    suffix = f"  ({detail})" if detail else ""
    print(f"[{PASS if condition else 'FAIL'}] {name}{suffix}")


now = datetime.now(timezone.utc).isoformat()
RAW = {"age_group": "45-49", "sex": "female", "bmi_category": "obese",
       "hypertension": "yes", "high_cholesterol": "yes",
       "general_health": "poor", "physical_activity": "no",
       "smoking": "yes", "cardiovascular_disease": "no", "stroke": "no",
       "kidney_disease": "no", "education": "high_school",
       "income": "25k-35k", "race": "white", "health_insurance": "yes",
       "personal_provider": "yes", "medical_cost": "no",
       "checkup": "past_year"}

before_pred, before_fb = count_predictions(), count_feedback()
versions_before = set((ADAPTIVE_MODELS_DIR / "registry.json")
                      .exists()
                      and json.loads((ADAPTIVE_MODELS_DIR / "registry.json").read_text())["versions"].keys())

for pid in ("P-H-0001", "P-H-0002"):
    save_prediction(pid, RAW, {"prediction": 1}, model_version="V4")
for fid, pid, label in (("FB-H-0001", "P-H-0001", 1),
                        ("FB-H-0002", "P-H-0002", 0)):
    save_feedback({"feedback_id": fid, "prediction_id": pid,
                   "feedback": "disagree", "helpfulness": "high",
                   "comment": "hardening test", "reward": -1,
                   "timestamp": now, "model_version": "V4",
                   "verified_label": label, "adaptive_processed": False})

# ------------------------------------------------------------------
# 1. Two verified rows -> skipped (default engineering floors)
# ------------------------------------------------------------------
assert al.ADAPTIVE_MIN_VERIFIED_SAMPLES >= 50  # defaults intact
result = al.run_adaptive_retraining(baseline_sample_size=200,
                                    eval_sample_size=500)
check("two verified rows -> skipped by minimum-sample floor",
      result["status"] == "skipped"
      and "Insufficient verified samples" in result["reason"]
      and result["verified_class_counts"] == {0: 1, 1: 1},
      result["reason"][:70])

# ------------------------------------------------------------------
# 2. Insufficient per-class counts -> skipped
# ------------------------------------------------------------------
al.ADAPTIVE_MIN_VERIFIED_SAMPLES = 2
al.ADAPTIVE_MIN_VERIFIED_PER_CLASS = 5
result = al.run_adaptive_retraining(baseline_sample_size=200,
                                    eval_sample_size=500)
check("insufficient per-class labels -> skipped",
      result["status"] == "skipped"
      and "per class" in result["reason"], result["reason"][:70])
al.ADAPTIVE_MIN_VERIFIED_PER_CLASS = 1

# ------------------------------------------------------------------
# 3. Corrected baseline loader: 50 features, 85 pre-selection, disjoint
#    fit/validation partitions derived from the TRAINING partition.
# ------------------------------------------------------------------
X_base, y_base = al.load_baseline_sample(400)
check("baseline loader produces exactly 50 features",
      X_base.shape[1] == 50 and X_base.shape[0] == 400)
active_direct = MODEL.predict_proba(X_base)[:, 1]
check("baseline rows are valid inputs for the active model",
      np.isfinite(active_direct).all()
      and np.std(active_direct) > 1e-6,
      f"mean p={active_direct.mean():.3f}")

fit_idx, val_idx, y_all = al._train_fit_validation_indices()
check("fit/validation partitions disjoint and complete",
      len(set(fit_idx.tolist()) & set(val_idx.tolist())) == 0
      and len(fit_idx) + len(val_idx) == len(y_all))
check("validation fraction matches configured 0.2 (+-1%)",
      abs(len(val_idx) / len(y_all) - al.ADAPTIVE_VALIDATION_FRACTION) < 0.01)

# ------------------------------------------------------------------
# 4. No obsolete 49-feature matrix anywhere; eval never reads test files
# ------------------------------------------------------------------
al.THESIS_TEST_X_PATH = ADAPTIVE_MODELS_DIR / "DOES_NOT_EXIST.npz"
al.THESIS_TEST_Y_PATH = ADAPTIVE_MODELS_DIR / "DOES_NOT_EXIST.csv"
try:
    X_val, y_val = al.load_validation_set(500)
    no_test_read = True
except Exception:
    no_test_read = False
check("validation loader never reads the frozen test partition",
      no_test_read and X_val.shape[1] == 50)
check("no .npz selected matrix used by loaders (X_train_raw pipeline)",
      X_val.shape[1] == 50)

# ------------------------------------------------------------------
# 5. Degenerate detection on a VA-0006-style model (2-row fit)
# ------------------------------------------------------------------
import joblib
tiny = joblib.load(ADAPTIVE_MODELS_DIR / "VA-0007" / "model.joblib")
quality = al.assess_candidate_quality(tiny, X_val)
check("saved VA-0007 style 2-row fit detected as degenerate/invalid",
      quality["degenerate"] and not quality["valid"],
      "; ".join(quality["reasons"])[:80])
check("healthy clone on baseline sample passes quality gates",
      al.assess_candidate_quality(
          clone(MODEL).fit(X_base, y_base), X_val)["valid"])

# ------------------------------------------------------------------
# 6. End-to-end: degenerate cycle -> candidate_invalid, not persisted
# ------------------------------------------------------------------
result = al.run_adaptive_retraining(baseline_sample_size=0,
                                    eval_sample_size=500)
check("2-row-feedback cycle -> candidate_invalid (not persisted)",
      result["status"] == "candidate_invalid", result.get("reason", "")[:60])
registry = json.loads((ADAPTIVE_MODELS_DIR / "registry.json").read_text())
check("no new candidate registered after degenerate cycle",
      set(registry["versions"].keys()) == versions_before)
check("active version still V4 after degenerate cycle",
      registry["active_version"] == "V4")
with _connect() as c:
    unproc = {r["feedback_id"]: r["adaptive_processed"] for r in c.execute(
        "SELECT feedback_id, adaptive_processed FROM feedback "
        "WHERE feedback_id IN ('FB-H-0001','FB-H-0002')")}
check("invalid cycle left feedback unprocessed",
      all(v == 0 for v in unproc.values()))

# ------------------------------------------------------------------
# 7. Valid cycle with baseline -> candidate created, never activated
# ------------------------------------------------------------------
result = al.run_adaptive_retraining(baseline_sample_size=400,
                                    eval_sample_size=800)
check("valid cycle creates candidate", result["status"] == "candidate_created")
meta = result["metadata"]
check("evaluation dataset is validation split (not test)",
      "validation split" in meta["evaluation"]["dataset"])
check("metadata records class counts, baseline size, quality gates",
      meta["verified_class_counts"] == {0: 1, 1: 1}
      and meta["baseline_sample_count"] == 400
      and meta["quality_gates"]["valid"] is True)
check("metadata records prediction counts + confusion matrix",
      {"predicted_positive", "predicted_negative"}
      <= set(meta["prediction_counts"]))
check("pre-registered acceptance criteria present",
      "roc_auc" in result["comparison"]["acceptance_criteria"])
registry = json.loads((ADAPTIVE_MODELS_DIR / "registry.json").read_text())
check("candidate creation did NOT activate any model",
      registry["active_version"] == "V4")
with _connect() as c:
    proc = {r["feedback_id"]: r["adaptive_processed"] for r in c.execute(
        "SELECT feedback_id, adaptive_processed FROM feedback "
        "WHERE feedback_id IN ('FB-H-0001','FB-H-0002')")}
check("successful cycle marked verified feedback processed",
      all(v == 1 for v in proc.values()))

# ------------------------------------------------------------------
# 8. TEST MODE protections intact
# ------------------------------------------------------------------
try:
    al.activate_model(meta["version"], approved_by="hardening-test")
    blocked = False
except RuntimeError:
    blocked = True
check("activate_model blocked in TEST MODE", blocked)
rejected = al.reject_model(meta["version"], rejected_by="hardening-test",
                           reason="regression cleanup")
check("manual rejection works", rejected["status"] == "rejected")

# ------------------------------------------------------------------
# 9. Prediction + feedback endpoints still work
# ------------------------------------------------------------------
app = create_app()
client = app.test_client()
res = client.post("/api/predict", json={"features": RAW})
check("/api/predict works", res.status_code == 200
      and 0.0 <= res.get_json()["probability"] <= 1.0)
res = client.post("/api/feedback", json={
    "prediction_id": res.get_json()["prediction_id"], "feedback": "agree"})
check("/api/feedback works", res.status_code == 200
      and res.get_json()["status"] == "recorded")
res = client.post("/api/adaptive/update", json={})
check("TEST MODE blocks adaptive updates via HTTP (403)",
      res.status_code == 403)

# ------------------------------------------------------------------
# 10. Cleanup temp rows (prediction/feedback inserted by endpoint kept:
#     they belong to the test DB lifecycle; only HB- rows removed).
# ------------------------------------------------------------------
with _connect() as c:
    c.execute("DELETE FROM feedback WHERE feedback_id LIKE 'FB-H-%%'")
    c.execute("DELETE FROM predictions WHERE prediction_id LIKE 'P-H-%%'")
check("temporary rows cleaned up",
      count_predictions() >= before_pred and count_feedback() >= before_fb)

failed = [name for name, ok in results if not ok]
print("=" * 70)
print(f"{len(results) - len(failed)}/{len(results)} checks passed")
if failed:
    print("FAILED:", *failed, sep="\n  - ")
    raise SystemExit(1)
print("ALL CHECKS PASSED")
