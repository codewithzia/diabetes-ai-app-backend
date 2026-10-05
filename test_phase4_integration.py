"""Phase 4 integration test: adaptive API + version management.

Requires a running backend in TEST MODE (python app.py).
Checks API protection, prediction regression, candidate lifecycle,
and verifies thesis baseline artifacts are byte-for-byte untouched.

Run:  python test_phase4_integration.py
"""

import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import requests

from app.config import (
    ADAPTIVE_MODELS_DIR, DATABASE_PATH, MODEL_PATH, MODEL_PATH,
    PREPROCESSOR_PATH, SELECTED_FEATURES_PATH, THESIS_DIR,
    THESIS_TEST_X_PATH, THESIS_TEST_Y_PATH,
)
from app.database import (
    count_feedback, count_predictions, save_feedback, save_prediction,
)
from app.services import adaptive_learning as al
from app.services import model_runtime

BASE = "http://127.0.0.1:5000"

results = []


def check(name, condition, detail=""):
    results.append((name, bool(condition)))
    print(f"[{'PASS' if condition else 'FAIL'}] {name}" +
          (f"  ({detail})" if detail else ""))


# ------------------------------------------------------------------
# 0. Snapshot thesis baseline artifacts (before)
# ------------------------------------------------------------------
THESIS_ARTIFACTS = [
    MODEL_PATH, PREPROCESSOR_PATH, SELECTED_FEATURES_PATH,
    THESIS_TEST_X_PATH, THESIS_TEST_Y_PATH,
    THESIS_TEST_X_PATH.with_name("X_train_selected.npz"),
    THESIS_TEST_Y_PATH.with_name("y_train.csv"),
    THESIS_DIR / "notebooks" / "08_adaptive_learning.ipynb",
    THESIS_DIR / "notebooks" / "09_rlhf_adaptive_learning.ipynb",
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


hashes_before = {p: sha256(p) for p in THESIS_ARTIFACTS}
print(f"Snapshotted {len(hashes_before)} thesis artifacts.")

# ------------------------------------------------------------------
# 1-3. API: test-mode protection + prediction regression
# ------------------------------------------------------------------
r = requests.post(f"{BASE}/api/adaptive/update", json={})
check("/api/adaptive/update blocked in TEST MODE (403)",
      r.status_code == 403, r.json().get("message", ""))
check("blocked response unchanged format",
      r.json().get("status") == "blocked" and r.json().get("mode") == "test")

r = requests.post(f"{BASE}/api/test/predict", json={"test_case_id": "TEST-002"})
p = r.json()
check("/api/test/predict works (200)", r.status_code == 200)
check("prediction threshold is 0.38", p.get("threshold") == 0.38)
check("prediction model_version is V4 (thesis active model)",
      p.get("model_version") == "V4")

# ------------------------------------------------------------------
# 4. Status endpoint: sanitised registry info
# ------------------------------------------------------------------
r = requests.get(f"{BASE}/api/adaptive/status")
s = r.json()
check("/api/adaptive/status 200 with expected keys",
      r.status_code == 200 and
      all(k in s for k in ("active_version", "latest_candidate",
                           "candidate_status", "adaptive_enabled",
                           "pending_verified_feedback", "mode")))
check("active_version is V4", s["active_version"] == "V4")
check("adaptive_enabled is False in test mode",
      s["adaptive_enabled"] is False)
check("status response exposes no filesystem paths",
      not any(sep in json.dumps(s) for sep in ("D:\\", "D:/", "\\Ml", ".joblib")))

# ------------------------------------------------------------------
# 5-8. Candidate lifecycle via service (internal invocation)
# ------------------------------------------------------------------
now = datetime.now(timezone.utc).isoformat()
before_pred, before_fb = count_predictions(), count_feedback()
raw = {"age_group": "60-64", "sex": "male", "bmi_category": "obese",
       "hypertension": "yes", "high_cholesterol": "yes",
       "general_health": "fair", "physical_activity": "no",
       "smoking": "no", "cardiovascular_disease": "no", "stroke": "no",
       "kidney_disease": "no", "education": "college_graduate",
       "income": "50k-75k", "race": "asian", "health_insurance": "yes",
       "personal_provider": "yes", "medical_cost": "no",
       "checkup": "past_year"}

for pid in ("P-P4-0001", "P-P4-0002"):
    save_prediction(pid, raw, {"prediction": 1}, model_version="V4")
for fid, pid, label in (("FB-P4-0001", "P-P4-0001", 1),
                        ("FB-P4-0002", "P-P4-0002", 0)):
    save_feedback({"feedback_id": fid, "prediction_id": pid,
                   "feedback": "disagree", "helpfulness": "high",
                   "comment": "phase4 verified", "reward": -1,
                   "timestamp": now, "model_version": "V4",
                   "verified_label": label, "adaptive_processed": False})

result = al.run_adaptive_retraining(eval_sample_size=1000)
meta = result.get("metadata", {})
check("candidate created",
      result.get("status") == "candidate_created", result.get("version", ""))
check("candidate NOT auto-activated (registry active still V4)",
      json.loads((ADAPTIVE_MODELS_DIR / "registry.json").read_text())
      ["active_version"] == "V4")
check("runtime still resolves thesis model",
      str(model_runtime._resolve_model_path()) == str(MODEL_PATH))

required = ["version", "parent_version", "created_at",
            "training_sample_count", "feedback_sample_count",
            "verified_feedback_count", "accuracy", "precision", "recall",
            "f1", "roc_auc", "pr_auc", "threshold", "mean_reward", "status"]
model_dir = ADAPTIVE_MODELS_DIR / meta.get("version", "")
check("model version artefact + metadata persisted with required fields",
      (model_dir / "model.joblib").exists()
      and (model_dir / "metadata.json").exists()
      and all(k in meta for k in required))
check("metadata threshold preserved (0.38)", meta.get("threshold") == 0.38)
check("metadata status distinguishes candidate", meta.get("status") == "candidate")
check("V0 never overwritten (no V0 dir in adaptive registry)",
      not (ADAPTIVE_MODELS_DIR / "V0").exists())

with sqlite3.connect(DATABASE_PATH) as c:
    ok = c.execute(
        "SELECT COUNT(*) FROM feedback "
        "WHERE feedback_id IN ('FB-P4-0001','FB-P4-0002') "
        "AND adaptive_processed = 1"
    ).fetchone()[0]
check("successful cycle marked verified feedback processed", ok == 2)

# Manual rejection (status distinguishes rejected candidates)
rej = al.reject_model(meta["version"], rejected_by="phase4-test",
                      reason="test rejection path")
check("manual rejection records status=rejected", rej["status"] == "rejected")
meta_after = json.loads((model_dir / "metadata.json").read_text())
check("metadata.json status updated to rejected",
      meta_after["status"] == "rejected")

# ------------------------------------------------------------------
# 9. Failed training does NOT mark feedback processed
# ------------------------------------------------------------------
save_prediction("P-P4-FAIL", raw, {"prediction": 0}, model_version="V4")
save_feedback({"feedback_id": "FB-P4-FAIL", "prediction_id": "P-P4-FAIL",
               "feedback": "agree", "helpfulness": "low",
               "comment": "single class", "reward": 1,
               "timestamp": now, "model_version": "V4",
               "verified_label": 1, "adaptive_processed": False})
try:
    al.run_adaptive_retraining(eval_sample_size=1000)
    failed_run = "no-error"
except ValueError as e:
    failed_run = str(e)
check("single-class label set rejected with error", failed_run != "no-error",
      failed_run[:60])
with sqlite3.connect(DATABASE_PATH) as c:
    unproc = c.execute(
        "SELECT adaptive_processed FROM feedback WHERE feedback_id='FB-P4-FAIL'"
    ).fetchone()[0]
check("failed cycle left feedback unprocessed", unproc == 0)

# ------------------------------------------------------------------
# 10. Cleanup temp rows; row counts restored
# ------------------------------------------------------------------
with sqlite3.connect(DATABASE_PATH) as c:
    c.execute("DELETE FROM feedback WHERE feedback_id LIKE 'FB-P4-%'")
    c.execute("DELETE FROM predictions WHERE prediction_id LIKE 'P-P4-%'")
check("database row counts restored",
      count_predictions() == before_pred and count_feedback() == before_fb)

# ------------------------------------------------------------------
# 11. Thesis artifacts byte-for-byte unchanged
# ------------------------------------------------------------------
hashes_after = {p: sha256(p) for p in THESIS_ARTIFACTS}
changed = [p.name for p in THESIS_ARTIFACTS
           if hashes_before[p] != hashes_after[p]]
check("no thesis baseline artifact modified", not changed,
      f"changed: {changed}" if changed else f"{len(hashes_after)} files verified")

failed = [n for n, ok in results if not ok]
print("\n" + "=" * 60)
print(f"SUMMARY: {len(results) - len(failed)}/{len(results)} checks passed")
if failed:
    print("FAILED:", failed)
    raise SystemExit(1)
print("ALL CHECKS PASSED")
