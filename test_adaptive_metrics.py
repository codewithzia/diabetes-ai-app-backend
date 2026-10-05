"""Test: /api/adaptive/metrics reports ACTUAL database counts.

Requires a running backend in TEST MODE (python app.py).
Inserts one temp prediction + one temp feedback, verifies the endpoint
reflects the increment, then removes only those temp rows.

Run:  python test_adaptive_metrics.py
"""

import json
import sqlite3
from datetime import datetime, timezone

import requests

from app.config import DATABASE_PATH
from app.database import count_feedback, count_predictions

BASE = "http://127.0.0.1:5000"
results = []


def check(name, condition, detail=""):
    results.append((name, bool(condition)))
    print(f"[{'PASS' if condition else 'FAIL'}] {name}" +
          (f"  ({detail})" if detail else ""))


before_p, before_f = count_predictions(), count_feedback()

r = requests.get(f"{BASE}/api/adaptive/metrics")
check("endpoint returns 200", r.status_code == 200)
data = r.json()

check("response has baseline_performance / database / model sections",
      all(k in data for k in ("baseline_performance", "database", "model")))

db = data["database"]
check("no legacy simulated offsets in response",
      "total_feedback" in db and "feedback_observations" not in json.dumps(db)
      and "50000" not in json.dumps(data["database"]),
      "no MODEL_INFO feedback_observations / +50000 padding")
check("database counts match actual persisted counts",
      db["total_predictions"] == before_p and db["total_feedback"] == before_f,
      f"api={db}, db=({before_p}, {before_f})")
check("mode reported", db.get("mode") == "test")
check("model section has active_version + last_update",
      "active_version" in data["model"] and "last_update" in data["model"])
check("baseline metrics preserved (static thesis results)",
      isinstance(data["baseline_performance"], list)
      and len(data["baseline_performance"]) == 5)

# --- Live-change proof: insert temp rows, expect +1 counts ---
now = datetime.now(timezone.utc).isoformat()
with sqlite3.connect(DATABASE_PATH) as c:
    c.execute(
        "INSERT INTO predictions (prediction_id, features, result, created_at,"
        " model_version) VALUES (?,?,?,?,?)",
        ("P-MET-TMP", "{}", "{}", now, "V4"))
    c.execute(
        "INSERT INTO feedback (feedback_id, prediction_id, feedback,"
        " helpfulness, comment, reward, created_at, model_version,"
        " verified_label, adaptive_processed, updated_at)"
        " VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        ("FB-MET-TMP", "P-MET-TMP", "agree", "low", "", 1, now, "V4",
         None, 0, now))

data2 = requests.get(f"{BASE}/api/adaptive/metrics").json()["database"]
check("endpoint reflects temp inserts (+1 each)",
      data2["total_predictions"] == before_p + 1
      and data2["total_feedback"] == before_f + 1,
      f"api={data2}")

with sqlite3.connect(DATABASE_PATH) as c:
    c.execute("DELETE FROM feedback WHERE feedback_id='FB-MET-TMP'")
    c.execute("DELETE FROM predictions WHERE prediction_id='P-MET-TMP'")

data3 = requests.get(f"{BASE}/api/adaptive/metrics").json()["database"]
check("counts restored after cleanup and endpoint reflects it",
      data3["total_predictions"] == before_p
      and data3["total_feedback"] == before_f)
check("row counts restored in database",
      count_predictions() == before_p and count_feedback() == before_f)


failed = [n for n, ok in results if not ok]
print("\n" + "=" * 60)
print(f"SUMMARY: {len(results) - len(failed)}/{len(results)} checks passed")
if failed:
    print("FAILED:", failed)
    raise SystemExit(1)
print("ALL CHECKS PASSED")
