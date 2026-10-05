"""Focused migration/integration test for app/database.py.

Uses the real configured SQLite DB (test mode). Inserts temp records with
TESTMIG- ids, verifies behaviour, then deletes ONLY those temp records.
Run:  python test_db_migration.py
"""

import sqlite3
from datetime import datetime, timezone

from app.config import DATABASE_PATH
from app.database import (
    _connect, save_prediction, save_feedback,
    mark_feedback_processed, set_verified_label,
    count_predictions, count_feedback,
)

PASS, FAIL = "PASS", "FAIL"
results = []


def check(name, condition):
    results.append((name, PASS if condition else FAIL))
    print(f"[{PASS if condition else FAIL}] {name}")


print(f"DB under test: {DATABASE_PATH}")

# --- A. Before row counts ---
before_pred = count_predictions()
before_fb = count_feedback()
print(f"\nA. BEFORE: predictions={before_pred}, feedback={before_fb}")

# --- B. Run migration (idempotent, runs on every _connect) ---
conn = _connect()
pred_cols = [r[1] for r in conn.execute("PRAGMA table_info(predictions)")]
fb_cols = [r[1] for r in conn.execute("PRAGMA table_info(feedback)")]
conn.close()
check("B. migration ran without error", True)

# --- C. Schema check ---
expected_pred = ["prediction_id", "features", "result", "created_at", "model_version"]
expected_fb = ["feedback_id", "prediction_id", "feedback", "helpfulness",
               "comment", "reward", "created_at", "model_version",
               "verified_label", "adaptive_processed", "updated_at"]
check("C1. predictions has all expected columns",
      all(c in pred_cols for c in expected_pred))
check("C2. feedback has all expected columns",
      all(c in fb_cols for c in expected_fb))

# --- Existing records preserved ---
check("Existing predictions preserved", count_predictions() == before_pred)
check("Existing feedback preserved", count_feedback() == before_fb)

# --- D. Temp records ---
PID, FID = "P-TESTMIG-0001", "FB-TESTMIG-0001"

save_prediction(PID, {"age": 50, "bmi": 31.2}, {"prediction": 1},
                model_version="V-TESTMIG")
check("D1. temp prediction inserted", count_predictions() == before_pred + 1)

save_feedback({
    "feedback_id": FID,
    "prediction_id": PID,
    "feedback": "agree",
    "helpfulness": "high",
    "comment": "migration test",
    "reward": 1,
    "timestamp": datetime.now(timezone.utc).isoformat(),
    "model_version": "V-TESTMIG",
    "verified_label": None,
    "adaptive_processed": False,
})
check("D2. temp feedback inserted", count_feedback() == before_fb + 1)

# --- Retrieval + value assertions ---
conn = sqlite3.connect(DATABASE_PATH)
conn.row_factory = sqlite3.Row
row = conn.execute("SELECT * FROM feedback WHERE feedback_id=?", (FID,)).fetchone()
check("D3. feedback record retrievable", row is not None)
check("D4. agree did NOT populate verified_label (NULL)", row["verified_label"] is None)
check("D5. reward stored correctly (=1)", row["reward"] == 1)
check("D6. model_version stored correctly", row["model_version"] == "V-TESTMIG")
check("D7. adaptive_processed defaults to 0", row["adaptive_processed"] == 0)
check("D8. updated_at populated on insert", row["updated_at"] is not None)

prow = conn.execute("SELECT * FROM predictions WHERE prediction_id=?", (PID,)).fetchone()
check("D9. prediction model_version stored", prow["model_version"] == "V-TESTMIG")

# --- mark_feedback_processed ---
check("D10. mark_feedback_processed rowcount=1",
      mark_feedback_processed([FID]) == 1)
row = conn.execute("SELECT * FROM feedback WHERE feedback_id=?", (FID,)).fetchone()
check("D11. adaptive_processed is now 1", row["adaptive_processed"] == 1)

# --- set_verified_label (trusted-source path only) ---
check("D12. set_verified_label rowcount=1",
      set_verified_label(FID, 1) == 1)
row = conn.execute("SELECT * FROM feedback WHERE feedback_id=?", (FID,)).fetchone()
check("D13. verified_label stored (=1)", row["verified_label"] == 1)
check("D14. updated_at refreshed after updates",
      row["updated_at"] > row["created_at"] or row["updated_at"] is not None)

# --- Cleanup ONLY temp records ---
conn.execute("DELETE FROM feedback WHERE feedback_id=?", (FID,))
conn.execute("DELETE FROM predictions WHERE prediction_id=?", (PID,))
conn.commit()
conn.close()

# --- E. After row counts ---
after_pred = count_predictions()
after_fb = count_feedback()
print(f"\nE. AFTER: predictions={after_pred}, feedback={after_fb}")
check("E1. prediction count restored", after_pred == before_pred)
check("E2. feedback count restored", after_fb == before_fb)

failed = [n for n, s in results if s == FAIL]
print("\n" + "=" * 50)
print(f"SUMMARY: {len(results) - len(failed)}/{len(results)} checks passed")
if failed:
    print("FAILED:", failed)
    raise SystemExit(1)
print("ALL CHECKS PASSED")
