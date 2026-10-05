"""Phase 5.3 - Track 2: backend-service validation of human-feedback-guided
adaptive learning (NB09-style cumulative progression).

Validates the REAL backend pipeline (app/services/adaptive_learning.py)
end-to-end in TEST MODE:

- feedback source: NB09 trailing-20% training pool (67,834 rows), seeded
  as prediction+feedback records in the TEST database only
- verified_label = true training label (supervised signal)
- agree/disagree + reward are simulated (+1/-1) but NEVER used as labels
- per batch k: cumulative quarters 1..k are eligible (earlier batches are
  adaptive_processed=1 and excluded; cumulative rows are re-inserted with
  fresh IDs so the service sees the cumulative set)
- baseline_sample_size=271332, eval_sample_size=None (full 84,792 test set)

Isolation: ADAPTIVE_MODELS_DIR is redirected to experiments/models_track2,
so the backend registry (models/adaptive/registry.json) is untouched.
No candidate is ever activated. Thesis artifacts are hash-verified.

Known, documented difference vs Track 1 (NOT compensated for):
- the backend rebuilds features through the production frontend path
  (build_model_input -> preprocessor -> 49 selected features). The frontend
  cannot express some BRFSS codes (e.g. _SMOKER3 2/3, PERSDOC3 2,
  _INCOMG1 6/7, _RACE 5/7) and always sets missingness indicators to 0,
  so seeded feature vectors deviate from X_train_selected for such rows.
- the service's baseline sample is a deterministic RANDOM 271,332-row
  subset of the training pool, not NB09's contiguous first 80%.

Run:  python experiments/run_phase5_track2.py
"""

import hashlib
import json
import os
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

# Isolation BEFORE importing the app: candidates/registry go to
# experiments/models_track2; backend registry untouched.
EXPERIMENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EXPERIMENT_DIR.parent))  # backend package root
os.environ["ADAPTIVE_MODELS_DIR"] = str(EXPERIMENT_DIR / "models_track2")
os.environ.setdefault("APP_MODE", "test")

import joblib
import numpy as np
import pandas as pd
from scipy import sparse

from app.config import (
    ADAPTIVE_MODELS_DIR, DATABASE_PATH, MODEL_PATH,
    PREPROCESSOR_PATH, SELECTED_FEATURES_PATH, THESIS_DIR,
)
from app.constants import FEATURE_MAPPINGS
from app.database import count_feedback, count_predictions
from app.services import adaptive_learning as al
from app.services import model_runtime

ML = THESIS_DIR / "data" / "processed" / "ml"
OUT = EXPERIMENT_DIR / "results"
OUT.mkdir(parents=True, exist_ok=True)
RESULTS_CSV = OUT / "phase5_backend_trajectory.csv"
PROVENANCE_JSON = OUT / "phase5_backend_provenance.json"

THRESHOLDS = (0.38, 0.50)
BASELINE_SAMPLE = 271332          # NB09 V0 initial size
QUARTERS = 4                      # NB09 trailing-20% quarters
ID_PREFIX = "T2"

results = []


def check(name, condition, detail=""):
    results.append((name, bool(condition)))
    print(f"[{'PASS' if condition else 'FAIL'}] {name}" +
          (f"  ({detail})" if detail else ""))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# 0. Safety snapshots
# ---------------------------------------------------------------------------
ARTIFACTS = [
    ML / "X_train_selected.npz", ML / "y_train.csv",
    ML / "X_test_selected.npz", ML / "y_test.csv",
    SELECTED_FEATURES_PATH, PREPROCESSOR_PATH, MODEL_PATH,
    THESIS_DIR / "notebooks" / "08_adaptive_learning.ipynb",
    THESIS_DIR / "notebooks" / "09_rlhf_adaptive_learning.ipynb",
]
hashes_before = {p: sha256(p) for p in ARTIFACTS}
before_p, before_f = count_predictions(), count_feedback()
production_db = DATABASE_PATH.with_name("production.db")
production_existed_before = production_db.exists()
production_hash_before = sha256(production_db) if production_existed_before else None
backup = EXPERIMENT_DIR / f"test_db_backup_phase5.db"
backup.write_bytes(DATABASE_PATH.read_bytes())
check("TEST MODE database in use", DATABASE_PATH.name == "test.db",
      str(DATABASE_PATH))
check("backend registry isolated to experiments dir",
      str(ADAPTIVE_MODELS_DIR).endswith("models_track2"))

# ---------------------------------------------------------------------------
# 1. Load datasets (read-only) and build the NB09 feedback pool
# ---------------------------------------------------------------------------
X_train = sparse.load_npz(ML / "X_train_selected.npz")
y_train = pd.read_csv(ML / "y_train.csv").iloc[:, 0].astype(int)
X_test = sparse.load_npz(ML / "X_test_selected.npz")
y_test = pd.read_csv(ML / "y_test.csv").iloc[:, 0].astype(int)
X_raw = pd.read_csv(ML / "X_train_raw.csv")

check("training set is 339,166 x 49", X_train.shape == (339166, 49))
check("test set is 84,792 x 49", X_test.shape == (84792, 49))
check("raw features align with labels", len(X_raw) == len(y_train))

n = len(y_train)
initial_n = int(n * 0.80)          # 271,332
pool = np.arange(initial_n, n)     # trailing 20% = NB09 feedback pool
quarters = list(np.array_split(pool, QUARTERS))
quarter_sizes = [len(q) for q in quarters]
check("feedback pool is 67,834 rows", len(pool) == 67834, str(quarter_sizes))
print(f"Feedback quarters (exact): {quarter_sizes}")

# Reverse map: BRFSS code -> a frontend category producing that code
CODE_TO_FE = {
    key: {code: value for value, code in codes.items()}
    for key, codes in FEATURE_MAPPINGS.items()
}
BRFSS_COLS = list(X_raw.columns)
FE_KEYS = list(FEATURE_MAPPINGS.keys())


def raw_row_to_frontend(idx: int) -> dict:
    """Frontend-style feature dict reproducing the raw BRFSS codes
    as far as the production form vocabulary allows."""
    row = X_raw.iloc[idx]
    out = {}
    for col, key in zip(BRFSS_COLS, FE_KEYS):
        value = row[col]
        if pd.isna(value):
            continue  # -> np.nan in build_model_input (documented deviation)
        fe_value = CODE_TO_FE[key].get(int(value))
        if fe_value is not None:
            out[key] = fe_value
    return out


# Fidelity check: how many seeded pool rows transform identically to
# the corresponding X_train_selected rows through the production path
rng = np.random.default_rng(RS := 42)
sample_idx = rng.choice(pool, size=1000, replace=False)
X_ref = X_train[sample_idx]
frames = [al.build_model_input(raw_row_to_frontend(int(i))) for i in sample_idx]
X_via_service = sparse.csr_matrix(
    model_runtime.PREPROCESSOR.transform(pd.concat(frames, ignore_index=True))
    [:, model_runtime.SELECTED_INDICES])
diff_counts = np.asarray((X_via_service - X_ref).power(2).sum(axis=1)).ravel()
exact_rows = int((diff_counts < 1e-12).sum())
fidelity = exact_rows / len(sample_idx)
print(f"Feature-path fidelity sample (1000 pool rows): "
      f"{exact_rows} exact matches ({fidelity * 100:.1f}%)")
check("fidelity measured (deviation documented, not compensated)", True,
      f"{fidelity * 100:.1f}% exact row match")

# ---------------------------------------------------------------------------
# 2. Simulate NB09-style agree/disagree + reward (metadata only)
#    agree iff the deployed V4 model's prediction (at 0.38) matches the
#    true label. NEVER used as the training label.
# ---------------------------------------------------------------------------
v4_model = joblib.load(MODEL_PATH)
p_pool = v4_model.predict_proba(X_train[pool])[:, 1]
pred_pool = (p_pool >= 0.38).astype(int)
y_pool = y_train.iloc[pool].to_numpy()
agree_pool = (pred_pool == y_pool).astype(int)

def pool_row(batch_pos: int):
    """Map position within the 67,834-row pool to (idx,label,agree)."""
    idx = int(pool[batch_pos])
    return idx, int(y_pool[batch_pos]), int(agree_pool[batch_pos])


# ---------------------------------------------------------------------------
# 3. Seed + run cumulative batches through the REAL service
# ---------------------------------------------------------------------------
def seed_cumulative(batch_no: int) -> int:
    """Insert cumulative quarters 1..batch_no with fresh IDs.

    Returns number of rows inserted this round (cumulative count).
    """
    upto = int(np.cumsum([len(q) for q in quarters[:batch_no]])[-1])
    now = datetime.now(timezone.utc).isoformat()
    with sqlite3.connect(DATABASE_PATH) as c:
        preds, fbs = [], []
        # round 1 inserts 0..upto; later rounds insert only with fresh IDs
        # (earlier rows are already adaptive_processed=1 in the DB).
        for j in range(upto):
            idx, label, agree = pool_row(j)
            pid, fid = f"{ID_PREFIX}-{batch_no}-P-{j:06d}", f"{ID_PREFIX}-{batch_no}-F-{j:06d}"
            preds.append((pid, json.dumps(raw_row_to_frontend(idx)),
                          json.dumps({"prediction": int(pred_pool[j])}),
                          now, "V4"))
            fbs.append((fid, pid, "agree" if agree else "disagree",
                        "neutral", "phase5-track2 seed", 1 if agree else -1,
                        now, "V4", label, 0, now))
        c.executemany(
            "INSERT INTO predictions (prediction_id, features, result,"
            " created_at, model_version) VALUES (?,?,?,?,?)", preds)
        c.executemany(
            "INSERT INTO feedback (feedback_id, prediction_id, feedback,"
            " helpfulness, comment, reward, created_at, model_version,"
            " verified_label, adaptive_processed, updated_at)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?)", fbs)
    return upto


check("runtime resolves deployed V4 (no activation can occur here)",
      str(model_runtime._resolve_model_path()) == str(MODEL_PATH))

candidates = []
active_metrics = {}
for thr in THRESHOLDS:
    active_metrics[thr] = al.evaluate_candidate_model(
        model_runtime.MODEL, X_test, y_test, threshold=thr)

for batch_no in range(1, QUARTERS + 1):
    cumulative = seed_cumulative(batch_no)
    expected_quarters = [len(q) for q in quarters[:batch_no]]
    eligible = al.get_eligible_feedback()
    check(f"batch {batch_no}: eligible verified feedback == cumulative "
          f"quarters ({sum(expected_quarters)})",
          len(eligible) == sum(expected_quarters), f"got {len(eligible)}")

    res = al.run_adaptive_retraining(
        baseline_sample_size=BASELINE_SAMPLE, eval_sample_size=None)
    check(f"batch {batch_no}: candidate created",
          res["status"] == "candidate_created", res.get("version", ""))
    meta = res["metadata"]
    version = meta["version"]
    candidates.append((version, meta))

    # safety after each batch
    registry = json.loads((ADAPTIVE_MODELS_DIR / "registry.json").read_text())
    check(f"batch {batch_no}: registry active remains V4",
          registry["active_version"] == "V4")
    check(f"batch {batch_no}: candidate NOT activated",
          meta["status"] == "candidate")
    check(f"batch {batch_no}: threshold in metadata is 0.38",
          meta["threshold"] == 0.38)
    check(f"batch {batch_no}: metadata has all required fields",
          all(k in meta for k in
              ("version", "parent_version", "created_at",
               "training_sample_count", "feedback_sample_count",
               "verified_feedback_count", "accuracy", "precision",
               "recall", "f1", "roc_auc", "pr_auc", "threshold",
               "mean_reward", "status")))
    after = len(al.get_eligible_feedback())
    check(f"batch {batch_no}: adaptive_processed consumed all eligible",
          after == 0)

# ---------------------------------------------------------------------------
# 4. Failure-safety probe: single-class labels must fail, stay unprocessed
# ---------------------------------------------------------------------------
now = datetime.now(timezone.utc).isoformat()
with sqlite3.connect(DATABASE_PATH) as c:
    c.execute("INSERT INTO predictions (prediction_id, features, result,"
              " created_at, model_version) VALUES (?,?,?,?,?)",
              (f"{ID_PREFIX}-FAIL-P", "{}", "{}", now, "V4"))
    c.execute("INSERT INTO feedback (feedback_id, prediction_id, feedback,"
              " helpfulness, comment, reward, created_at, model_version,"
              " verified_label, adaptive_processed, updated_at)"
              " VALUES (?,?,?,?,?,?,?,?,?,?,?)",
              (f"{ID_PREFIX}-FAIL-F", f"{ID_PREFIX}-FAIL-P", "agree",
               "low", "single-class probe", 1, now, "V4", 1, 0, now))
try:
    al.run_adaptive_retraining(baseline_sample_size=1000, eval_sample_size=2000)
    failed_cycle = "no-error"
except (ValueError, Exception) as e:
    failed_cycle = type(e).__name__
with sqlite3.connect(DATABASE_PATH) as c:
    unproc = c.execute("SELECT adaptive_processed FROM feedback WHERE"
                       " feedback_id=?", (f"{ID_PREFIX}-FAIL-F",)).fetchone()[0]
check("failed training leaves feedback unprocessed", unproc == 0,
      failed_cycle)

# ---------------------------------------------------------------------------
# 5. Evaluate every candidate at BOTH thresholds (0.50 read-only from the
#    persisted artifact, same evaluation function) + deltas vs active V4.
# ---------------------------------------------------------------------------
rows = []
for thr in THRESHOLDS:
    m = active_metrics[thr]
    rows.append({"version": "V4 (active)", "threshold": thr,
                 "training_size": 339166, "is_candidate": False, **m})

for version, meta in candidates:
    model = joblib.load(ADAPTIVE_MODELS_DIR / version / "model.joblib")
    for thr in THRESHOLDS:
        m = al.evaluate_candidate_model(model, X_test, y_test, threshold=thr)
        a = active_metrics[thr]
        row = {"version": version, "threshold": thr,
               "training_size": meta["training_sample_count"],
               "verified_feedback_count": meta["verified_feedback_count"],
               "is_candidate": True, **m}
        for k in ("accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"):
            row[f"delta_{k}_vs_V4"] = round(m[k] - a[k], 6)
        rows.append(row)

traj = pd.DataFrame(rows)
traj.round(6).to_csv(RESULTS_CSV, index=False)

for thr in THRESHOLDS:
    print("\n" + "=" * 80)
    print(f"TRACK 2 RESULTS AT THRESHOLD {thr}")
    print("=" * 80)
    cols = ["version", "training_size", "accuracy", "precision", "recall",
            "f1", "roc_auc", "pr_auc", "tn", "fp", "fn", "tp"]
    print(traj[traj["threshold"] == thr][cols].round(4).to_string(index=False))
    print(f"\nDeltas vs active V4 (threshold {thr}):")
    dcols = ["version"] + [c for c in traj.columns if c.startswith("delta_")]
    print(traj[(traj["threshold"] == thr) & (traj["is_candidate"])]
          [dcols].round(6).to_string(index=False))

# ---------------------------------------------------------------------------
# 6. Integrity + cleanup
# ---------------------------------------------------------------------------
changed = [p.name for p in ARTIFACTS if sha256(p) != hashes_before[p]]
check("thesis artifacts byte-identical", not changed,
      f"changed: {changed}" if changed else f"{len(ARTIFACTS)} files")

production_after = production_db.exists()
production_ok = (production_after == production_existed_before) and (
    not production_after or sha256(production_db) == production_hash_before)
check("production database untouched", production_ok)

versions = [v for v, _ in candidates]
nums = [int(v.removeprefix("VA-")) for v in versions]
check("candidate versions unique + sequential",
      len(set(versions)) == QUARTERS and nums == sorted(nums) and
      nums == list(range(nums[0], nums[0] + QUARTERS)), str(versions))

with sqlite3.connect(DATABASE_PATH) as c:
    c.execute("DELETE FROM feedback WHERE feedback_id LIKE ?", (f"{ID_PREFIX}-%",))
    c.execute("DELETE FROM predictions WHERE prediction_id LIKE ?", (f"{ID_PREFIX}-%",))
check("test database cleaned (counts restored)",
      count_predictions() == before_p and count_feedback() == before_f,
      f"before=({before_p},{before_f}) after=({count_predictions()},{count_feedback()})")

# ---------------------------------------------------------------------------
# 7. Provenance
# ---------------------------------------------------------------------------
provenance = {
    "experiment": "Phase 5.3 Track 2 - backend-service validation of "
                  "human-feedback-guided adaptive learning",
    "created_at": datetime.now(timezone.utc).isoformat(),
    "mode": "test (database/test.db, seeded rows cleaned afterward)",
    "registry_isolation": str(ADAPTIVE_MODELS_DIR),
    "activation": "none - all candidates remain status=candidate",
    "feedback_source": "NB09 trailing-20% pool (rows 271,332..339,166 of "
                       "the thesis training set), 4 contiguous quarters",
    "label_rule": "verified_label = true training label; agree/disagree "
                  "and reward simulated (+1/-1) and never used as labels",
    "baseline_sample_size": BASELINE_SAMPLE,
    "eval_sample_size": None,
    "evaluation_rows": int(len(y_test)),
    "thresholds": list(THRESHOLDS),
    "active_model": "rlhf_adaptive_logistic_regression_final.joblib (V4)",
    "candidate_versions": versions,
    "quarter_sizes": quarter_sizes,
    "feature_path_fidelity_exact_match_share": round(fidelity, 4),
    "known_track2_vs_track1_differences": [
        "features rebuilt via production frontend path; unmapped BRFSS "
        "codes (_SMOKER3 2/3, PERSDOC3 2, _INCOMG1 6/7, _RACE 5/7) become "
        "NaN and missingness indicators are always 0",
        "baseline sample is a deterministic random 271,332-row subset, "
        "not NB09's contiguous first 80%",
    ],
    "outputs": [str(RESULTS_CSV), str(PROVENANCE_JSON)],
}
PROVENANCE_JSON.write_text(json.dumps(provenance, indent=2), encoding="utf-8")

failed = [n for n, ok in results if not ok]
print("\n" + "=" * 80)
print(f"SUMMARY: {len(results) - len(failed)}/{len(results)} checks passed")
if failed:
    print("FAILED:", failed)
    sys.exit(1)
print("ALL CHECKS PASSED - results reported raw, no success/failure call.")
