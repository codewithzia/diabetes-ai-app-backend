# Phase 5.1 — Experimental Protocol: Human-Feedback-Guided Adaptive Learning Validation

**Thesis:** Development of an Adaptive AI System for Diabetes Prediction through Reinforcement Learning
**Status:** Protocol only — the final experiment has NOT been run.
**Derived from:** Notebook 08 (`08_adaptive_learning.ipynb`), Notebook 09 (`09_rlhf_adaptive_learning.ipynb`), notebooks 05/06/06b artifacts, and the Flask backend (Phases 1–4).

Terminology used throughout: **human-feedback-guided adaptive learning** / **human-feedback-guided incremental retraining**. No deep RLHF claim is made.

---

## A. Existing Notebook 08 methodology (adaptive learning)

Source: `notebooks/08_adaptive_learning.ipynb`

- **Partitioning:** stratified `train_test_split` of the **training set only** (339,166 obs): 80% initial training (271,332) + 20% simulated feedback pool (67,834), `random_state=42`, stratified on `y_train`. The test set is never used for training (asserted disjointness of indices).
- **Feedback generation:** the feedback pool is shuffled (`rng(42)`) and `np.array_split` into **4 batches** (~16,959 each; prevalences 13.36–13.93%). Feedback = the pool's true training labels (simulated future verified labels).
- **Labels:** batch labels are the true BRFSS training labels — used directly as supervised targets.
- **Reward:** not modeled in NB08. (Reward appears only in NB09.)
- **Model factory:** `LogisticRegression(max_iter=1000, class_weight='balanced', solver='liblinear', random_state=42)`.
- **Cumulative training:** V0 trains on initial 80%. V1 = V0-data + batch 1, …, V4 = all 339,166. Each version is retrained **from scratch** (new model object, not incremental weight updates).
- **Evaluation dataset:** full frozen test set `X_test_selected.npz` / `y_test.csv` (84,792 obs), evaluated **after every version** at **threshold 0.50**. NB08 markdown explicitly acknowledges repeated test evaluation for trajectory display only: *"we will not use those test results to decide when/how to retrain."*
- **Metrics:** accuracy, precision, recall, F1, ROC-AUC, PR-AUC (+ confusion matrix for the final model).
- **Version sequence:** V0 (initial) → V1 → V2 → V3 → V4; final model saved as `adaptive_logistic_regression_final.joblib`.
- **Change metrics:** F1/recall/PR-AUC deltas vs V0; final-model confusion matrix was `[[53177, 20092], [2652, 8871]]` at 0.50 (recall ≈ 0.77, consistent with `class_weight='balanced'`).

## B. Existing Notebook 09 methodology (human-feedback-guided adaptive learning)

Source: `notebooks/09_rlhf_adaptive_learning.ipynb` + recorded cell outputs.

- **Base model:** attempts to load a saved model by `rglob` under `data/`; **this lookup failed** in the recorded run, so **NB09 actually trained the base from scratch**: `LogisticRegression(max_iter=2000, class_weight=None, solver='liblinear', random_state=42)` on the full 339,166×49 training matrix. (Confirmed in cell 8 output.)
- **Feedback schema:** `user_id, prediction, prediction_probability, feedback, reward, verified_label, timestamp`.
- **Feedback generation (simulated):** the base model predicts all training rows (threshold 0.50); `feedback = 1 (agree)` iff `prediction == y_train`, else `0 (disagree)`. Recorded distribution: agree 295,103 / disagree 44,063.
- **Reward:** `feedback_to_reward`: agree → **+1.0**, disagree → **−1.0**. Reward is computed over the whole training set and summarized per batch (mean reward, positive/negative rates) — but **reward is never used in training**.
- **Labels:** `verified_label = y_train` (true labels). The methodology note explicitly states: *"User agreement/disagreement is treated as a feedback signal. Verified labels are required for medically grounded model retraining in a real deployment."*
- **Batches:** cell 16 splits `feedback_df` (all 339,166 rows) into 4 batches of ~84,792. **However, the retraining loop (cell 17) does not use those batches** — it re-splits the trailing 20% of training indices (contiguous, NOT shuffled): V0 on first 271,332 rows, then adds contiguous quarters of the remaining 67,834 rows.
- **Cumulative training:** same structure as NB08 — `clone(base_model).fit(X_current ∪ feedback batch)` per version, from scratch each time.
- **Evaluation:** full frozen test set, threshold **0.50**, after each version. Recorded trajectory: accuracy ≈ 0.8696 flat, recall ≈ 0.18 (unweighted LR at 0.50), ROC-AUC ≈ 0.8265 across V0–V4.
- **Metrics:** accuracy, precision, recall, F1, ROC-AUC, PR-AUC, tn/fp/fn/tp.
- **Output artifacts:** `rlhf_feedback_simulation.csv`, `rlhf_reward_summary.csv`, `rlhf_adaptive_learning_results.csv`, `static_vs_adaptive_comparison.csv`, `rlhf_adaptive_logistic_regression_final.joblib` (**this is the backend's active V4 model**), `rlhf_adaptive_learning_metadata.json`.

## C. Recommended Phase 5 experiment design

**Comparison:** frozen thesis baseline **vs** human-feedback-guided adaptive versions.

Two experiment tracks, both faithful to existing artifacts; Track 1 is the thesis-reproducibility anchor, Track 2 validates the backend service.

### Track 1 — Notebook-faithful replication (primary thesis experiment)
Re-run the NB09 protocol verbatim, in a **new** standalone script (do NOT modify NB09):
1. Load `X_train_selected.npz` / `y_train.csv` and frozen test set.
2. Rebuild the NB09 base model: `LogisticRegression(max_iter=2000, class_weight=None, solver='liblinear', random_state=42)` on full training — this reproduces the recorded NB09 run and yields the model equivalent to the deployed V4 artifact.
3. V0 = `clone(base).fit(first 80% of training indices)`; V1–V4 = cumulative contiguous quarters of the trailing 20%; evaluate every version on the **full frozen test set (84,792)**.
4. Report static-vs-adaptive comparison (V0 vs V4) exactly as NB09 cell 18.

### Track 2 — Backend-service validation (system-level)
Use the Phase 3–4 pipeline end-to-end with realistic feedback volume:
1. Seed the database with prediction/feedback records whose `verified_label` values come from the NB09 feedback pool (labels from training data — NOT from agree/disagree, NOT from reward).
2. Call `run_adaptive_retraining(eval_sample_size=None)` (full 84,792 test set) with `baseline_sample_size` set to include the V0-initial-data sample, producing candidates VA-0001…
3. Verify: candidate metadata metrics equal notebook-method metrics; no auto-activation; `adaptive_processed` gating holds.
4. Report candidate-vs-active deltas from the service's `compare_models`.

**Decision deferred (ambiguity, see H/I):** which evaluation threshold(s) to report. Recommended: report at **0.38** (deployed production threshold, fixed by thesis decision) as primary, plus **0.50** (notebook comparability) as a sensitivity column. Do not recalculate the threshold unless the thesis methodology chapter formally adopts 06b's 0.62.

## D. Dataset partitioning (as existing artifacts define it)

| Split | Size | Source artifact | Role |
|---|---|---|---|
| Training pool | 339,166 | `X_train_selected.npz` / `y_train.csv` | initial training + feedback pool |
| — V0 initial | 271,332 (80%) | first/stratified split per notebook | initial model training |
| — Feedback pool | 67,834 (20%) | same | V1–V4 batches (4 × ~16,959 NB08 / contiguous quarters NB09) |
| Validation (06b) | 67,834 | notebook 06/06b (internal to baseline training) | threshold selection history — frozen, do not reuse |
| **Final test** | **84,792** | `X_test_selected.npz` / `y_test.csv` | **evaluation only, never trained on** |

Training columns: 49 selected processed features (from 84, from 18 original predictors) — fixed.

## E. Feedback-generation methodology

Existing (notebook-simulated):
- NB08: feedback = true labels of a held-out 20% training slice; no reward modeling.
- NB09: feedback stream = model predictions over the training pool; `agree` iff prediction == true label; reward = ±1; `verified_label` = true label.

Backend (real deployment):
- Users submit agree/disagree + helpfulness + comment → reward ±1 → stored. **agree/disagree and reward are never labels.**
- Only `verified_label` (trusted/verified source, e.g. clinician-confirmed or the NB09 pool labels for the validation experiment) may supervise retraining.
- `adaptive_processed` flips to 1 only after a candidate is successfully trained, evaluated and persisted.

## F. Model-version sequence

Notebook lineage: **V0** (initial 80%) → **V1** (+batch 1) → **V2** (+batch 2) → **V3** (+batch 3) → **V4** (full 339,166; deployed to the backend).
Backend lineage: `V4` (active, frozen thesis artifact) → `VA-0001`, `VA-0002`, … candidates (registry-gated, manual activation only).
Both sequences: retrain from scratch on cumulative data each version; evaluation on frozen test set after every version.

## G. Evaluation methodology (to reuse unchanged)

Exact function from NB09 cell 9 / NB08 cell 7 (also implemented in `app/services/adaptive_learning.py::evaluate_candidate_model`):
```
accuracy, precision, recall, f1         (at chosen threshold)
roc_auc, pr_auc                          (threshold-free)
confusion matrix (tn, fp, fn, tp)
```
Dataset: full frozen test set, 84,792 samples, read-only.

## H. Leakage risks & ambiguities (explicit)

1. **Test-set reuse for trajectory:** both notebooks evaluate every version on the test set. This is display/evaluation only, no test-driven retraining decisions — but the thesis must state this for the Phase 5 write-up.
2. **NB09 feedback simulation uses in-sample predictions:** agree/disagree was computed from predictions on the same training rows later added as feedback. Because *labels* used in training are the true labels (not the feedback), this does not leak test data, but it makes the simulated agree-rate dependent on training-fit. To be flagged, not silently reused.
3. **NB09 batch discrepancy:** cell 16 creates batches over all 339,166 rows, but cell 17 retrains on the contiguous trailing 20% instead. The recorded V1–V4 results reflect the cell-17 split. Phase 5 must follow **cell 17** for reproducibility and note the discrepancy. *(Ambiguity — do not assume intent.)*
4. **Threshold ambiguity:** NB08/NB09 evaluate at 0.50; notebook 06b selected **0.62** (max validation F1); the deployed system uses **0.38** (recall-oriented; at 0.38 the 06b test row shows recall 0.8694). No notebook explicitly derives 0.38. → report metrics at the deployed 0.38 as primary, 0.50 for notebook comparability, and reference 0.62 as 06b's optimum. Do not silently pick one.
5. **Class weighting ambiguity:** NB08 used `class_weight='balanced'` (recall ≈ 0.77); NB09/thesis final model used `class_weight=None` (recall ≈ 0.18 at 0.50). Static-vs-adaptive deltas depend heavily on which configuration is used. Phase 5 Track 1 follows **NB09** (`class_weight=None`) because NB09 produced the deployed V4 artifact.
6. NB09 loaded no saved model (rglob under `data/` found none) — the deployed V4 model equals a from-scratch fit on the full training set, which is reproducible from the artifacts.

## I. Exact commands/code for the final experiment (NOT yet run)

New file (to be created when Phase 5.2 is approved): `backend/experiments/run_phase5_validation.py` — read-only against the thesis tree, writes results to `backend/experiments/results/`. Skeleton (complete, no invention):

```python
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, average_precision_score, confusion_matrix)

ML = Path(r"D:\MSc\BRFSS_Diabetes_Thesis\data\processed\ml")
OUT = Path("experiments/results"); OUT.mkdir(parents=True, exist_ok=True)
RS = 42

X_train = sparse.load_npz(ML / "X_train_selected.npz")
y_train = pd.read_csv(ML / "y_train.csv").iloc[:, 0]
X_test  = sparse.load_npz(ML / "X_test_selected.npz")
y_test  = pd.read_csv(ML / "y_test.csv").iloc[:, 0]
assert X_test.shape[0] == 84792

def evaluate(model, X, y, threshold):
    p = model.predict_proba(X)[:, 1]
    yhat = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yhat).ravel()
    return dict(accuracy=accuracy_score(y, yhat),
                precision=precision_score(y, yhat, zero_division=0),
                recall=recall_score(y, yhat, zero_division=0),
                f1=f1_score(y, yhat, zero_division=0),
                roc_auc=roc_auc_score(y, p),
                pr_auc=average_precision_score(y, p),
                tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp))

# NB09-faithful: base trained on full training set (class_weight=None)
base = LogisticRegression(max_iter=2000, class_weight=None,
                          solver="liblinear", random_state=RS).fit(X_train, y_train)

# NB09 cell-17 split: contiguous first 80% for V0, trailing 20% in 4 quarters
n = len(y_train)
initial_idx = np.arange(int(n * 0.80))
remaining = np.arange(int(n * 0.80), n)
batches = np.array_split(remaining, 4)

results = []
X_cur = X_train[initial_idx]; y_cur = y_train.iloc[initial_idx].copy()
for version in ["V0", "V1", "V2", "V3", "V4"]:
    if version != "V0":
        b = batches.pop(0)
        X_cur = sparse.vstack([X_cur, X_train[b]])
        y_cur = pd.concat([y_cur, y_train.iloc[b]], ignore_index=True)
    model = clone(base).fit(X_cur, y_cur)
    for thr in (0.38, 0.50):   # deployed threshold + notebook comparability
        results.append({"version": version, "threshold": thr,
                        "training_size": len(y_cur),
                        **evaluate(model, X_test, y_test, thr)})
    # Note: V4 is evaluated from this reproduction; the deployed
    # rlhf_adaptive_logistic_regression_final.joblib is NOT overwritten.

df = pd.DataFrame(results)
df.to_csv(OUT / "phase5_version_trajectory.csv", index=False)
print(df.round(4))
print(df[df.version.isin(["V0","V4"])].to_string())
```

Run command: `python experiments/run_phase5_validation.py`

Track 2 (backend service):
```python
# with backend in TEST MODE (test.db):
#   seed verified feedback from the NB09 pool labels, then:
from app.services import adaptive_learning as al
result = al.run_adaptive_retraining(baseline_sample_size=271332,
                                    eval_sample_size=None)   # full 84,792
# assert no auto-activation; inspect models/adaptive/<VA>/metadata.json
```

---

**Protocol author note:** nothing in this document modifies the backend, thesis artifacts, Notebook 08/09, the model, preprocessing, or the database schema. The final experiment is intentionally not executed here.
