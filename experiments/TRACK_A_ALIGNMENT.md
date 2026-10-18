# Track A Alignment — Change Log and Reproduction Guide

Branch: `track-a-alignment` (based on `before-resampling` @ `453dffa`).
Thesis repo untouched: `before-resampling` @ `9fdccf9` (all artifacts
byte-identical; see `experiments/baseline_before_track_a/baseline_snapshot.json`).

## What changed and why

| File | Change |
|---|---|
| `app/config.py` | `MODEL_PATH` default switched from the NB09 artifact (`rlhf_adaptive_logistic_regression_final.joblib`, `class_weight=None`, `max_iter=2000`) to the **Track A NB08 artifact** (`adaptive_logistic_regression_final.joblib`, `class_weight='balanced'`, `liblinear`, `max_iter=1000`, `random_state=42`, 49 features). Added `THESIS_TRAIN_*` paths, `ADAPTIVE_EVAL_SOURCE` (default `validation`), baseline/minimum/seed envs, `REVIEWER_API_KEY`. |
| `app/constants.py` | `OPTIMIZED_THRESHOLD` 0.38 -> **0.50** (Track A; the 0.38 value had no documented derivation — `PHASE5_EXPERIMENTAL_PROTOCOL.md` H.4). `MODEL_INFO` records the Track A parameters/track. Hard-coded `PERFORMANCE_METRICS`/`FEEDBACK_METRICS`/`MODEL_COMPARISON` explicitly labelled **demo/illustrative only — not experiment results**. |
| `app/services/model_runtime.py` | `build_model_input` now implements notebook-03 missingness semantics: absent keys and values outside `FEATURE_MAPPINGS` become `NaN` (mode-imputed) **and** the corresponding `<col>_missing` indicator is set to 1, instead of all 17 indicators being hard-coded to 0. Unrepresentable BRFSS categories are therefore recorded as missing rather than silently valid. |
| `app/services/adaptive_learning.py` | Candidate training now **requires baseline data**: `baseline_sample_size` defaults to the dev-partition size (271,332) and must be `> 0`. Baseline rows are drawn deterministically (seed 42) from the **dev partition only** (validation rows never enter candidate training). Added minimum `min_feedback_total`/`min_feedback_per_class` floors (env-configurable, defaults 1000/100), `ConvergenceWarning`-aware **quality gates** (finite/non-zero coefficients, finite in-range non-constant probabilities, feature-count match) executed before persistence and enforced **fail-closed** by `activate_model`, a thread+process lock preventing concurrent cycles, and `eval_source` support (`validation` default — leakage-safe; `test` legacy). |
| `app/routes/feedback.py` | `POST /api/feedback/<id>/verify` now requires header `X-Reviewer-Key` matching the `REVIEWER_API_KEY` env var; fails closed (503 unconfigured, 401 bad/missing key). |
| `app/routes/adaptive.py` | Metrics endpoint labels the constants block as `demo` data; `/api/adaptive/update` returns `busy` (409) on concurrent cycles and `rejected` on quality-gate failure. |
| `app/routes/model.py` | `/api/model/comparison` wraps the hard-coded table as `{data_source: "demo ...", rows: [...]}` instead of returning it as if it were experiment output. |
| `models/adaptive/registry.json` | `active_model_path` -> Track A artifact; `previous_active_model_path` preserved the NB09 path for rollback. |
| `test_data/test_patients.json` | Rewritten to the current 18-field frontend vocabulary (previous file used a legacy schema that silently decoded mostly to NaN). |
| `test_adaptive_service.py`, `test_phase4_integration.py` | Updated for threshold 0.50 and the mandatory-baseline + minimum-floor call signature. |
| `test_track_a_alignment.py` | **New**: 32 checks covering artifact params, threshold, missingness indicators, feature order, API schema, reviewer auth, minimum floors, quality gates, provenance metadata, rollback, concurrency lock, registry integrity. |
| `.env` (local only, gitignored) | `REVIEWER_API_KEY` added for the dev environment. |

## Reproduction commands

```bash
# TESTS (in-process, isolated; sqlite test db)
python test_track_a_alignment.py            # 32 checks
python test_adaptive_service.py             # 26 checks
python test_db_migration.py                 # 21 checks
# HTTP tests need a running server (py app.py):
python test_phase4_integration.py           # 23 checks
python test_adaptive_metrics.py             # 10 checks

# EXPERIMENTS
python experiments/run_track_a_baseline.py  # Experiment A
python experiments/run_track_a_adaptive.py  # Experiment B
python experiments/run_track_a_parity.py    # Experiment C
```

## Regenerated results (all actual runs, sklearn 1.9.0)

### Experiment A — baseline verification (`results/track_a_baseline_metrics.csv`)

Track A reference @ 0.50 on frozen test (84,792): acc 0.7318, prec 0.3063,
rec 0.7699, F1 0.4382, ROC-AUC 0.8268, PR-AUC 0.4219 — **exact match to
recorded NB08 V4** (max abs diff 0.000000). Same model on the 67,834-row
validation holdout: acc 0.7363, rec 0.7730, F1 0.4434, ROC 0.8278, PR 0.4217.

### Experiment B — adaptive trajectory (`results/track_a_adaptive_trajectory.csv`)

`historical_track_a` reproduces `adaptive_learning_results.csv` V0-V4
exactly (max abs diff 0.000000). `improved_devval` evaluates each version
on the held-out validation set and the final model once on the frozen
test set: V4 acc 0.7334, rec 0.7664, F1 0.4387, ROC 0.8268.

### Experiment C — parity (`results/track_a_backend_parity.csv`)

500 sampled training rows through `POST /api/predict`:
183 fully representable -> **100% exact parity** (max |Δp| = 0.0);
317 unrepresentable rows deviate by design (unrepresentable codes now
set `<col>_missing=1`, deviating from the raw-row reference with
documented missing semantics; 26.2% happened to be identical anyway).

## Rollback

`git checkout before-resampling` or restore
`models/adaptive/registry.json` and `MODEL_PATH` to the NB09 artifact —
the snapshot in `experiments/baseline_before_track_a/` records prior state.
