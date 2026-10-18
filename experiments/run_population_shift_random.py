"""Leakage-controlled state-held-out adaptation with label-blind feedback.

The historical thesis files are read-only inputs. Every run writes to a
fresh timestamped directory below experiments/results/population_shift_random_feedback_fixed_split/.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
import scipy
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from population_shift_random_design import (
    SEED,
    TARGET_STATES,
    make_random_partitions,
    normalize_state,
    sample_feedback_random,
)


THESIS_DIR = Path(r"D:\MSc\BRFSS_Diabetes_Thesis")
EXPERIMENT_DIR = Path(__file__).resolve().parent
REFERENCE_RUN = EXPERIMENT_DIR / "results" / "population_shift" / "run_20261018T064829965426Z_seed42"
OUTPUT_ROOT = EXPERIMENT_DIR / "results" / "population_shift_random_feedback_fixed_split"
INPUT_PATH = THESIS_DIR / "data" / "processed" / "brfss_2021_diabetes_research_base.csv"
RAW_FEATURES = (
    "_AGEG5YR", "_SEX", "_RACE", "_EDUCAG", "_INCOMG1", "_BMI5CAT",
    "_SMOKER3", "_TOTINDA", "_RFHYPE6", "_RFCHOL3", "_MICHD",
    "CVDSTRK3", "CHCKDNY2", "_RFHLTH", "_HLTHPLN", "PERSDOC3",
    "MEDCOST1", "CHECKUP1",
)
MISSING_FEATURES = tuple(c for c in RAW_FEATURES if c != "_SEX")
FEEDBACK_SIZES = (0, 500, 1_000, 2_500, 5_000, 10_000, 20_000, 30_000)
FEEDBACK_REPLICATES = 5
PRIMARY_FEEDBACK_SIZE = 10_000
TOP_K_FEATURES = 49
C_GRID = (0.1, 1.0, 10.0)
BOOTSTRAP_REPLICATES = 1_000


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_model_frame(raw: pd.DataFrame) -> pd.DataFrame:
    """Add the notebook-style missing indicators without using state/labels."""
    frame = raw.loc[:, RAW_FEATURES].copy()
    for column in MISSING_FEATURES:
        frame[f"{column}_missing"] = frame[column].isna().astype("int8")
    return frame


def fit_preprocessor(source_train: pd.DataFrame):
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
    ])
    indicators = [f"{c}_missing" for c in MISSING_FEATURES]
    transformer = ColumnTransformer(
        [("categorical", categorical, list(RAW_FEATURES)),
         ("missing", "passthrough", indicators)],
        sparse_threshold=1.0,
        verbose_feature_names_out=True,
    )
    # The only fit call for preprocessing is on source_train.
    transformed = transformer.fit_transform(source_train)
    return transformer, transformed


def fit_feature_selector(X_source_train, y_source_train):
    # Every encoded column is categorical/binary. The fixed top-49 MI rule is
    # intentionally predeclared and fit only on source training observations.
    selector = SelectKBest(
        score_func=lambda X, y: mutual_info_classif(
            X, y, discrete_features=True, random_state=SEED
        ),
        k=min(TOP_K_FEATURES, X_source_train.shape[1]),
    )
    return selector, selector.fit_transform(X_source_train, y_source_train)


def fit_model(X, y, c_value):
    model = LogisticRegression(
        C=float(c_value), class_weight="balanced", solver="liblinear",
        max_iter=1_000, random_state=SEED,
    )
    model.fit(X, y)
    return model


def choose_threshold(y, probabilities):
    precision, recall, thresholds = precision_recall_curve(y, probabilities)
    if not len(thresholds):
        return 0.5
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-15)
    return float(thresholds[int(np.argmax(f1))])


def evaluate(model, X, y, threshold):
    probabilities = model.predict_proba(X)[:, 1]
    predictions = (probabilities >= threshold).astype(np.int8)
    tn, fp, fn, tp = confusion_matrix(y, predictions, labels=[0, 1]).ravel()
    metrics = {
        "n": int(len(y)),
        "positive": int(np.sum(y)),
        "accuracy": float(accuracy_score(y, predictions)),
        "precision": float(precision_score(y, predictions, zero_division=0)),
        "recall": float(recall_score(y, predictions, zero_division=0)),
        "f1": float(f1_score(y, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, probabilities)),
        "pr_auc": float(average_precision_score(y, probabilities)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }
    return metrics, probabilities, predictions


def paired_bootstrap_pr_delta(y, baseline_p, adaptive_p, seed, repetitions=BOOTSTRAP_REPLICATES):
    """Stratified paired bootstrap CI for adaptive-minus-baseline PR-AUC."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y, dtype=np.int8)
    class_indices = [np.flatnonzero(y == cls) for cls in (0, 1)]
    deltas = np.empty(repetitions, dtype=float)
    for b in range(repetitions):
        sample = np.concatenate([
            rng.choice(indices, size=len(indices), replace=True)
            for indices in class_indices
        ])
        deltas[b] = (
            average_precision_score(y[sample], adaptive_p[sample])
            - average_precision_score(y[sample], baseline_p[sample])
        )
    return {
        "replicates": int(repetitions),
        "delta_pr_auc": float(average_precision_score(y, adaptive_p)
                               - average_precision_score(y, baseline_p)),
        "ci_95_low": float(np.quantile(deltas, 0.025)),
        "ci_95_high": float(np.quantile(deltas, 0.975)),
        "bootstrap_seed": int(seed),
    }


def exact_mcnemar_p(baseline_predictions, adaptive_predictions, y):
    """Two-sided exact McNemar p value from paired disagreement counts."""
    y = np.asarray(y, dtype=np.int8)
    base_error = np.asarray(baseline_predictions) != y
    adapt_error = np.asarray(adaptive_predictions) != y
    b = int(np.sum(base_error & ~adapt_error))
    c = int(np.sum(~base_error & adapt_error))
    n = b + c
    if n == 0:
        p = 1.0
    else:
        tail_k = min(b, c)
        log_terms = [math.lgamma(n + 1) - math.lgamma(k + 1)
                     - math.lgamma(n - k + 1) - n * math.log(2)
                     for k in range(tail_k + 1)]
        peak = max(log_terms)
        lower_tail = math.exp(peak) * sum(math.exp(x - peak) for x in log_terms)
        p = min(1.0, 2.0 * lower_tail)
    return {"baseline_only_corrected": b, "adaptive_only_corrected": c,
            "discordant_pairs": n, "p_value_two_sided_exact": float(p)}


def _assignments_frame(partitions, states):
    assignment = np.empty(len(states), dtype=object)
    assignment[:] = ""
    for split, indices in partitions.items():
        assignment[np.asarray(indices, dtype=int)] = split
    return pd.DataFrame({
        "row_index": np.arange(len(states), dtype=np.int64),
        "state": [normalize_state(s) for s in states],
        "split": assignment,
    })


def run(output_root: Path, seed=SEED):
    if not INPUT_PATH.is_file():
        raise FileNotFoundError(f"Required BRFSS input not found: {INPUT_PATH}")
    reference_provenance_path = REFERENCE_RUN / "provenance.json"
    reference_assignments_path = REFERENCE_RUN / "split_assignments.csv"
    if not reference_provenance_path.is_file() or not reference_assignments_path.is_file():
        raise FileNotFoundError(
            "The completed label-stratified reference run is required to keep "
            "the target test rows identical."
        )
    reference_provenance = json.loads(
        reference_provenance_path.read_text(encoding="utf-8")
    )
    current_input_hash = sha256(INPUT_PATH)
    if current_input_hash != reference_provenance.get("input_sha256"):
        raise ValueError("Input data hash differs from the reference run")
    run_id = datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%S%fZ") + f"_seed{seed}"
    out = output_root / run_id
    # _STATE defines the geographic split only; survey weights are retained
    # as metadata but excluded from predictors and primary unweighted metrics.
    data = pd.read_csv(INPUT_PATH, usecols=["diabetes", "_STATE", *RAW_FEATURES])
    data = data.dropna(subset=["diabetes", "_STATE"]).reset_index(drop=True)
    y = data["diabetes"].astype("int8").to_numpy()
    states = data["_STATE"].map(normalize_state).to_numpy()
    partitions = make_random_partitions(states, y.tolist(), seed=seed)
    assignment_frame = _assignments_frame(partitions, states)
    reference_assignments = pd.read_csv(reference_assignments_path)
    expected_split_rows = {
        name: set(map(int, rows)) for name, rows in
        reference_assignments.groupby("split")["row_index"].agg(list).items()
    }
    current_split_rows = {
        name: set(map(int, rows)) for name, rows in partitions.items()
    }
    if current_split_rows != expected_split_rows:
        raise ValueError("Current split assignments do not match the reference run")

    # State and label columns are omitted from all model frames.
    raw_features = data.loc[:, RAW_FEATURES]
    frames = make_model_frame(raw_features)
    source_train = np.asarray(partitions["source_train"], dtype=int)
    source_val = np.asarray(partitions["source_validation"], dtype=int)
    source_test = np.asarray(partitions["source_test"], dtype=int)
    target_pool = np.asarray(partitions["target_feedback_pool"], dtype=int)
    target_test = np.asarray(partitions["target_test"], dtype=int)

    preprocessor, X_source_train_all = fit_preprocessor(frames.iloc[source_train])
    selector, X_source_train = fit_feature_selector(
        X_source_train_all, y[source_train]
    )

    def transform(indices):
        return selector.transform(preprocessor.transform(frames.iloc[indices]))

    X_source_val = transform(source_val)
    X_source_test = transform(source_test)
    X_target_test = transform(target_test)

    # Model selection and threshold choice use source validation only.
    candidates = []
    for c_value in C_GRID:
        candidate = fit_model(X_source_train, y[source_train], c_value)
        val_probability = candidate.predict_proba(X_source_val)[:, 1]
        candidates.append((average_precision_score(y[source_val], val_probability),
                           c_value, candidate, val_probability))
    candidates.sort(key=lambda item: (-item[0], item[1]))
    validation_ap, chosen_c, baseline_model, val_probability = candidates[0]
    threshold = choose_threshold(y[source_val], val_probability)

    base_source, base_source_p, base_source_hat = evaluate(
        baseline_model, X_source_test, y[source_test], threshold
    )
    base_target, base_target_p, base_target_hat = evaluate(
        baseline_model, X_target_test, y[target_test], threshold
    )
    metric_rows = []
    for domain, metrics in (("source_test", base_source), ("target_test", base_target)):
        metric_rows.append({"feedback_size": 0, "replicate": 0,
                            "domain": domain, **metrics})

    effective_sizes = [n for n in FEEDBACK_SIZES[1:] if n <= len(target_pool)]
    if PRIMARY_FEEDBACK_SIZE not in effective_sizes:
        raise ValueError(
            f"Target feedback pool ({len(target_pool)}) is too small for the "
            f"predeclared primary size ({PRIMARY_FEEDBACK_SIZE})."
        )
    primary = None
    for size in effective_sizes:
        for replicate in range(FEEDBACK_REPLICATES):
            feedback_seed = int(seed + size * 17 + replicate)
            feedback_indices = sample_feedback_random(
                target_pool.tolist(), size, feedback_seed
            )
            X_feedback = transform(np.asarray(feedback_indices, dtype=int))
            X_combined = sparse.vstack([X_source_train, X_feedback], format="csr")
            y_combined = np.concatenate([y[source_train], y[feedback_indices]])
            adaptive = fit_model(X_combined, y_combined, chosen_c)
            for domain, X_eval, y_eval in (
                ("target_test", X_target_test, y[target_test]),
                ("source_test", X_source_test, y[source_test]),
            ):
                metrics, _, _ = evaluate(adaptive, X_eval, y_eval, threshold)
                metric_rows.append({"feedback_size": size,
                                    "replicate": replicate,
                                    "feedback_seed": feedback_seed,
                                    "domain": domain, **metrics})
            if size == PRIMARY_FEEDBACK_SIZE and replicate == 0:
                target_metrics, target_p, target_hat = evaluate(
                    adaptive, X_target_test, y[target_test], threshold
                )
                source_metrics, source_p, source_hat = evaluate(
                    adaptive, X_source_test, y[source_test], threshold
                )
                primary = {
                    "target_test": {
                        "paired_bootstrap_pr_auc": paired_bootstrap_pr_delta(
                            y[target_test], base_target_p, target_p, seed + 701
                        ),
                        "mcnemar": exact_mcnemar_p(
                            base_target_hat, target_hat, y[target_test]
                        ),
                    },
                    "source_test": {
                        "paired_bootstrap_pr_auc": paired_bootstrap_pr_delta(
                            y[source_test], base_source_p, source_p, seed + 702
                        ),
                        "mcnemar": exact_mcnemar_p(
                            base_source_hat, source_hat, y[source_test]
                        ),
                    },
                    "feedback_size": size,
                    "replicate": replicate,
                    "feedback_seed": feedback_seed,
                    "target_metrics": target_metrics,
                    "source_metrics": source_metrics,
                }

    # Create an output folder only after model fitting and evaluation succeed,
    # so failed attempts do not leave folders that look like completed runs.
    out.mkdir(parents=True, exist_ok=False)
    assignment_frame.to_csv(out / "split_assignments.csv", index=False)
    metric_frame = pd.DataFrame(metric_rows)
    metric_frame.to_csv(out / "metrics_by_feedback.csv", index=False)
    aggregate_rows = []
    numeric_metrics = ("accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc")
    for (size, domain), group in metric_frame.groupby(["feedback_size", "domain"]):
        row = {"feedback_size": int(size), "domain": domain,
               "replicates": int(group["replicate"].nunique())}
        for metric in numeric_metrics:
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_sd"] = float(group[metric].std(ddof=1)) if len(group) > 1 else 0.0
        aggregate_rows.append(row)
    pd.DataFrame(aggregate_rows).to_csv(out / "feedback_size_summary.csv", index=False)
    selected_names = preprocessor.get_feature_names_out()[selector.get_support()]
    pd.DataFrame({"processed_feature_index": np.flatnonzero(selector.get_support()),
                  "selected_feature": selected_names}).to_csv(
                      out / "selected_features.csv", index=False
                  )
    summary = {
        "experiment": "BRFSS 2021 state-held-out adaptation with random feedback",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "primary_metric": "unweighted target-test PR-AUC",
        "primary_feedback_size": PRIMARY_FEEDBACK_SIZE,
        "primary_inference_replicate": 0,
        "baseline_target_test_metrics": base_target,
        "baseline_source_test_metrics": base_source,
        "source_validation_best_pr_auc": float(validation_ap),
        "selected_C": float(chosen_c),
        "threshold_selected_on_source_validation_for_maximum_F1": float(threshold),
        "primary_significance": primary,
    }
    (out / "significance.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    split_summary = {
        name: {"n": len(indices),
               "positive": int(np.sum(y[np.asarray(indices, dtype=int)])),
               "states": sorted({states[i] for i in indices}, key=int)}
        for name, indices in partitions.items()
    }
    provenance = {
        "experiment": "leakage-controlled BRFSS population shift",
        "seed": int(seed),
        "input": str(INPUT_PATH),
        "input_sha256": current_input_hash,
        "fixed_split_reference": {
            "run_directory": str(REFERENCE_RUN),
            "target_test_rows_identical": True,
            "target_test_row_count": int(len(partitions["target_test"])),
            "reference_split_assignments_sha256": sha256(reference_assignments_path),
        },
        "code_sha256": {
            "experiment": sha256(Path(__file__).resolve()),
            "split_design": sha256(Path(__file__).with_name("population_shift_random_design.py")),
        },
        "git_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[1],
            check=True, capture_output=True, text=True,
        ).stdout.strip(),
        "package_versions": {
            "python": __import__("platform").python_version(),
            "numpy": np.__version__, "pandas": pd.__version__,
            "scipy": scipy.__version__, "scikit_learn": sklearn.__version__,
        },
        "target_states": list(TARGET_STATES),
        "state_used_only_for_partitioning": True,
        "predictors": list(RAW_FEATURES),
        "label": "diabetes (binary)",
        "splits": split_summary,
        "split_assignment_file": "split_assignments.csv",
        "source_train_fitted_preprocessing": "most-frequent imputation + one-hot encoding; explicit missing flags",
        "source_train_fitted_feature_selection": {
            "method": "mutual information, top 49 processed features",
            "n_selected": int(len(selected_names)),
        },
        "model": "LogisticRegression(class_weight=balanced, solver=liblinear, max_iter=1000)",
        "model_selection": {"C_grid": list(C_GRID),
                            "selected_C": float(chosen_c),
                            "criterion": "source validation PR-AUC"},
        "threshold": {"value": float(threshold),
                      "selection": "maximum source validation F1"},
        "feedback_sizes": [0, *effective_sizes],
        "feedback_sampling": "uniform random without replacement; target labels are not consulted when selecting feedback rows",
        "target_pool_test_assignment": "same outcome-stratified split as reference run; fixed to preserve identical test respondents",
        "feedback_replicates_per_nonzero_size": FEEDBACK_REPLICATES,
        "primary_inference": {
            "size": PRIMARY_FEEDBACK_SIZE,
            "replicate": 0,
            "paired_stratified_bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "paired_test": "exact two-sided McNemar on thresholded errors",
        },
        "metrics": "unweighted; survey weights and state are not model features",
        "artifacts": ["split_assignments.csv", "metrics_by_feedback.csv",
                      "feedback_size_summary.csv",
                      "selected_features.csv", "significance.json"],
        "limitations": [
            "BRFSS 2021 is cross-sectional; feedback labels are retrospective simulated verified labels, not prospective clinician feedback.",
            "Within-target-state feedback and test respondents share geography, so results estimate adaptation after some target-domain labels, not transfer to unseen states after feedback.",
            "Primary metrics are unweighted and do not use the full BRFSS complex-survey variance design.",
        ],
    }
    (out / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=OUTPUT_ROOT,
    )
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    result_dir = run(args.output_root, args.seed)
    print(f"Saved new population-shift experiment outputs to {result_dir}")


if __name__ == "__main__":
    main()
