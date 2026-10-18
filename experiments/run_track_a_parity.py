"""Experiment C - Backend service integration / notebook parity.

Compares the backend prediction path (Flask test client, full stack
including persistence to the TEST sqlite DB) against the notebook-side
pipeline (raw BRFSS row -> preprocessor -> selected 49 -> LR proba)
on identical inputs, plus negative-input handling.

Method
------
* Samples rows from the frozen test set, reverses BRFSS codes to the
  frontend vocabulary where representable, and submits each through
  POST /api/predict via Flask's test client.
* Reference probability = Track A model on the notebook-constructed row
  (raw codes -> preprocessor -> selected features).
* For rows whose codes the frontend vocabulary CANNOT represent
  (e.g. _SMOKER3=2/3, PERSDOC3=2, _INCOMG1=6/7, _RACE=5/7), the backend
  now sets the corresponding <col>_missing indicator = 1 and imputes the
  mode - exactly notebook-03 missing semantics. Parity is reported
  separately for representable vs unrepresentable rows.
* Tolerance: |backend_p - notebook_p| <= 1e-9 counts as parity
  (responses round probability to 4dp; rounded-vs-raw delta is reported
  separately with tolerance 5e-5).

Run:  python experiments/run_track_a_parity.py
"""

import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# Environment BEFORE importing the app package.
os.environ["APP_MODE"] = "test"
os.environ["DB_ENGINE"] = "sqlite"
os.environ["REVIEWER_API_KEY"] = "parity-test-key"
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from scipy import sparse

from app import create_app
from app.config import MODEL_PATH, PREPROCESSOR_PATH, SELECTED_FEATURES_PATH
from app.constants import (
    FEATURE_MAPPINGS, FRONTEND_TO_BRFSS, OPTIMIZED_THRESHOLD,
)
from app.database import _connect
from app.services import model_runtime as mr

RS = 42
N_SAMPLE = 500
THESIS = Path(r"D:\MSc\BRFSS_Diabetes_Thesis")
ML = THESIS / "data" / "processed" / "ml"
OUT = Path(__file__).resolve().parent / "results"
OUT.mkdir(parents=True, exist_ok=True)

RAW_PATH = ML / "X_train_raw.csv"
X_SEL_PATH = ML / "X_train_selected.npz"
Y_PATH = ML / "y_train.csv"

CODE_TO_FE = {
    key: {int(v): k for k, v in mapping.items()}
    for key, mapping in FEATURE_MAPPINGS.items()
}


def raw_row_to_frontend(row) -> tuple[dict, list[str]]:
    """BRFSS coded row -> frontend payload; returns unrepresentable keys."""
    out, unrepr = {}, []
    for key, column in FRONTEND_TO_BRFSS.items():
        val = row[column]
        if pd.isna(val):
            out[key] = None
            continue
        fe = CODE_TO_FE.get(key, {}).get(int(val))
        if fe is None:
            unrepr.append(key)
        out[key] = fe
    return out, unrepr


def notebook_pipeline_proba(raw_df_row) -> float:
    """Notebook-side path: raw row -> preprocessor -> selected -> proba."""
    Xp = mr.PREPROCESSOR.transform(raw_df_row)
    Xs = Xp[:, mr.SELECTED_INDICES]
    return float(mr.MODEL.predict_proba(Xs)[0, 1])


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    app = create_app()
    client = app.test_client()

    raw = pd.read_csv(RAW_PATH)
    sel_cols = list(mr.PREPROCESSOR.feature_names_in_)
    missing_cols = [c for c in sel_cols if c.endswith("_missing")]

    rng = np.random.default_rng(RS)
    sample_idx = np.sort(rng.choice(len(raw), size=N_SAMPLE, replace=False))
    y_ref = pd.read_csv(Y_PATH).iloc[:, 0].astype(int)

    rows = []
    unrepr_count = 0
    backend_ids = []
    print(f"Sampling {N_SAMPLE} raw training rows through the live API...")
    for i, idx in enumerate(sample_idx):
        row = raw.iloc[idx]
        payload, unrepr = raw_row_to_frontend(row)
        if unrepr:
            unrepr_count += 1

        # Backend (full stack: route -> service -> runtime -> DB persist)
        r = client.post("/api/predict", json={"features": payload})
        status = r.status_code
        body = r.get_json() or {}
        if status == 200:
            backend_ids.append(body["prediction_id"])

        # Reference: identical row through the notebook-side path.
        # X_train_raw.csv holds only the 18 raw predictors; the 17
        # <col>_missing indicators are derived from raw NaN exactly as
        # in notebook 03 (before imputation).
        ref_row = {c: row[c] for c in FRONTEND_TO_BRFSS.values()}
        for c in missing_cols:
            base = c[: -len("_missing")]
            ref_row[c] = 1.0 if pd.isna(row.get(base)) else 0.0
        df_row = pd.DataFrame([ref_row], columns=sel_cols)
        ref_p = notebook_pipeline_proba(df_row)

        # Backend-internal input row (missing flags as built)
        mi = mr.build_model_input(
            {k: v for k, v in payload.items()})
        n_missing = int((mi[missing_cols].iloc[0] == 1.0).sum())
        backend_p_raw = mr.predict_probability(mi)

        rows.append({
            "sample": i, "train_row_idx": int(idx),
            "n_unrepresentable_fields": len(unrepr),
            "unrepresentable_fields": "|".join(unrepr),
            "n_missing_flags_set": n_missing,
            "http_status": status,
            "api_probability": body.get("probability"),
            "api_prediction": body.get("prediction"),
            "api_risk_category": body.get("risk_category"),
            "backend_inprocess_probability": round(backend_p_raw, 10),
            "notebook_probability": round(ref_p, 10),
            "abs_diff_inprocess": abs(backend_p_raw - ref_p),
            "y_true": int(y_ref.iloc[idx]),
        })

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "track_a_backend_parity.csv", index=False)

    ok = df[df.http_status == 200]
    repr_ok = ok[ok.n_unrepresentable_fields == 0]
    unrepr_ok = ok[ok.n_unrepresentable_fields > 0]

    def parity(sub, tol):
        if len(sub) == 0:
            return float("nan"), float("nan")
        d = (sub.backend_inprocess_probability - sub.notebook_probability).abs()
        return float(d.max()), float((d <= tol).mean())

    mx_r, frac_r = parity(repr_ok, 1e-9)
    mx_u, frac_u = parity(unrepr_ok, 1e-9)
    # API rounds probability to 4dp -> separate tolerance
    api_diff = (ok.api_probability - ok.notebook_probability).abs()

    print(f"\nRepresentable rows ({len(repr_ok)}): "
          f"max|d|={mx_r:.2e} parity={frac_r*100:.1f}%")
    print(f"Unrepresentable rows ({len(unrepr_ok)}): "
          f"max|d|={mx_u:.2e} parity={frac_u*100:.1f}%")
    print(f"API-level (4dp-rounded p): max|d|={api_diff.max():.2e}")

    # ----------------- negative input tests -----------------
    neg = {}
    r = client.post("/api/test/predict", json={"test_case_id": "INVALID"})
    neg["unknown_test_case"] = r.status_code
    r = client.post("/api/predict", data="{bad json",
                    content_type="application/json")
    neg["malformed_json"] = r.status_code
    r = client.post("/api/predict", json={"features": {}})
    neg["empty_features"] = (r.status_code,
                             mr.build_model_input({})[
                                 [c for c in missing_cols]].iloc[0].sum())
    r = client.post("/api/predict",
                    json={"features": {"smoking": "zzz"}})
    neg["unknown_value_only"] = r.status_code
    # verify auth (covered in unit tests; record again here)
    r = client.post("/api/feedback/FB-X/verify",
                    json={"outcome": "diabetes"})
    neg["verify_no_key"] = r.status_code
    r = client.post("/api/feedback/FB-X/verify",
                    json={"outcome": "diabetes"},
                    headers={"X-Reviewer-Key": "parity-test-key"})
    neg["verify_with_key_unknown_id"] = r.status_code

    print("Negative tests:", neg)

    # Cleanup persisted API rows
    with _connect() as c:
        for pid in backend_ids:
            c.execute("DELETE FROM predictions WHERE prediction_id=?", (pid,))
    print(f"Cleaned {len(backend_ids)} persisted API prediction rows.")

    summary = {
        "experiment": "Experiment C - backend/notebook prediction parity",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "n_samples": int(len(df)),
        "n_http_200": int(len(ok)),
        "n_unrepresentable_rows": int(unrepr_count),
        "representable_parity": {
            "tolerance": 1e-9,
            "max_abs_diff": mx_r,
            "fraction_within": frac_r,
        },
        "unrepresentable_rows": {
            "tolerance": 1e-9,
            "max_abs_diff": mx_u,
            "fraction_within": frac_u,
            "note": "backend sets <col>_missing=1 for unrepresentable "
                    "fields; deviations vs raw-row notebook path are "
                    "expected and intentional (missing semantics)",
        },
        "api_rounded_parity": {
            "tolerance": 5e-5,
            "max_abs_diff": float(api_diff.max()) if len(ok) else None,
        },
        "negative_tests": neg,
        "threshold": OPTIMIZED_THRESHOLD,
        "artifact_sha256": {
            "model": sha256(MODEL_PATH),
            "preprocessor": sha256(PREPROCESSOR_PATH),
            "selected_features": sha256(SELECTED_FEATURES_PATH),
        },
        "outputs": [str(OUT / "track_a_backend_parity.csv")],
    }
    (OUT / "track_a_backend_parity_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nSaved: {OUT / 'track_a_backend_parity.csv'}")
    print(f"Saved: {OUT / 'track_a_backend_parity_summary.json'}")


if __name__ == "__main__":
    main()
