"""Trained model runtime: BRFSS preprocessing + logistic regression inference.

Loads the fitted artifacts from the leakage-safe final experiment
(notebook 05b, results_leakage_safe):
- ColumnTransformer (impute + one-hot, 18 raw + 18 missingness flags -> 85 features)
- Feature selection indices (85 -> 50 selected features)
- Final logistic regression model (50 features)

Missingness flags follow the exact training-time logic
(``{column}__missing`` = 1 iff the raw value is NaN), computed
deterministically row by row; never learned from other rows.
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from ..config import (
    ADAPTIVE_REGISTRY_PATH,
    MODEL_PATH,
    PREPROCESSOR_PATH,
    SELECTED_FEATURES_PATH,
)
from ..constants import FEATURE_MAPPINGS, FRONTEND_TO_BRFSS, BRFSS_TO_FRONTEND

# Original 18 BRFSS predictor columns (training-time order) and the
# deterministic missingness flag column for each one (double underscore).
RAW_COLUMNS = list(FRONTEND_TO_BRFSS.values())
MISSING_FLAG_COLUMNS = [f"{column}__missing" for column in RAW_COLUMNS]

# Expected pipeline dimensions (validated at load time from the artifacts
# in experiment_metadata.json).
EXPECTED_PROCESSED_FEATURES = 85
EXPECTED_SELECTED_FEATURES = 50


def _load_artifacts():
    """Load and validate preprocessor, model and selected features once."""
    for path in (PREPROCESSOR_PATH, MODEL_PATH, SELECTED_FEATURES_PATH):
        if not Path(path).exists():
            raise RuntimeError(
                f"Model artifact is missing: {Path(path).name}. "
                "Check DIABETES_MODEL_DIR / artifact path configuration."
            )

    try:
        preprocessor = joblib.load(PREPROCESSOR_PATH)
        model = joblib.load(MODEL_PATH)
    except Exception as exc:  # corrupted or sklearn-incompatible artifact
        raise RuntimeError(
            f"Failed to load model artifacts (incompatible or corrupted "
            f"joblib files). Retrain or reinstall the scikit-learn version "
            f"used during training. Original error: {type(exc).__name__}"
        ) from exc

    # Preprocessor input contract: 18 raw columns + 18 __missing flags.
    input_columns = list(preprocessor.feature_names_in_)
    expected_inputs = RAW_COLUMNS + MISSING_FLAG_COLUMNS
    if input_columns != expected_inputs:
        raise RuntimeError(
            "Preprocessor input columns do not match the expected 18 BRFSS "
            "columns followed by the 18 __missing flag columns."
        )

    # Preprocessor output must be exactly 85 features with no NaNs.
    probe = pd.DataFrame([{c: np.nan for c in input_columns}], columns=input_columns)
    processed_width = preprocessor.transform(probe).shape[1]
    if processed_width != EXPECTED_PROCESSED_FEATURES:
        raise RuntimeError(
            f"Preprocessor produced {processed_width} features; expected "
            f"{EXPECTED_PROCESSED_FEATURES}."
        )

    selected = pd.read_csv(SELECTED_FEATURES_PATH)
    selected_indices = selected["selected_feature_index"].astype(int).to_numpy()
    selected_names = selected["selected_feature_name"].astype(str).tolist()

    if len(selected_indices) != EXPECTED_SELECTED_FEATURES:
        raise RuntimeError(
            f"Selected features file contains {len(selected_indices)} rows; "
            f"expected {EXPECTED_SELECTED_FEATURES}."
        )
    if len(selected_names) != len(set(selected_names)):
        raise RuntimeError("Selected feature names contain duplicates.")
    if int(selected_indices.max()) >= EXPECTED_PROCESSED_FEATURES:
        raise RuntimeError("Selected feature index exceeds preprocessor width.")

    return preprocessor, model, input_columns, selected_indices, selected_names


def _resolve_model_path():
    """Active model from the adaptive registry, else the thesis model.

    Falls back to MODEL_PATH whenever no registry/activation exists, so
    default behaviour is exactly the frozen thesis model.
    """
    try:
        if ADAPTIVE_REGISTRY_PATH.exists():
            registry = json.loads(
                ADAPTIVE_REGISTRY_PATH.read_text(encoding="utf-8")
            )
            active = Path(registry.get("active_model_path", ""))
            if active.exists():
                return active
    except (OSError, json.JSONDecodeError):
        pass
    return MODEL_PATH


PREPROCESSOR, _BASE_MODEL, INPUT_COLUMNS, SELECTED_INDICES, SELECTED_FEATURE_NAMES = (
    _load_artifacts()
)
MODEL = joblib.load(_resolve_model_path())

if len(SELECTED_INDICES) != MODEL.n_features_in_:
    raise RuntimeError(
        f"Selected features ({len(SELECTED_INDICES)}) do not match "
        f"model input size ({MODEL.n_features_in_})."
    )


def build_model_input(raw_features: dict) -> pd.DataFrame:
    """Build one model input row (36 columns) from frontend feature values.

    Missing keys become NaN and receive a 1 in their ``__missing`` flag,
    exactly as during training. Unsupported categorical values for a
    present key are rejected with ValueError (never silently imputed).
    """
    row = {}
    for key, column in FRONTEND_TO_BRFSS.items():
        value = raw_features.get(key)
        if value is None or value == "":
            row[column] = np.nan
            continue
        mapping = FEATURE_MAPPINGS.get(key, {})
        if value not in mapping:
            raise ValueError(
                f"Unsupported value '{value}' for feature '{key}'. "
                f"Supported values: {sorted(mapping)}."
            )
        row[column] = float(mapping[value])

    # Same deterministic, rowwise missingness flags used at training time.
    for column in RAW_COLUMNS:
        row[f"{column}__missing"] = float(pd.isna(row[column]))

    return pd.DataFrame(
        [{column: row.get(column, np.nan) for column in INPUT_COLUMNS}],
        columns=INPUT_COLUMNS,
    )


def add_missingness_flags(frame: pd.DataFrame) -> pd.DataFrame:
    """Append deterministic 0/1 missingness flags (training-time logic).

    Used by the adaptive-learning service when scoring raw BRFSS frames
    (e.g. the frozen thesis test split).
    """
    result = frame[RAW_COLUMNS].copy()
    for column in RAW_COLUMNS:
        result[f"{column}__missing"] = result[column].isna().astype(np.int8)
    return result[INPUT_COLUMNS]


def _transform_selected(model_input: pd.DataFrame):
    processed = PREPROCESSOR.transform(model_input)
    if processed.shape[1] != EXPECTED_PROCESSED_FEATURES:
        raise RuntimeError(
            f"Preprocessor output width {processed.shape[1]} != "
            f"{EXPECTED_PROCESSED_FEATURES}."
        )
    return processed[:, SELECTED_INDICES]


def predict_probability(model_input: pd.DataFrame) -> float:
    """Return P(diabetes) from the trained logistic regression model."""
    probability = float(MODEL.predict_proba(_transform_selected(model_input))[0, 1])
    if not np.isfinite(probability):
        raise RuntimeError("Model returned a non-finite probability.")
    return probability


def generate_explanations(model_input: pd.DataFrame, raw_features: dict, top_n: int = 8) -> list:
    """Coefficient-based explanation factors from the trained model.

    These are Logistic Regression feature contributions (associations),
    not causal explanations.
    """
    from ..constants import FEATURE_DISPLAY_NAMES, FEATURE_EXPLANATIONS

    selected = _transform_selected(model_input)
    selected_row = selected.toarray()[0] if hasattr(selected, "toarray") else np.asarray(selected)[0]
    coefficients = MODEL.coef_[0]

    contributions_by_column = {}
    for name, value, coefficient in zip(SELECTED_FEATURE_NAMES, selected_row, coefficients):
        if value == 0 or name.endswith("_missing"):
            continue
        if not name.startswith("categorical__"):
            continue  # only one-hot BRFSS columns map back to form fields
        column = name.removeprefix("categorical__").rsplit("_", 1)[0]
        contributions_by_column[column] = (
            contributions_by_column.get(column, 0.0) + float(coefficient) * float(value)
        )

    explanations = []
    for column, contribution in contributions_by_column.items():
        feature = BRFSS_TO_FRONTEND.get(column)
        if feature is None:
            continue

        display_name = FEATURE_DISPLAY_NAMES.get(feature, feature)
        raw_value = raw_features.get(feature)
        if raw_value:
            value_label = str(raw_value).replace("_", " ").capitalize()
            display_name = f"{display_name}: {value_label}"

        explanations.append({
            "feature": feature,
            "display_name": display_name,
            "direction": "increase" if contribution > 0 else "decrease",
            "contribution": round(abs(contribution) * 100, 2),
            "explanation": FEATURE_EXPLANATIONS.get(
                feature, f"Contribution of {feature} to the prediction."
            ),
        })

    explanations.sort(key=lambda item: item["contribution"], reverse=True)
    return explanations[:top_n]
