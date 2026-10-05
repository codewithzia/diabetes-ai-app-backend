"""Trained model runtime: BRFSS preprocessing + logistic regression inference.

Loads the fitted artifacts produced by the research notebooks:
- ColumnTransformer (impute + one-hot encode, 35 -> 84 features)
- Feature selection indices (84 -> 49 selected features)
- Final logistic regression model (49 features)
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

# Missingness indicator columns (every BRFSS input except _SEX)
MISSING_INDICATOR_COLUMNS = [
    f"{column}_missing"
    for column in FRONTEND_TO_BRFSS.values()
    if column != "_SEX"
]

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


PREPROCESSOR = joblib.load(PREPROCESSOR_PATH)
MODEL = joblib.load(_resolve_model_path())

_selected_features = pd.read_csv(SELECTED_FEATURES_PATH)
SELECTED_INDICES = (
    _selected_features["feature_index_in_processed_matrix"]
    .astype(int)
    .to_numpy()
)
SELECTED_FEATURE_NAMES = _selected_features["selected_feature_name"].tolist()

INPUT_COLUMNS = list(PREPROCESSOR.feature_names_in_)

assert len(SELECTED_INDICES) == MODEL.n_features_in_, (
    f"Selected features ({len(SELECTED_INDICES)}) do not match "
    f"model input size ({MODEL.n_features_in_})."
)


def build_model_input(raw_features: dict) -> pd.DataFrame:
    """Build one model input row (35 columns) from frontend feature values."""
    row = {}
    for key, column in FRONTEND_TO_BRFSS.items():
        code = FEATURE_MAPPINGS.get(key, {}).get(raw_features.get(key))
        row[column] = float(code) if code is not None else np.nan

    # The form requires every field, so no value is missing.
    for column in MISSING_INDICATOR_COLUMNS:
        row[column] = 0.0

    return pd.DataFrame(
        [{column: row.get(column, np.nan) for column in INPUT_COLUMNS}],
        columns=INPUT_COLUMNS,
    )


def _transform_selected(model_input: pd.DataFrame):
    processed = PREPROCESSOR.transform(model_input)
    return processed[:, SELECTED_INDICES]


def predict_probability(model_input: pd.DataFrame) -> float:
    """Return P(diabetes) from the trained logistic regression model."""
    return float(MODEL.predict_proba(_transform_selected(model_input))[0, 1])


def generate_explanations(model_input: pd.DataFrame, raw_features: dict, top_n: int = 8) -> list:
    """Coefficient-based explanation factors from the trained model."""
    from ..constants import FEATURE_DISPLAY_NAMES, FEATURE_EXPLANATIONS

    selected = _transform_selected(model_input)
    selected_row = selected.toarray()[0] if hasattr(selected, "toarray") else np.asarray(selected)[0]
    coefficients = MODEL.coef_[0]

    contributions_by_column = {}
    for name, value, coefficient in zip(SELECTED_FEATURE_NAMES, selected_row, coefficients):
        if value == 0 or name.endswith("_missing"):
            continue
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
