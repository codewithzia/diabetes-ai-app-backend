"""Request-schema validation for prediction inputs.

The prediction route previously accepted any payload: ``features`` was
extracted with ``data.get("features", data)`` (falling back to the raw
request body), and the model input builder silently imputed every
unmappable value. An empty ``{}`` therefore produced a fully-valid
prediction (~0.17 probability) with all 17 missingness indicators hard-
coded — a fabricated result, not a model prediction on real input.

Rules implemented here (documented contract for POST /api/predict):

* ``features`` must be present and be a non-empty JSON object.
* Exactly the 18 known frontend keys are accepted; unknown keys are
  rejected (fail-loud, not silently ignored).
* Every known key must be present (required fields).
* Each value must be either
    - ``null``  -> explicit missing value; the model input then sets the
      corresponding ``<col>_missing`` indicator = 1 and the fitted
      preprocessor imputes the mode. This preserves the notebook-03
      missing-value semantics, so a caller can represent a legitimate
      BRFSS missing/non-response state without inventing a category; or
    - a string contained in ``FEATURE_MAPPINGS[key]``. Unsupported
      category strings (e.g. values representing BRFSS codes the form
      vocabulary cannot express) are REJECTED — they are not mapped to
      a wrong category and are not silently treated as missing.
* Numbers, booleans, arrays, objects and non-finite values are invalid
  types and rejected.

``FeatureValidationError`` is raised before any model inference or
persistence, so invalid requests never create prediction-history rows
or become eligible for adaptive training.
"""

from ..constants import FEATURE_MAPPINGS, FRONTEND_TO_BRFSS

REQUIRED_KEYS = set(FRONTEND_TO_BRFSS)


class FeatureValidationError(ValueError):
    """Invalid prediction input; carries a list of per-field errors."""

    def __init__(self, errors):
        super().__init__("Invalid feature input.")
        self.errors = list(errors)


def validate_features(features) -> dict:
    """Validate the ``features`` payload; raises FeatureValidationError."""
    if not isinstance(features, dict):
        raise FeatureValidationError(
            ["'features' must be a JSON object containing the "
             "18 BRFSS predictor fields."]
        )
    if not features:
        raise FeatureValidationError(
            ["'features' must not be empty; the 18 BRFSS predictor "
             "fields are required."]
        )

    errors = []

    unknown = sorted(set(features) - REQUIRED_KEYS)
    if unknown:
        errors.append(
            "unknown field(s): " + ", ".join(unknown)
        )

    for key in sorted(REQUIRED_KEYS - set(features)):
        errors.append(f"missing required field: '{key}'")

    for key in sorted(REQUIRED_KEYS & set(features)):
        value = features[key]
        if value is None:
            continue  # explicit missing -> missingness indicator
        if not isinstance(value, str) or not value:
            errors.append(
                f"'{key}': expected a category string or null, "
                f"got {type(value).__name__!r}"
            )
            continue
        if value not in FEATURE_MAPPINGS[key]:
            allowed = ", ".join(sorted(FEATURE_MAPPINGS[key]))
            errors.append(
                f"'{key}': unsupported value {value!r} "
                f"(allowed: {allowed}, or null for missing)"
            )

    if errors:
        raise FeatureValidationError(errors)
    return features
