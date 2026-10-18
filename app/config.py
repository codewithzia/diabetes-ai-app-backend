"""Application configuration."""

import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent

# Load variables from backend/.env into the environment (system env vars win).
load_dotenv(BASE_DIR / ".env")

ALLOWED_MODES = {"test", "production"}

APP_MODE = os.getenv("APP_MODE", "test").lower()

# "prod" is accepted as an alias for "production" (e.g. APP_MODE=prod).
if APP_MODE == "prod":
    APP_MODE = "production"

if APP_MODE not in ALLOWED_MODES:
    raise ValueError(
        f"Invalid APP_MODE: {APP_MODE}. "
        f"Allowed values: {ALLOWED_MODES}"
    )

CORS_ORIGINS = ["http://localhost:4200", "http://127.0.0.1:4200"]

TEST_DATA_PATH = BASE_DIR / "test_data" / "test_patients.json"

# Separate database per mode so test and production data stay isolated:
#   test mode       -> database/test.db           (SQLite engine)
#   production mode -> database/production.db     (SQLite engine)
#   test mode       -> TEST_DB_NAME               (MySQL engine)
#   production mode -> PROD_DB_NAME               (MySQL engine)
DATABASE_DIR = BASE_DIR / "database"
DATABASE_PATH = DATABASE_DIR / f"{APP_MODE}.db"

# ---------------------------------------------------------------------------
# Database engine selection.
#
#   DB_ENGINE=mysql   -> MySQL (primary deployment)
#   DB_ENGINE=sqlite  -> original SQLite files (rollback path only)
#
# MySQL credentials/hosts are NEVER hard-coded; they come from environment
# variables (see .env.example). The database NAME is mapped explicitly by
# APP_MODE — TEST mode can never silently fall back to the production
# database: if the required variable is missing the app fails at startup.
# ---------------------------------------------------------------------------
ALLOWED_DB_ENGINES = {"mysql", "sqlite"}

DB_ENGINE = os.getenv("DB_ENGINE", "mysql").lower()
if DB_ENGINE not in ALLOWED_DB_ENGINES:
    raise ValueError(
        f"Invalid DB_ENGINE: {DB_ENGINE}. "
        f"Allowed values: {ALLOWED_DB_ENGINES}"
    )

if DB_ENGINE == "mysql":
    _required_mysql_vars = ["DB_HOST", "DB_USER", "TEST_DB_NAME", "PROD_DB_NAME"]
    _missing = [v for v in _required_mysql_vars if not os.getenv(v)]
    if _missing:
        raise RuntimeError(
            "DB_ENGINE=mysql but required environment variables are "
            f"missing: {', '.join(_missing)}. See .env.example."
        )
    DB_HOST = os.getenv("DB_HOST")
    DB_PORT = int(os.getenv("DB_PORT", "3306"))
    DB_USER = os.getenv("DB_USER")
    # An empty password is allowed (e.g. local WAMP root) but the variable
    # itself is still read only from the environment.
    DB_PASSWORD = os.getenv("DB_PASSWORD", "")
    TEST_DB_NAME = os.getenv("TEST_DB_NAME")
    PROD_DB_NAME = os.getenv("PROD_DB_NAME")

    # Explicit environment mapping — never silently falls back.
    MYSQL_DB_NAME = TEST_DB_NAME if APP_MODE == "test" else PROD_DB_NAME

# Trained model artifacts. Override via environment variables to change
# the location without touching code, e.g.:
#   $env:PREPROCESSOR_PATH = "D:\path\to\brfss_preprocessor_final.joblib"
#   $env:MODEL_PATH        = "D:\path\to\model.joblib"
THESIS_DIR = Path(os.getenv("THESIS_DIR", r"D:\MSc\BRFSS_Diabetes_Thesis"))

PREPROCESSOR_PATH = Path(os.getenv(
    "PREPROCESSOR_PATH",
    THESIS_DIR / "data" / "processed" / "ml" / "brfss_preprocessor_final.joblib",
))

# Track A reference model (notebook 08, before-resampling branch):
#   LogisticRegression(class_weight='balanced', solver='liblinear',
#                      max_iter=1000, random_state=42), 49 selected features.
# Previously this pointed at the NB09 artifact
# (rlhf_adaptive_logistic_regression_final.joblib, class_weight=None);
# the old path is preserved in experiments/baseline_before_track_a/.
MODEL_PATH = Path(os.getenv(
    "MODEL_PATH",
    THESIS_DIR / "results" / "ml" / "adaptive_logistic_regression_final.joblib",
))

SELECTED_FEATURES_PATH = Path(os.getenv(
    "SELECTED_FEATURES_PATH",
    THESIS_DIR / "data" / "processed" / "ml" / "selected_feature_names.csv",
))

# Frozen thesis train/test artifacts (READ-ONLY; used by the adaptive
# learning service for candidate evaluation only, never modified).
THESIS_TEST_X_PATH = Path(os.getenv(
    "THESIS_TEST_X_PATH",
    THESIS_DIR / "data" / "processed" / "ml" / "X_test_selected.npz",
))
THESIS_TEST_Y_PATH = Path(os.getenv(
    "THESIS_TEST_Y_PATH",
    THESIS_DIR / "data" / "processed" / "ml" / "y_test.csv",
))
THESIS_TRAIN_X_PATH = Path(os.getenv(
    "THESIS_TRAIN_X_PATH",
    THESIS_DIR / "data" / "processed" / "ml" / "X_train_selected.npz",
))
THESIS_TRAIN_Y_PATH = Path(os.getenv(
    "THESIS_TRAIN_Y_PATH",
    THESIS_DIR / "data" / "processed" / "ml" / "y_train.csv",
))

# ---------------------------------------------------------------------------
# Adaptive-learning controls.
#
# Baseline data is MANDATORY for candidate training: every candidate is
# fitted on (baseline sample from the development partition) + (verified
# feedback). Training on feedback alone is no longer permitted.
#
# Evaluation source for candidate comparison:
#   "validation" (default) -> deterministic stratified 20% holdout carved
#       from the TRAINING partition (same split as notebooks 06/06b).
#       Candidate training never sees these rows, so candidate evaluation
#       stays leakage-safe and the frozen test set is not reused.
#   "test" -> legacy behaviour (frozen thesis test set); kept only for
#       historical Track-2 reproduction, never for unbiased reporting.
# ---------------------------------------------------------------------------
ADAPTIVE_EVAL_SOURCE = os.getenv("ADAPTIVE_EVAL_SOURCE", "validation").lower()
if ADAPTIVE_EVAL_SOURCE not in ("validation", "test"):
    raise ValueError(
        f"Invalid ADAPTIVE_EVAL_SOURCE: {ADAPTIVE_EVAL_SOURCE}."
    )

VALIDATION_SPLIT_SIZE = float(os.getenv("VALIDATION_SPLIT_SIZE", "0.20"))
ADAPTIVE_RANDOM_STATE = int(os.getenv("ADAPTIVE_RANDOM_STATE", "42"))

DEFAULT_BASELINE_SAMPLE_SIZE = int(
    os.getenv("ADAPTIVE_BASELINE_SAMPLE_SIZE", "271332")
)
MIN_VERIFIED_FEEDBACK_TOTAL = int(
    os.getenv("ADAPTIVE_MIN_FEEDBACK_TOTAL", "1000")
)
MIN_VERIFIED_FEEDBACK_PER_CLASS = int(
    os.getenv("ADAPTIVE_MIN_FEEDBACK_PER_CLASS", "100")
)

# Reviewer authorization for verified-outcome submission and any future
# activation route. If REVIEWER_API_KEY is not configured the verification
# endpoint is disabled (fail-closed) rather than open.
REVIEWER_API_KEY = os.getenv("REVIEWER_API_KEY", "")

# Adaptive model registry (human-feedback-guided candidate versions).
# Lives inside the backend so thesis artifacts are never overwritten.
ADAPTIVE_MODELS_DIR = Path(os.getenv(
    "ADAPTIVE_MODELS_DIR", BASE_DIR / "models" / "adaptive",
))
ADAPTIVE_REGISTRY_PATH = ADAPTIVE_MODELS_DIR / "registry.json"
