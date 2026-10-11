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

# Trained model artifacts (leakage-safe final experiment, notebook 05b).
# Override the directory or individual paths via environment variables, e.g.:
#   $env:DIABETES_MODEL_DIR = "D:\path\to\artifacts"
#   $env:MODEL_PATH         = "D:\path\to\model.joblib"
THESIS_DIR = Path(os.getenv("THESIS_DIR", r"D:\MSc\BRFSS_Diabetes_Thesis"))

DIABETES_MODEL_DIR = Path(os.getenv(
    "DIABETES_MODEL_DIR",
    THESIS_DIR / "results" / "ml" / "results_leakage_safe",
))

PREPROCESSOR_PATH = Path(os.getenv(
    "PREPROCESSOR_PATH",
    DIABETES_MODEL_DIR / "leakage_safe_preprocessor.joblib",
))

MODEL_PATH = Path(os.getenv(
    "MODEL_PATH",
    DIABETES_MODEL_DIR / "leakage_safe_logistic_regression.joblib",
))

SELECTED_FEATURES_PATH = Path(os.getenv(
    "SELECTED_FEATURES_PATH",
    DIABETES_MODEL_DIR / "final_selected_features.csv",
))

EXPERIMENT_METADATA_PATH = DIABETES_MODEL_DIR / "experiment_metadata.json"

# Deployment threshold. Chosen on validation split: 0.63 maximizes F1,
# 0.465 targets >=80% recall. Override via DIABETES_THRESHOLD.
PREDICTION_THRESHOLD = float(os.getenv("DIABETES_THRESHOLD", "0.63"))

# Frozen thesis train/test artifacts (READ-ONLY, never modified).
# The TEST split stays fully isolated: it is never used for training,
# candidate evaluation, threshold selection, or candidate acceptance.
THESIS_TEST_X_PATH = Path(os.getenv(
    "THESIS_TEST_X_PATH",
    THESIS_DIR / "data" / "processed" / "ml" / "X_test_raw.csv",
))

# Raw training partition (READ-ONLY). A deterministic, stratified 80/20
# split (random_state=42, matching experiment_metadata.json
# validation_size=0.2) derives a FIT portion for baseline sampling and a
# VALIDATION portion for candidate evaluation — so validation rows never
# enter candidate training data and the frozen test set is untouched.
THESIS_TRAIN_X_PATH = Path(os.getenv(
    "THESIS_TRAIN_X_PATH",
    THESIS_DIR / "data" / "processed" / "ml" / "X_train_raw.csv",
))
THESIS_TRAIN_Y_PATH = Path(os.getenv(
    "THESIS_TRAIN_Y_PATH",
    THESIS_DIR / "data" / "processed" / "ml" / "y_train.csv",
))

# Adaptive-learning engineering safeguards (configurable via env vars).
# These are NOT tuned model parameters; they are data-floor policies.
# Calibrate per deployment: set total/per-class minima to the smallest
# counts where a logistic-regression refit is meaningful (rule of thumb:
# >=10 events per feature for stability => 50 features needs >= 500
# events per class for a from-scratch fit; smaller floors are acceptable
# when a large verified baseline sample is prepended).
ADAPTIVE_MIN_VERIFIED_SAMPLES = int(os.getenv("ADAPTIVE_MIN_VERIFIED_SAMPLES", "50"))
ADAPTIVE_MIN_VERIFIED_PER_CLASS = int(os.getenv("ADAPTIVE_MIN_VERIFIED_PER_CLASS", "20"))
ADAPTIVE_BASELINE_SAMPLE_SIZE = int(os.getenv("ADAPTIVE_BASELINE_SAMPLE_SIZE", "5000"))
ADAPTIVE_VALIDATION_FRACTION = float(os.getenv("ADAPTIVE_VALIDATION_FRACTION", "0.2"))
THESIS_TEST_Y_PATH = Path(os.getenv(
    "THESIS_TEST_Y_PATH",
    THESIS_DIR / "data" / "processed" / "ml" / "y_test.csv",
))

# Adaptive model registry (human-feedback-guided candidate versions).
# Lives inside the backend so thesis artifacts are never overwritten.
ADAPTIVE_MODELS_DIR = Path(os.getenv(
    "ADAPTIVE_MODELS_DIR", BASE_DIR / "models" / "adaptive",
))
ADAPTIVE_REGISTRY_PATH = ADAPTIVE_MODELS_DIR / "registry.json"
