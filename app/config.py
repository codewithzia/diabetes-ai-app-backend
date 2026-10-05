"""Application configuration."""

import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent

# Load variables from backend/.env into the environment (system env vars win).
load_dotenv(BASE_DIR / ".env")

ALLOWED_MODES = {"test", "production"}

APP_MODE = os.getenv("APP_MODE", "test").lower()

if APP_MODE not in ALLOWED_MODES:
    raise ValueError(
        f"Invalid APP_MODE: {APP_MODE}. "
        f"Allowed values: {ALLOWED_MODES}"
    )

CORS_ORIGINS = ["http://localhost:4200", "http://127.0.0.1:4200"]

TEST_DATA_PATH = BASE_DIR / "test_data" / "test_patients.json"

# Separate database per mode so test and production data stay isolated:
#   test mode       -> database/test.db
#   production mode -> database/production.db
DATABASE_DIR = BASE_DIR / "database"
DATABASE_PATH = DATABASE_DIR / f"{APP_MODE}.db"

# Trained model artifacts. Override via environment variables to change
# the location without touching code, e.g.:
#   $env:PREPROCESSOR_PATH = "D:\path\to\brfss_preprocessor_final.joblib"
#   $env:MODEL_PATH        = "D:\path\to\model.joblib"
THESIS_DIR = Path(os.getenv("THESIS_DIR", r"D:\MSc\BRFSS_Diabetes_Thesis"))

PREPROCESSOR_PATH = Path(os.getenv(
    "PREPROCESSOR_PATH",
    THESIS_DIR / "data" / "processed" / "ml" / "brfss_preprocessor_final.joblib",
))

MODEL_PATH = Path(os.getenv(
    "MODEL_PATH",
    THESIS_DIR / "results" / "ml" / "rlhf_adaptive_logistic_regression_final.joblib",
))

SELECTED_FEATURES_PATH = Path(os.getenv(
    "SELECTED_FEATURES_PATH",
    THESIS_DIR / "data" / "processed" / "ml" / "selected_feature_names.csv",
))
