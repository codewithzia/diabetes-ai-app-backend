"""Application configuration."""

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

ALLOWED_MODES = {"test", "production"}

APP_MODE = os.getenv("APP_MODE", "test").lower()

if APP_MODE not in ALLOWED_MODES:
    raise ValueError(
        f"Invalid APP_MODE: {APP_MODE}. "
        f"Allowed values: {ALLOWED_MODES}"
    )

CORS_ORIGINS = ["http://localhost:4200", "http://127.0.0.1:4200"]

TEST_DATA_PATH = BASE_DIR / "test_data" / "test_patients.json"
