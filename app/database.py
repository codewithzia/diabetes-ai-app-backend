"""SQLite persistence for predictions and human feedback.

Each application mode uses its own database file (see config.DATABASE_PATH),
so data recorded in TEST MODE never mixes with PRODUCTION data.
"""

import json
import sqlite3
from datetime import datetime, timezone

from .config import DATABASE_DIR, DATABASE_PATH

SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    prediction_id TEXT PRIMARY KEY,
    features      TEXT NOT NULL,
    result        TEXT NOT NULL,
    created_at    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS feedback (
    feedback_id   TEXT PRIMARY KEY,
    prediction_id TEXT,
    feedback      TEXT,
    helpfulness   TEXT,
    comment       TEXT,
    reward        INTEGER,
    created_at    TEXT NOT NULL
);
"""


def _connect() -> sqlite3.Connection:
    DATABASE_DIR.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DATABASE_PATH)
    connection.executescript(SCHEMA)
    return connection


def save_prediction(prediction_id: str, features: dict, result: dict) -> None:
    with _connect() as connection:
        connection.execute(
            "INSERT INTO predictions (prediction_id, features, result, created_at)"
            " VALUES (?, ?, ?, ?)",
            (
                prediction_id,
                json.dumps(features),
                json.dumps(result),
                datetime.now(timezone.utc).isoformat(),
            ),
        )


def save_feedback(record: dict) -> None:
    with _connect() as connection:
        connection.execute(
            "INSERT INTO feedback (feedback_id, prediction_id, feedback,"
            " helpfulness, comment, reward, created_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                record["feedback_id"],
                record["prediction_id"],
                record["feedback"],
                record["helpfulness"],
                record["comment"],
                record["reward"],
                record["timestamp"],
            ),
        )


def _count(table: str) -> int:
    with _connect() as connection:
        row = connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
    return row[0]


def count_predictions() -> int:
    return _count("predictions")


def count_feedback() -> int:
    return _count("feedback")
