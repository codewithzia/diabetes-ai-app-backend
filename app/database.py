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
    created_at    TEXT NOT NULL,
    model_version TEXT
);

CREATE TABLE IF NOT EXISTS feedback (
    feedback_id        TEXT PRIMARY KEY,
    prediction_id      TEXT,
    feedback           TEXT,
    helpfulness        TEXT,
    comment            TEXT,
    reward             INTEGER,
    created_at         TEXT NOT NULL,
    model_version      TEXT,
    verified_label     INTEGER,
    adaptive_processed INTEGER NOT NULL DEFAULT 0,
    updated_at         TEXT
);
"""

# Additive migrations for databases created before the extended schema.
# ALTER TABLE ADD COLUMN is used so existing rows are preserved untouched.
_MIGRATIONS = {
    "predictions": [
        "ALTER TABLE predictions ADD COLUMN model_version TEXT",
    ],
    "feedback": [
        "ALTER TABLE feedback ADD COLUMN model_version TEXT",
        "ALTER TABLE feedback ADD COLUMN verified_label INTEGER",
        "ALTER TABLE feedback ADD COLUMN adaptive_processed INTEGER NOT NULL DEFAULT 0",
        "ALTER TABLE feedback ADD COLUMN updated_at TEXT",
    ],
}


def _migrate(connection: sqlite3.Connection) -> None:
    for table, statements in _MIGRATIONS.items():
        existing = {
            row[1]
            for row in connection.execute(f"PRAGMA table_info({table})")
        }
        for statement in statements:
            # Column name is the token after ADD COLUMN.
            column = statement.split("ADD COLUMN")[1].split()[0]
            if column not in existing:
                connection.execute(statement)


def _connect() -> sqlite3.Connection:
    DATABASE_DIR.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DATABASE_PATH)
    connection.executescript(SCHEMA)
    _migrate(connection)
    return connection


def save_prediction(
    prediction_id: str,
    features: dict,
    result: dict,
    model_version: str | None = None,
) -> None:
    with _connect() as connection:
        connection.execute(
            "INSERT INTO predictions (prediction_id, features, result,"
            " created_at, model_version) VALUES (?, ?, ?, ?, ?)",
            (
                prediction_id,
                json.dumps(features),
                json.dumps(result),
                datetime.now(timezone.utc).isoformat(),
                model_version,
            ),
        )


def save_feedback(record: dict) -> None:
    """Persist a feedback record.

    NOTE: record["verified_label"] must only ever come from a trusted /
    verified clinical source. It must NEVER be derived from the user's
    agree/disagree feedback, which is only a reward signal.
    """
    now = datetime.now(timezone.utc).isoformat()
    with _connect() as connection:
        connection.execute(
            "INSERT INTO feedback (feedback_id, prediction_id, feedback,"
            " helpfulness, comment, reward, created_at, model_version,"
            " verified_label, adaptive_processed, updated_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                record["feedback_id"],
                record["prediction_id"],
                record["feedback"],
                record["helpfulness"],
                record["comment"],
                record["reward"],
                record["timestamp"],
                record.get("model_version"),
                record.get("verified_label"),
                1 if record.get("adaptive_processed") else 0,
                record.get("updated_at", now),
            ),
        )


def mark_feedback_processed(feedback_ids: list[str]) -> int:
    """Mark feedback records as consumed by an adaptive training batch."""
    now = datetime.now(timezone.utc).isoformat()
    with _connect() as connection:
        cursor = connection.executemany(
            "UPDATE feedback SET adaptive_processed = 1, updated_at = ?"
            " WHERE feedback_id = ?",
            [(now, feedback_id) for feedback_id in feedback_ids],
        )
        return cursor.rowcount


def set_verified_label(feedback_id: str, verified_label: int) -> int:
    """Set a trusted/verified clinical label for a feedback record.

    This is the ONLY path through which verified_label may be written,
    and it must only be called with labels from a verified source.
    """
    now = datetime.now(timezone.utc).isoformat()
    with _connect() as connection:
        cursor = connection.execute(
            "UPDATE feedback SET verified_label = ?, updated_at = ?"
            " WHERE feedback_id = ?",
            (verified_label, now, feedback_id),
        )
        return cursor.rowcount


def get_verified_feedback(only_unprocessed: bool = True) -> list[dict]:
    """Feedback rows that may serve as supervised labels.

    Only rows with a verified_label from a trusted source are returned.
    agree/disagree feedback and reward are reward signals only and are
    never used as training labels.
    """
    query = "SELECT * FROM feedback WHERE verified_label IS NOT NULL"
    if only_unprocessed:
        query += " AND adaptive_processed = 0"
    with _connect() as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(query).fetchall()
    return [dict(row) for row in rows]


def get_predictions_by_ids(prediction_ids: list[str]) -> dict[str, dict]:
    """Fetch prediction rows (id -> row dict) for the given ids."""
    if not prediction_ids:
        return {}
    placeholders = ",".join("?" for _ in prediction_ids)
    with _connect() as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            f"SELECT * FROM predictions WHERE prediction_id IN ({placeholders})",
            prediction_ids,
        ).fetchall()
    return {row["prediction_id"]: dict(row) for row in rows}


def _count(table: str) -> int:
    with _connect() as connection:
        row = connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
    return row[0]


def count_predictions() -> int:
    return _count("predictions")


def count_feedback() -> int:
    return _count("feedback")
