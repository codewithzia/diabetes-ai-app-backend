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
    result = {}
    with _connect() as connection:
        connection.row_factory = sqlite3.Row
        # Chunk to stay below SQLite's variable limit for large batches.
        for start in range(0, len(prediction_ids), 500):
            chunk = prediction_ids[start:start + 500]
            placeholders = ",".join("?" for _ in chunk)
            rows = connection.execute(
                f"SELECT * FROM predictions"
                f" WHERE prediction_id IN ({placeholders})",
                chunk,
            ).fetchall()
            result.update({row["prediction_id"]: dict(row) for row in rows})
    return result


def list_predictions(limit: int = 200) -> list[dict]:
    """Read-only prediction history with aggregated feedback status.

    Returns newest-first rows. Feedback is summarised (latest record,
    plus counters) — agree/disagree is a signal only; verification is
    reported as a boolean flag, never exposed as a label value here.
    """
    with _connect() as connection:
        connection.row_factory = sqlite3.Row
        predictions = connection.execute(
            "SELECT prediction_id, features, result, created_at, model_version"
            " FROM predictions ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        feedback_rows = connection.execute(
            "SELECT prediction_id, feedback, helpfulness, comment, reward,"
            " verified_label, adaptive_processed, created_at"
            " FROM feedback ORDER BY created_at ASC"
        ).fetchall()

    feedback_by_prediction: dict[str, list[dict]] = {}
    for row in feedback_rows:
        feedback_by_prediction.setdefault(row["prediction_id"], []).append(
            {
                "feedback": row["feedback"],
                "helpfulness": row["helpfulness"],
                "comment": row["comment"],
                "reward": row["reward"],
                "has_verified_label": row["verified_label"] is not None,
                "adaptive_processed": bool(row["adaptive_processed"]),
                "created_at": row["created_at"],
            }
        )

    history = []
    for row in predictions:
        result = json.loads(row["result"]) if row["result"] else {}
        entries = feedback_by_prediction.get(row["prediction_id"], [])
        history.append({
            "prediction_id": row["prediction_id"],
            "created_at": row["created_at"],
            "model_version": row["model_version"],
            "prediction": result.get("prediction"),
            "probability": result.get("probability"),
            "risk_category": result.get("risk_category"),
            "threshold": result.get("threshold"),
            "feedback_count": len(entries),
            "latest_feedback": entries[-1]["feedback"] if entries else None,
            "latest_helpfulness": entries[-1]["helpfulness"] if entries else None,
            "latest_feedback_at": entries[-1]["created_at"] if entries else None,
            "has_verified_label": any(e["has_verified_label"] for e in entries),
            "adaptive_processed": any(e["adaptive_processed"] for e in entries),
        })
    return history


def get_prediction_detail(prediction_id: str) -> dict | None:
    """Read-only full record for one prediction: inputs, result and
    all associated feedback entries (verified status as flag only)."""
    with _connect() as connection:
        connection.row_factory = sqlite3.Row
        row = connection.execute(
            "SELECT prediction_id, features, result, created_at, model_version"
            " FROM predictions WHERE prediction_id = ?",
            (prediction_id,),
        ).fetchone()
        if row is None:
            return None
        feedback_rows = connection.execute(
            "SELECT feedback_id, feedback, helpfulness, comment, reward,"
            " verified_label, adaptive_processed, created_at"
            " FROM feedback WHERE prediction_id = ? ORDER BY created_at ASC",
            (prediction_id,),
        ).fetchall()

    return {
        "prediction_id": row["prediction_id"],
        "created_at": row["created_at"],
        "model_version": row["model_version"],
        "features": json.loads(row["features"]) if row["features"] else {},
        "result": json.loads(row["result"]) if row["result"] else {},
        "feedback": [
            {
                "feedback_id": fb["feedback_id"],
                "feedback": fb["feedback"],
                "helpfulness": fb["helpfulness"],
                "comment": fb["comment"],
                "reward": fb["reward"],
                "has_verified_label": fb["verified_label"] is not None,
                "adaptive_processed": bool(fb["adaptive_processed"]),
                "created_at": fb["created_at"],
            }
            for fb in feedback_rows
        ],
    }


def _count(table: str) -> int:
    with _connect() as connection:
        row = connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
    return row[0]


def count_predictions() -> int:
    return _count("predictions")


def count_feedback() -> int:
    return _count("feedback")
