"""Persistence for predictions and human feedback.

Two engines are supported, selected by config.DB_ENGINE:

  mysql   -> MySQL databases selected by APP_MODE
             (TEST_DB_NAME / PROD_DB_NAME; see config.py)
  sqlite  -> original per-mode SQLite files (rollback path only)

Either way, TEST and PRODUCTION data stay in completely separate
databases — the environment mapping is explicit in config.py and never
silently falls back.
"""

import json
import sqlite3
from datetime import datetime, timezone

from .config import (
    APP_MODE,
    DATABASE_DIR,
    DATABASE_PATH,
    DB_ENGINE,
)

if DB_ENGINE == "mysql":
    import pymysql
    from pymysql.cursors import DictCursor

    from .config import (
        DB_HOST,
        DB_PASSWORD,
        DB_PORT,
        DB_USER,
        MYSQL_DB_NAME,
    )

# SQLite DDL (unchanged semantics — TEXT columns keep ISO-8601 strings
# and JSON text exactly as before).
SQLITE_SCHEMA = """
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

# MySQL DDL — minimal mapping of the same schema. TEXT/VARCHAR keep the
# stored ISO timestamps and JSON text byte-identical; INTEGER keeps the
# reward/verified_label/adaptive_processed semantics unchanged.
MYSQL_SCHEMA = [
    """
CREATE TABLE IF NOT EXISTS predictions (
    prediction_id VARCHAR(64)  NOT NULL PRIMARY KEY,
    features      TEXT         NOT NULL,
    result        TEXT         NOT NULL,
    created_at    VARCHAR(64)  NOT NULL,
    model_version VARCHAR(32)  NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
""",
    """
CREATE TABLE IF NOT EXISTS feedback (
    feedback_id        VARCHAR(64) NOT NULL PRIMARY KEY,
    prediction_id      VARCHAR(64) NULL,
    feedback           VARCHAR(32) NULL,
    helpfulness        VARCHAR(32) NULL,
    comment            TEXT        NULL,
    reward             INT         NULL,
    created_at         VARCHAR(64) NOT NULL,
    model_version      VARCHAR(32) NULL,
    verified_label     INT         NULL,
    adaptive_processed INT         NOT NULL DEFAULT 0,
    updated_at         VARCHAR(64) NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
""",
]

# Additive SQLite migrations for databases created before the extended
# schema. ALTER TABLE ADD COLUMN is used so existing rows are preserved
# untouched. (MySQL tables are created with the full schema already.)
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

_IS_MYSQL = DB_ENGINE == "mysql"

# Dialect-specific SQL fragments (kept to the absolute minimum).
_MAX_FEEDBACK_SUFFIX_SQL = (
    "SELECT MAX(CAST(SUBSTR(feedback_id, 4) AS UNSIGNED)) AS max_suffix"
    " FROM feedback" if _IS_MYSQL else
    "SELECT MAX(CAST(SUBSTR(feedback_id, 4) AS INTEGER)) AS max_suffix"
    " FROM feedback"
)


def _migrate(connection: sqlite3.Connection) -> None:
    """SQLite-only additive migrations (no-op for MySQL)."""
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


class _MySQLConnection:
    """Thin adapter giving a PyMySQL connection the sqlite3-style API
    used by this module: connection.execute(sql, params),
    connection.executemany(sql, seq) and 'with' commit/rollback.

    Rows are returned as dicts (DictCursor), matching the named-column
    access used everywhere below. '?' placeholders are translated to
    '%s'; none of the queries embed a literal '?' inside string values.
    """

    # Accept (and ignore) row_factory assignments for API compatibility.
    row_factory = None

    def __init__(self, database: str | None = None):
        self._conn = pymysql.connect(
            host=DB_HOST,
            port=DB_PORT,
            user=DB_USER,
            password=DB_PASSWORD,
            database=database,
            charset="utf8mb4",
            cursorclass=DictCursor,
            autocommit=False,
        )

    @staticmethod
    def _t(sql: str) -> str:
        return sql.replace("?", "%s")

    def execute(self, sql: str, params=()):
        cursor = self._conn.cursor()
        cursor.execute(self._t(sql), params)
        return cursor

    def executemany(self, sql: str, seq):
        cursor = self._conn.cursor()
        cursor.executemany(self._t(sql), seq)
        return cursor

    def executescript(self, sql: str) -> None:  # pragma: no cover
        for statement in sql.split(";"):
            statement = statement.strip()
            if statement:
                self.execute(statement)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            if exc_type is None:
                self._conn.commit()
            else:
                self._conn.rollback()
        finally:
            self._conn.close()


def _connect():
    """Open a connection to the mode-appropriate database."""
    if _IS_MYSQL:
        connection = _MySQLConnection(database=MYSQL_DB_NAME)
        for statement in MYSQL_SCHEMA:
            connection.execute(statement)
        return connection

    DATABASE_DIR.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row
    connection.executescript(SQLITE_SCHEMA)
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


def next_feedback_id() -> str:
    """Generate a unique feedback ID based on the highest existing suffix.

    Count-based IDs break when rows are removed (e.g. demo cleanup), so
    derive the next ID from MAX instead of COUNT to guarantee uniqueness.
    """
    with _connect() as connection:
        row = connection.execute(_MAX_FEEDBACK_SUFFIX_SQL).fetchone()
    highest = row["max_suffix"] or 0
    return f"FB-{highest + 1:06d}"


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


def set_verified_label(feedback_id: str, verified_label: int | None) -> int:
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
        rows = connection.execute(query).fetchall()
    return [dict(row) for row in rows]


def get_predictions_by_ids(prediction_ids: list[str]) -> dict[str, dict]:
    """Fetch prediction rows (id -> row dict) for the given ids."""
    if not prediction_ids:
        return {}
    result = {}
    with _connect() as connection:
        # Chunk to stay below the driver variable limit for large batches.
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
                "verified_label": fb["verified_label"],
                "has_verified_label": fb["verified_label"] is not None,
                "adaptive_processed": bool(fb["adaptive_processed"]),
                "created_at": fb["created_at"],
            }
            for fb in feedback_rows
        ],
    }


def delete_demo_predictions(prediction_ids: list[str]) -> dict:
    """Delete explicitly identified demo/test predictions and their feedback.

    SAFETY: only ever called from the test-mode-only cleanup route with an
    explicit list of prediction IDs known to be demo/test records. Thesis
    datasets, model artifacts and research records are never touched —
    only rows in the active mode-specific database.
    """
    if not prediction_ids:
        return {"deleted_predictions": 0, "deleted_feedback": 0, "not_found": []}
    with _connect() as connection:
        existing = {
            row["prediction_id"]
            for row in connection.execute(
                "SELECT prediction_id FROM predictions"
                f" WHERE prediction_id IN ({','.join('?' for _ in prediction_ids)})",
                prediction_ids,
            )
        }
        not_found = [p for p in prediction_ids if p not in existing]
        targets = [p for p in prediction_ids if p in existing]
        if not targets:
            return {"deleted_predictions": 0, "deleted_feedback": 0, "not_found": not_found}
        placeholders = ",".join("?" for _ in targets)
        deleted_feedback = connection.execute(
            f"DELETE FROM feedback WHERE prediction_id IN ({placeholders})",
            targets,
        ).rowcount
        deleted_predictions = connection.execute(
            f"DELETE FROM predictions WHERE prediction_id IN ({placeholders})",
            targets,
        ).rowcount
    return {
        "deleted_predictions": deleted_predictions,
        "deleted_feedback": deleted_feedback,
        "not_found": not_found,
    }


def _count(table: str) -> int:
    with _connect() as connection:
        row = connection.execute(
            f"SELECT COUNT(*) AS n FROM {table}"
        ).fetchone()
    return row["n"]


def count_predictions() -> int:
    return _count("predictions")


def count_feedback() -> int:
    return _count("feedback")


def database_info() -> dict:
    """Sanitised engine/database description for startup diagnostics.

    Never includes credentials.
    """
    if _IS_MYSQL:
        return {
            "engine": "mysql",
            "host": DB_HOST,
            "port": DB_PORT,
            "database": MYSQL_DB_NAME,
            "mode": APP_MODE,
        }
    return {
        "engine": "sqlite",
        "path": str(DATABASE_PATH),
        "mode": APP_MODE,
    }
