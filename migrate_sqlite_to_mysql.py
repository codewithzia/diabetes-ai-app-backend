"""Idempotent SQLite -> MySQL migration for the diabetes app.

Usage (from the backend directory, MySQL credentials in .env):

    py migrate_sqlite_to_mysql.py            # migrate + verify both envs
    py migrate_sqlite_to_mysql.py --verify   # verify only

Safety properties:
  - SQLite source files are opened READ-ONLY and are never modified.
  - MySQL target databases are created if missing (CREATE DATABASE
    IF NOT EXISTS), tables likewise.
  - Existing primary keys are never overwritten blindly: identical rows
    are skipped; conflicting rows are reported as CONFLICT and skipped.
  - Each table is committed only after all its rows are staged;
    any failure rolls that table back.
  - A full by-primary-key, column-by-column verification runs after
    the copy and is also available standalone via --verify.
"""

import argparse
import os
import sqlite3
import sys
from pathlib import Path

import pymysql
from dotenv import load_dotenv
from pymysql.cursors import DictCursor

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

# --- Configuration ---------------------------------------------------------

DB_HOST = os.getenv("DB_HOST")
DB_PORT = int(os.getenv("DB_PORT", "3306"))
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")
TEST_DB_NAME = os.getenv("TEST_DB_NAME")
PROD_DB_NAME = os.getenv("PROD_DB_NAME")

_REQUIRED = ["DB_HOST", "DB_USER", "TEST_DB_NAME", "PROD_DB_NAME"]
_missing = [v for v in _REQUIRED if not os.getenv(v)]
if _missing:
    sys.exit(
        "Missing required environment variables: "
        + ", ".join(_missing)
        + " (see .env.example)"
    )

# Explicit environment mapping — SQLite file -> MySQL database name.
ENVIRONMENTS = [
    {
        "env": "TEST",
        "sqlite": BASE_DIR / "database" / "test.db",
        "mysql_db": TEST_DB_NAME,
    },
    {
        "env": "PROD",
        "sqlite": BASE_DIR / "database" / "production.db",
        "mysql_db": PROD_DB_NAME,
    },
]

TABLES = {
    "predictions": {
        "pk": "prediction_id",
        "columns": [
            "prediction_id", "features", "result", "created_at",
            "model_version",
        ],
        "ddl": """
CREATE TABLE IF NOT EXISTS predictions (
    prediction_id VARCHAR(64)  NOT NULL PRIMARY KEY,
    features      TEXT         NOT NULL,
    result        TEXT         NOT NULL,
    created_at    VARCHAR(64)  NOT NULL,
    model_version VARCHAR(32)  NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4""",
    },
    "feedback": {
        "pk": "feedback_id",
        "columns": [
            "feedback_id", "prediction_id", "feedback", "helpfulness",
            "comment", "reward", "created_at", "model_version",
            "verified_label", "adaptive_processed", "updated_at",
        ],
        "ddl": """
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
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4""",
    },
}


# --- Helpers ---------------------------------------------------------------

def mysql_connect(database: str | None = None):
    return pymysql.connect(
        host=DB_HOST,
        port=DB_PORT,
        user=DB_USER,
        password=DB_PASSWORD,
        database=database,
        charset="utf8mb4",
        cursorclass=DictCursor,
        autocommit=False,
    )


def sqlite_connect_readonly(path: Path) -> sqlite3.Connection | None:
    """Open a SQLite file strictly read-only. An empty/0-byte file (or a
    file with no tables) is reported as an empty source."""
    if not path.exists() or path.stat().st_size == 0:
        return None
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def sqlite_table_exists(conn, table: str) -> bool:
    row = conn.execute(
        "SELECT COUNT(*) AS n FROM sqlite_master"
        " WHERE type='table' AND name=?",
        (table,),
    ).fetchone()
    return row["n"] > 0


def read_sqlite_rows(conn, table: str, columns: list[str]) -> list[dict]:
    cols = ", ".join(columns)
    return [
        dict(row)
        for row in conn.execute(f"SELECT {cols} FROM {table}").fetchall()
    ]


def _norm(value):
    """Normalise a value for cross-engine comparison."""
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return value
    return str(value)


def rows_equal(a: dict, b: dict, columns: list[str]) -> bool:
    return all(_norm(a.get(c)) == _norm(b.get(c)) for c in columns)


# --- Migration -------------------------------------------------------------

def migrate_environment(env: dict) -> dict:
    """Migrate one environment; returns a report dict."""
    report = {"env": env["env"], "tables": {}, "sqlite_path": str(env["sqlite"]),
              "mysql_db": env["mysql_db"]}

    src = sqlite_connect_readonly(env["sqlite"])

    # Create target database + tables.
    admin = mysql_connect()
    with admin.cursor() as cur:
        cur.execute(
            f"CREATE DATABASE IF NOT EXISTS `{env['mysql_db']}`"
            " CHARACTER SET utf8mb4"
        )
    admin.commit()
    admin.close()

    dst = mysql_connect(database=env["mysql_db"])
    try:
        for table, spec in TABLES.items():
            with dst.cursor() as cur:
                cur.execute(spec["ddl"])
        dst.commit()

        for table, spec in TABLES.items():
            pk, cols = spec["pk"], spec["columns"]
            stats = {"sqlite_rows": 0, "inserted": 0, "skipped_identical": 0,
                     "conflicts": []}

            source_rows = []
            if src is not None and sqlite_table_exists(src, table):
                source_rows = read_sqlite_rows(src, table, cols)
            stats["sqlite_rows"] = len(source_rows)

            with dst.cursor() as cur:
                cur.execute(f"SELECT * FROM {table}")
                existing = {row[pk]: row for row in cur.fetchall()}

            insert_sql = (
                f"INSERT INTO {table} ({', '.join(cols)})"
                f" VALUES ({', '.join(['%s'] * len(cols))})"
            )
            try:
                with dst.cursor() as cur:
                    for row in source_rows:
                        row_pk = row[pk]
                        if row_pk in existing:
                            if rows_equal(row, existing[row_pk], cols):
                                stats["skipped_identical"] += 1
                            else:
                                stats["conflicts"].append(row_pk)
                            continue
                        cur.execute(
                            insert_sql,
                            tuple(row[c] for c in cols),
                        )
                        stats["inserted"] += 1
                dst.commit()
            except Exception:
                dst.rollback()
                raise
            report["tables"][table] = stats
    finally:
        dst.close()
        if src is not None:
            src.close()
    return report


# --- Verification ----------------------------------------------------------

def verify_environment(env: dict) -> dict:
    """Compare SQLite source and MySQL target by primary key."""
    result = {"env": env["env"], "ok": True, "tables": {}}
    src = sqlite_connect_readonly(env["sqlite"])
    dst = mysql_connect(database=env["mysql_db"])
    try:
        for table, spec in TABLES.items():
            pk, cols = spec["pk"], spec["columns"]
            source_rows = []
            if src is not None and sqlite_table_exists(src, table):
                source_rows = read_sqlite_rows(src, table, cols)
            src_by_pk = {r[pk]: r for r in source_rows}

            with dst.cursor() as cur:
                cur.execute(f"SELECT * FROM {table}")
                dst_rows = cur.fetchall()
            dst_by_pk = {r[pk]: r for r in dst_rows}

            missing = [k for k in src_by_pk if k not in dst_by_pk]
            extra = [k for k in dst_by_pk if k not in src_by_pk]
            mismatched = [
                k for k in src_by_pk
                if k in dst_by_pk and not rows_equal(src_by_pk[k], dst_by_pk[k], cols)
            ]
            dup_check = len(dst_rows) != len(dst_by_pk)

            ok = not missing and not mismatched and not dup_check
            result["ok"] = result["ok"] and ok
            result["tables"][table] = {
                "sqlite_rows": len(source_rows),
                "mysql_rows": len(dst_rows),
                "difference": len(source_rows) - len(dst_rows),
                "missing_in_mysql": missing,
                "extra_in_mysql": extra,
                "mismatched_rows": mismatched,
                "duplicate_pks": dup_check,
                "ok": ok,
            }
    finally:
        dst.close()
        if src is not None:
            src.close()
    return result


# --- Reporting -------------------------------------------------------------

def print_report(migration: dict, verification: dict) -> None:
    env = verification["env"]
    print("=" * 60)
    print(f"{env} DATABASE")
    print("=" * 60)
    print("SQLite:")
    print(f"  path: {migration['sqlite_path']}")
    print("MySQL:")
    print(f"  database: {migration['mysql_db']}")
    print()
    for table in TABLES:
        m = migration["tables"].get(table, {})
        v = verification["tables"].get(table, {})
        print(f"  {env} {table}:")
        print(f"    SQLite:      {v.get('sqlite_rows')}")
        print(f"    MySQL:       {v.get('mysql_rows')}")
        print(f"    Difference:  {v.get('difference')}")
        print(f"    Inserted:    {m.get('inserted', '-')}")
        print(f"    Skipped (already identical): {m.get('skipped_identical', '-')}")
        if m.get("conflicts"):
            print(f"    CONFLICTS (kept existing MySQL row): {m['conflicts']}")
        if v.get("missing_in_mysql"):
            print(f"    MISSING in MySQL: {v['missing_in_mysql']}")
        if v.get("extra_in_mysql"):
            print(f"    Extra in MySQL (not in SQLite): {v['extra_in_mysql']}")
        if v.get("mismatched_rows"):
            print(f"    MISMATCHED rows: {v['mismatched_rows']}")
        if v.get("duplicate_pks"):
            print("    DUPLICATE primary keys detected in MySQL!")
        print(f"    Verified:    {'OK' if v.get('ok') else 'FAILED'}")
        print()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true",
                        help="run verification only (no copy)")
    args = parser.parse_args()

    all_ok = True
    for env in ENVIRONMENTS:
        print(f"\n>>> {env['env']}: {env['sqlite'].name} -> {env['mysql_db']}")
        if args.verify:
            migration = {"sqlite_path": str(env["sqlite"]),
                         "mysql_db": env["mysql_db"], "tables": {}}
        else:
            migration = migrate_environment(env)
        verification = verify_environment(env)
        print_report(migration, verification)
        all_ok = all_ok and verification["ok"]

    print("=" * 60)
    print("MIGRATION VERIFICATION:", "PASSED" if all_ok else "FAILED")
    print("=" * 60)
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
