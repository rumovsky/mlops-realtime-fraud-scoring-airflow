"""Shared storage and calendar operations for the three DAGs."""

import calendar
import csv
import hashlib
import json
import os
from datetime import date, datetime
from pathlib import Path

import psycopg2
from psycopg2.extras import Json, execute_values


DATA_PATH = Path("/opt/airflow/data/train.csv")
WORK_DIR = Path("/opt/airflow/work")
MODEL_DIR = Path("/opt/airflow/models")
REPORT_DIR = Path("/opt/airflow/reports")
WINDOW_MONTHS = int(os.getenv("WINDOW_MONTHS", "3"))
HISTORY_MONTHS = int(os.getenv("HISTORY_MONTHS", "12"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "5000"))
LOCK_ID = 31003


def connect():
    return psycopg2.connect(os.getenv("TRAINING_DSN", "postgresql://fraud:fraud@postgres:5432/fraud"))


def shift_month(month, count):
    number = month.year * 12 + month.month - 1 + count
    return date(number // 12, number % 12 + 1, 1)


def month_start(value):
    return date(value.year, value.month, 1)


def run_directory(run_id):
    path = WORK_DIR / hashlib.sha256(run_id.encode()).hexdigest()[:16]
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, data):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def current_model():
    return read_json(MODEL_DIR / "current.json")


def ensure_source():
    """Initialize the labelled store from CSV once; a retry checks the committed state.

    Reads train.csv, writes the last WINDOW_MONTHS and an initial month pointer
    in one database transaction. Concurrent DAGs serialize on an advisory lock.
    """
    if WINDOW_MONTHS < 2 or HISTORY_MONTHS < WINDOW_MONTHS:
        raise ValueError("Need WINDOW_MONTHS >= 2 and HISTORY_MONTHS >= WINDOW_MONTHS")
    with connect() as connection, connection.cursor() as cursor:
        cursor.execute("SELECT pg_advisory_xact_lock(%s)", (LOCK_ID,))
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS labelled_transactions (
                transaction_id TEXT PRIMARY KEY,
                transaction_time TIMESTAMP NOT NULL,
                data JSONB NOT NULL,
                target SMALLINT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS labelled_time_idx
                ON labelled_transactions (transaction_time);
            CREATE TABLE IF NOT EXISTS training_state (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                current_month DATE NOT NULL,
                model_version TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS month_batches (
                month DATE PRIMARY KEY,
                row_count INTEGER NOT NULL
            );
        """)
        cursor.execute("SELECT current_month FROM training_state WHERE id = 1")
        if cursor.fetchone():
            return
        with DATA_PATH.open(newline="", encoding="utf-8-sig") as source:
            latest = max(datetime.fromisoformat(row["transaction_time"]) for row in csv.DictReader(source))
        latest_month = month_start(latest)
        first_month = shift_month(latest_month, 1 - WINDOW_MONTHS)
        buckets = {shift_month(first_month, n): [] for n in range(WINDOW_MONTHS)}
        with DATA_PATH.open(newline="", encoding="utf-8-sig") as source:
            for index, row in enumerate(csv.DictReader(source)):
                timestamp = datetime.fromisoformat(row["transaction_time"])
                month = month_start(timestamp)
                if month in buckets and len(buckets[month]) < BATCH_SIZE:
                    target = int(row.pop("target"))
                    buckets[month].append((f"source-{index}", timestamp, Json(row), target))
        for month, rows in buckets.items():
            if not rows:
                raise ValueError(f"No source rows for {month}")
            execute_values(cursor, """
                INSERT INTO labelled_transactions (transaction_id, transaction_time, data, target)
                VALUES %s ON CONFLICT DO NOTHING
            """, rows, page_size=1000)
            cursor.execute("INSERT INTO month_batches VALUES (%s, %s)", (month, len(rows)))
        cursor.execute("INSERT INTO training_state VALUES (1, %s, %s)", (latest_month, current_model()["version"]))


def append_month():
    """Copy a batch to the next month and prune history, without moving the training pointer.

    Reads the latest committed month, writes deterministic IDs and a batch marker
    in one transaction. Repeating the task returns without appending duplicates.
    """
    with connect() as connection, connection.cursor() as cursor:
        cursor.execute("SELECT pg_advisory_xact_lock(%s)", (LOCK_ID,))
        cursor.execute("SELECT current_month FROM training_state WHERE id = 1")
        current = cursor.fetchone()[0]
        cursor.execute("SELECT max(month) FROM month_batches")
        source_month = cursor.fetchone()[0] or current
        next_month = shift_month(source_month, 1)
        cursor.execute("SELECT 1 FROM month_batches WHERE month = %s", (next_month,))
        if cursor.fetchone():
            return
        cursor.execute("""
            SELECT transaction_id, transaction_time, data, target
            FROM labelled_transactions
            WHERE transaction_time >= %s AND transaction_time < %s
            ORDER BY transaction_id LIMIT %s
        """, (source_month, next_month, BATCH_SIZE))
        source_rows = cursor.fetchall()
        if not source_rows:
            raise ValueError(f"No rows to copy from {current}")
        rows = []
        for index, (_, timestamp, data, target) in enumerate(source_rows):
            day = min(timestamp.day, calendar.monthrange(next_month.year, next_month.month)[1])
            shifted = timestamp.replace(year=next_month.year, month=next_month.month, day=day)
            data["transaction_time"] = shifted.isoformat(sep=" ")
            rows.append((f"{next_month.isoformat()}-{index}", shifted, Json(data), target))
        execute_values(cursor, """
            INSERT INTO labelled_transactions (transaction_id, transaction_time, data, target)
            VALUES %s ON CONFLICT DO NOTHING
        """, rows, page_size=1000)
        cursor.execute("INSERT INTO month_batches VALUES (%s, %s)", (next_month, len(rows)))
        cutoff = shift_month(next_month, 1 - HISTORY_MONTHS)
        cursor.execute("DELETE FROM labelled_transactions WHERE transaction_time < %s", (cutoff,))
        cursor.execute("DELETE FROM month_batches WHERE month < %s", (cutoff,))
        print(f"Added {len(rows)} rows for {next_month} from {source_month}; retained history from {cutoff}")


def window_rows():
    with connect() as connection, connection.cursor() as cursor:
        cursor.execute("SELECT pg_advisory_xact_lock(%s)", (LOCK_ID,))
        cursor.execute("SELECT current_month FROM training_state WHERE id = 1")
        current = cursor.fetchone()[0]
        cursor.execute("SELECT max(month) FROM month_batches")
        latest = max(current, cursor.fetchone()[0])
        earliest = shift_month(latest, 1 - WINDOW_MONTHS)
        cursor.execute("""
            SELECT transaction_id, transaction_time, data, target FROM labelled_transactions
            WHERE transaction_time >= %s AND transaction_time < %s
            ORDER BY transaction_time, transaction_id
        """, (earliest, shift_month(latest, 1)))
        return current, latest, earliest, cursor.fetchall()
