"""Load chat_logs.csv into a typed SQLite database.

Blank numeric fields are stored as NULL, so SQL treats them as missing
(COUNT and AVG skip them). Blank text fields (response, error) stay as
empty strings: an empty response is a meaningful value in this dataset.
"""

import csv
import sqlite3
from contextlib import closing
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
CSV_PATH = PROJECT_DIR / "chat_logs.csv"
DB_PATH = PROJECT_DIR / "database.sqlite"

SCHEMA = """
CREATE TABLE chat_logs (
    id INTEGER,
    timestamp TEXT,
    model TEXT,
    category TEXT,
    prompt TEXT,
    response TEXT,
    latency_ms REAL,
    input_tokens INTEGER,
    output_tokens INTEGER,
    error TEXT,
    judge_score INTEGER
);
"""

COLUMNS = [
    "id",
    "timestamp",
    "model",
    "category",
    "prompt",
    "response",
    "latency_ms",
    "input_tokens",
    "output_tokens",
    "error",
    "judge_score",
]

NUMERIC_COLUMNS = {"id", "latency_ms", "input_tokens", "output_tokens", "judge_score"}


def to_db_value(column, value):
    """Blank numeric fields become NULL; everything else is kept as read."""
    if column in NUMERIC_COLUMNS and value.strip() == "":
        return None
    return value


def main():
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Input CSV not found: {CSV_PATH}")

    if DB_PATH.exists():
        DB_PATH.unlink()

    with closing(sqlite3.connect(DB_PATH)) as connection:
        connection.executescript(SCHEMA)

        with CSV_PATH.open("r", newline="", encoding="utf-8-sig") as csv_file:
            reader = csv.DictReader(csv_file)

            if reader.fieldnames != COLUMNS:
                raise ValueError(f"Unexpected CSV columns: {reader.fieldnames}")

            placeholders = ", ".join("?" for _ in COLUMNS)
            column_names = ", ".join(COLUMNS)
            insert_sql = (
                f"INSERT INTO chat_logs ({column_names}) "
                f"VALUES ({placeholders})"
            )

            rows = [
                tuple(to_db_value(column, row[column]) for column in COLUMNS)
                for row in reader
            ]
            connection.executemany(insert_sql, rows)

        connection.commit()

        count = connection.execute("SELECT COUNT(*) FROM chat_logs").fetchone()[0]

    print(f"Database created: {DB_PATH}")
    print(f"Rows loaded: {count}")


if __name__ == "__main__":
    main()
