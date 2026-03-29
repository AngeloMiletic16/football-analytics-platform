from __future__ import annotations

import argparse
from pathlib import Path

from src.storage.clickhouse_client import get_clickhouse_client


def run_sql_file(sql_path: str) -> None:
    path = Path(sql_path)
    if not path.exists():
        raise FileNotFoundError(f"SQL file not found: {path}")

    sql = path.read_text(encoding="utf-8")
    client = get_clickhouse_client()

    result = client.query(sql)

    if result.column_names:
        print(" | ".join(result.column_names))

    for row in result.result_rows:
        print(" | ".join(str(value) for value in row))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sql-path", required=True)
    args = parser.parse_args()

    run_sql_file(args.sql_path)