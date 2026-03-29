from __future__ import annotations

from pathlib import Path

from src.storage.clickhouse_client import get_clickhouse_client


DDL_DIR = Path("sql/ddl")


def run_sql_file(client, path: Path) -> None:
    sql = path.read_text(encoding="utf-8")
    client.command(sql)


def ensure_tables() -> None:
    client = get_clickhouse_client()

    ddl_files = sorted(DDL_DIR.glob("*.sql"))
    if not ddl_files:
        raise FileNotFoundError(f"No DDL files found in {DDL_DIR.resolve()}")

    for ddl_file in ddl_files:
        print(f"Applying DDL: {ddl_file}")
        run_sql_file(client, ddl_file)

    print("All tables ensured successfully.")


if __name__ == "__main__":
    ensure_tables()