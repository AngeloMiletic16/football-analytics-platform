import clickhouse_connect
from pathlib import Path


def main() -> None:
    client = clickhouse_connect.get_client(
        host="localhost",
        port=8123,
        username="admin",
        password="admin123",
        database="football_analytics",
    )

    project_root = Path(__file__).resolve().parents[2]
    sql_path = project_root / "sql" / "create_tables.sql"

    query = sql_path.read_text(encoding="utf-8")
    client.command(query)
    print("Table created successfully.")


if __name__ == "__main__":
    main()