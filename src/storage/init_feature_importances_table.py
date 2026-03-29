from pathlib import Path
import clickhouse_connect


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def main() -> None:
    client = clickhouse_connect.get_client(
        host="localhost",
        port=8123,
        username="admin",
        password="admin123",
        database="football_analytics",
    )

    sql_path = get_project_root() / "sql" / "create_feature_importances_table.sql"
    query = sql_path.read_text(encoding="utf-8")

    client.command(query)
    print("feature_importances table created successfully.")


if __name__ == "__main__":
    main()