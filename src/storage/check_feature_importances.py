import clickhouse_connect


def main() -> None:
    client = clickhouse_connect.get_client(
        host="localhost",
        port=8123,
        username="admin",
        password="admin123",
        database="football_analytics",
    )

    result = client.query(
        """
        SELECT
            model_name,
            task_type,
            target_name,
            dataset_name,
            feature_name,
            importance,
            rank
        FROM feature_importances
        ORDER BY created_at DESC, model_name, rank
        """
    )

    for row in result.result_rows:
        print(row)


if __name__ == "__main__":
    main()