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
            split_name,
            accuracy,
            roc_auc,
            log_loss,
            mae,
            rmse,
            r2,
            notes
        FROM model_metrics
        ORDER BY created_at DESC
        """
    )

    for row in result.result_rows:
        print(row)


if __name__ == "__main__":
    main()