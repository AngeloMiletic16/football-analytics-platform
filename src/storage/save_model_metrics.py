from __future__ import annotations

from datetime import datetime
from uuid import uuid4

import pandas as pd
import clickhouse_connect


def get_client():
    return clickhouse_connect.get_client(
        host="localhost",
        port=8123,
        username="admin",
        password="admin123",
        database="football_analytics",
    )


def build_rows() -> list[dict]:
    run_id = str(uuid4())
    created_at = datetime.now()

    rows = [
        {
            "run_id": run_id,
            "created_at": created_at,
            "model_name": "dummy_classifier_most_frequent",
            "task_type": "binary_classification",
            "target_name": "home_win",
            "dataset_name": "matches_v1_clean",
            "split_name": "test",
            "accuracy": 0.5847489618724047,
            "roc_auc": None,
            "log_loss": None,
            "mae": None,
            "rmse": None,
            "r2": None,
            "precision_macro": 0.29,
            "recall_macro": 0.50,
            "f1_macro": 0.37,
            "notes": "Baseline model using most frequent class.",
        },
        {
            "run_id": run_id,
            "created_at": created_at,
            "model_name": "logistic_regression",
            "task_type": "binary_classification",
            "target_name": "home_win",
            "dataset_name": "matches_v1_clean",
            "split_name": "test",
            "accuracy": 0.6851642129105323,
            "roc_auc": 0.7382528317389518,
            "log_loss": None,
            "mae": None,
            "rmse": None,
            "r2": None,
            "precision_macro": 0.68,
            "recall_macro": 0.66,
            "f1_macro": 0.67,
            "notes": "Binary home_win model.",
        },
        {
            "run_id": run_id,
            "created_at": created_at,
            "model_name": "random_forest",
            "task_type": "binary_classification",
            "target_name": "home_win",
            "dataset_name": "matches_v1_clean",
            "split_name": "test",
            "accuracy": 0.6881842204605512,
            "roc_auc": 0.7385415810787019,
            "log_loss": None,
            "mae": None,
            "rmse": None,
            "r2": None,
            "precision_macro": 0.68,
            "recall_macro": 0.67,
            "f1_macro": 0.67,
            "notes": "Best binary home_win model.",
        },
        {
            "run_id": run_id,
            "created_at": created_at,
            "model_name": "multiclass_logistic_regression",
            "task_type": "multiclass_classification",
            "target_name": "ft_result",
            "dataset_name": "matches_v1_clean",
            "split_name": "test",
            "accuracy": 0.5341638354095886,
            "roc_auc": None,
            "log_loss": 0.9672821142929461,
            "mae": None,
            "rmse": None,
            "r2": None,
            "precision_macro": 0.37,
            "recall_macro": 0.47,
            "f1_macro": 0.40,
            "notes": "Best multiclass accuracy model.",
        },
        {
            "run_id": run_id,
            "created_at": created_at,
            "model_name": "multiclass_random_forest",
            "task_type": "multiclass_classification",
            "target_name": "ft_result",
            "dataset_name": "matches_v1_clean",
            "split_name": "test",
            "accuracy": 0.5198187995469988,
            "roc_auc": None,
            "log_loss": 0.976567201854026,
            "mae": None,
            "rmse": None,
            "r2": None,
            "precision_macro": 0.50,
            "recall_macro": 0.50,
            "f1_macro": 0.50,
            "notes": "More balanced multiclass model, better on draws.",
        },
        {
            "run_id": run_id,
            "created_at": created_at,
            "model_name": "logistic_regression",
            "task_type": "binary_classification",
            "target_name": "over_2_5",
            "dataset_name": "matches_v1_clean",
            "split_name": "test",
            "accuracy": 0.5979614949037373,
            "roc_auc": 0.6221498988442293,
            "log_loss": None,
            "mae": None,
            "rmse": None,
            "r2": None,
            "precision_macro": 0.59,
            "recall_macro": 0.58,
            "f1_macro": 0.58,
            "notes": "Best over/under model.",
        },
        {
            "run_id": run_id,
            "created_at": created_at,
            "model_name": "random_forest",
            "task_type": "binary_classification",
            "target_name": "over_2_5",
            "dataset_name": "matches_v1_clean",
            "split_name": "test",
            "accuracy": 0.5964514911287279,
            "roc_auc": 0.6155851074823121,
            "log_loss": None,
            "mae": None,
            "rmse": None,
            "r2": None,
            "precision_macro": 0.60,
            "recall_macro": 0.58,
            "f1_macro": 0.57,
            "notes": "Alternative over/under model.",
        },
        {
            "run_id": run_id,
            "created_at": created_at,
            "model_name": "linear_regression",
            "task_type": "regression",
            "target_name": "total_goals",
            "dataset_name": "matches_v1_clean",
            "split_name": "test",
            "accuracy": None,
            "roc_auc": None,
            "log_loss": None,
            "mae": 1.3046831399021563,
            "rmse": 1.6237575175098802,
            "r2": 0.06539486512907189,
            "precision_macro": None,
            "recall_macro": None,
            "f1_macro": None,
            "notes": "Best total goals regression result.",
        },
        {
            "run_id": run_id,
            "created_at": created_at,
            "model_name": "random_forest_regressor",
            "task_type": "regression",
            "target_name": "total_goals",
            "dataset_name": "matches_v1_clean",
            "split_name": "test",
            "accuracy": None,
            "roc_auc": None,
            "log_loss": None,
            "mae": 1.307066333781128,
            "rmse": 1.6289694964740802,
            "r2": 0.05938539633884343,
            "precision_macro": None,
            "recall_macro": None,
            "f1_macro": None,
            "notes": "Alternative total goals regression result.",
        },
    ]
    return rows


def main() -> None:
    client = get_client()
    rows = build_rows()
    df = pd.DataFrame(rows)

    client.insert_df("model_metrics", df)

    print(f"Inserted {len(df)} rows into model_metrics.")


if __name__ == "__main__":
    main()