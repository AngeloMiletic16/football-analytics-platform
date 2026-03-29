from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.storage.clickhouse_client import get_clickhouse_client


def load_metadata(metadata_path: str | Path) -> dict[str, Any]:
    with open(metadata_path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_tree_importances(
    model,
    feature_names: list[str],
    run_id: str,
    model_name: str,
    model_version: str,
    target_name: str,
    task_type: str,
    split_name: str,
) -> pd.DataFrame:
    importances = np.asarray(model.feature_importances_, dtype=float)

    df = pd.DataFrame(
        {
            "run_id": run_id,
            "model_name": model_name,
            "model_version": model_version,
            "target_name": target_name,
            "task_type": task_type,
            "split_name": split_name,
            "importance_source": "feature_importance",
            "class_name": None,
            "feature_name": feature_names,
            "importance_value": importances,
            "abs_importance_value": np.abs(importances),
            "raw_coefficient": None,
        }
    )

    df = df.sort_values("importance_value", ascending=False).reset_index(drop=True)
    df["importance_rank"] = np.arange(1, len(df) + 1)
    return df


def extract_binary_linear_importances(
    model,
    feature_names: list[str],
    run_id: str,
    model_name: str,
    model_version: str,
    target_name: str,
    task_type: str,
    split_name: str,
) -> pd.DataFrame:
    coef = np.asarray(model.coef_).ravel().astype(float)

    df = pd.DataFrame(
        {
            "run_id": run_id,
            "model_name": model_name,
            "model_version": model_version,
            "target_name": target_name,
            "task_type": task_type,
            "split_name": split_name,
            "importance_source": "coefficient",
            "class_name": None,
            "feature_name": feature_names,
            "importance_value": np.abs(coef),
            "abs_importance_value": np.abs(coef),
            "raw_coefficient": coef,
        }
    )

    df = df.sort_values("abs_importance_value", ascending=False).reset_index(drop=True)
    df["importance_rank"] = np.arange(1, len(df) + 1)
    return df


def extract_multiclass_linear_importances(
    model,
    feature_names: list[str],
    class_names: list[str],
    run_id: str,
    model_name: str,
    model_version: str,
    target_name: str,
    task_type: str,
    split_name: str,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    coef_matrix = np.asarray(model.coef_, dtype=float)

    for class_idx, class_name in enumerate(class_names):
        class_coef = coef_matrix[class_idx]

        class_df = pd.DataFrame(
            {
                "run_id": run_id,
                "model_name": model_name,
                "model_version": model_version,
                "target_name": target_name,
                "task_type": task_type,
                "split_name": split_name,
                "importance_source": "coefficient",
                "class_name": str(class_name),
                "feature_name": feature_names,
                "importance_value": np.abs(class_coef),
                "abs_importance_value": np.abs(class_coef),
                "raw_coefficient": class_coef,
            }
        )

        class_df = class_df.sort_values("abs_importance_value", ascending=False).reset_index(drop=True)
        class_df["importance_rank"] = np.arange(1, len(class_df) + 1)
        rows.append(class_df)

    return pd.concat(rows, ignore_index=True)


def build_feature_importances_dataframe(model, metadata: dict[str, Any]) -> pd.DataFrame:
    feature_names = metadata["feature_names"]
    run_id = metadata["run_id"]
    model_name = metadata["model_name"]
    model_version = metadata.get("model_version", "v1")
    target_name = metadata["target_name"]
    task_type = metadata["task_type"]
    split_name = metadata.get("split_name", "test")

    if hasattr(model, "feature_importances_"):
        return extract_tree_importances(
            model=model,
            feature_names=feature_names,
            run_id=run_id,
            model_name=model_name,
            model_version=model_version,
            target_name=target_name,
            task_type=task_type,
            split_name=split_name,
        )

    if hasattr(model, "coef_"):
        coef = np.asarray(model.coef_)
        if coef.ndim == 2 and coef.shape[0] > 1:
            class_names = metadata.get("class_names")
            if class_names is None:
                class_names = [str(c) for c in model.classes_]

            return extract_multiclass_linear_importances(
                model=model,
                feature_names=feature_names,
                class_names=class_names,
                run_id=run_id,
                model_name=model_name,
                model_version=model_version,
                target_name=target_name,
                task_type=task_type,
                split_name=split_name,
            )

        return extract_binary_linear_importances(
            model=model,
            feature_names=feature_names,
            run_id=run_id,
            model_name=model_name,
            model_version=model_version,
            target_name=target_name,
            task_type=task_type,
            split_name=split_name,
        )

    raise ValueError(f"Unsupported model type for feature importance extraction: {type(model)}")


def delete_existing_run_if_present(df: pd.DataFrame) -> None:
    if df.empty:
        return

    client = get_clickhouse_client()
    run_id = df["run_id"].iloc[0]
    model_name = df["model_name"].iloc[0]
    target_name = df["target_name"].iloc[0]

    delete_sql = f"""
    ALTER TABLE feature_importances
    DELETE WHERE run_id = '{run_id}'
      AND model_name = '{model_name}'
      AND target_name = '{target_name}'
    """
    client.command(delete_sql)


def write_feature_importances_to_clickhouse(df: pd.DataFrame) -> None:
    client = get_clickhouse_client()

    insert_columns = [
        "run_id",
        "model_name",
        "model_version",
        "target_name",
        "task_type",
        "split_name",
        "importance_source",
        "class_name",
        "feature_name",
        "importance_value",
        "abs_importance_value",
        "raw_coefficient",
        "importance_rank",
    ]

    insert_df = df[insert_columns].copy()

    client.insert_df(
        table="feature_importances",
        df=insert_df,
    )


def main(model_path: str, metadata_path: str, replace_existing: bool = True) -> None:
    model = joblib.load(model_path)
    metadata = load_metadata(metadata_path)

    df = build_feature_importances_dataframe(model=model, metadata=metadata)

    if replace_existing:
        delete_existing_run_if_present(df)

    write_feature_importances_to_clickhouse(df)

    print("Feature importances written successfully.")
    print(df.head(10).to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--metadata-path", required=True)
    parser.add_argument("--no-replace-existing", action="store_true")
    args = parser.parse_args()

    main(
        model_path=args.model_path,
        metadata_path=args.metadata_path,
        replace_existing=not args.no_replace_existing,
    )