from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from src.models.registry import MODEL_REGISTRY
from src.storage.clickhouse_client import get_clickhouse_client


def load_metadata(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_dataset(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["match_date"] = pd.to_datetime(df["match_date"]).dt.date
    return df


def build_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["elo_diff"] = df["home_elo"] - df["away_elo"]
    df["form3_diff"] = df["form3_home"] - df["form3_away"]
    df["form5_diff"] = df["form5_home"] - df["form5_away"]

    df["elo_abs_diff"] = (df["home_elo"] - df["away_elo"]).abs()
    df["form3_abs_diff"] = (df["form3_home"] - df["form3_away"]).abs()
    df["form5_abs_diff"] = (df["form5_home"] - df["form5_away"]).abs()
    df["odd_abs_diff"] = (df["odd_home"] - df["odd_away"]).abs()

    return df


def build_base_output_frame(
    df: pd.DataFrame,
    prediction_run_id: str,
    model_name: str,
    model_version: str,
    target_name: str,
    task_type: str,
) -> pd.DataFrame:
    out = pd.DataFrame(
        {
            "prediction_run_id": prediction_run_id,
            "model_name": model_name,
            "model_version": model_version,
            "target_name": target_name,
            "task_type": task_type,
            "match_id": df["match_id"].astype(str),
            "match_date": df["match_date"],
            "division": df["division"].astype(str),
            "home_team": df["home_team"].astype(str),
            "away_team": df["away_team"].astype(str),
            "actual_home_goals": df["ft_home"] if "ft_home" in df.columns else None,
            "actual_away_goals": df["ft_away"] if "ft_away" in df.columns else None,
            "actual_ft_result": df["ft_result"] if "ft_result" in df.columns else None,
            "actual_home_win": df["home_win"] if "home_win" in df.columns else None,
            "actual_over_2_5": df["over_2_5"] if "over_2_5" in df.columns else None,
            "actual_total_goals": df["total_goals"] if "total_goals" in df.columns else None,
            "predicted_label": None,
            "predicted_value": None,
            "predicted_probability": None,
            "home_win_probability": None,
            "draw_probability": None,
            "away_win_probability": None,
            "over_2_5_probability": None,
            "under_2_5_probability": None,
            "absolute_error": None,
            "is_correct": None,
        }
    )
    return out


def build_predictions_dataframe(model, metadata: dict[str, Any], df: pd.DataFrame) -> pd.DataFrame:
    feature_names = metadata["feature_names"]
    X = df[feature_names].copy()

    prediction_run_id = metadata.get("prediction_run_id", metadata["run_id"])
    model_name = metadata["model_name"]
    model_version = metadata.get("model_version", "v1")
    target_name = metadata["target_name"]
    task_type = metadata["task_type"]

    out = build_base_output_frame(
        df=df,
        prediction_run_id=prediction_run_id,
        model_name=model_name,
        model_version=model_version,
        target_name=target_name,
        task_type=task_type,
    )

    if task_type == "binary_classification":
        preds = model.predict(X)
        probs = model.predict_proba(X)

        class_to_index = {cls: idx for idx, cls in enumerate(model.classes_)}
        positive_class = metadata.get("positive_class_label", 1)
        positive_idx = class_to_index[positive_class]

        out["predicted_label"] = preds.astype(str)
        out["predicted_value"] = preds.astype(float)
        out["predicted_probability"] = probs[:, positive_idx]

        if target_name == "home_win":
            out["home_win_probability"] = probs[:, positive_idx]

        actual_col = f"actual_{target_name}"
        if actual_col in out.columns:
            out["is_correct"] = (out[actual_col] == preds).astype("uint8")

        return out

    raise ValueError(f"Unsupported task_type for this script version: {task_type}")


def delete_existing_prediction_run_if_present(df: pd.DataFrame) -> None:
    if df.empty:
        return

    client = get_clickhouse_client()

    prediction_run_id = df["prediction_run_id"].iloc[0]
    target_name = df["target_name"].iloc[0]
    model_name = df["model_name"].iloc[0]

    delete_sql = f"""
    ALTER TABLE match_predictions
    DELETE WHERE prediction_run_id = '{prediction_run_id}'
      AND target_name = '{target_name}'
      AND model_name = '{model_name}'
    """
    client.command(delete_sql)


def write_predictions_to_clickhouse(df: pd.DataFrame) -> None:
    client = get_clickhouse_client()

    insert_columns = [
        "prediction_run_id",
        "model_name",
        "model_version",
        "target_name",
        "task_type",
        "match_id",
        "match_date",
        "division",
        "home_team",
        "away_team",
        "actual_home_goals",
        "actual_away_goals",
        "actual_ft_result",
        "actual_home_win",
        "actual_over_2_5",
        "actual_total_goals",
        "predicted_label",
        "predicted_value",
        "predicted_probability",
        "home_win_probability",
        "draw_probability",
        "away_win_probability",
        "over_2_5_probability",
        "under_2_5_probability",
        "absolute_error",
        "is_correct",
    ]

    insert_df = df[insert_columns].copy()

    client.insert_df(
        table="match_predictions",
        df=insert_df,
    )


def main(model_key: str, dataset_path: str, replace_existing: bool = True) -> None:
    if model_key not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model_key: {model_key}")

    registry_item = MODEL_REGISTRY[model_key]
    model = joblib.load(registry_item["model_path"])
    metadata = load_metadata(registry_item["metadata_path"])

    metadata["target_name"] = registry_item["target_name"]
    metadata["task_type"] = registry_item["task_type"]
    metadata["positive_class_label"] = registry_item.get("positive_class_label", 1)

    df = load_dataset(dataset_path)
    df = build_derived_features(df)

    pred_df = build_predictions_dataframe(model=model, metadata=metadata, df=df)

    if replace_existing:
        delete_existing_prediction_run_if_present(pred_df)

    write_predictions_to_clickhouse(pred_df)

    print("Match predictions written successfully.")
    print(pred_df.head(10).to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-key", required=True)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--no-replace-existing", action="store_true")
    args = parser.parse_args()

    main(
        model_key=args.model_key,
        dataset_path=args.dataset_path,
        replace_existing=not args.no_replace_existing,
    )