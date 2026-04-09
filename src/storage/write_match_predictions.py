from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import clickhouse_connect
import joblib
import numpy as np
import pandas as pd

from src.models.registry import get_model_config


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate match predictions from a saved model and write them to ClickHouse."
    )
    parser.add_argument(
        "--model-name",
        type=str,
        required=True,
        help="Registry model name, e.g. over_2_5_lr_v1",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=None,
        help="Optional override for dataset path. If omitted, dataset_path from metadata is used.",
    )
    parser.add_argument(
        "--allow-duplicates",
        action="store_true",
        help="Allow inserting rows even if predictions for the same model already exist.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def resolve_path(path_str: str | None) -> Path | None:
    if path_str is None:
        return None

    p = Path(path_str)
    if p.is_absolute():
        return p
    return PROJECT_ROOT / p


def get_clickhouse_client():
    host = os.getenv("CLICKHOUSE_HOST", "localhost")
    port = int(os.getenv("CLICKHOUSE_PORT", "8123"))
    username = os.getenv("CLICKHOUSE_USER", "default")
    password = os.getenv("CLICKHOUSE_PASSWORD", "")
    database = os.getenv("CLICKHOUSE_DATABASE", "default")

    return clickhouse_connect.get_client(
        host=host,
        port=port,
        username=username,
        password=password,
        database=database,
    )


def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    return pd.read_csv(path)


def build_supported_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    numeric_candidates = [
        "home_elo",
        "away_elo",
        "form3_home",
        "form5_home",
        "form3_away",
        "form5_away",
        "odd_home",
        "odd_draw",
        "odd_away",
    ]
    for col in numeric_candidates:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if {"home_elo", "away_elo"}.issubset(df.columns):
        df["elo_diff"] = df["home_elo"] - df["away_elo"]
        df["elo_abs_diff"] = df["elo_diff"].abs()
        df["elo_sum"] = df["home_elo"] + df["away_elo"]

    if {"form3_home", "form3_away"}.issubset(df.columns):
        df["form3_diff"] = df["form3_home"] - df["form3_away"]
        df["form3_abs_diff"] = df["form3_diff"].abs()
        df["form3_sum"] = df["form3_home"] + df["form3_away"]

    if {"form5_home", "form5_away"}.issubset(df.columns):
        df["form5_diff"] = df["form5_home"] - df["form5_away"]
        df["form5_abs_diff"] = df["form5_diff"].abs()
        df["form5_sum"] = df["form5_home"] + df["form5_away"]

    if {"odd_home", "odd_away"}.issubset(df.columns):
        df["odd_abs_diff"] = (df["odd_home"] - df["odd_away"]).abs()

    if {"odd_home", "odd_draw", "odd_away"}.issubset(df.columns):
        odd_home = df["odd_home"].replace(0, np.nan)
        odd_draw = df["odd_draw"].replace(0, np.nan)
        odd_away = df["odd_away"].replace(0, np.nan)

        inv_home = 1.0 / odd_home
        inv_draw = 1.0 / odd_draw
        inv_away = 1.0 / odd_away
        inv_sum = inv_home + inv_draw + inv_away

        df["implied_home_prob"] = inv_home / inv_sum
        df["implied_draw_prob"] = inv_draw / inv_sum
        df["implied_away_prob"] = inv_away / inv_sum

    return df


def prepare_model_frame(
    df: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = build_supported_features(df)

    missing_features = [c for c in feature_columns if c not in df.columns]
    if missing_features:
        raise ValueError(
            f"Dataset is missing required feature columns after feature engineering: {missing_features}"
        )

    if target_column not in df.columns:
        raise ValueError(f"Dataset is missing target column: {target_column}")

    model_df = df.copy()
    model_df = model_df.dropna(subset=feature_columns + [target_column]).reset_index(drop=True)

    if model_df.empty:
        raise ValueError("No rows left after dropping NaNs for required features and target.")

    X = model_df[feature_columns].copy()
    return model_df, X


def get_estimator_from_pipeline(model):
    if hasattr(model, "named_steps") and model.named_steps:
        last_step_name = list(model.named_steps.keys())[-1]
        return model.named_steps[last_step_name]
    return model


def build_predictions_frame(
    source_df: pd.DataFrame,
    model,
    metadata: dict[str, Any],
    model_name: str,
) -> pd.DataFrame:
    feature_columns = metadata["feature_columns"]
    target_column = metadata["target"]
    algorithm = metadata.get("algorithm", "unknown")
    task_type = metadata.get("task_type", "classification")
    problem_type = metadata.get("problem_type", "unknown")

    working_df, X = prepare_model_frame(
        df=source_df,
        feature_columns=feature_columns,
        target_column=target_column,
    )

    y_true = working_df[target_column].copy()
    y_pred = model.predict(X)

    if not hasattr(model, "predict_proba"):
        raise ValueError("Model does not support predict_proba, but this pipeline expects probabilities.")

    proba = model.predict_proba(X)
    estimator = get_estimator_from_pipeline(model)
    classes = list(estimator.classes_)

    pred_class_idx = proba.argmax(axis=1)
    predicted_label = np.array(classes, dtype=object)[pred_class_idx]
    predicted_probability = proba[np.arange(len(proba)), pred_class_idx]

    positive_class_probability = np.full(len(working_df), np.nan)
    if set(classes) == {0, 1}:
        positive_idx = classes.index(1)
        positive_class_probability = proba[:, positive_idx]

    out = pd.DataFrame()

    passthrough_columns = [
        "match_id",
        "division",
        "season",
        "match_date",
        "home_team",
        "away_team",
        "ft_home",
        "ft_away",
        "total_goals",
    ]
    for col in passthrough_columns:
        if col in working_df.columns:
            out[col] = working_df[col]

    if "match_date" in out.columns:
        out["match_date"] = pd.to_datetime(out["match_date"], errors="coerce")

    out["model_name"] = model_name
    out["target"] = target_column
    out["algorithm"] = algorithm
    out["task_type"] = task_type
    out["problem_type"] = problem_type

    out["true_label"] = y_true.astype(str)
    out["actual_value"] = y_true.astype(str)

    out["predicted_label"] = pd.Series(predicted_label).astype(str)
    out["predicted_value"] = pd.Series(predicted_label).astype(str)
    out["predicted_class"] = pd.Series(predicted_label).astype(str)

    out["predicted_probability"] = predicted_probability.astype(float)
    out["probability"] = predicted_probability.astype(float)
    out["confidence"] = predicted_probability.astype(float)
    out["positive_class_probability"] = positive_class_probability.astype(float)

    out["is_correct"] = (pd.Series(predicted_label).astype(str) == y_true.astype(str)).astype(int)

    created_at = datetime.now(timezone.utc).replace(tzinfo=None)
    out["created_at"] = created_at
    out["created_at_utc"] = created_at

    return out


def get_table_schema(client, table_name: str) -> dict[str, str]:
    result = client.query(f"DESCRIBE TABLE {table_name}")
    return {row[0]: row[1] for row in result.result_rows}


def get_table_columns(client, table_name: str) -> list[str]:
    return list(get_table_schema(client, table_name).keys())


def existing_prediction_count(client, model_name: str, target: str) -> int:
    table_columns = get_table_columns(client, "match_predictions")

    where_clauses = []

    if "model_name" in table_columns:
        where_clauses.append(f"model_name = '{model_name}'")

    if "target" in table_columns:
        where_clauses.append(f"target = '{target}'")

    if not where_clauses:
        return 0

    query = (
        "SELECT count() "
        "FROM match_predictions "
        f"WHERE {' AND '.join(where_clauses)}"
    )
    result = client.query(query)
    return int(result.result_rows[0][0])

def is_clickhouse_string_type(ch_type: str) -> bool:
    return any(token in ch_type for token in ["String", "Enum", "UUID"])


def is_clickhouse_int_type(ch_type: str) -> bool:
    int_tokens = [
        "Int8", "Int16", "Int32", "Int64", "Int128", "Int256",
        "UInt8", "UInt16", "UInt32", "UInt64", "UInt128", "UInt256",
    ]
    return any(token in ch_type for token in int_tokens)


def is_clickhouse_float_type(ch_type: str) -> bool:
    return any(token in ch_type for token in ["Float32", "Float64", "Decimal"])


def is_clickhouse_datetime_type(ch_type: str) -> bool:
    return "DateTime" in ch_type


def is_clickhouse_date_type(ch_type: str) -> bool:
    return ("Date" in ch_type) and ("DateTime" not in ch_type)


def align_dataframe_to_clickhouse_schema(
    df: pd.DataFrame,
    table_schema: dict[str, str],
) -> pd.DataFrame:
    aligned = df.copy()

    for col in aligned.columns:
        if col not in table_schema:
            continue

        ch_type = table_schema[col]

        if is_clickhouse_string_type(ch_type):
            aligned[col] = aligned[col].apply(
                lambda x: None if pd.isna(x) else str(x)
            )

        elif is_clickhouse_int_type(ch_type):
            aligned[col] = pd.to_numeric(aligned[col], errors="coerce")

        elif is_clickhouse_float_type(ch_type):
            aligned[col] = pd.to_numeric(aligned[col], errors="coerce")

        elif is_clickhouse_datetime_type(ch_type):
            aligned[col] = pd.to_datetime(aligned[col], errors="coerce")

        elif is_clickhouse_date_type(ch_type):
            aligned[col] = pd.to_datetime(aligned[col], errors="coerce").dt.date

    return aligned


def normalize_for_insert(df: pd.DataFrame) -> pd.DataFrame:
    normalized = df.copy()

    for col in normalized.columns:
        if pd.api.types.is_datetime64_any_dtype(normalized[col]):
            normalized[col] = normalized[col].apply(
                lambda x: None if pd.isna(x) else x.to_pydatetime()
            )
        else:
            normalized[col] = normalized[col].astype(object)
            normalized[col] = normalized[col].where(pd.notna(normalized[col]), None)

    return normalized


def insert_predictions(client, predictions_df: pd.DataFrame, table_name: str = "match_predictions") -> int:
    table_schema = get_table_schema(client, table_name)
    table_columns = list(table_schema.keys())

    insert_columns = [col for col in predictions_df.columns if col in table_columns]

    if not insert_columns:
        raise ValueError(
            f"No overlapping columns between predictions dataframe and ClickHouse table '{table_name}'."
        )

    final_df = predictions_df[insert_columns].copy()
    final_df = align_dataframe_to_clickhouse_schema(final_df, table_schema)
    final_df = normalize_for_insert(final_df)

    rows = final_df.values.tolist()

    client.insert(
        table=table_name,
        data=rows,
        column_names=insert_columns,
    )
    return len(rows)


def main() -> None:
    args = parse_args()

    print(f"Loading registry config for model: {args.model_name}")
    config = get_model_config(args.model_name)

    artifact_path = Path(config["artifact_path"])
    metadata_path = Path(config["metadata_path"])

    if not artifact_path.exists():
        raise FileNotFoundError(f"Model artifact not found: {artifact_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    metadata = load_json(metadata_path)

    dataset_path = (
        resolve_path(args.dataset_path)
        if args.dataset_path
        else resolve_path(metadata.get("dataset_path"))
    )
    if dataset_path is None:
        raise ValueError("Could not resolve dataset path.")

    print(f"Model artifact: {artifact_path}")
    print(f"Metadata path: {metadata_path}")
    print(f"Dataset path: {dataset_path}")

    print("Loading model...")
    model = joblib.load(artifact_path)

    print("Loading dataset...")
    df = load_dataset(dataset_path)

    print("Generating predictions...")
    predictions_df = build_predictions_frame(
        source_df=df,
        model=model,
        metadata=metadata,
        model_name=args.model_name,
    )

    print(f"Prepared rows for insert: {len(predictions_df)}")
    print(f"Target: {metadata['target']}")
    print(f"Algorithm: {metadata.get('algorithm')}")

    client = get_clickhouse_client()
    print("Connected to ClickHouse.")

    if not args.allow_duplicates:
        existing_count = existing_prediction_count(
            client=client,
            model_name=args.model_name,
            target=metadata["target"],
        )
        if existing_count > 0:
            raise ValueError(
                f"match_predictions already contains {existing_count} rows for "
                f"model_name='{args.model_name}' and target='{metadata['target']}'. "
                "Use --allow-duplicates only if you intentionally want another insert."
            )

    inserted_rows = insert_predictions(client, predictions_df, table_name="match_predictions")

    print(f"Inserted rows into ClickHouse: {inserted_rows}")
    print("Done.")


if __name__ == "__main__":
    main()