from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List

import joblib
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "matches_v1_clean.csv"
MODEL_OUTPUT_PATH = PROJECT_ROOT / "artifacts" / "models" / "total_goals_lr_v1.joblib"
METADATA_OUTPUT_PATH = PROJECT_ROOT / "artifacts" / "metadata" / "total_goals_lr_v1.json"

TARGET_COLUMN = "total_goals"
MODEL_NAME = "total_goals_lr_v1"
RANDOM_STATE = 42
TEST_SIZE = 0.20


BASE_FEATURE_COLUMNS = [
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

DERIVED_FEATURE_COLUMNS = [
    "elo_diff",
    "form3_diff",
    "form5_diff",
    "elo_abs_diff",
    "form3_abs_diff",
    "form5_abs_diff",
    "odd_abs_diff",
    "implied_home_prob",
    "implied_draw_prob",
    "implied_away_prob",
    "elo_sum",
    "form3_sum",
    "form5_sum",
]

FEATURE_COLUMNS = BASE_FEATURE_COLUMNS + DERIVED_FEATURE_COLUMNS


@dataclass
class TrainingMetadata:
    model_name: str
    target: str
    task_type: str
    problem_type: str
    algorithm: str
    artifact_path: str
    dataset_path: str
    trained_at_utc: str
    random_state: int
    test_size: float
    train_rows: int
    test_rows: int
    feature_columns: List[str]
    metrics: dict
    estimator_step_name: str
    feature_importance_type: str


def ensure_parent_dirs() -> None:
    MODEL_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    METADATA_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)


def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path)

    required_columns = set(BASE_FEATURE_COLUMNS + [TARGET_COLUMN])
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(
            f"Dataset is missing required columns: {sorted(missing_columns)}"
        )

    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["elo_diff"] = df["home_elo"] - df["away_elo"]
    df["form3_diff"] = df["form3_home"] - df["form3_away"]
    df["form5_diff"] = df["form5_home"] - df["form5_away"]

    df["elo_abs_diff"] = df["elo_diff"].abs()
    df["form3_abs_diff"] = df["form3_diff"].abs()
    df["form5_abs_diff"] = df["form5_diff"].abs()

    df["odd_abs_diff"] = (df["odd_home"] - df["odd_away"]).abs()

    inv_home = 1.0 / df["odd_home"].replace(0, pd.NA)
    inv_draw = 1.0 / df["odd_draw"].replace(0, pd.NA)
    inv_away = 1.0 / df["odd_away"].replace(0, pd.NA)

    inv_sum = inv_home + inv_draw + inv_away

    df["implied_home_prob"] = inv_home / inv_sum
    df["implied_draw_prob"] = inv_draw / inv_sum
    df["implied_away_prob"] = inv_away / inv_sum

    df["elo_sum"] = df["home_elo"] + df["away_elo"]
    df["form3_sum"] = df["form3_home"] + df["form3_away"]
    df["form5_sum"] = df["form5_home"] + df["form5_away"]

    return df


def prepare_training_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    df = build_features(df)

    before_drop = len(df)
    model_df = df[FEATURE_COLUMNS + [TARGET_COLUMN]].dropna().copy()
    after_drop = len(model_df)

    dropped_rows = before_drop - after_drop
    if dropped_rows > 0:
        print(f"Dropped rows due to missing values: {dropped_rows}")

    X = model_df[FEATURE_COLUMNS].copy()
    y = pd.to_numeric(model_df[TARGET_COLUMN], errors="coerce").copy()

    return X, y


def build_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", LinearRegression()),
        ]
    )


def evaluate_model(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    rmse = mse ** 0.5

    metrics = {
        "mae": round(float(mean_absolute_error(y_test, y_pred)), 6),
        "rmse": round(float(rmse), 6),
        "r2": round(float(r2_score(y_test, y_pred)), 6),
    }
    return metrics


def save_metadata(metadata: TrainingMetadata) -> None:
    with METADATA_OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(asdict(metadata), f, indent=2, ensure_ascii=False)


def main() -> None:
    print("Loading dataset...")
    df = load_dataset(DATA_PATH)

    print("Preparing features...")
    X, y = prepare_training_frame(df)

    print(f"Total rows for modeling: {len(X)}")
    print(f"Target mean ({TARGET_COLUMN}): {y.mean():.4f}")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    print(f"Training rows: {len(X_train)}")
    print(f"Test rows: {len(X_test)}")

    model = build_pipeline()

    print("Training Linear Regression...")
    model.fit(X_train, y_train)

    print("Evaluating model...")
    metrics = evaluate_model(model, X_test, y_test)

    print(f"Test MAE: {metrics['mae']:.4f}")
    print(f"Test RMSE: {metrics['rmse']:.4f}")
    print(f"Test R^2: {metrics['r2']:.4f}")

    ensure_parent_dirs()

    print(f"Saving model to: {MODEL_OUTPUT_PATH}")
    joblib.dump(model, MODEL_OUTPUT_PATH)

    metadata = TrainingMetadata(
        model_name=MODEL_NAME,
        target=TARGET_COLUMN,
        task_type="regression",
        problem_type="regression",
        algorithm="linear_regression",
        artifact_path=str(MODEL_OUTPUT_PATH.relative_to(PROJECT_ROOT)),
        dataset_path=str(DATA_PATH.relative_to(PROJECT_ROOT)),
        trained_at_utc=datetime.now(timezone.utc).isoformat(),
        random_state=RANDOM_STATE,
        test_size=TEST_SIZE,
        train_rows=len(X_train),
        test_rows=len(X_test),
        feature_columns=FEATURE_COLUMNS,
        metrics=metrics,
        estimator_step_name="model",
        feature_importance_type="coefficient_abs",
    )

    print(f"Saving metadata to: {METADATA_OUTPUT_PATH}")
    save_metadata(metadata)

    print("Done.")
    print(f"Model artifact created: {MODEL_OUTPUT_PATH}")
    print(f"Metadata artifact created: {METADATA_OUTPUT_PATH}")


if __name__ == "__main__":
    main()