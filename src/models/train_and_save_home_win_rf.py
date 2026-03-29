from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split


DATA_PATH = Path("data/processed/matches_v1_clean.csv")
MODEL_OUTPUT_PATH = Path("artifacts/models/home_win_rf_v1.joblib")


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
]

FEATURE_COLUMNS = BASE_FEATURE_COLUMNS + DERIVED_FEATURE_COLUMNS
TARGET_COLUMN = "home_win"


def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    return pd.read_csv(path)


def build_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    required_base_columns = [
        "home_elo",
        "away_elo",
        "form3_home",
        "form5_home",
        "form3_away",
        "form5_away",
        "odd_home",
        "odd_away",
    ]

    missing = [col for col in required_base_columns if col not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns for derived features: {missing}")

    df = df.copy()

    df["elo_diff"] = df["home_elo"] - df["away_elo"]
    df["form3_diff"] = df["form3_home"] - df["form3_away"]
    df["form5_diff"] = df["form5_home"] - df["form5_away"]

    df["elo_abs_diff"] = (df["home_elo"] - df["away_elo"]).abs()
    df["form3_abs_diff"] = (df["form3_home"] - df["form3_away"]).abs()
    df["form5_abs_diff"] = (df["form5_home"] - df["form5_away"]).abs()
    df["odd_abs_diff"] = (df["odd_home"] - df["odd_away"]).abs()

    return df


def validate_columns(df: pd.DataFrame) -> None:
    required_columns = FEATURE_COLUMNS + [TARGET_COLUMN]
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns for training: {missing}")


def train_model(df: pd.DataFrame) -> RandomForestClassifier:
    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)

    print(f"Training rows: {len(X_train)}")
    print(f"Test rows: {len(X_test)}")
    print(f"Test accuracy: {accuracy:.4f}")
    print(f"Test ROC AUC: {roc_auc:.4f}")

    return model


def save_model(model: RandomForestClassifier, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_path)
    print(f"Model saved to: {output_path}")


def main() -> None:
    df = load_data(DATA_PATH)
    df = build_derived_features(df)
    validate_columns(df)

    model = train_model(df)
    save_model(model, MODEL_OUTPUT_PATH)


if __name__ == "__main__":
    main()