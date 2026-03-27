from pathlib import Path
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from src.features.build_features import add_model_features


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_clean_data() -> pd.DataFrame:
    file_path = get_project_root() / "data" / "processed" / "matches_v1_clean.csv"
    df = pd.read_csv(file_path, parse_dates=["match_date"])
    return df


def time_split(df: pd.DataFrame):
    train_df = df[df["match_date"] < "2022-01-01"].copy()
    val_df = df[(df["match_date"] >= "2022-01-01") & (df["match_date"] < "2024-01-01")].copy()
    test_df = df[df["match_date"] >= "2024-01-01"].copy()
    return train_df, val_df, test_df


def main() -> None:
    df = load_clean_data()
    df = add_model_features(df)

    train_df, val_df, test_df = time_split(df)

    feature_cols = [
        "home_elo",
        "away_elo",
        "form3_home",
        "form5_home",
        "form3_away",
        "form5_away",
        "odd_home",
        "odd_draw",
        "odd_away",
        "elo_diff",
        "form3_diff",
        "form5_diff",
    ]

    target_col = "home_win"

    X_train = train_df[feature_cols]
    y_train = train_df[target_col]

    X_val = val_df[feature_cols]
    y_val = val_df[target_col]

    X_test = test_df[feature_cols]
    y_test = test_df[target_col]

    model = DummyClassifier(strategy="most_frequent")
    model.fit(X_train, y_train)

    val_pred = model.predict(X_val)
    test_pred = model.predict(X_test)

    print("Train shape:", train_df.shape)
    print("Validation shape:", val_df.shape)
    print("Test shape:", test_df.shape)

    print("\nValidation accuracy:", accuracy_score(y_val, val_pred))
    print("Test accuracy:", accuracy_score(y_test, test_pred))

    print("\nValidation confusion matrix:")
    print(confusion_matrix(y_val, val_pred))

    print("\nTest confusion matrix:")
    print(confusion_matrix(y_test, test_pred))

    print("\nTest classification report:")
    print(classification_report(y_test, test_pred, zero_division=0))


if __name__ == "__main__":
    main()