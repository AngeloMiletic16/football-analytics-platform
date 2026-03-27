from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.features.build_features import add_model_features


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_clean_data() -> pd.DataFrame:
    file_path = get_project_root() / "data" / "processed" / "matches_v1_clean.csv"
    return pd.read_csv(file_path, parse_dates=["match_date"])


def time_split(df: pd.DataFrame):
    train_df = df[df["match_date"] < "2022-01-01"].copy()
    val_df = df[(df["match_date"] >= "2022-01-01") & (df["match_date"] < "2024-01-01")].copy()
    test_df = df[df["match_date"] >= "2024-01-01"].copy()
    return train_df, val_df, test_df


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    df = add_model_features(df)

    df["draw_target"] = (df["ft_result"] == "D").astype(int)
    df["home_vs_away_target"] = df["ft_result"].map({"H": 1, "A": 0})

    return df


def train_draw_model(X_train: pd.DataFrame, y_train: pd.Series):
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=3000,
                    random_state=42,
                    class_weight="balanced",
                ),
            ),
        ]
    )
    model.fit(X_train, y_train)
    return model


def train_home_away_model(X_train: pd.DataFrame, y_train: pd.Series):
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=3000,
                    random_state=42,
                ),
            ),
        ]
    )
    model.fit(X_train, y_train)
    return model


def hierarchical_predict(draw_model, home_away_model, X: pd.DataFrame, draw_threshold: float = 0.5):
    draw_prob = draw_model.predict_proba(X)[:, 1]
    non_draw_mask = draw_prob < draw_threshold

    final_pred = pd.Series(index=X.index, dtype="object")
    final_pred.loc[~non_draw_mask] = "D"

    if non_draw_mask.sum() > 0:
        home_away_pred = home_away_model.predict(X.loc[non_draw_mask])
        final_pred.loc[non_draw_mask] = pd.Series(home_away_pred, index=X.loc[non_draw_mask].index).map(
            {1: "H", 0: "A"}
        )

    return final_pred, draw_prob


def main() -> None:
    df = load_clean_data()
    df = make_features(df)

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
        "elo_abs_diff",
        "form3_abs_diff",
        "form5_abs_diff",
        "odd_abs_diff",
    ]

    X_train_draw = train_df[feature_cols]
    y_train_draw = train_df["draw_target"]

    X_val = val_df[feature_cols]
    y_val = val_df["ft_result"]

    X_test = test_df[feature_cols]
    y_test = test_df["ft_result"]

    # second-stage model trains only on non-draw matches
    non_draw_train = train_df[train_df["ft_result"] != "D"].copy()
    X_train_home_away = non_draw_train[feature_cols]
    y_train_home_away = non_draw_train["home_vs_away_target"]

    draw_model = train_draw_model(X_train_draw, y_train_draw)
    home_away_model = train_home_away_model(X_train_home_away, y_train_home_away)

    val_pred, _ = hierarchical_predict(draw_model, home_away_model, X_val, draw_threshold=0.70)
    test_pred, _ = hierarchical_predict(draw_model, home_away_model, X_test, draw_threshold=0.70)

    print("Validation accuracy:", accuracy_score(y_val, val_pred))
    print("Test accuracy:", accuracy_score(y_test, test_pred))

    print("\nValidation confusion matrix:")
    print(confusion_matrix(y_val, val_pred, labels=["H", "D", "A"]))

    print("\nTest confusion matrix:")
    print(confusion_matrix(y_test, test_pred, labels=["H", "D", "A"]))

    print("\nTest classification report:")
    print(classification_report(y_test, test_pred, zero_division=0))


if __name__ == "__main__":
    main()