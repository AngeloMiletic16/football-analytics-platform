from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
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


def add_goal_features(df: pd.DataFrame) -> pd.DataFrame:
    featured = add_model_features(df.copy())

    featured["implied_home_prob"] = 1 / featured["odd_home"]
    featured["implied_draw_prob"] = 1 / featured["odd_draw"]
    featured["implied_away_prob"] = 1 / featured["odd_away"]

    featured["elo_sum"] = featured["home_elo"] + featured["away_elo"]
    featured["form3_sum"] = featured["form3_home"] + featured["form3_away"]
    featured["form5_sum"] = featured["form5_home"] + featured["form5_away"]

    return featured


def evaluate_regression(name: str, model, X_val, y_val, X_test, y_test) -> None:
    val_pred = model.predict(X_val)
    test_pred = model.predict(X_test)

    val_mae = mean_absolute_error(y_val, val_pred)
    val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
    val_r2 = r2_score(y_val, val_pred)

    test_mae = mean_absolute_error(y_test, test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
    test_r2 = r2_score(y_test, test_pred)

    print(f"\n===== {name} =====")

    print("Validation MAE:", val_mae)
    print("Validation RMSE:", val_rmse)
    print("Validation R2:", val_r2)

    print("\nTest MAE:", test_mae)
    print("Test RMSE:", test_rmse)
    print("Test R2:", test_r2)

    preview_df = pd.DataFrame(
        {
            "actual_total_goals": y_test.values[:10],
            "predicted_total_goals": test_pred[:10],
        }
    )
    print("\nSample predictions:")
    print(preview_df)


def main() -> None:
    df = load_clean_data()
    df = add_goal_features(df)

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
        "implied_home_prob",
        "implied_draw_prob",
        "implied_away_prob",
        "elo_sum",
        "form3_sum",
        "form5_sum",
    ]

    target_col = "total_goals"

    X_train = train_df[feature_cols]
    y_train = train_df[target_col]

    X_val = val_df[feature_cols]
    y_val = val_df[target_col]

    X_test = test_df[feature_cols]
    y_test = test_df[target_col]

    print("Train shape:", train_df.shape)
    print("Validation shape:", val_df.shape)
    print("Test shape:", test_df.shape)

    linear_model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("regressor", LinearRegression()),
        ]
    )

    rf_model = RandomForestRegressor(
        n_estimators=400,
        max_depth=10,
        min_samples_split=20,
        min_samples_leaf=10,
        random_state=42,
        n_jobs=-1,
    )

    linear_model.fit(X_train, y_train)
    rf_model.fit(X_train, y_train)

    evaluate_regression("Linear Regression Total Goals", linear_model, X_val, y_val, X_test, y_test)
    evaluate_regression("Random Forest Total Goals", rf_model, X_val, y_val, X_test, y_test)

    feature_importance = pd.DataFrame(
        {
            "feature": feature_cols,
            "importance": rf_model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)

    print("\n===== Random Forest Regressor Feature Importances =====")
    print(feature_importance)


if __name__ == "__main__":
    main()