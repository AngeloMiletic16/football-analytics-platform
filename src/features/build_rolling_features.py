from pathlib import Path

import pandas as pd


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_clean_data() -> pd.DataFrame:
    file_path = get_project_root() / "data" / "processed" / "matches_v1_clean.csv"
    df = pd.read_csv(file_path, parse_dates=["match_date"])
    return df.sort_values("match_date").reset_index(drop=True)


def add_team_match_context(df: pd.DataFrame) -> pd.DataFrame:
    enriched = df.copy()

    enriched["home_points"] = enriched["ft_result"].map({"H": 3, "D": 1, "A": 0})
    enriched["away_points"] = enriched["ft_result"].map({"H": 0, "D": 1, "A": 3})

    enriched["home_is_draw"] = (enriched["ft_result"] == "D").astype(int)
    enriched["away_is_draw"] = (enriched["ft_result"] == "D").astype(int)

    return enriched


def add_rolling_home_features(df: pd.DataFrame) -> pd.DataFrame:
    enriched = df.copy()

    enriched["home_avg_goals_scored_last_5"] = (
        enriched.groupby("home_team")["ft_home"]
        .transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean())
    )

    enriched["home_avg_goals_conceded_last_5"] = (
        enriched.groupby("home_team")["ft_away"]
        .transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean())
    )

    enriched["home_points_last_5"] = (
        enriched.groupby("home_team")["home_points"]
        .transform(lambda s: s.shift(1).rolling(5, min_periods=1).sum())
    )

    enriched["home_draw_rate_last_5"] = (
        enriched.groupby("home_team")["home_is_draw"]
        .transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean())
    )

    return enriched


def add_rolling_away_features(df: pd.DataFrame) -> pd.DataFrame:
    enriched = df.copy()

    enriched["away_avg_goals_scored_last_5"] = (
        enriched.groupby("away_team")["ft_away"]
        .transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean())
    )

    enriched["away_avg_goals_conceded_last_5"] = (
        enriched.groupby("away_team")["ft_home"]
        .transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean())
    )

    enriched["away_points_last_5"] = (
        enriched.groupby("away_team")["away_points"]
        .transform(lambda s: s.shift(1).rolling(5, min_periods=1).sum())
    )

    enriched["away_draw_rate_last_5"] = (
        enriched.groupby("away_team")["away_is_draw"]
        .transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean())
    )

    return enriched


def build_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    featured = add_team_match_context(df)
    featured = add_rolling_home_features(featured)
    featured = add_rolling_away_features(featured)

    featured = featured.drop(columns=["home_points", "away_points", "home_is_draw", "away_is_draw"])

    return featured


def save_featured_data(df: pd.DataFrame, file_name: str = "matches_v2_features.csv") -> None:
    output_path = get_project_root() / "data" / "processed" / file_name
    df.to_csv(output_path, index=False)


def main() -> None:
    df = load_clean_data()
    featured_df = build_rolling_features(df)

    print("Input shape:", df.shape)
    print("Output shape:", featured_df.shape)
    print(featured_df.head(10))

    save_featured_data(featured_df)
    print("Saved to data/processed/matches_v2_features.csv")


if __name__ == "__main__":
    main()