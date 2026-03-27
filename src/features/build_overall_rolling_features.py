from pathlib import Path

import pandas as pd


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_clean_data() -> pd.DataFrame:
    file_path = get_project_root() / "data" / "processed" / "matches_v1_clean.csv"
    df = pd.read_csv(file_path, parse_dates=["match_date"])
    return df.sort_values("match_date").reset_index(drop=True)


def build_team_long_table(df: pd.DataFrame) -> pd.DataFrame:
    home_df = df[
        ["match_id", "match_date", "home_team", "ft_home", "ft_away", "ft_result"]
    ].copy()
    home_df = home_df.rename(
        columns={
            "home_team": "team",
            "ft_home": "goals_scored",
            "ft_away": "goals_conceded",
        }
    )
    home_df["is_draw"] = (home_df["ft_result"] == "D").astype(int)
    home_df["points"] = home_df["ft_result"].map({"H": 3, "D": 1, "A": 0})
    home_df["side"] = "home"

    away_df = df[
        ["match_id", "match_date", "away_team", "ft_home", "ft_away", "ft_result"]
    ].copy()
    away_df = away_df.rename(
        columns={
            "away_team": "team",
            "ft_away": "goals_scored",
            "ft_home": "goals_conceded",
        }
    )
    away_df["is_draw"] = (away_df["ft_result"] == "D").astype(int)
    away_df["points"] = away_df["ft_result"].map({"H": 0, "D": 1, "A": 3})
    away_df["side"] = "away"

    team_df = pd.concat([home_df, away_df], ignore_index=True)
    team_df = team_df.sort_values(["team", "match_date", "match_id"]).reset_index(drop=True)

    return team_df


def add_team_overall_rolling(team_df: pd.DataFrame) -> pd.DataFrame:
    enriched = team_df.copy()

    enriched["avg_goals_scored_last_5_overall"] = (
        enriched.groupby("team")["goals_scored"]
        .transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean())
    )

    enriched["avg_goals_conceded_last_5_overall"] = (
        enriched.groupby("team")["goals_conceded"]
        .transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean())
    )

    enriched["points_last_5_overall"] = (
        enriched.groupby("team")["points"]
        .transform(lambda s: s.shift(1).rolling(5, min_periods=1).sum())
    )

    enriched["draw_rate_last_5_overall"] = (
        enriched.groupby("team")["is_draw"]
        .transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean())
    )

    return enriched


def merge_overall_features_back(df: pd.DataFrame, team_df: pd.DataFrame) -> pd.DataFrame:
    enriched = df.copy()

    home_features = team_df[team_df["side"] == "home"][
        [
            "match_id",
            "avg_goals_scored_last_5_overall",
            "avg_goals_conceded_last_5_overall",
            "points_last_5_overall",
            "draw_rate_last_5_overall",
        ]
    ].copy()

    home_features = home_features.rename(
        columns={
            "avg_goals_scored_last_5_overall": "home_avg_goals_scored_last_5_overall",
            "avg_goals_conceded_last_5_overall": "home_avg_goals_conceded_last_5_overall",
            "points_last_5_overall": "home_points_last_5_overall",
            "draw_rate_last_5_overall": "home_draw_rate_last_5_overall",
        }
    )

    away_features = team_df[team_df["side"] == "away"][
        [
            "match_id",
            "avg_goals_scored_last_5_overall",
            "avg_goals_conceded_last_5_overall",
            "points_last_5_overall",
            "draw_rate_last_5_overall",
        ]
    ].copy()

    away_features = away_features.rename(
        columns={
            "avg_goals_scored_last_5_overall": "away_avg_goals_scored_last_5_overall",
            "avg_goals_conceded_last_5_overall": "away_avg_goals_conceded_last_5_overall",
            "points_last_5_overall": "away_points_last_5_overall",
            "draw_rate_last_5_overall": "away_draw_rate_last_5_overall",
        }
    )

    enriched = enriched.merge(home_features, on="match_id", how="left")
    enriched = enriched.merge(away_features, on="match_id", how="left")

    return enriched


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    team_df = build_team_long_table(df)
    team_df = add_team_overall_rolling(team_df)
    enriched = merge_overall_features_back(df, team_df)

    return enriched


def save_featured_data(df: pd.DataFrame, file_name: str = "matches_v3_features.csv") -> None:
    output_path = get_project_root() / "data" / "processed" / file_name
    df.to_csv(output_path, index=False)


def main() -> None:
    df = load_clean_data()
    featured_df = build_features(df)

    print("Input shape:", df.shape)
    print("Output shape:", featured_df.shape)
    print(featured_df.head(10))

    save_featured_data(featured_df)
    print("Saved to data/processed/matches_v3_features.csv")


if __name__ == "__main__":
    main()