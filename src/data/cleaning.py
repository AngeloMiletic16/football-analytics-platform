from pathlib import Path
import pandas as pd


TOP5_DIVISIONS = ["E0", "SP1", "I1", "D1", "F1"]

V1_COLUMNS = [
    "Division",
    "MatchDate",
    "HomeTeam",
    "AwayTeam",
    "HomeElo",
    "AwayElo",
    "Form3Home",
    "Form5Home",
    "Form3Away",
    "Form5Away",
    "FTHome",
    "FTAway",
    "FTResult",
    "OddHome",
    "OddDraw",
    "OddAway",
]


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def clean_matches_v1(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()

    cleaned = cleaned[V1_COLUMNS].copy()

    cleaned["MatchDate"] = pd.to_datetime(cleaned["MatchDate"], errors="coerce")

    cleaned = cleaned[cleaned["Division"].isin(TOP5_DIVISIONS)]
    cleaned = cleaned[cleaned["MatchDate"] >= "2015-01-01"]

    cleaned = cleaned.dropna(subset=V1_COLUMNS)

    cleaned = cleaned.rename(
        columns={
            "Division": "division",
            "MatchDate": "match_date",
            "HomeTeam": "home_team",
            "AwayTeam": "away_team",
            "HomeElo": "home_elo",
            "AwayElo": "away_elo",
            "Form3Home": "form3_home",
            "Form5Home": "form5_home",
            "Form3Away": "form3_away",
            "Form5Away": "form5_away",
            "FTHome": "ft_home",
            "FTAway": "ft_away",
            "FTResult": "ft_result",
            "OddHome": "odd_home",
            "OddDraw": "odd_draw",
            "OddAway": "odd_away",
        }
    )

    cleaned["home_win"] = (cleaned["ft_result"] == "H").astype(int)
    cleaned["draw"] = (cleaned["ft_result"] == "D").astype(int)
    cleaned["away_win"] = (cleaned["ft_result"] == "A").astype(int)
    cleaned["total_goals"] = cleaned["ft_home"] + cleaned["ft_away"]
    cleaned["over_2_5"] = (cleaned["total_goals"] > 2.5).astype(int)

    cleaned = cleaned.sort_values("match_date").reset_index(drop=True)

    cleaned.insert(0, "match_id", range(1, len(cleaned) + 1))

    return cleaned


def save_processed_matches(df: pd.DataFrame, file_name: str = "matches_v1_clean.csv") -> None:
    output_path = get_project_root() / "data" / "processed" / file_name
    df.to_csv(output_path, index=False)