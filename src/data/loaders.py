from pathlib import Path
import pandas as pd


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_raw_matches(file_name: str = "matches.csv") -> pd.DataFrame:
    file_path = get_project_root() / "data" / "raw" / file_name
    return pd.read_csv(
        file_path,
        low_memory=False,
        dtype={"MatchTime": "string"},
    )