import pandas as pd


def add_model_features(df: pd.DataFrame) -> pd.DataFrame:
    featured = df.copy()

    featured["elo_diff"] = featured["home_elo"] - featured["away_elo"]
    featured["form3_diff"] = featured["form3_home"] - featured["form3_away"]
    featured["form5_diff"] = featured["form5_home"] - featured["form5_away"]

    featured["elo_abs_diff"] = (featured["home_elo"] - featured["away_elo"]).abs()
    featured["form3_abs_diff"] = (featured["form3_home"] - featured["form3_away"]).abs()
    featured["form5_abs_diff"] = (featured["form5_home"] - featured["form5_away"]).abs()
    featured["odd_abs_diff"] = (featured["odd_home"] - featured["odd_away"]).abs()

    return featured