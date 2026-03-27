CREATE TABLE IF NOT EXISTS matches_raw
(
    match_date Date,
    season String,
    league String,
    home_team String,
    away_team String,
    home_goals UInt8,
    away_goals UInt8
)
ENGINE = MergeTree
ORDER BY (league, season, match_date);