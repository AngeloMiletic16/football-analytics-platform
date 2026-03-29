CREATE TABLE IF NOT EXISTS match_predictions
(
    inserted_at DateTime DEFAULT now(),
    prediction_run_id String,
    model_name String,
    model_version String,
    target_name String,
    task_type String,

    match_id String,
    match_date Date,
    division String,
    home_team String,
    away_team String,

    actual_home_goals Nullable(Int32),
    actual_away_goals Nullable(Int32),
    actual_ft_result Nullable(String),
    actual_home_win Nullable(UInt8),
    actual_over_2_5 Nullable(UInt8),
    actual_total_goals Nullable(Float64),

    predicted_label Nullable(String),
    predicted_value Nullable(Float64),

    predicted_probability Nullable(Float64),
    home_win_probability Nullable(Float64),
    draw_probability Nullable(Float64),
    away_win_probability Nullable(Float64),
    over_2_5_probability Nullable(Float64),
    under_2_5_probability Nullable(Float64),

    absolute_error Nullable(Float64),
    is_correct Nullable(UInt8)
)
ENGINE = MergeTree
ORDER BY (target_name, model_name, model_version, match_date, match_id);