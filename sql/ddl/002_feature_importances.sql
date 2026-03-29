CREATE TABLE IF NOT EXISTS feature_importances
(
    inserted_at DateTime DEFAULT now(),
    run_id String,
    model_name String,
    model_version String,
    target_name String,
    task_type String,                  -- binary_classification / multiclass_classification / regression
    split_name String,                 -- test / validation / full_train
    importance_source String,          -- feature_importance / coefficient
    class_name Nullable(String),       -- za multiclass logistic može biti H / D / A
    feature_name String,
    importance_value Float64,          -- standardizirano "glavno" polje
    abs_importance_value Float64,      -- korisno za linear/multiclass
    raw_coefficient Nullable(Float64), -- samo za linear modele
    importance_rank UInt32
)
ENGINE = MergeTree
ORDER BY (model_name, model_version, target_name, split_name, importance_rank, feature_name);