CREATE TABLE IF NOT EXISTS feature_importances
(
    run_id UUID,
    created_at DateTime DEFAULT now(),

    model_name String,
    task_type String,
    target_name String,
    dataset_name String,

    feature_name String,
    importance Float64,

    rank UInt16,
    notes String
)
ENGINE = MergeTree
ORDER BY (task_type, target_name, model_name, rank, created_at);