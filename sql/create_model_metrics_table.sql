CREATE TABLE IF NOT EXISTS model_metrics
(
    run_id UUID,
    created_at DateTime DEFAULT now(),

    model_name String,
    task_type String,
    target_name String,
    dataset_name String,
    split_name String,

    accuracy Nullable(Float64),
    roc_auc Nullable(Float64),
    log_loss Nullable(Float64),
    mae Nullable(Float64),
    rmse Nullable(Float64),
    r2 Nullable(Float64),

    precision_macro Nullable(Float64),
    recall_macro Nullable(Float64),
    f1_macro Nullable(Float64),

    notes String
)
ENGINE = MergeTree
ORDER BY (task_type, target_name, model_name, split_name, created_at);