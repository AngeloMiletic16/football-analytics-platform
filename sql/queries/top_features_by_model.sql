SELECT
    model_name,
    model_version,
    target_name,
    feature_name,
    importance_value,
    importance_rank
FROM feature_importances
WHERE importance_rank <= 10
ORDER BY model_name, target_name, importance_rank;