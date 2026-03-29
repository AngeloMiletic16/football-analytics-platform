SELECT
    model_name,
    target_name,
    round(avg(is_correct), 4) AS accuracy,
    count() AS n_predictions
FROM match_predictions
WHERE is_correct IS NOT NULL
GROUP BY model_name, target_name
ORDER BY accuracy DESC, n_predictions DESC;