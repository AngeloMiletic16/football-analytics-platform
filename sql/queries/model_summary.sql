SELECT
    model_name,
    model_version,
    target_name,
    task_type,
    min(match_date) AS first_match_date,
    max(match_date) AS last_match_date,
    count() AS predictions,
    round(avg(toFloat64(is_correct)), 4) AS accuracy,
    round(avg(predicted_probability), 4) AS avg_confidence,
    countIf(predicted_probability >= 0.70) AS high_conf_predictions,
    round(countIf(predicted_probability >= 0.70) / count(), 4) AS high_conf_share,
    round(avgIf(toFloat64(is_correct), predicted_probability >= 0.70), 4) AS high_conf_accuracy
FROM match_predictions
GROUP BY
    model_name,
    model_version,
    target_name,
    task_type
ORDER BY
    accuracy DESC,
    predictions DESC;