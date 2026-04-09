SELECT
    model_name,
    predicted_label,
    multiIf(
        predicted_probability < 0.50, '0.00-0.49',
        predicted_probability < 0.60, '0.50-0.59',
        predicted_probability < 0.70, '0.60-0.69',
        predicted_probability < 0.80, '0.70-0.79',
        predicted_probability < 0.90, '0.80-0.89',
        '0.90-1.00'
    ) AS confidence_bucket,
    count() AS predictions,
    round(avg(is_correct), 4) AS accuracy
FROM match_predictions
GROUP BY
    model_name,
    predicted_label,
    confidence_bucket
ORDER BY
    model_name,
    confidence_bucket,
    predicted_label;