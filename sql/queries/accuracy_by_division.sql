SELECT
    model_name,
    ifNull(nullIf(division, ''), 'UNKNOWN') AS division,
    count() AS predictions,
    round(avg(is_correct), 4) AS accuracy
FROM match_predictions
GROUP BY
    model_name,
    division
HAVING predictions >= 30
ORDER BY
    model_name,
    accuracy DESC,
    predictions DESC;