SELECT
    model_name,
    toStartOfMonth(toDate(match_date)) AS month,
    count() AS predictions,
    round(avg(is_correct), 4) AS accuracy
FROM match_predictions
WHERE match_date IS NOT NULL
GROUP BY
    model_name,
    month
HAVING predictions >= 30
ORDER BY
    model_name,
    month;