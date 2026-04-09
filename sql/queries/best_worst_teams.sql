SELECT
    model_name,
    team_name,
    count() AS predictions,
    round(avg(is_correct), 4) AS accuracy
FROM
(
    SELECT
        model_name,
        home_team AS team_name,
        is_correct
    FROM match_predictions

    UNION ALL

    SELECT
        model_name,
        away_team AS team_name,
        is_correct
    FROM match_predictions
)
GROUP BY
    model_name,
    team_name
HAVING predictions >= 40
ORDER BY
    model_name,
    accuracy DESC,
    predictions DESC;