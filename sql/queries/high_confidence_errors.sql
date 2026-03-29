SELECT
    model_name,
    target_name,
    match_date,
    division,
    home_team,
    away_team,
    actual_home_win,
    predicted_label,
    predicted_probability,
    is_correct
FROM match_predictions
WHERE target_name = 'home_win'
  AND is_correct = 0
  AND predicted_probability IS NOT NULL
ORDER BY predicted_probability DESC
LIMIT 20;