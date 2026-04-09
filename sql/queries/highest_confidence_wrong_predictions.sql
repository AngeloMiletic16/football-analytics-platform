highest_confidence_wrong_predictionsSELECT
    model_name,
    model_version,
    target_name,
    match_date,
    division,
    home_team,
    away_team,
    multiIf(
        target_name = 'ft_result', actual_ft_result,
        target_name = 'home_win', toString(actual_home_win),
        target_name = 'over_2_5', toString(actual_over_2_5),
        NULL
    ) AS actual_label,
    predicted_label,
    predicted_probability,
    is_correct
FROM match_predictions
WHERE is_correct = 0
ORDER BY
    predicted_probability DESC,
    match_date DESC
LIMIT 100;