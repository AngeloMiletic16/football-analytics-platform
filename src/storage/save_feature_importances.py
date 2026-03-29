from __future__ import annotations

from datetime import datetime
from uuid import uuid4

import pandas as pd
import clickhouse_connect


def get_client():
    return clickhouse_connect.get_client(
        host="localhost",
        port=8123,
        username="admin",
        password="admin123",
        database="football_analytics",
    )


def build_feature_importances() -> list[dict]:
    run_id = str(uuid4())
    created_at = datetime.now()

    rows: list[dict] = []

    # Random Forest - home_win
    home_win_rf = [
        ("odd_away", 0.247723),
        ("odd_home", 0.243743),
        ("elo_diff", 0.152092),
        ("home_elo", 0.092528),
        ("odd_draw", 0.073783),
        ("away_elo", 0.069408),
        ("form5_diff", 0.035374),
        ("form3_diff", 0.020875),
        ("form5_away", 0.019349),
        ("form5_home", 0.018663),
        ("form3_home", 0.013849),
    ]

    for rank, (feature_name, importance) in enumerate(home_win_rf, start=1):
        rows.append(
            {
                "run_id": run_id,
                "created_at": created_at,
                "model_name": "random_forest",
                "task_type": "binary_classification",
                "target_name": "home_win",
                "dataset_name": "matches_v1_clean",
                "feature_name": feature_name,
                "importance": importance,
                "rank": rank,
                "notes": "Best binary home_win model feature importances.",
            }
        )

    # Random Forest - over_2_5
    over_under_rf = [
        ("implied_draw_prob", 0.101751),
        ("odd_draw", 0.095242),
        ("elo_abs_diff", 0.076224),
        ("odd_abs_diff", 0.067411),
        ("elo_diff", 0.066253),
        ("home_elo", 0.062581),
        ("away_elo", 0.060703),
        ("elo_sum", 0.059552),
        ("implied_home_prob", 0.045204),
        ("odd_away", 0.044719),
        ("implied_away_prob", 0.044631),
        ("odd_home", 0.043085),
        ("form5_sum", 0.030229),
        ("form5_diff", 0.026937),
        ("form5_abs_diff", 0.025597),
    ]

    for rank, (feature_name, importance) in enumerate(over_under_rf, start=1):
        rows.append(
            {
                "run_id": run_id,
                "created_at": created_at,
                "model_name": "random_forest",
                "task_type": "binary_classification",
                "target_name": "over_2_5",
                "dataset_name": "matches_v1_clean",
                "feature_name": feature_name,
                "importance": importance,
                "rank": rank,
                "notes": "Random Forest over/under 2.5 feature importances.",
            }
        )

    # Random Forest Regressor - total_goals
    total_goals_rf = [
        ("odd_draw", 0.141353),
        ("implied_draw_prob", 0.133345),
        ("odd_abs_diff", 0.082529),
        ("home_elo", 0.077606),
        ("elo_sum", 0.071791),
        ("away_elo", 0.069353),
        ("elo_abs_diff", 0.060540),
        ("elo_diff", 0.048346),
        ("form5_sum", 0.036427),
        ("form5_away", 0.028234),
        ("form3_sum", 0.027155),
        ("form5_home", 0.024191),
        ("form5_abs_diff", 0.024053),
        ("form5_diff", 0.022923),
        ("form3_abs_diff", 0.020327),
    ]

    for rank, (feature_name, importance) in enumerate(total_goals_rf, start=1):
        rows.append(
            {
                "run_id": run_id,
                "created_at": created_at,
                "model_name": "random_forest_regressor",
                "task_type": "regression",
                "target_name": "total_goals",
                "dataset_name": "matches_v1_clean",
                "feature_name": feature_name,
                "importance": importance,
                "rank": rank,
                "notes": "Random Forest total goals regressor feature importances.",
            }
        )

    # Multiclass Random Forest - ft_result
    multiclass_rf = [
        ("odd_home", 0.123773),
        ("odd_away", 0.113616),
        ("elo_diff", 0.090426),
        ("odd_abs_diff", 0.071020),
        ("home_elo", 0.059171),
        ("away_elo", 0.056389),
        ("elo_abs_diff", 0.053086),
        ("odd_draw", 0.051510),
        ("recent_goals_scored_diff_overall", 0.025141),
        ("form5_diff", 0.021850),
        ("recent_points_diff_overall", 0.021462),
        ("away_avg_goals_scored_last_5_overall", 0.020702),
        ("recent_goals_conceded_diff_overall", 0.020099),
        ("home_avg_goals_scored_last_5_overall", 0.019481),
        ("home_avg_goals_conceded_last_5_overall", 0.018723),
    ]

    for rank, (feature_name, importance) in enumerate(multiclass_rf, start=1):
        rows.append(
            {
                "run_id": run_id,
                "created_at": created_at,
                "model_name": "multiclass_random_forest",
                "task_type": "multiclass_classification",
                "target_name": "ft_result",
                "dataset_name": "matches_v3_features",
                "feature_name": feature_name,
                "importance": importance,
                "rank": rank,
                "notes": "Multiclass Random Forest feature importances on v3 features.",
            }
        )

    return rows


def main() -> None:
    client = get_client()
    rows = build_feature_importances()
    df = pd.DataFrame(rows)

    client.insert_df("feature_importances", df)

    print(f"Inserted {len(df)} rows into feature_importances.")


if __name__ == "__main__":
    main()