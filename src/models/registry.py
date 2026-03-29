from __future__ import annotations

MODEL_REGISTRY = {
    "home_win_rf_v1": {
        "model_path": "artifacts/models/home_win_rf_v1.joblib",
        "metadata_path": "artifacts/metadata/home_win_rf_v1.json",
        "target_name": "home_win",
        "task_type": "binary_classification",
        "positive_class_label": 1,
    },
}