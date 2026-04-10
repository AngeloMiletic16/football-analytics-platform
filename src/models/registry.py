from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


MODEL_REGISTRY = {
    "home_win_rf_v1": {
        "model_name": "home_win_rf_v1",
        "target": "home_win",
        "task_type": "classification",
        "problem_type": "binary_classification",
        "algorithm": "random_forest",
        "artifact_path": PROJECT_ROOT / "artifacts" / "models" / "home_win_rf_v1.joblib",
        "metadata_path": PROJECT_ROOT / "artifacts" / "metadata" / "home_win_rf_v1.json",
    },
    "over_2_5_lr_v1": {
        "model_name": "over_2_5_lr_v1",
        "target": "over_2_5",
        "task_type": "classification",
        "problem_type": "binary_classification",
        "algorithm": "logistic_regression",
        "artifact_path": PROJECT_ROOT / "artifacts" / "models" / "over_2_5_lr_v1.joblib",
        "metadata_path": PROJECT_ROOT / "artifacts" / "metadata" / "over_2_5_lr_v1.json",
    },
    "ft_result_lr_v1": {
        "model_name": "ft_result_lr_v1",
        "target": "ft_result",
        "task_type": "classification",
        "problem_type": "multiclass_classification",
        "algorithm": "logistic_regression",
        "artifact_path": PROJECT_ROOT / "artifacts" / "models" / "ft_result_lr_v1.joblib",
        "metadata_path": PROJECT_ROOT / "artifacts" / "metadata" / "ft_result_lr_v1.json",
    },
    "total_goals_lr_v1": {
        "model_name": "total_goals_lr_v1",
        "target": "total_goals",
        "task_type": "regression",
        "problem_type": "regression",
        "algorithm": "linear_regression",
        "artifact_path": PROJECT_ROOT / "artifacts" / "models" / "total_goals_lr_v1.joblib",
        "metadata_path": PROJECT_ROOT / "artifacts" / "metadata" / "total_goals_lr_v1.json",
    },
}


def get_model_config(model_name: str) -> dict:
    if model_name not in MODEL_REGISTRY:
        available_models = ", ".join(sorted(MODEL_REGISTRY.keys()))
        raise ValueError(
            f"Unknown model_name='{model_name}'. Available models: {available_models}"
        )
    return MODEL_REGISTRY[model_name]


def list_model_names() -> list[str]:
    return sorted(MODEL_REGISTRY.keys())