"""Thin wrapper around MLflow run logging for experiment tracking."""
import mlflow

from biobert_sentiment import config


def log_run(
    run_name: str,
    params: dict,
    metrics: dict,
    experiment_name: str = config.MLFLOW_EXPERIMENT_NAME,
) -> None:
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=run_name):
        for key, value in params.items():
            mlflow.log_param(key, value)
        for key, value in metrics.items():
            mlflow.log_metric(key, value)
