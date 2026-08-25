"""Thin wrapper around MLflow run logging for experiment tracking."""
import os

import mlflow

from biobert_sentiment import config

# MLflow 3.x requires explicit opt-in for filesystem tracking backend
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")


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
