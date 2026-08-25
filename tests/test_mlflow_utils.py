import mlflow

from biobert_sentiment import mlflow_utils


def test_log_run_records_params_and_metrics(tmp_path, monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"file://{tmp_path}")
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
    mlflow.set_tracking_uri(f"file://{tmp_path}")

    mlflow_utils.log_run(
        run_name="unit_test_run",
        params={"model": "biobert", "epochs": 3, "lora": False},
        metrics={"f1": 0.9, "precision": 0.88, "recall": 0.91},
        experiment_name="unit-test-experiment",
    )

    client = mlflow.tracking.MlflowClient(tracking_uri=f"file://{tmp_path}")
    experiment = client.get_experiment_by_name("unit-test-experiment")
    assert experiment is not None
    runs = client.search_runs(experiment.experiment_id)
    assert len(runs) == 1
    run = runs[0]
    assert run.data.params["model"] == "biobert"
    assert run.data.params["lora"] == "False"
    assert run.data.metrics["f1"] == 0.9
