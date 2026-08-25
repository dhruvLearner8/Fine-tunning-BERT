# scripts/05_compare_experiments.py
"""Step 6: pull all three MLflow runs and print the comparison table; save the best model."""
import shutil

import mlflow

from biobert_sentiment import config

RUN_NAMES = ["zero_shot", "full_finetune", "lora_finetune"]
DISPLAY_NAMES = {
    "zero_shot": "Base BioBERT",
    "full_finetune": "Full Fine-tuning",
    "lora_finetune": "LoRA Fine-tuning",
}
MODEL_DIRS = {
    "full_finetune": config.MODELS_DIR / "full_finetune",
    "lora_finetune": config.MODELS_DIR / "lora_finetune",
}


def _latest_run(client, experiment_id, run_name):
    runs = client.search_runs(
        experiment_id,
        filter_string=f"tags.mlflow.runName = '{run_name}'",
        order_by=["start_time DESC"],
        max_results=1,
    )
    if not runs:
        raise RuntimeError(f"No MLflow run found named '{run_name}'. Run the corresponding experiment script first.")
    return runs[0]


def main():
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(config.MLFLOW_EXPERIMENT_NAME)
    if experiment is None:
        raise RuntimeError(f"MLflow experiment '{config.MLFLOW_EXPERIMENT_NAME}' not found. Run the experiment scripts first.")

    rows = []
    best_run_name, best_f1 = None, -1.0
    for run_name in RUN_NAMES:
        run = _latest_run(client, experiment.experiment_id, run_name)
        m = run.data.metrics
        rows.append(
            (
                DISPLAY_NAMES[run_name],
                m.get("f1", 0.0),
                m.get("precision", 0.0),
                m.get("recall", 0.0),
                int(m.get("trainable_params", 0)),
                m.get("train_time_seconds", 0.0),
            )
        )
        if run_name != "zero_shot" and m.get("f1", 0.0) > best_f1:
            best_f1, best_run_name = m.get("f1", 0.0), run_name

    header = f"{'Model':<20} | {'F1':<6} | {'Precision':<9} | {'Recall':<6} | {'Params Trained':<15} | {'Time (s)'}"
    print(header)
    print("-" * len(header))
    for name, f1, precision, recall, params, seconds in rows:
        print(f"{name:<20} | {f1:<6.3f} | {precision:<9.3f} | {recall:<6.3f} | {params:<15,} | {seconds:.1f}")

    print(f"\nBest fine-tuned model: {DISPLAY_NAMES[best_run_name]} (F1={best_f1:.3f})")
    src_dir = MODEL_DIRS[best_run_name]
    if config.FINAL_MODEL_DIR.exists():
        shutil.rmtree(config.FINAL_MODEL_DIR)
    shutil.copytree(src_dir, config.FINAL_MODEL_DIR)
    print(f"Saved best model to {config.FINAL_MODEL_DIR}")


if __name__ == "__main__":
    main()
