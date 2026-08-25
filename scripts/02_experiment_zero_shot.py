# scripts/02_experiment_zero_shot.py
"""Experiment 1: zero-shot BioBERT baseline, no training."""
import time

import numpy as np
from transformers import Trainer

from biobert_sentiment import config, data, evaluate, mlflow_utils, models, tokenize
from biobert_sentiment.train import build_training_args


def main():
    _, _, test_df = data.load_splits(config.PROCESSED_DATA_DIR)
    tokenizer = tokenize.get_tokenizer()
    test_dataset = tokenize.tokenize_dataset(test_df, tokenizer)

    model = models.load_base_model()
    args = build_training_args(config.MODELS_DIR / "zero_shot", epochs=0)
    trainer = Trainer(model=model, args=args, eval_dataset=test_dataset, processing_class=tokenizer, compute_metrics=evaluate.compute_metrics)

    start = time.time()
    predictions = trainer.predict(test_dataset)
    elapsed = time.time() - start

    y_pred = np.argmax(predictions.predictions, axis=-1)
    y_true = predictions.label_ids

    print(evaluate.build_classification_report(y_true, y_pred))
    print("Confusion matrix:")
    print(evaluate.build_confusion_matrix(y_true, y_pred))

    metrics = evaluate.compute_metrics((predictions.predictions, y_true))
    mlflow_utils.log_run(
        run_name="zero_shot",
        params={"model": "biobert", "epochs": 0, "learning_rate": "n/a", "lora": False},
        metrics={**metrics, "trainable_params": 0, "train_time_seconds": elapsed},
    )
    print(f"\nF1: {metrics['f1']:.3f}  Precision: {metrics['precision']:.3f}  Recall: {metrics['recall']:.3f}")


if __name__ == "__main__":
    main()
