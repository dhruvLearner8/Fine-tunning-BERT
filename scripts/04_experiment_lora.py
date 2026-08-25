# scripts/04_experiment_lora.py
"""Experiment 3: LoRA fine-tuning (query/value adapters only)."""
import time

import numpy as np

from biobert_sentiment import config, data, evaluate, mlflow_utils, models, tokenize
from biobert_sentiment.train import build_training_args, run_training


def main():
    train_df, val_df, test_df = data.load_splits(config.PROCESSED_DATA_DIR)
    tokenizer = tokenize.get_tokenizer()
    train_dataset = tokenize.tokenize_dataset(train_df, tokenizer)
    val_dataset = tokenize.tokenize_dataset(val_df, tokenizer)
    test_dataset = tokenize.tokenize_dataset(test_df, tokenizer)

    model = models.load_lora_model()
    trainable, total, pct = models.get_trainable_param_counts(model)
    print(f"Trainable params: {trainable:,} / {total:,} ({pct:.2f}%)")
    model.print_trainable_parameters()

    output_dir = config.MODELS_DIR / "lora_finetune"
    args = build_training_args(output_dir)

    start = time.time()
    trainer = run_training(model, tokenizer, train_dataset, val_dataset, args)
    elapsed = time.time() - start

    predictions = trainer.predict(test_dataset)
    y_pred = np.argmax(predictions.predictions, axis=-1)
    y_true = predictions.label_ids

    print(evaluate.build_classification_report(y_true, y_pred))
    print("Confusion matrix:")
    print(evaluate.build_confusion_matrix(y_true, y_pred))

    metrics = evaluate.compute_metrics((predictions.predictions, y_true))
    mlflow_utils.log_run(
        run_name="lora_finetune",
        params={
            "model": "biobert",
            "epochs": config.EPOCHS,
            "learning_rate": config.LEARNING_RATE,
            "lora": True,
            "lora_r": config.LORA_R,
            "lora_alpha": config.LORA_ALPHA,
        },
        metrics={**metrics, "trainable_params": trainable, "train_time_seconds": elapsed},
    )

    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print(f"\nSaved LoRA adapter to {output_dir}")
    print(f"F1: {metrics['f1']:.3f}  Precision: {metrics['precision']:.3f}  Recall: {metrics['recall']:.3f}")


if __name__ == "__main__":
    main()
