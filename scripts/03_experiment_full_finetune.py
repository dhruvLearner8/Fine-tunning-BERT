# scripts/03_experiment_full_finetune.py
"""Experiment 2: full fine-tuning of all BioBERT parameters."""
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

    model = models.load_base_model()
    trainable, total, pct = models.get_trainable_param_counts(model)
    print(f"Trainable params: {trainable:,} / {total:,} ({pct:.2f}%)")

    output_dir = config.MODELS_DIR / "full_finetune"
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

    correct_idx = np.where(y_pred == y_true)[0][:5]
    wrong_idx = np.where(y_pred != y_true)[0][:5]
    print("\n5 correct predictions:")
    for i in correct_idx:
        print(f"  [{config.ID2LABEL[int(y_true[i])]}] {test_df.iloc[int(i)]['text'][:120]}")
    print("\n5 wrong predictions:")
    for i in wrong_idx:
        print(
            f"  true={config.ID2LABEL[int(y_true[i])]} pred={config.ID2LABEL[int(y_pred[i])]} "
            f"{test_df.iloc[int(i)]['text'][:120]}"
        )

    metrics = evaluate.compute_metrics((predictions.predictions, y_true))
    mlflow_utils.log_run(
        run_name="full_finetune",
        params={"model": "biobert", "epochs": config.EPOCHS, "learning_rate": config.LEARNING_RATE, "lora": False},
        metrics={**metrics, "trainable_params": trainable, "train_time_seconds": elapsed},
    )

    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print(f"\nSaved model to {output_dir}")
    print(f"F1: {metrics['f1']:.3f}  Precision: {metrics['precision']:.3f}  Recall: {metrics['recall']:.3f}")


if __name__ == "__main__":
    main()
