"""Shared Trainer setup for full fine-tuning and LoRA fine-tuning."""
from pathlib import Path

from transformers import Trainer, TrainingArguments

from biobert_sentiment import config
from biobert_sentiment.evaluate import compute_metrics


def build_training_args(
    output_dir: Path,
    epochs: int = config.EPOCHS,
    lr: float = config.LEARNING_RATE,
    batch_size: int = config.BATCH_SIZE,
) -> TrainingArguments:
    return TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        report_to=[],
        logging_steps=10,
    )


def run_training(model, tokenizer, train_dataset, val_dataset, training_args: TrainingArguments) -> Trainer:
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    return trainer
