"""Metric computation, confusion matrices, and classification reports."""
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support

from biobert_sentiment import config


def compute_metrics(eval_pred) -> dict:
    if hasattr(eval_pred, "predictions"):
        predictions, labels = eval_pred.predictions, eval_pred.label_ids
    else:
        predictions, labels = eval_pred
    preds = np.argmax(predictions, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="macro", zero_division=0
    )
    return {"f1": f1, "precision": precision, "recall": recall}


def build_classification_report(y_true, y_pred) -> str:
    return classification_report(y_true, y_pred, target_names=config.LABEL_NAMES, zero_division=0)


def build_confusion_matrix(y_true, y_pred) -> np.ndarray:
    return confusion_matrix(y_true, y_pred, labels=list(range(config.NUM_LABELS)))
