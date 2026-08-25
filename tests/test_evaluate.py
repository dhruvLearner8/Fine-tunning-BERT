import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from transformers import EvalPrediction

from biobert_sentiment.evaluate import build_classification_report, build_confusion_matrix, compute_metrics


def test_compute_metrics_matches_manual_sklearn_calculation():
    y_true = np.array([0, 0, 1, 1, 2, 2])
    y_pred = np.array([0, 1, 1, 1, 2, 0])
    logits = np.zeros((len(y_pred), 3))
    for i, p in enumerate(y_pred):
        logits[i, p] = 10.0

    result = compute_metrics(EvalPrediction(predictions=logits, label_ids=y_true))

    expected_p, expected_r, expected_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    assert result["f1"] == expected_f1
    assert result["precision"] == expected_p
    assert result["recall"] == expected_r


def test_compute_metrics_accepts_plain_tuple():
    logits = np.array([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0]])
    labels = np.array([0, 1])
    result = compute_metrics((logits, labels))
    assert result["f1"] == 1.0


def test_build_confusion_matrix_shape_and_diagonal():
    y_true = np.array([0, 1, 2, 0])
    y_pred = np.array([0, 1, 2, 0])
    cm = build_confusion_matrix(y_true, y_pred)
    assert cm.shape == (3, 3)
    assert cm.trace() == 4


def test_build_classification_report_includes_label_names():
    report = build_classification_report(np.array([0, 1, 2]), np.array([0, 1, 2]))
    assert "NEGATIVE" in report and "NEUTRAL" in report and "POSITIVE" in report
