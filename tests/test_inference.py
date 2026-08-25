import numpy as np

from biobert_sentiment.inference import logits_to_prediction


def test_logits_to_prediction_picks_highest_confidence_class():
    result = logits_to_prediction(np.array([5.0, 0.0, 0.0]))
    assert result["label"] == "NEGATIVE"
    assert result["confidences"]["NEGATIVE"] > 0.9
    assert abs(sum(result["confidences"].values()) - 1.0) < 1e-6


def test_logits_to_prediction_labels_match_config():
    result = logits_to_prediction(np.array([0.0, 0.0, 5.0]))
    assert result["label"] == "POSITIVE"
    assert set(result["confidences"].keys()) == {"NEGATIVE", "NEUTRAL", "POSITIVE"}
