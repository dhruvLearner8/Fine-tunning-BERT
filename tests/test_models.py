import pytest

from biobert_sentiment import models


@pytest.mark.integration
def test_load_base_model_is_fully_trainable():
    model = models.load_base_model()
    trainable, total, pct = models.get_trainable_param_counts(model)
    assert trainable == total
    assert pct == pytest.approx(100.0)


@pytest.mark.integration
def test_load_lora_model_trains_a_small_fraction():
    model = models.load_lora_model()
    trainable, total, pct = models.get_trainable_param_counts(model)
    assert trainable < total
    assert pct < 1.0
