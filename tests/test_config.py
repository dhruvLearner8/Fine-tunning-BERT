import importlib

from biobert_sentiment import config


def test_get_device_returns_valid_choice():
    assert config.get_device() in ("cuda", "mps", "cpu")


def test_label_maps_match_spec_bucketing():
    assert config.LABEL2ID == {"NEGATIVE": 0, "NEUTRAL": 1, "POSITIVE": 2}
    assert config.ID2LABEL == {0: "NEGATIVE", 1: "NEUTRAL", 2: "POSITIVE"}


def test_subset_size_parses_env_var(monkeypatch):
    monkeypatch.setenv("SUBSET_SIZE", "none")
    importlib.reload(config)
    assert config.SUBSET_SIZE is None

    monkeypatch.setenv("SUBSET_SIZE", "250")
    importlib.reload(config)
    assert config.SUBSET_SIZE == 250

    monkeypatch.delenv("SUBSET_SIZE", raising=False)
    importlib.reload(config)
    assert config.SUBSET_SIZE == 600
