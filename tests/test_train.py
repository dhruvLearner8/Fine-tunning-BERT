from pathlib import Path

from biobert_sentiment.train import build_training_args


def test_build_training_args_uses_config_defaults():
    args = build_training_args(Path("/tmp/biobert-test-defaults"))
    assert args.num_train_epochs == 3
    assert args.learning_rate == 2e-5
    assert args.per_device_train_batch_size == 16
    assert args.eval_strategy == "epoch"
    assert args.metric_for_best_model == "f1"


def test_build_training_args_respects_overrides():
    args = build_training_args(Path("/tmp/biobert-test-overrides"), epochs=1, lr=1e-4, batch_size=8)
    assert args.num_train_epochs == 1
    assert args.learning_rate == 1e-4
    assert args.per_device_train_batch_size == 8
