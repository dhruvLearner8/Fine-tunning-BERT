import pytest
import pandas as pd

from biobert_sentiment import config, data


@pytest.mark.integration
def test_load_raw_dataset_finds_confirmed_dataset():
    df, dataset_id = data.load_raw_dataset(split="train[:20]")
    assert dataset_id == "lewtun/drug-reviews"
    assert list(df.columns) == ["text", "rating"]
    assert len(df) == 20
    assert df["rating"].between(1, 10).all()


@pytest.mark.integration
def test_load_raw_dataset_raises_when_no_candidate_matches():
    with pytest.raises(RuntimeError):
        data.load_raw_dataset(candidates=["surrey-nlp/PLOD-filtered"], split="train[:5]")


def test_convert_ratings_to_labels_boundaries():
    df = pd.DataFrame({"text": ["a", "b", "c", "d", "e", "f"], "rating": [1, 4, 5, 6, 7, 10]})
    out = data.convert_ratings_to_labels(df)
    assert list(out["label"]) == [
        config.LABEL2ID["NEGATIVE"],
        config.LABEL2ID["NEGATIVE"],
        config.LABEL2ID["NEUTRAL"],
        config.LABEL2ID["NEUTRAL"],
        config.LABEL2ID["POSITIVE"],
        config.LABEL2ID["POSITIVE"],
    ]


def _synthetic_labeled_df(n_per_class=100):
    rows = []
    for label, rating in ((0, 2), (1, 5), (2, 9)):
        for i in range(n_per_class):
            rows.append({"text": f"label{label}-{i}", "rating": rating})
    df = pd.DataFrame(rows)
    return data.convert_ratings_to_labels(df)


def test_stratified_split_sizes_and_proportions():
    df = _synthetic_labeled_df(n_per_class=100)  # 300 rows, balanced
    train_df, val_df, test_df = data.stratified_split(df, subset_size=None, seed=42)

    assert len(train_df) + len(val_df) + len(test_df) == 300
    assert abs(len(train_df) - 240) <= 3
    assert abs(len(val_df) - 30) <= 3
    assert abs(len(test_df) - 30) <= 3

    for split_df in (train_df, val_df, test_df):
        counts = split_df["label"].value_counts()
        assert set(counts.index) == {0, 1, 2}


def test_stratified_split_respects_subset_size():
    df = _synthetic_labeled_df(n_per_class=100)  # 300 rows
    train_df, val_df, test_df = data.stratified_split(df, subset_size=60, seed=42)
    assert len(train_df) + len(val_df) + len(test_df) == 60


def test_save_and_load_splits_round_trip(tmp_path):
    df = _synthetic_labeled_df(n_per_class=10)
    train_df, val_df, test_df = data.stratified_split(df, subset_size=None, seed=42)
    data.save_splits(train_df, val_df, test_df, tmp_path)

    loaded_train, loaded_val, loaded_test = data.load_splits(tmp_path)
    assert len(loaded_train) == len(train_df)
    assert len(loaded_val) == len(val_df)
    assert len(loaded_test) == len(test_df)
    assert set(loaded_train.columns) >= {"text", "rating", "label"}
