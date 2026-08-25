import pytest

from biobert_sentiment import data


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
