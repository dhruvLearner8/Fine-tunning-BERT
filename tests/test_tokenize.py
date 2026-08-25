import pandas as pd

from biobert_sentiment import config, tokenize


def test_get_tokenizer_loads_biobert():
    tokenizer = tokenize.get_tokenizer()
    assert tokenizer.model_max_length >= config.MAX_LENGTH


def test_tokenize_dataset_shapes_and_columns():
    df = pd.DataFrame({"text": ["The drug was killing it", "Terrible side effects"], "label": [2, 0]})
    tokenizer = tokenize.get_tokenizer()
    dataset = tokenize.tokenize_dataset(df, tokenizer)

    assert len(dataset) == 2
    assert "labels" in dataset.column_names
    assert "label" not in dataset.column_names
    row = dataset[0]
    assert row["input_ids"].shape[0] == config.MAX_LENGTH
    assert row["attention_mask"].shape[0] == config.MAX_LENGTH
    assert dataset[0]["text"] == "The drug was killing it"
