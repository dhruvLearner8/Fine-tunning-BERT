"""BioBERT tokenizer loading and dataset tokenization."""
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from biobert_sentiment import config


def get_tokenizer() -> PreTrainedTokenizerBase:
    return AutoTokenizer.from_pretrained(config.BIOBERT_MODEL_NAME)


def tokenize_dataset(df: pd.DataFrame, tokenizer: PreTrainedTokenizerBase) -> Dataset:
    """Convert a text(+label) dataframe into a tokenized HF Dataset ready for Trainer."""
    dataset = Dataset.from_pandas(df.reset_index(drop=True))

    def _tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=config.MAX_LENGTH,
        )

    dataset = dataset.map(_tokenize, batched=True)
    if "label" in dataset.column_names:
        dataset = dataset.rename_column("label", "labels")
    dataset.set_format(
        type="torch",
        columns=[c for c in ("input_ids", "attention_mask", "token_type_ids", "labels") if c in dataset.column_names],
        output_all_columns=True,
    )
    return dataset
