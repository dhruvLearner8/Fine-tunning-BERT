"""Dataset loading, label conversion, and stratified splitting."""
from pathlib import Path
from typing import Optional

import pandas as pd
from datasets import load_dataset

TEXT_COLUMN_CANDIDATES = ("review", "review_text", "text")
RATING_COLUMN_CANDIDATES = ("rating", "score")

DATASET_CANDIDATES = [
    "lewtun/drug-reviews",
    "surrey-nlp/PLOD-filtered",
    "health_fact",
    "BI55/MedText",
]


def _find_column(columns, candidates):
    for candidate in candidates:
        if candidate in columns:
            return candidate
    return None


def load_raw_dataset(candidates: Optional[list] = None, split: str = "train"):
    """Try each candidate HF dataset id in order; return (df, dataset_id) for
    the first one exposing a text column and a numeric rating column."""
    candidates = candidates or DATASET_CANDIDATES
    last_error = None
    for dataset_id in candidates:
        try:
            ds = load_dataset(dataset_id, split=split)
        except Exception as exc:  # noqa: BLE001 - trying several unrelated sources on purpose
            last_error = exc
            continue
        text_col = _find_column(ds.column_names, TEXT_COLUMN_CANDIDATES)
        rating_col = _find_column(ds.column_names, RATING_COLUMN_CANDIDATES)
        if text_col is None or rating_col is None:
            continue
        df = ds.to_pandas()[[text_col, rating_col]].rename(
            columns={text_col: "text", rating_col: "rating"}
        )
        df = df.dropna(subset=["text", "rating"]).reset_index(drop=True)
        return df, dataset_id
    raise RuntimeError(
        f"None of {candidates} exposed a text+rating schema. Last error: {last_error}"
    )
