"""Dataset loading, label conversion, and stratified splitting."""
from pathlib import Path
from typing import Optional

import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split

from biobert_sentiment import config

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


def convert_ratings_to_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Bucket the 1-10 rating column into NEGATIVE(0)/NEUTRAL(1)/POSITIVE(2)."""

    def bucket(rating: float) -> int:
        if rating <= 4:
            return config.LABEL2ID["NEGATIVE"]
        if rating <= 6:
            return config.LABEL2ID["NEUTRAL"]
        return config.LABEL2ID["POSITIVE"]

    out = df.copy()
    out["label"] = out["rating"].apply(bucket)
    return out


def stratified_split(df: pd.DataFrame, subset_size: Optional[int] = None, seed: int = 42):
    """Stratify on `label`, optionally subsample first, then split 80/10/10."""
    working = df
    if subset_size is not None and subset_size < len(df):
        working, _ = train_test_split(
            df, train_size=subset_size, stratify=df["label"], random_state=seed
        )
    train_df, temp_df = train_test_split(
        working, test_size=0.2, stratify=working["label"], random_state=seed
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, stratify=temp_df["label"], random_state=seed
    )
    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def save_splits(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, split_df in (("train", train_df), ("val", val_df), ("test", test_df)):
        split_df.to_json(out_dir / f"{name}.jsonl", orient="records", lines=True)


def load_splits(out_dir: Path):
    return tuple(
        pd.read_json(out_dir / f"{name}.jsonl", orient="records", lines=True)
        for name in ("train", "val", "test")
    )
