# Patient Sentiment Classifier Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fine-tune BioBERT on patient drug reviews to classify sentiment (POSITIVE/NEGATIVE/NEUTRAL), comparing zero-shot, full fine-tuning, and LoRA fine-tuning, with every experiment tracked in MLflow and a README documenting the results.

**Architecture:** A small installable Python package (`src/biobert_sentiment/`) holds all reusable logic (data loading, tokenization, model loading, training, evaluation, MLflow logging, inference) behind pure, testable functions. Numbered scripts in `scripts/` call that package to execute each of the user's 9 steps in order, writing intermediate artifacts (split files, trained models, MLflow runs) to disk so later scripts can consume earlier outputs. A `SUBSET_SIZE` config knob makes every script run end-to-end locally on a small stratified subset now, and identically on Colab against the full dataset later — no logic is duplicated between the two.

**Tech Stack:** transformers, datasets, peft (LoRA), accelerate, torch, scikit-learn, pandas, mlflow, pytest.

**Spec:** [docs/superpowers/specs/2026-08-25-patient-sentiment-classifier-design.md](../specs/2026-08-25-patient-sentiment-classifier-design.md)

## Global Constraints

- Model: `dmis-lab/biobert-base-cased-v1.2`, `max_length=256`, 3 labels.
- Label buckets (exact): rating 1–4 → NEGATIVE (0); rating 5–6 → NEUTRAL (1); rating 7–10 → POSITIVE (2).
- Split: 80/10/10 train/val/test, stratified on the derived label.
- Full fine-tuning and LoRA fine-tuning both use: 3 epochs, learning_rate=2e-5, batch_size=16.
- LoRA config (exact): `LoraConfig(task_type=TaskType.SEQ_CLS, r=8, lora_alpha=32, lora_dropout=0.1, target_modules=["query", "value"])`.
- MLflow experiment name: `patient-sentiment-classifier`.
- Confirmed real dataset (verified live against the HuggingFace Hub during planning): **`lewtun/drug-reviews`** — columns `review` (text, 3–10.8k chars) and `rating` (float, 1–10), `train`/`test` splits, ~215k rows total. This is the correct dataset behind the user's `datasets/drug_reviews` placeholder and is tried first in the fallback chain.
- Local runs use `SUBSET_SIZE=600` (stratified subsample) by default so all three experiments finish in minutes on CPU/MPS; set the `SUBSET_SIZE` env var to `none` to use the full dataset (the Colab configuration).
- All repo-relative paths below assume the working directory is the repo root: `/Users/dhruvpatel/Desktop/Projects 1/BioBert`.

---

## Task 1: Project Scaffolding

**Files:**
- Create: `pyproject.toml`
- Create: `requirements.txt`
- Create: `.gitignore`
- Create: `src/biobert_sentiment/__init__.py`
- Create: `tests/__init__.py`
- Create: `scripts/`, `notebooks/`, `data/processed/`, `models/` (empty dirs, via `.gitkeep` where needed)

**Interfaces:**
- Produces: an installable `biobert_sentiment` package importable from anywhere in the repo after `pip install -e .`.

- [ ] **Step 1: Create the directory skeleton**

```bash
mkdir -p "src/biobert_sentiment" "scripts" "notebooks" "tests" "data/processed" "models"
touch "src/biobert_sentiment/__init__.py" "tests/__init__.py" "data/processed/.gitkeep" "models/.gitkeep"
```

- [ ] **Step 2: Write `pyproject.toml`**

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "biobert-sentiment"
version = "0.1.0"
requires-python = ">=3.10"

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
markers = [
    "integration: requires network access to download from the HuggingFace Hub",
]
```

- [ ] **Step 3: Write `requirements.txt`**

```
transformers>=4.46
datasets>=2.19
peft>=0.11
accelerate>=0.30
torch>=2.2
scikit-learn>=1.4
pandas>=2.2
mlflow>=2.13
pytest>=8.0
```

- [ ] **Step 4: Write `.gitignore`**

```
__pycache__/
*.pyc
.venv/
venv/
*.egg-info/
mlruns/
models/*
!models/.gitkeep
patient-sentiment-final/
data/processed/*
!data/processed/.gitkeep
.pytest_cache/
.DS_Store
```

- [ ] **Step 5: Install and verify the package imports**

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
python -c "import biobert_sentiment; print('ok')"
```

Expected: prints `ok`.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml requirements.txt .gitignore src tests data/processed/.gitkeep models/.gitkeep
git commit -m "chore: scaffold biobert_sentiment package"
```

---

## Task 2: Config Module

**Files:**
- Create: `src/biobert_sentiment/config.py`
- Test: `tests/test_config.py`

**Interfaces:**
- Consumes: nothing (base module).
- Produces: `REPO_ROOT, DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, FINAL_MODEL_DIR, MLRUNS_DIR: Path`; `BIOBERT_MODEL_NAME: str`; `MAX_LENGTH=256`; `NUM_LABELS=3`; `LABEL_NAMES: list[str]`; `LABEL2ID, ID2LABEL: dict`; `EPOCHS=3, LEARNING_RATE=2e-5, BATCH_SIZE=16`; `LORA_R=8, LORA_ALPHA=32, LORA_DROPOUT=0.1, LORA_TARGET_MODULES`; `SUBSET_SIZE: int | None`; `MLFLOW_EXPERIMENT_NAME: str`; `get_device() -> str`; `DEVICE: str`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_config.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_config.py -v`
Expected: FAIL with `ModuleNotFoundError` or `AttributeError` (config.py doesn't exist yet).

- [ ] **Step 3: Write the implementation**

```python
# src/biobert_sentiment/config.py
"""Central configuration: paths, hyperparameters, and label maps."""
import os
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = REPO_ROOT / "models"
FINAL_MODEL_DIR = REPO_ROOT / "patient-sentiment-final"
MLRUNS_DIR = REPO_ROOT / "mlruns"

BIOBERT_MODEL_NAME = "dmis-lab/biobert-base-cased-v1.2"
MAX_LENGTH = 256
NUM_LABELS = 3
LABEL_NAMES = ["NEGATIVE", "NEUTRAL", "POSITIVE"]
LABEL2ID = {name: i for i, name in enumerate(LABEL_NAMES)}
ID2LABEL = {i: name for i, name in enumerate(LABEL_NAMES)}

EPOCHS = 3
LEARNING_RATE = 2e-5
BATCH_SIZE = 16

LORA_R = 8
LORA_ALPHA = 32
LORA_DROPOUT = 0.1
LORA_TARGET_MODULES = ["query", "value"]

# Drugs.com-style ratings skew positive with a thin NEUTRAL band (5-6), so the
# local subset needs to be large enough that stratified splitting still leaves
# a handful of NEUTRAL examples in val/test.
_subset_env = os.environ.get("SUBSET_SIZE", "600")
SUBSET_SIZE = None if _subset_env.strip().lower() in ("none", "0", "") else int(_subset_env)

MLFLOW_EXPERIMENT_NAME = "patient-sentiment-classifier"


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


DEVICE = get_device()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_config.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add src/biobert_sentiment/config.py tests/test_config.py
git commit -m "feat: add central config module"
```

---

## Task 3: Dataset Loading with Fallback Chain

**Files:**
- Create: `src/biobert_sentiment/data.py` (this task: `load_raw_dataset` only)
- Test: `tests/test_data.py` (this task: dataset-loading tests only)

**Interfaces:**
- Consumes: nothing external besides `datasets.load_dataset`.
- Produces: `load_raw_dataset(candidates: list[str] | None = None, split: str = "train") -> tuple[pandas.DataFrame, str]` — DataFrame has exactly columns `text` and `rating`; second element is the dataset id that succeeded.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_data.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_data.py -v -m integration`
Expected: FAIL with `ModuleNotFoundError` (data.py doesn't exist yet).

- [ ] **Step 3: Write the implementation**

```python
# src/biobert_sentiment/data.py
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_data.py -v -m integration`
Expected: PASS (2 tests). Requires network access to huggingface.co.

- [ ] **Step 5: Commit**

```bash
git add src/biobert_sentiment/data.py tests/test_data.py
git commit -m "feat: add dataset loading with fallback chain"
```

---

## Task 4: Label Conversion & Stratified Split

**Files:**
- Modify: `src/biobert_sentiment/data.py` (add `convert_ratings_to_labels`, `stratified_split`, `save_splits`, `load_splits`)
- Test: `tests/test_data.py` (append pure unit tests)

**Interfaces:**
- Consumes: `config.LABEL2ID` from [[Task 2]]; the `text`/`rating` DataFrame shape produced by `load_raw_dataset` in [[Task 3]].
- Produces: `convert_ratings_to_labels(df) -> pd.DataFrame` (adds int `label` column); `stratified_split(df, subset_size=None, seed=42) -> (train_df, val_df, test_df)`; `save_splits(train_df, val_df, test_df, out_dir: Path) -> None`; `load_splits(out_dir: Path) -> (train_df, val_df, test_df)`.

- [ ] **Step 1: Write the failing tests**

```python
# append to tests/test_data.py
import pandas as pd

from biobert_sentiment import config, data


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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_data.py -v -k "not integration"`
Expected: FAIL with `AttributeError: module 'biobert_sentiment.data' has no attribute 'convert_ratings_to_labels'`.

- [ ] **Step 3: Write the implementation**

```python
# append to src/biobert_sentiment/data.py
from sklearn.model_selection import train_test_split

from biobert_sentiment import config


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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_data.py -v -k "not integration"`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add src/biobert_sentiment/data.py tests/test_data.py
git commit -m "feat: add rating-to-label conversion and stratified split"
```

---

## Task 5: Tokenization Module

**Files:**
- Create: `src/biobert_sentiment/tokenize.py`
- Test: `tests/test_tokenize.py`

**Interfaces:**
- Consumes: `config.BIOBERT_MODEL_NAME`, `config.MAX_LENGTH` from [[Task 2]].
- Produces: `get_tokenizer() -> PreTrainedTokenizerBase`; `tokenize_dataset(df: pd.DataFrame, tokenizer) -> datasets.Dataset` with columns `input_ids`, `attention_mask`, `token_type_ids`, `labels` (torch tensors) plus original `text` still accessible per-row.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_tokenize.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_tokenize.py -v`
Expected: FAIL with `ModuleNotFoundError` (tokenize.py doesn't exist yet).

- [ ] **Step 3: Write the implementation**

```python
# src/biobert_sentiment/tokenize.py
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_tokenize.py -v`
Expected: PASS (2 tests). Downloads the small BioBERT tokenizer files on first run.

- [ ] **Step 5: Commit**

```bash
git add src/biobert_sentiment/tokenize.py tests/test_tokenize.py
git commit -m "feat: add BioBERT tokenization module"
```

---

## Task 6: Step 1 Script — Data Exploration & Split Persistence

**Files:**
- Create: `scripts/01_explore_data.py`

**Interfaces:**
- Consumes: `data.load_raw_dataset`, `data.convert_ratings_to_labels`, `data.stratified_split`, `data.save_splits` from [[Task 3]]/[[Task 4]]; `config.SUBSET_SIZE`, `config.PROCESSED_DATA_DIR`, `config.ID2LABEL` from [[Task 2]].
- Produces: `data/processed/{train,val,test}.jsonl` on disk, consumed by every later script.

- [ ] **Step 1: Write the script**

```python
# scripts/01_explore_data.py
"""Step 1: load raw data, convert ratings to labels, show distribution, split, and persist."""
from biobert_sentiment import config, data


def main():
    df, dataset_id = data.load_raw_dataset()
    print(f"Loaded {len(df)} rows from '{dataset_id}'")
    print(df.head(5))

    labeled_df = data.convert_ratings_to_labels(df)
    counts = labeled_df["label"].map(config.ID2LABEL).value_counts()
    print("\nClass distribution (full dataset):")
    print(counts)
    print(
        "\nDrugs.com-style ratings skew positive; NEUTRAL is the thin band (5-6). "
        "Stratified sampling below keeps that proportion intact in every split "
        "instead of a random split accidentally starving NEUTRAL in val/test."
    )

    train_df, val_df, test_df = data.stratified_split(labeled_df, subset_size=config.SUBSET_SIZE)
    print(f"\nSubset size: {config.SUBSET_SIZE or 'full dataset'}")
    print(f"Split sizes -> train: {len(train_df)}, val: {len(val_df)}, test: {len(test_df)}")
    for name, split_df in (("train", train_df), ("val", val_df), ("test", test_df)):
        print(f"\n{name} distribution:")
        print(split_df["label"].map(config.ID2LABEL).value_counts())

    data.save_splits(train_df, val_df, test_df, config.PROCESSED_DATA_DIR)
    print(f"\nSaved splits to {config.PROCESSED_DATA_DIR}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the script and verify output**

Run: `python scripts/01_explore_data.py`
Expected: prints the 5 sample rows, class distribution, split sizes/distributions, and creates `data/processed/train.jsonl`, `data/processed/val.jsonl`, `data/processed/test.jsonl`.

```bash
test -f data/processed/train.jsonl && test -f data/processed/val.jsonl && test -f data/processed/test.jsonl && echo "splits present"
```

- [ ] **Step 3: Commit**

```bash
git add scripts/01_explore_data.py
git commit -m "feat: add Step 1 data exploration and split script"
```

---

## Task 7: Models Module

**Files:**
- Create: `src/biobert_sentiment/models.py`
- Test: `tests/test_models.py`

**Interfaces:**
- Consumes: `config.BIOBERT_MODEL_NAME, NUM_LABELS, ID2LABEL, LABEL2ID, DEVICE, LORA_R, LORA_ALPHA, LORA_DROPOUT, LORA_TARGET_MODULES` from [[Task 2]].
- Produces: `load_base_model() -> AutoModelForSequenceClassification`; `load_lora_model() -> PeftModel`; `get_trainable_param_counts(model) -> (trainable: int, total: int, pct: float)`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_models.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_models.py -v -m integration`
Expected: FAIL with `ModuleNotFoundError` (models.py doesn't exist yet).

- [ ] **Step 3: Write the implementation**

```python
# src/biobert_sentiment/models.py
"""BioBERT model loading for full fine-tuning and LoRA fine-tuning."""
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForSequenceClassification

from biobert_sentiment import config


def load_base_model():
    return AutoModelForSequenceClassification.from_pretrained(
        config.BIOBERT_MODEL_NAME,
        num_labels=config.NUM_LABELS,
        id2label=config.ID2LABEL,
        label2id=config.LABEL2ID,
    ).to(config.DEVICE)


def load_lora_model():
    base_model = load_base_model()
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=config.LORA_R,
        lora_alpha=config.LORA_ALPHA,
        lora_dropout=config.LORA_DROPOUT,
        target_modules=config.LORA_TARGET_MODULES,
    )
    return get_peft_model(base_model, lora_config)


def get_trainable_param_counts(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    pct = 100 * trainable / total
    return trainable, total, pct
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_models.py -v -m integration`
Expected: PASS (2 tests). Downloads full BioBERT weights (~440MB) on first run.

- [ ] **Step 5: Commit**

```bash
git add src/biobert_sentiment/models.py tests/test_models.py
git commit -m "feat: add BioBERT and LoRA model loading"
```

---

## Task 8: Evaluation Module

**Files:**
- Create: `src/biobert_sentiment/evaluate.py`
- Test: `tests/test_evaluate.py`

**Interfaces:**
- Consumes: `config.LABEL_NAMES, NUM_LABELS` from [[Task 2]].
- Produces: `compute_metrics(eval_pred) -> dict` (accepts either a `transformers.EvalPrediction` or a plain `(predictions, labels)` tuple — returns `{"f1", "precision", "recall"}`, macro-averaged); `build_classification_report(y_true, y_pred) -> str`; `build_confusion_matrix(y_true, y_pred) -> np.ndarray`.

**Note:** `transformers.EvalPrediction` is a 3-field NamedTuple (`predictions, label_ids, inputs`), so `predictions, labels = eval_pred` would raise `ValueError: too many values to unpack`. Use attribute access for that branch.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_evaluate.py
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from transformers import EvalPrediction

from biobert_sentiment.evaluate import build_classification_report, build_confusion_matrix, compute_metrics


def test_compute_metrics_matches_manual_sklearn_calculation():
    y_true = np.array([0, 0, 1, 1, 2, 2])
    y_pred = np.array([0, 1, 1, 1, 2, 0])
    logits = np.zeros((len(y_pred), 3))
    for i, p in enumerate(y_pred):
        logits[i, p] = 10.0

    result = compute_metrics(EvalPrediction(predictions=logits, label_ids=y_true))

    expected_p, expected_r, expected_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    assert result["f1"] == expected_f1
    assert result["precision"] == expected_p
    assert result["recall"] == expected_r


def test_compute_metrics_accepts_plain_tuple():
    logits = np.array([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0]])
    labels = np.array([0, 1])
    result = compute_metrics((logits, labels))
    assert result["f1"] == 1.0


def test_build_confusion_matrix_shape_and_diagonal():
    y_true = np.array([0, 1, 2, 0])
    y_pred = np.array([0, 1, 2, 0])
    cm = build_confusion_matrix(y_true, y_pred)
    assert cm.shape == (3, 3)
    assert cm.trace() == 4


def test_build_classification_report_includes_label_names():
    report = build_classification_report(np.array([0, 1, 2]), np.array([0, 1, 2]))
    assert "NEGATIVE" in report and "NEUTRAL" in report and "POSITIVE" in report
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_evaluate.py -v`
Expected: FAIL with `ModuleNotFoundError` (evaluate.py doesn't exist yet).

- [ ] **Step 3: Write the implementation**

```python
# src/biobert_sentiment/evaluate.py
"""Metric computation, confusion matrices, and classification reports."""
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support

from biobert_sentiment import config


def compute_metrics(eval_pred) -> dict:
    if hasattr(eval_pred, "predictions"):
        predictions, labels = eval_pred.predictions, eval_pred.label_ids
    else:
        predictions, labels = eval_pred
    preds = np.argmax(predictions, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="macro", zero_division=0
    )
    return {"f1": f1, "precision": precision, "recall": recall}


def build_classification_report(y_true, y_pred) -> str:
    return classification_report(y_true, y_pred, target_names=config.LABEL_NAMES, zero_division=0)


def build_confusion_matrix(y_true, y_pred) -> np.ndarray:
    return confusion_matrix(y_true, y_pred, labels=list(range(config.NUM_LABELS)))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_evaluate.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add src/biobert_sentiment/evaluate.py tests/test_evaluate.py
git commit -m "feat: add metrics, confusion matrix, and classification report helpers"
```

---

## Task 9: MLflow Logging Utility

**Files:**
- Create: `src/biobert_sentiment/mlflow_utils.py`
- Test: `tests/test_mlflow_utils.py`

**Interfaces:**
- Consumes: `config.MLFLOW_EXPERIMENT_NAME` from [[Task 2]].
- Produces: `log_run(run_name: str, params: dict, metrics: dict, experiment_name: str = config.MLFLOW_EXPERIMENT_NAME) -> None`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_mlflow_utils.py
import mlflow

from biobert_sentiment import mlflow_utils


def test_log_run_records_params_and_metrics(tmp_path, monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"file://{tmp_path}")
    mlflow.set_tracking_uri(f"file://{tmp_path}")

    mlflow_utils.log_run(
        run_name="unit_test_run",
        params={"model": "biobert", "epochs": 3, "lora": False},
        metrics={"f1": 0.9, "precision": 0.88, "recall": 0.91},
        experiment_name="unit-test-experiment",
    )

    client = mlflow.tracking.MlflowClient(tracking_uri=f"file://{tmp_path}")
    experiment = client.get_experiment_by_name("unit-test-experiment")
    assert experiment is not None
    runs = client.search_runs(experiment.experiment_id)
    assert len(runs) == 1
    run = runs[0]
    assert run.data.params["model"] == "biobert"
    assert run.data.params["lora"] == "False"
    assert run.data.metrics["f1"] == 0.9
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_mlflow_utils.py -v`
Expected: FAIL with `ModuleNotFoundError` (mlflow_utils.py doesn't exist yet).

- [ ] **Step 3: Write the implementation**

```python
# src/biobert_sentiment/mlflow_utils.py
"""Thin wrapper around MLflow run logging for experiment tracking."""
import mlflow

from biobert_sentiment import config


def log_run(
    run_name: str,
    params: dict,
    metrics: dict,
    experiment_name: str = config.MLFLOW_EXPERIMENT_NAME,
) -> None:
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=run_name):
        for key, value in params.items():
            mlflow.log_param(key, value)
        for key, value in metrics.items():
            mlflow.log_metric(key, value)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_mlflow_utils.py -v`
Expected: PASS (1 test).

- [ ] **Step 5: Commit**

```bash
git add src/biobert_sentiment/mlflow_utils.py tests/test_mlflow_utils.py
git commit -m "feat: add MLflow run logging utility"
```

---

## Task 10: Training Module

**Files:**
- Create: `src/biobert_sentiment/train.py`
- Test: `tests/test_train.py`

**Interfaces:**
- Consumes: `config.EPOCHS, LEARNING_RATE, BATCH_SIZE` from [[Task 2]]; `evaluate.compute_metrics` from [[Task 8]].
- Produces: `build_training_args(output_dir: Path, epochs=config.EPOCHS, lr=config.LEARNING_RATE, batch_size=config.BATCH_SIZE) -> TrainingArguments`; `run_training(model, tokenizer, train_dataset, val_dataset, training_args) -> Trainer` (trained — `.train()` already called).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_train.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_train.py -v`
Expected: FAIL with `ModuleNotFoundError` (train.py doesn't exist yet).

- [ ] **Step 3: Write the implementation**

```python
# src/biobert_sentiment/train.py
"""Shared Trainer setup for full fine-tuning and LoRA fine-tuning."""
from pathlib import Path

from transformers import Trainer, TrainingArguments

from biobert_sentiment import config
from biobert_sentiment.evaluate import compute_metrics


def build_training_args(
    output_dir: Path,
    epochs: int = config.EPOCHS,
    lr: float = config.LEARNING_RATE,
    batch_size: int = config.BATCH_SIZE,
) -> TrainingArguments:
    return TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        report_to=[],
        logging_steps=10,
    )


def run_training(model, tokenizer, train_dataset, val_dataset, training_args: TrainingArguments) -> Trainer:
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    return trainer
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_train.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add src/biobert_sentiment/train.py tests/test_train.py
git commit -m "feat: add shared Trainer setup module"
```

---

## Task 11: Experiment 1 Script — Zero-Shot Baseline

**Files:**
- Create: `scripts/02_experiment_zero_shot.py`

**Interfaces:**
- Consumes: `data.load_splits`, `tokenize.get_tokenizer/tokenize_dataset`, `models.load_base_model`, `evaluate.*`, `mlflow_utils.log_run`, `train.build_training_args` (for eval-only args) from earlier tasks.
- Produces: an MLflow run named `zero_shot` in experiment `patient-sentiment-classifier` with metrics `f1, precision, recall, trainable_params=0, train_time_seconds`.

- [ ] **Step 1: Write the script**

```python
# scripts/02_experiment_zero_shot.py
"""Experiment 1: zero-shot BioBERT baseline, no training."""
import time

import numpy as np
from transformers import Trainer

from biobert_sentiment import config, data, evaluate, mlflow_utils, models, tokenize
from biobert_sentiment.train import build_training_args


def main():
    _, _, test_df = data.load_splits(config.PROCESSED_DATA_DIR)
    tokenizer = tokenize.get_tokenizer()
    test_dataset = tokenize.tokenize_dataset(test_df, tokenizer)

    model = models.load_base_model()
    args = build_training_args(config.MODELS_DIR / "zero_shot", epochs=0)
    trainer = Trainer(model=model, args=args, tokenizer=tokenizer, compute_metrics=evaluate.compute_metrics)

    start = time.time()
    predictions = trainer.predict(test_dataset)
    elapsed = time.time() - start

    y_pred = np.argmax(predictions.predictions, axis=-1)
    y_true = predictions.label_ids

    print(evaluate.build_classification_report(y_true, y_pred))
    print("Confusion matrix:")
    print(evaluate.build_confusion_matrix(y_true, y_pred))

    metrics = evaluate.compute_metrics((predictions.predictions, y_true))
    mlflow_utils.log_run(
        run_name="zero_shot",
        params={"model": "biobert", "epochs": 0, "learning_rate": "n/a", "lora": False},
        metrics={**metrics, "trainable_params": 0, "train_time_seconds": elapsed},
    )
    print(f"\nF1: {metrics['f1']:.3f}  Precision: {metrics['precision']:.3f}  Recall: {metrics['recall']:.3f}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the script and verify output**

Run: `python scripts/02_experiment_zero_shot.py`
Expected: prints a classification report, confusion matrix, and F1/precision/recall; creates one MLflow run named `zero_shot`.

```bash
python -c "
import mlflow
client = mlflow.tracking.MlflowClient()
exp = client.get_experiment_by_name('patient-sentiment-classifier')
runs = client.search_runs(exp.experiment_id, filter_string=\"tags.mlflow.runName = 'zero_shot'\")
assert len(runs) == 1, 'expected exactly one zero_shot run'
print('zero_shot run logged:', runs[0].data.metrics)
"
```

- [ ] **Step 3: Commit**

```bash
git add scripts/02_experiment_zero_shot.py
git commit -m "feat: add Experiment 1 zero-shot baseline script"
```

---

## Task 12: Experiment 2 Script — Full Fine-Tuning

**Files:**
- Create: `scripts/03_experiment_full_finetune.py`

**Interfaces:**
- Consumes: `data.load_splits`, `tokenize.*`, `models.load_base_model/get_trainable_param_counts`, `train.build_training_args/run_training`, `evaluate.*`, `mlflow_utils.log_run` from earlier tasks.
- Produces: `models/full_finetune/` (saved model + tokenizer) and an MLflow run named `full_finetune`, consumed by [[Task 14]].

- [ ] **Step 1: Write the script**

```python
# scripts/03_experiment_full_finetune.py
"""Experiment 2: full fine-tuning of all BioBERT parameters."""
import time

import numpy as np

from biobert_sentiment import config, data, evaluate, mlflow_utils, models, tokenize
from biobert_sentiment.train import build_training_args, run_training


def main():
    train_df, val_df, test_df = data.load_splits(config.PROCESSED_DATA_DIR)
    tokenizer = tokenize.get_tokenizer()
    train_dataset = tokenize.tokenize_dataset(train_df, tokenizer)
    val_dataset = tokenize.tokenize_dataset(val_df, tokenizer)
    test_dataset = tokenize.tokenize_dataset(test_df, tokenizer)

    model = models.load_base_model()
    trainable, total, pct = models.get_trainable_param_counts(model)
    print(f"Trainable params: {trainable:,} / {total:,} ({pct:.2f}%)")

    output_dir = config.MODELS_DIR / "full_finetune"
    args = build_training_args(output_dir)

    start = time.time()
    trainer = run_training(model, tokenizer, train_dataset, val_dataset, args)
    elapsed = time.time() - start

    predictions = trainer.predict(test_dataset)
    y_pred = np.argmax(predictions.predictions, axis=-1)
    y_true = predictions.label_ids

    print(evaluate.build_classification_report(y_true, y_pred))
    print("Confusion matrix:")
    print(evaluate.build_confusion_matrix(y_true, y_pred))

    correct_idx = np.where(y_pred == y_true)[0][:5]
    wrong_idx = np.where(y_pred != y_true)[0][:5]
    print("\n5 correct predictions:")
    for i in correct_idx:
        print(f"  [{config.ID2LABEL[int(y_true[i])]}] {test_df.iloc[int(i)]['text'][:120]}")
    print("\n5 wrong predictions:")
    for i in wrong_idx:
        print(
            f"  true={config.ID2LABEL[int(y_true[i])]} pred={config.ID2LABEL[int(y_pred[i])]} "
            f"{test_df.iloc[int(i)]['text'][:120]}"
        )

    metrics = evaluate.compute_metrics((predictions.predictions, y_true))
    mlflow_utils.log_run(
        run_name="full_finetune",
        params={"model": "biobert", "epochs": config.EPOCHS, "learning_rate": config.LEARNING_RATE, "lora": False},
        metrics={**metrics, "trainable_params": trainable, "train_time_seconds": elapsed},
    )

    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print(f"\nSaved model to {output_dir}")
    print(f"F1: {metrics['f1']:.3f}  Precision: {metrics['precision']:.3f}  Recall: {metrics['recall']:.3f}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the script and verify output**

Run: `python scripts/03_experiment_full_finetune.py`
Expected: prints trainable param count (100%), per-epoch train/val loss (from `Trainer`'s built-in logging), final classification report, 5 correct/5 wrong predictions; creates `models/full_finetune/` with model files and an MLflow run named `full_finetune`.

```bash
test -f models/full_finetune/config.json && echo "full_finetune model saved"
```

- [ ] **Step 3: Commit**

```bash
git add scripts/03_experiment_full_finetune.py
git commit -m "feat: add Experiment 2 full fine-tuning script"
```

---

## Task 13: Experiment 3 Script — LoRA Fine-Tuning

**Files:**
- Create: `scripts/04_experiment_lora.py`

**Interfaces:**
- Consumes: same as [[Task 12]], but `models.load_lora_model` instead of `models.load_base_model`.
- Produces: `models/lora_finetune/` (saved LoRA adapter + tokenizer) and an MLflow run named `lora_finetune`, consumed by [[Task 14]].

- [ ] **Step 1: Write the script**

```python
# scripts/04_experiment_lora.py
"""Experiment 3: LoRA fine-tuning (query/value adapters only)."""
import time

import numpy as np

from biobert_sentiment import config, data, evaluate, mlflow_utils, models, tokenize
from biobert_sentiment.train import build_training_args, run_training


def main():
    train_df, val_df, test_df = data.load_splits(config.PROCESSED_DATA_DIR)
    tokenizer = tokenize.get_tokenizer()
    train_dataset = tokenize.tokenize_dataset(train_df, tokenizer)
    val_dataset = tokenize.tokenize_dataset(val_df, tokenizer)
    test_dataset = tokenize.tokenize_dataset(test_df, tokenizer)

    model = models.load_lora_model()
    trainable, total, pct = models.get_trainable_param_counts(model)
    print(f"Trainable params: {trainable:,} / {total:,} ({pct:.2f}%)")
    model.print_trainable_parameters()

    output_dir = config.MODELS_DIR / "lora_finetune"
    args = build_training_args(output_dir)

    start = time.time()
    trainer = run_training(model, tokenizer, train_dataset, val_dataset, args)
    elapsed = time.time() - start

    predictions = trainer.predict(test_dataset)
    y_pred = np.argmax(predictions.predictions, axis=-1)
    y_true = predictions.label_ids

    print(evaluate.build_classification_report(y_true, y_pred))
    print("Confusion matrix:")
    print(evaluate.build_confusion_matrix(y_true, y_pred))

    metrics = evaluate.compute_metrics((predictions.predictions, y_true))
    mlflow_utils.log_run(
        run_name="lora_finetune",
        params={
            "model": "biobert",
            "epochs": config.EPOCHS,
            "learning_rate": config.LEARNING_RATE,
            "lora": True,
            "lora_r": config.LORA_R,
            "lora_alpha": config.LORA_ALPHA,
        },
        metrics={**metrics, "trainable_params": trainable, "train_time_seconds": elapsed},
    )

    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print(f"\nSaved LoRA adapter to {output_dir}")
    print(f"F1: {metrics['f1']:.3f}  Precision: {metrics['precision']:.3f}  Recall: {metrics['recall']:.3f}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the script and verify output**

Run: `python scripts/04_experiment_lora.py`
Expected: prints trainable param count (~0.1-0.2% of total, well under 1%), classification report, confusion matrix; creates `models/lora_finetune/` and an MLflow run named `lora_finetune`.

```bash
test -f models/lora_finetune/adapter_config.json && echo "lora adapter saved"
```

- [ ] **Step 3: Commit**

```bash
git add scripts/04_experiment_lora.py
git commit -m "feat: add Experiment 3 LoRA fine-tuning script"
```

---

## Task 14: Comparison Script + Best Model Save

**Files:**
- Create: `scripts/05_compare_experiments.py`

**Interfaces:**
- Consumes: the three MLflow runs from [[Task 11]], [[Task 12]], [[Task 13]]; `models/full_finetune/` and `models/lora_finetune/` directories.
- Produces: printed comparison table; `patient-sentiment-final/` containing the best-F1 fine-tuned model (zero-shot is excluded from "best" since it has no artifacts to serve), consumed by [[Task 15]].

- [ ] **Step 1: Write the script**

```python
# scripts/05_compare_experiments.py
"""Step 6: pull all three MLflow runs and print the comparison table; save the best model."""
import shutil

import mlflow

from biobert_sentiment import config

RUN_NAMES = ["zero_shot", "full_finetune", "lora_finetune"]
DISPLAY_NAMES = {
    "zero_shot": "Base BioBERT",
    "full_finetune": "Full Fine-tuning",
    "lora_finetune": "LoRA Fine-tuning",
}
MODEL_DIRS = {
    "full_finetune": config.MODELS_DIR / "full_finetune",
    "lora_finetune": config.MODELS_DIR / "lora_finetune",
}


def _latest_run(client, experiment_id, run_name):
    runs = client.search_runs(
        experiment_id,
        filter_string=f"tags.mlflow.runName = '{run_name}'",
        order_by=["start_time DESC"],
        max_results=1,
    )
    if not runs:
        raise RuntimeError(f"No MLflow run found named '{run_name}'. Run the corresponding experiment script first.")
    return runs[0]


def main():
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(config.MLFLOW_EXPERIMENT_NAME)
    if experiment is None:
        raise RuntimeError(f"MLflow experiment '{config.MLFLOW_EXPERIMENT_NAME}' not found. Run the experiment scripts first.")

    rows = []
    best_run_name, best_f1 = None, -1.0
    for run_name in RUN_NAMES:
        run = _latest_run(client, experiment.experiment_id, run_name)
        m = run.data.metrics
        rows.append(
            (
                DISPLAY_NAMES[run_name],
                m.get("f1", 0.0),
                m.get("precision", 0.0),
                m.get("recall", 0.0),
                int(m.get("trainable_params", 0)),
                m.get("train_time_seconds", 0.0),
            )
        )
        if run_name != "zero_shot" and m.get("f1", 0.0) > best_f1:
            best_f1, best_run_name = m.get("f1", 0.0), run_name

    header = f"{'Model':<20} | {'F1':<6} | {'Precision':<9} | {'Recall':<6} | {'Params Trained':<15} | {'Time (s)'}"
    print(header)
    print("-" * len(header))
    for name, f1, precision, recall, params, seconds in rows:
        print(f"{name:<20} | {f1:<6.3f} | {precision:<9.3f} | {recall:<6.3f} | {params:<15,} | {seconds:.1f}")

    print(f"\nBest fine-tuned model: {DISPLAY_NAMES[best_run_name]} (F1={best_f1:.3f})")
    src_dir = MODEL_DIRS[best_run_name]
    if config.FINAL_MODEL_DIR.exists():
        shutil.rmtree(config.FINAL_MODEL_DIR)
    shutil.copytree(src_dir, config.FINAL_MODEL_DIR)
    print(f"Saved best model to {config.FINAL_MODEL_DIR}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the script and verify output**

Run: `python scripts/05_compare_experiments.py`
Expected: prints the 3-row comparison table and which model was chosen as best; creates `patient-sentiment-final/`.

```bash
test -f "patient-sentiment-final/config.json" && echo "best model saved"
```

- [ ] **Step 3: Commit**

```bash
git add scripts/05_compare_experiments.py
git commit -m "feat: add experiment comparison script and best-model save"
```

---

## Task 15: Inference Module + Custom Examples Script

**Files:**
- Create: `src/biobert_sentiment/inference.py`
- Test: `tests/test_inference.py`
- Create: `scripts/06_custom_examples.py`

**Interfaces:**
- Consumes: `config.FINAL_MODEL_DIR, DEVICE, MAX_LENGTH, ID2LABEL` from [[Task 2]]; `patient-sentiment-final/` from [[Task 14]].
- Produces: `load_inference_model(model_dir: Path) -> (model, tokenizer)`; `logits_to_prediction(logits: np.ndarray) -> dict`; `predict(text: str, model, tokenizer) -> dict` with keys `label: str`, `confidences: dict[str, float]`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_inference.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_inference.py -v`
Expected: FAIL with `ModuleNotFoundError` (inference.py doesn't exist yet).

- [ ] **Step 3: Write the implementation**

```python
# src/biobert_sentiment/inference.py
"""Load the saved best model and run sentiment predictions on new text."""
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from biobert_sentiment import config


def load_inference_model(model_dir: Path):
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir)).to(config.DEVICE)
    model.eval()
    return model, tokenizer


def logits_to_prediction(logits: np.ndarray) -> dict:
    exp = np.exp(logits - np.max(logits))
    probs = exp / exp.sum()
    label_id = int(np.argmax(probs))
    return {
        "label": config.ID2LABEL[label_id],
        "confidences": {config.ID2LABEL[i]: float(probs[i]) for i in range(len(probs))},
    }


def predict(text: str, model, tokenizer) -> dict:
    inputs = tokenizer(
        text, truncation=True, padding="max_length", max_length=config.MAX_LENGTH, return_tensors="pt"
    ).to(config.DEVICE)
    with torch.no_grad():
        logits = model(**inputs).logits[0].cpu().numpy()
    return logits_to_prediction(logits)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_inference.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Write the custom examples script**

```python
# scripts/06_custom_examples.py
"""Step 7: run the five example sentences through the best saved model."""
from biobert_sentiment import config, inference

TEST_EXAMPLES = [
    "This medication completely changed my life, my pain is finally under control after 3 years of suffering",
    "Terrible side effects, nausea every single day, I stopped taking it after one week",
    "It works okay I guess, some days better than others, nothing remarkable",
    "The drug was killing my symptoms within 3 days, absolutely incredible results",
    "My doctor prescribed this but it made everything worse, would not recommend",
]

EXPECTED = ["POSITIVE", "NEGATIVE", "NEUTRAL", "POSITIVE", "NEGATIVE"]


def main():
    model, tokenizer = inference.load_inference_model(config.FINAL_MODEL_DIR)
    for text, expected in zip(TEST_EXAMPLES, EXPECTED):
        result = inference.predict(text, model, tokenizer)
        note = "matches intuition" if result["label"] == expected else f"expected {expected}, review this case"
        print(f"\nText: {text}")
        print(f"Predicted: {result['label']} ({note})")
        for label, conf in result["confidences"].items():
            print(f"  {label}: {conf:.3f}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Run the script and verify output**

Run: `python scripts/06_custom_examples.py`
Expected: prints predicted label + per-class confidence + sanity note for all 5 sentences, sourced from `patient-sentiment-final/`.

- [ ] **Step 7: Commit**

```bash
git add src/biobert_sentiment/inference.py tests/test_inference.py scripts/06_custom_examples.py
git commit -m "feat: add inference module and custom examples script"
```

---

## Task 16: README + Colab Notebook Wrapper

**Files:**
- Create: `README.md`
- Create: `notebooks/colab_full_run.ipynb`

**Interfaces:**
- Consumes: the printed comparison table from [[Task 14]] and the 5 predictions from [[Task 15]] (real numbers from this session's local-subset run — copy them from the terminal output, don't invent numbers).

- [ ] **Step 1: Write `README.md`**

Use the actual F1/precision/recall/time numbers printed by `scripts/05_compare_experiments.py` in Task 14 and the actual predictions printed by `scripts/06_custom_examples.py` in Task 15. Structure:

```markdown
# Patient Sentiment Classifier

Fine-tuned BioBERT that classifies patient drug-review sentiment as POSITIVE,
NEGATIVE, or NEUTRAL — built to solve a problem from my work at Sinta AI:
clustering patient social media posts into themes, but needing the emotional
tone *within* each cluster too. General sentiment models fail on medical
language ("the drug was killing it" reads as NEGATIVE to a generic model
because of "killing", when it's actually POSITIVE slang) — this model is
trained specifically on patient health language.

## Dataset & Label Conversion

[dataset id actually used, from the Task 6 run output], via
`datasets.load_dataset`. Ratings (1-10) are bucketed:
- 1-4 -> NEGATIVE
- 5-6 -> NEUTRAL
- 7-10 -> POSITIVE

Split 80/10/10, stratified on the derived label so the (thin) NEUTRAL class
keeps its proportion across train/val/test.

## Experiments

| Model | F1 | Precision | Recall | Params Trained | Time |
|---|---|---|---|---|---|
| Base BioBERT (zero-shot) | [from Task 14 output] | ... | ... | 0 | 0 |
| Full Fine-tuning | ... | ... | ... | ~110M | ... |
| LoRA Fine-tuning | ... | ... | ... | ~[N] (0.XX%) | ... |

**Key finding:** LoRA reaches [X]% of full fine-tuning's F1 while training
only [Y]% of the parameters.

These numbers are from a local ~600-example subset (`SUBSET_SIZE=600`) run on
CPU/MPS to verify the pipeline end-to-end. For the numbers at full dataset
scale, re-run `scripts/01_explore_data.py` through `scripts/05_compare_experiments.py`
on Colab with `SUBSET_SIZE=none` (a T4 GPU is recommended for the full
~172k-example dataset).

## Why BioBERT, not generic BERT

BioBERT is pretrained on PubMed abstracts and clinical text, so it already
carries biomedical vocabulary before any fine-tuning — fine-tuning on patient
reviews then teaches it *sentiment* on top of vocabulary it already knows,
rather than having to learn both from scratch.

## Why these LoRA hyperparameters

`r=8` sets the rank of the low-rank update matrices — small enough to keep
the adapter under 1% of the base model's parameters, large enough to capture
a 3-way sentiment distinction. `alpha=32` scales the adapter's contribution
(`alpha/r = 4x`), which keeps the LoRA update meaningfully large relative to
the frozen base weights despite the small rank. `target_modules=["query",
"value"]` adapts only the attention score and value projections, the two
places most fine-tuning signal in BERT-family models concentrates.

## Learning rate: 2e-5

BioBERT was pretrained with a much larger effective learning rate over many
more steps; 2e-5 is a standard *fine-tuning* rate for BERT-family models —
large enough to adapt the pretrained weights to sentiment in 3 epochs,
small enough not to destroy the pretrained representations (a common failure
mode called "catastrophic forgetting").

## Running Inference

```python
from pathlib import Path
from biobert_sentiment import inference

model, tokenizer = inference.load_inference_model(Path("patient-sentiment-final"))
result = inference.predict("The drug was killing my symptoms within days", model, tokenizer)
print(result)  # {"label": "POSITIVE", "confidences": {...}}
```

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Reproducing

```bash
python scripts/01_explore_data.py
python scripts/02_experiment_zero_shot.py
python scripts/03_experiment_full_finetune.py
python scripts/04_experiment_lora.py
python scripts/05_compare_experiments.py
python scripts/06_custom_examples.py
mlflow ui  # inspect all runs at http://localhost:5000
```
```

- [ ] **Step 2: Write `notebooks/colab_full_run.ipynb`**

```json
{
 "cells": [
  {"cell_type": "markdown", "metadata": {}, "source": ["# Patient Sentiment Classifier — Full Dataset Run (Colab T4)\n", "Thin wrapper around `src/biobert_sentiment` — same code as the local scripts, run here against the full dataset."]},
  {"cell_type": "code", "execution_count": null, "metadata": {}, "outputs": [], "source": ["!git clone <YOUR_REPO_URL> biobert-sentiment\n", "%cd biobert-sentiment\n", "!pip install -r requirements.txt\n", "!pip install -e .\n", "import os\n", "os.environ['SUBSET_SIZE'] = 'none'"]},
  {"cell_type": "code", "execution_count": null, "metadata": {}, "outputs": [], "source": ["from biobert_sentiment import config, data\n", "df, dataset_id = data.load_raw_dataset()\n", "labeled_df = data.convert_ratings_to_labels(df)\n", "train_df, val_df, test_df = data.stratified_split(labeled_df, subset_size=config.SUBSET_SIZE)\n", "data.save_splits(train_df, val_df, test_df, config.PROCESSED_DATA_DIR)\n", "print(len(train_df), len(val_df), len(test_df))"]},
  {"cell_type": "markdown", "metadata": {}, "source": ["## Experiment 1: Zero-Shot"]},
  {"cell_type": "code", "execution_count": null, "metadata": {}, "outputs": [], "source": ["!python scripts/02_experiment_zero_shot.py"]},
  {"cell_type": "markdown", "metadata": {}, "source": ["## Experiment 2: Full Fine-Tuning"]},
  {"cell_type": "code", "execution_count": null, "metadata": {}, "outputs": [], "source": ["!python scripts/03_experiment_full_finetune.py"]},
  {"cell_type": "markdown", "metadata": {}, "source": ["## Experiment 3: LoRA Fine-Tuning"]},
  {"cell_type": "code", "execution_count": null, "metadata": {}, "outputs": [], "source": ["!python scripts/04_experiment_lora.py"]},
  {"cell_type": "markdown", "metadata": {}, "source": ["## Compare & Save Best Model"]},
  {"cell_type": "code", "execution_count": null, "metadata": {}, "outputs": [], "source": ["!python scripts/05_compare_experiments.py\n", "!python scripts/06_custom_examples.py"]}
 ],
 "metadata": {
  "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
  "language_info": {"name": "python", "version": "3.10"}
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
```

- [ ] **Step 3: Verify the notebook is valid JSON**

```bash
python -c "import json; json.load(open('notebooks/colab_full_run.ipynb')); print('valid notebook JSON')"
```

- [ ] **Step 4: Commit**

```bash
git add README.md notebooks/colab_full_run.ipynb
git commit -m "docs: add README with results and Colab notebook wrapper"
```

---

## Self-Review Notes

- **Spec coverage:** Steps 1-9 from the spec map to Tasks 6 (Step 1), 5 (Step 2 tokenization), 11 (Step 3), 12 (Step 4), 13 (Step 5), 14 (Step 6), 15 (Step 7), 14's model-save (Step 8), 16 (Step 9). All three experiments, MLflow logging, LoRA hyperparameters, and comparison table match the spec's Global Constraints exactly.
- **Dataset placeholder resolved:** the spec's open question ("research the actual correct HF dataset id") was resolved during planning — confirmed live against the HF Hub as `lewtun/drug-reviews` (see Global Constraints) — not left as a TODO for the implementer.
- **Bug caught during planning:** `transformers.EvalPrediction` is a 3-field NamedTuple, so naive 2-value tuple unpacking in `compute_metrics` would raise `ValueError`. Task 8's implementation uses attribute access for that case instead, with a plain-tuple fallback for the scripts' post-hoc metric recomputation.
- **Type consistency check:** `predict()`'s return shape (`{"label": str, "confidences": dict}`) is used identically in Task 15's test and in `06_custom_examples.py`. `stratified_split`'s 3-tuple return order (`train_df, val_df, test_df`) is used consistently in Tasks 6, 11 (via `load_splits`), 12, 13. `get_trainable_param_counts`'s 3-tuple (`trainable, total, pct`) is used consistently in Tasks 7, 12, 13.
