# Patient Sentiment Classifier

Fine-tuned BioBERT that classifies patient drug-review sentiment as POSITIVE,
NEGATIVE, or NEUTRAL — built to solve a problem from my work at Sinta AI:
clustering patient social media posts into themes, but needing the emotional
tone *within* each cluster too. General sentiment models fail on medical
language ("the drug was killing it" reads as NEGATIVE to a generic model
because of "killing", when it's actually POSITIVE slang) — this model is
trained specifically on patient health language.

## Dataset & Label Conversion

[`lewtun/drug-reviews`](https://huggingface.co/datasets/lewtun/drug-reviews)
from HuggingFace, loaded via `datasets.load_dataset`. The `train` split has
161,297 rows. Ratings (1-10) are bucketed:
- 1-4 -> NEGATIVE
- 5-6 -> NEUTRAL
- 7-10 -> POSITIVE

Across the full dataset this label conversion produces a class distribution
of POSITIVE 66.3% / NEGATIVE 24.9% / NEUTRAL 8.9% — a meaningfully imbalanced
3-way problem, with NEUTRAL the rare class by a wide margin.

Split 80/10/10, stratified on the derived label so the (thin) NEUTRAL class
keeps its proportion across train/val/test. The local pipeline-verification
run used `SUBSET_SIZE=600` (a stratified subsample of the full dataset),
giving 480/60/60 train/val/test examples.

## Experiments

| Model | F1 | Precision | Recall | Params Trained | Time |
|---|---|---|---|---|---|
| Base BioBERT (zero-shot) | 0.305 | 0.391 | 0.347 | 0 (0%) | 0s |
| Full Fine-tuning | 0.424 | 0.477 | 0.428 | 108,312,579 (100%) | 106s |
| LoRA Fine-tuning | 0.267 | 0.222 | 0.333 | 297,219 (0.27%) | 73s |

Best model selected: **Full Fine-tuning** (highest F1), saved to
[`patient-sentiment-final/`](patient-sentiment-final/).

**Key finding:** on this local run, LoRA (0.267 F1) underperformed full
fine-tuning (0.424 F1). Take that gap with a large grain of salt, though —
it is almost certainly an artifact of the tiny local subset, **not**
representative of how LoRA compares to full fine-tuning in general. The
subset has only 480 training examples, of which just 43 are NEUTRAL; with
that little data and that much class imbalance, LoRA's much smaller
parameter budget (0.27% of the model) simply doesn't get enough gradient
signal to find a good low-rank update, while full fine-tuning's much larger
capacity can still (barely) fit the pattern. This is the opposite of what
the standard LoRA literature reports at realistic dataset scale, where LoRA
is expected to be competitive with full fine-tuning (often within a few
points of F1) while training a tiny fraction of the parameters and in a
fraction of the time.

Both experiments also show the **NEUTRAL class collapsing to 0 recall** on
this run. With only 43 NEUTRAL examples in the 480-example training set
(~9% of the data), neither model sees enough NEUTRAL examples to learn the
class, and both end up predicting NEGATIVE/POSITIVE for every NEUTRAL test
example. This is a known limitation of the local subset and is expected to
resolve once trained on the full dataset, where NEUTRAL has ~14,000+
examples to learn from.

**These numbers are from a local ~600-example subset (`SUBSET_SIZE=600`) run
on CPU/MPS to verify the pipeline end-to-end — they are a pipeline sanity
check, not the final benchmark.** Do not read the LoRA-vs-full-finetuning
gap above as a claim like "LoRA achieves X% of full fine-tuning quality" —
at this scale the comparison is dominated by data starvation, not by any
property of LoRA itself. For the real comparison numbers, re-run
`scripts/01_explore_data.py` through `scripts/05_compare_experiments.py` on
Colab with `SUBSET_SIZE=none` against the full ~172k-example dataset (see
`notebooks/colab_full_run.ipynb`; a T4 GPU is recommended).

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

On the 5 hand-written custom examples run through the saved model
(`scripts/06_custom_examples.py`), 4 out of 5 predictions matched the
intuitively-expected label. The one miss was `"It works okay I guess, some
days better than others, nothing remarkable"` (expected NEUTRAL), which was
predicted POSITIVE with confidences NEGATIVE 0.150 / NEUTRAL 0.103 /
POSITIVE 0.747 — consistent with the NEUTRAL-class weakness noted above:
a model trained on only 43 NEUTRAL examples doesn't yet have a strong
NEUTRAL signal to fall back on for genuinely lukewarm language.

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

To reproduce the full-dataset run instead of the local pipeline-verification
subset, see `notebooks/colab_full_run.ipynb`.
