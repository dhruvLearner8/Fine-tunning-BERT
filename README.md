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
keeps its proportion across train/val/test: 129,038 / 16,130 / 16,129
examples on the full dataset.

## Experiments — Full Dataset Results

These are the real, full-scale numbers, trained on Colab against the
complete 161,297-row dataset (`SUBSET_SIZE=none`), not a subsample.

| Model | F1 | Precision | Recall | Params Trained | Time |
|---|---|---|---|---|---|
| Base BioBERT (zero-shot) | 0.054 | 0.030 | 0.333 | 0 (0%) | 254s (~4 min) |
| **Full Fine-tuning** | **0.765** | **0.766** | **0.765** | 108,312,579 (100%) | 18,220s (~5h 4m) |
| LoRA Fine-tuning | 0.557 | 0.528 | 0.589 | 297,219 (0.27%) | 14,977s (~4h 10m) |

**Winner: Full Fine-tuning**, by a wide margin — F1 gap of 0.208 over LoRA.
LoRA reaches about 73% of full fine-tuning's quality here, not the ~97%+ a
"within 3% F1" target would need. This is the opposite of what a smaller,
480-example pipeline-verification run initially suggested (see below) — the
full-dataset result is the one to trust.

Full fine-tuning's per-class breakdown on the 16,130-example test set:

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| NEGATIVE | 0.86 | 0.86 | 0.86 | 4,008 |
| NEUTRAL | 0.49 | 0.48 | 0.49 | 1,435 |
| POSITIVE | 0.95 | 0.95 | 0.95 | 10,687 |

Overall accuracy 88%, weighted F1 0.88. NEUTRAL is the weakest class by a
wide margin even with ~13x more NEUTRAL training examples than the local
subset had — expected, since ratings 5-6 are a genuinely ambiguous "mixed"
sentiment band, not a data-volume problem the way it looked at small scale.
Its errors split roughly evenly toward NEGATIVE and POSITIVE (357 vs. 383
misclassifications out of 1,435), consistent with real ambiguity rather than
a systematic bias toward one side.

### Why LoRA underperformed full fine-tuning at full scale

This is a real, generalizable finding on this task, not a small-sample
artifact — worth understanding rather than explaining away:

- **Capacity mismatch.** `r=8` restricted to only the `query`/`value`
  attention projections gives LoRA a genuinely small hypothesis space
  (297,219 trainable parameters, 0.27% of the model). On a noisy
  480-example run that ceiling didn't matter much because full fine-tuning
  couldn't extract a strong signal either. At 129,038 real training
  examples, full fine-tuning's ability to adjust every layer — including
  the earlier encoder layers this LoRA config never touches — clearly
  starts to matter, and the two approaches diverge sharply instead of
  converging.
- **Time savings were smaller than the parameter-count reduction implies.**
  LoRA trained 0.27% of the parameters but took 82% of full fine-tuning's
  wall-clock time (14,977s vs. 18,220s). LoRA still runs the *full forward
  pass* through the entire frozen 108M-parameter model every step — its
  savings are in backward-pass compute and optimizer memory, not in
  forward-pass time, which is what dominates wall-clock training time here.
- **What would likely close the gap**: a higher rank, broader target
  modules (adding `key` and `output.dense`, not just `query`/`value`), or a
  separate — likely higher — learning rate for the LoRA adapter than the
  `2e-5` tuned for full fine-tuning. This run doesn't prove LoRA can't match
  full fine-tuning on this task, only that this specific, deliberately
  conservative default configuration (see "Why these LoRA hyperparameters"
  below) doesn't, at this scale.

### A note on `patient-sentiment-final/`

The model saved in this repo's `patient-sentiment-final/` directory (and
gitignored, since it's a large binary artifact) is the **local
pipeline-verification model**, trained on the 480-example subset described
below — not the full-dataset model in the table above. The full-dataset
full-fine-tuning checkpoint (the actual best model) was lost to a Colab
session disconnect before it finished syncing to persistent storage; only
its metrics survived, because they were logged to MLflow and read back out
before the loss was discovered. The full-dataset LoRA checkpoint did
survive (backed up to Google Drive mid-run, once the notebook was hardened
against disconnects — see below) but isn't committed here for the same
gitignore reason. Reproducing the full run end-to-end via
`notebooks/colab_full_run.ipynb` regenerates both.

## Local Pipeline-Verification Run

Before spending Colab GPU time, every experiment was run once locally
(`SUBSET_SIZE=600`, a stratified subsample of the full dataset, giving
480/60/60 train/val/test) to verify the pipeline end-to-end on CPU/MPS.

| Model | F1 | Precision | Recall | Params Trained | Time |
|---|---|---|---|---|---|
| Base BioBERT (zero-shot) | 0.051 | 0.028 | 0.333 | 0 (0%) | 1.0s |
| Full Fine-tuning | 0.267 | 0.222 | 0.333 | 108,312,579 (100%) | 105.5s |
| LoRA Fine-tuning | 0.249 | 0.263 | 0.417 | 297,219 (0.27%) | 73.5s |

**These numbers should not be read as a benchmark** — they exist only to
prove the code runs correctly end-to-end before committing hours of GPU
time to it. With only 480 training examples (43 NEUTRAL), both fine-tuned
models collapsed toward the majority POSITIVE class rather than learning a
real decision boundary, and the LoRA-vs-full-finetuning gap here (0.018 F1)
is noise: a prior run of the *identical* code and data, differing only in
an unseeded random classification-head initialization, swung zero-shot F1
from 0.051 to 0.305 by chance alone. That swing is why `set_seed(42)` is
now pinned in `models.load_base_model()` — the same code and data now
reproduce the same result every time, which is precisely what made the
real full-dataset comparison above trustworthy enough to draw a conclusion
from.

## Why BioBERT, not generic BERT

BioBERT is pretrained on PubMed abstracts and clinical text, so it already
carries biomedical vocabulary before any fine-tuning — fine-tuning on patient
reviews then teaches it *sentiment* on top of vocabulary it already knows,
rather than having to learn both from scratch.

## Why these LoRA hyperparameters

`r=8` sets the rank of the low-rank update matrices — small enough to keep
the adapter under 1% of the base model's parameters, large enough to capture
a 3-way sentiment distinction on paper. `alpha=32` scales the adapter's
contribution (`alpha/r = 4x`), which keeps the LoRA update meaningfully
large relative to the frozen base weights despite the small rank.
`target_modules=["query", "value"]` adapts only the attention score and
value projections, the two places most fine-tuning signal in BERT-family
models concentrates for many tasks. This is a deliberately conservative,
textbook-default configuration — the full-dataset results above show it
undershoots full fine-tuning by a real margin on this task, which is itself
a useful finding about where this specific config's limits are, not a
failure of LoRA as a technique.

## Learning rate: 2e-5

BioBERT was pretrained with a much larger effective learning rate over many
more steps; 2e-5 is a standard *fine-tuning* rate for BERT-family models —
large enough to adapt the pretrained weights to sentiment in 3 epochs,
small enough not to destroy the pretrained representations (a common failure
mode called "catastrophic forgetting"). The same rate was used for both full
fine-tuning and LoRA to keep the comparison controlled — one plausible way
to close some of LoRA's gap above would be giving the adapter its own,
likely higher, learning rate instead of reusing the rate tuned for updating
100% of the parameters.

## Engineering Challenges Along the Way

Real production-adjacent friction hit while building this, worth
understanding rather than glossing over:

- **Unpinned dependencies drift underneath you.** `requirements.txt` pins
  floors (`transformers>=4.46`, `mlflow>=2.13`), not exact versions. Between
  writing the plan and running the code, `pip` resolved `transformers==5.15.1`
  and `mlflow==3.15.1` — both several major versions ahead. `transformers` 5.x
  removed the `Trainer(tokenizer=...)` kwarg entirely in favor of
  `processing_class=`; `mlflow` 3.x deprecated the file-based tracking
  backend and now raises unless `MLFLOW_ALLOW_FILE_STORE=true` is set. Both
  are fixed at the source (`train.py`'s `Trainer` construction;
  `config.py` setting the env var and tracking URI at import time) so every
  caller inherits the fix instead of patching it per-script.
- **Colab's preinstalled packages can be incompatible with what your own
  requirements pull in.** Running on Colab surfaced two more version
  collisions this local run never hit: `datasets`' `TorchFormatter`
  unconditionally probes `torchvision.io.VideoReader` for video-column
  support this project never needs, and Colab's preinstalled `torchvision`
  had removed it; `peft`'s LoRA dispatch similarly checks for a `torchao`
  quantization backend this project never uses, and raised because Colab's
  preinstalled `torchao` was older than the installed `peft` expected.
  Neither library is actually used here, so both are uninstalled in the
  Colab notebook's setup cell rather than fought into version alignment.
- **No random seed means no reproducibility, even for a "no training"
  baseline.** The zero-shot experiment loads a fresh, randomly-initialized
  classification head with no training at all — but with no seed, that
  random initialization alone made F1 swing from 0.051 to 0.305 across two
  otherwise-identical runs. `set_seed(42)` in `models.load_base_model()`
  fixed this for all three experiments, not just zero-shot.
- **A silent-failure bug found by code review, not by testing.**
  `scripts/04_experiment_lora.py` originally saved LoRA's adapter-only
  weights via `trainer.save_model()`. If LoRA had won the model comparison,
  the plain `AutoModelForSequenceClassification.from_pretrained()` call in
  `inference.py` would have loaded that adapter-only directory as a
  freshly-initialized, wrong-sized 2-label classifier — no crash, just
  confident-looking wrong predictions missing an entire class. Fixed by
  merging the adapter into the base model (`merge_and_unload()`) before
  saving, plus a defense-in-depth `num_labels` assertion in
  `load_inference_model()` so a similarly-malformed checkpoint fails loudly
  in the future instead of silently.
- **Colab's compute is ephemeral, and free-tier GPU quota runs out.** A
  session disconnect mid-run cost the full-dataset full-fine-tuning
  checkpoint entirely (see "A note on `patient-sentiment-final/`" above) —
  its weights only ever existed in that VM's `/content/` disk, which is
  wiped on disconnect. The notebook originally saved results to Google
  Drive only once, at the very end of the whole pipeline; it now mounts
  Drive up front and saves each experiment's model immediately after it
  finishes, so a disconnect anywhere in the run only costs that one step's
  work, not everything before it. Separately, the ~5-hour full-fine-tuning
  run (no mixed precision, `fp16` never enabled) consumed enough of
  Colab's free-tier GPU quota to briefly block further training —
  `fp16=True` would likely have roughly halved that training time on a T4.

## Running Inference

```python
from pathlib import Path
from biobert_sentiment import inference

model, tokenizer = inference.load_inference_model(Path("patient-sentiment-final"))
result = inference.predict("The drug was killing my symptoms within days", model, tokenizer)
print(result)  # {"label": "POSITIVE", "confidences": {...}}
```

This loads whichever model is currently saved at `patient-sentiment-final/`
— by default the local pipeline-verification model (see above), since the
full-dataset checkpoint isn't committed to the repo. To run inference
against the full-dataset model instead, download the checkpoint from
`notebooks/colab_full_run.ipynb`'s Drive backup and point
`load_inference_model` at that directory.

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
subset, use `notebooks/colab_full_run.ipynb` on Colab with a T4 GPU. It
handles the environment quirks above automatically (torchvision/torchao
removal, incremental Drive backups after each experiment) — a plain local
clone of this repo does not need those workarounds, since they're specific
to Colab's preinstalled package set.
