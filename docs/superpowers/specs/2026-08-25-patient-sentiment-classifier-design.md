# Patient Sentiment Classifier — Design Spec

Date: 2026-08-25

## Motivation

Portfolio project derived from work at Sinta AI, where patient social media
posts were clustered into topical themes but the emotional tone within each
cluster (not just the topic) was needed. General-purpose sentiment models
misread medical language because everyday negative-sounding words are used
positively in patient speech:

- "The drug was killing it" → POSITIVE (general models read "killing" → NEGATIVE)
- "My symptoms were dying down" → POSITIVE (general models flag "dying" → NEGATIVE)
- "The treatment was sick" → ambiguous slang, general models default NEGATIVE

Goal: fine-tune BioBERT (`dmis-lab/biobert-base-cased-v1.2`) on patient drug
reviews to classify sentiment as POSITIVE / NEGATIVE / NEUTRAL, and compare
zero-shot, full fine-tuning, and LoRA fine-tuning.

## Success Criteria

- Three experiments run end-to-end and compared in one table.
- LoRA achieves within 3% F1 of full fine-tuning.
- All experiments tracked in MLflow (params, metrics, per-run).
- Custom example sentences classified with per-class confidence shown.
- Clean repo with README explaining the project, results, and how to run inference.
- Every design decision (BioBERT vs BERT, LoRA hyperparams, learning rate,
  stratified split) is explained inline, not just implemented.

## Execution Model: Local-Dev / Colab-Portable

This session runs locally on macOS with no NVIDIA GPU (CPU/Apple MPS only).
Google Colab's free T4 GPU is the target for full-scale final numbers. To
avoid two divergent codebases:

- All logic lives in `src/biobert_sentiment/`, imported by both local scripts
  and the Colab notebook — no logic is duplicated or copy-pasted into cells.
- `config.py` exposes `SUBSET_SIZE` (int or `None`) and device selection
  (auto-detects `mps` > `cpu` locally; `cuda` on Colab, no code change needed).
- **This session**: all three experiments are actually executed locally
  against a small stratified subset (~400 examples) so real MLflow runs,
  metrics, confusion matrices, and a first-pass README are produced now.
- **On Colab**: the user reruns the identical `scripts/0N_*.py` files with
  `SUBSET_SIZE=None` against the full dataset for final reported numbers.
  `notebooks/colab_full_run.ipynb` is a thin wrapper that calls the same
  `src/` functions cell-by-cell.

## Dataset

Target: a HuggingFace dataset of patient drug reviews with free-text review
+ a numerical rating (1–10), so rating can be bucketed into sentiment labels.

Resolution order (first implementation step — verify programmatically, do
not assume any of these load without checking):
1. Research and confirm the actual correct HF dataset id for the well-known
   Drugs.com patient reviews dataset (the user's literal `datasets/drug_reviews`
   is not a valid HF id — missing an org namespace).
2. Fall back to the user-supplied alternatives in order: `surrey-nlp/PLOD-filtered`,
   `health_fact`, `BI55/MedText` — checked for a review-text + rating schema
   (most of these are not actually rating-labeled review datasets, so they're
   expected to fail this check, but are tried as specified).
3. If nothing in (1)–(2) yields review text + numeric rating, generate ~300
   synthetic patient-review examples locally (no Gemini API access in this
   environment — flagged to the user rather than silently substituted, since
   synthetic data changes the nature of the "real world data" claim in the
   README).

Label conversion (applied uniformly regardless of which source is used):
- Rating 1–4 → NEGATIVE (0)
- Rating 5–6 → NEUTRAL (1)
- Rating 7–10 → POSITIVE (2)

Split: 80/10/10 train/val/test, stratified on the derived label so class
proportions are preserved across splits (important because drug review
ratings are typically skewed positive/negative with a thin neutral band).

## Repo Structure

```
BioBert/
  README.md
  requirements.txt
  .gitignore
  src/biobert_sentiment/
    __init__.py
    config.py         # paths, hyperparams, label map, SUBSET_SIZE, device
    data.py            # dataset loading w/ fallback chain, label conversion, stratified split
    tokenize.py         # BioBERT tokenizer wrapper (max_length=256)
    models.py            # load_base_model(), load_lora_model()
    train.py              # TrainingArguments + Trainer setup, run_experiment()
    evaluate.py            # F1/precision/recall, confusion matrix, classification report
    mlflow_utils.py         # log_run() helper wrapping mlflow.start_run()
    inference.py             # load saved model, predict(text) -> label + per-class confidence
  scripts/
    01_explore_data.py        # Step 1: load, show samples, class distribution, split
    02_experiment_zero_shot.py # Step 3: Experiment 1
    03_experiment_full_finetune.py # Step 4: Experiment 2
    04_experiment_lora.py     # Step 5: Experiment 3
    05_compare_experiments.py # Step 6: comparison table from MLflow runs
    06_custom_examples.py     # Step 7: run the 5 provided test sentences
  notebooks/
    colab_full_run.ipynb      # thin wrapper around src/, for Colab GPU full runs
  tests/
    test_data.py               # label conversion, stratified split proportions
    test_tokenize.py            # tokenization shape/keys
  mlruns/                       # MLflow local tracking store (gitignored)
  patient-sentiment-final/      # saved best model (gitignored, large binaries)
```

## Experiments

All three share the same label set, tokenizer (`max_length=256`), and eval
metrics (F1/precision/recall, macro-averaged given 3 classes, plus per-class
breakdown and confusion matrix).

1. **Zero-shot baseline** — `dmis-lab/biobert-base-cased-v1.2` with an
   untrained classification head, evaluated directly on the test set. No
   training. Expected F1 ~0.55–0.65 (a fresh classification head is close to
   random, floor set by whatever signal survives from pretraining).

2. **Full fine-tuning** — all ~110M params trainable. 3 epochs, lr=2e-5,
   batch size 16. Eval after each epoch (train vs val loss). Final test-set
   evaluation, classification report, 5 correct + 5 incorrect predictions
   shown. Expected F1 ~0.87–0.91.

3. **LoRA fine-tuning** — `peft` `LoraConfig(task_type=SEQ_CLS, r=8,
   lora_alpha=32, lora_dropout=0.1, target_modules=["query","value"])` on top
   of a fresh BioBERT load. Same training args as (2). Print trainable
   parameter count (expected ~0.13% of 110M). Expected F1 ~0.85–0.89.

Each run explains inline (in the script's comments/README, not just code):
why BioBERT over generic BERT (biomedical pretraining corpus), what lr=2e-5
means for fine-tuning vs. pretraining-scale rates, why r=8/alpha=32 for LoRA,
and why stratified splitting matters given class imbalance.

## MLflow Tracking

Experiment name: `patient-sentiment-classifier`. One run per experiment
(`zero_shot`, `full_finetune`, `lora_finetune`), logging params (`model`,
`epochs`, `learning_rate`, `lora`, `batch_size`) and metrics (`f1`,
`precision`, `recall`, per-class variants, `trainable_params`,
`train_time_seconds`). Local file-based tracking store (`./mlruns`) — no
remote tracking server in scope.

## Comparison & Inference

`05_compare_experiments.py` reads back all three MLflow runs and prints the
table specified by the user (model, F1, precision, recall, params trained,
time). `06_custom_examples.py` runs the 5 given sentences through the
best-performing saved model, printing predicted label + confidence per class
and a one-line sanity note on whether the prediction matches intuition.

## Testing

TDD applies to the pure, fast, deterministic logic: rating→label conversion
boundaries, stratified split proportions, tokenizer output shape/keys. Model
training itself is not unit-tested (inherently an integration-scale run) —
it's verified by actually executing all three experiments locally against
the subset and inspecting real metrics/outputs.

## Out of Scope

- Remote MLflow tracking server / model registry.
- CI/CD, Docker packaging.
- Serving the model behind an API (README documents local inference only).
- GitHub remote push (local git only per user's request this session).
