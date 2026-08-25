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
