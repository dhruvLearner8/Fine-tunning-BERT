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
    assert model.config.num_labels == config.NUM_LABELS, (
        f"Loaded model has {model.config.num_labels} labels, expected {config.NUM_LABELS} "
        f"— the saved model at {model_dir} may be corrupted or adapter-only."
    )
    return model, tokenizer


def logits_to_prediction(logits: np.ndarray) -> dict:
    exp = np.exp(logits - np.max(logits))
    probs = exp / exp.sum()
    label_id = int(np.argmax(probs))
    return {
        "label": config.ID2LABEL[label_id],
        "confidences": {config.ID2LABEL[i]: float(probs[i]) for i in range(config.NUM_LABELS)},
    }


def predict(text: str, model, tokenizer) -> dict:
    inputs = tokenizer(
        text, truncation=True, padding="max_length", max_length=config.MAX_LENGTH, return_tensors="pt"
    ).to(config.DEVICE)
    with torch.no_grad():
        logits = model(**inputs).logits[0].cpu().numpy()
    return logits_to_prediction(logits)
