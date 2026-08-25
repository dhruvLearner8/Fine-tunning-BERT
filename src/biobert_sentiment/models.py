"""BioBERT model loading for full fine-tuning and LoRA fine-tuning."""
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForSequenceClassification, set_seed

from biobert_sentiment import config


def load_base_model():
    set_seed(42)
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
