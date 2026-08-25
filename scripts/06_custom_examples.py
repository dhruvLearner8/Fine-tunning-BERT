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
