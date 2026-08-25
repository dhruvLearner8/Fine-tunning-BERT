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
