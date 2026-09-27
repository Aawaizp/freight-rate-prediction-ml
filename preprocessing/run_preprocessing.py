"""Runs the FreightPreprocessor on train-test.csv and validation.csv.

Fits the preprocessor on train-test.csv only, saves the learned medians
to preprocessing/artifacts/preprocessing_stats.json, then applies the
same transformation to both files and writes the results to
data/processed/. Raw CSVs under data/ are never modified.
"""
from pathlib import Path

import pandas as pd

from preprocess import FreightPreprocessor

DATA_DIR = Path(__file__).parent.parent / "data"
OUT_DIR = DATA_DIR / "processed"


def main():
    train = pd.read_csv(DATA_DIR / "train-test.csv")
    validation = pd.read_csv(DATA_DIR / "validation.csv")

    pre = FreightPreprocessor().fit(train)
    pre.save()

    train_clean = pre.transform(train)
    validation_clean = pre.transform(validation)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train_clean.to_csv(OUT_DIR / "train_clean.csv", index=False)
    validation_clean.to_csv(OUT_DIR / "validation_clean.csv", index=False)

    print("Weight median (from train):", pre.weight_median_)
    print("Market index median (from train):", pre.market_index_median_)
    print("Train clean shape:", train_clean.shape)
    print("Validation clean shape:", validation_clean.shape)
    print("\nTrain clean columns:", list(train_clean.columns))


if __name__ == "__main__":
    main()
