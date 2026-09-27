"""Runs FeatureEngineer on the preprocessed train/validation data.

Reads data/processed/train_clean.csv and validation_clean.csv (produced
by preprocessing/run_preprocessing.py), fits FeatureEngineer on the
training split, saves the learned equipment categories to
feature_engineering/artifacts/feature_engineering_stats.json, then
writes fully numeric, model-ready tables to data/processed/.
"""
from pathlib import Path

import pandas as pd

from feature_engineering import FeatureEngineer

DATA_DIR = Path(__file__).parent.parent / "data"
PROCESSED_DIR = DATA_DIR / "processed"


def main():
    train_clean = pd.read_csv(PROCESSED_DIR / "train_clean.csv")
    validation_clean = pd.read_csv(PROCESSED_DIR / "validation_clean.csv")

    fe = FeatureEngineer().fit(train_clean)
    fe.save()

    train_features = fe.transform(train_clean)
    validation_features = fe.transform(validation_clean)

    train_features.to_csv(PROCESSED_DIR / "train_model_ready.csv", index=False)
    validation_features.to_csv(PROCESSED_DIR / "validation_model_ready.csv", index=False)

    print("Equipment categories (from train):", fe.equipment_categories_)
    print("\nTrain model-ready shape:", train_features.shape)
    print("Validation model-ready shape:", validation_features.shape)
    print("\nTrain model-ready columns:")
    print(list(train_features.columns))


if __name__ == "__main__":
    main()
