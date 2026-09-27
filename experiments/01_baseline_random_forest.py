"""Baseline tree-based model for freight rate prediction.

Trains and evaluates using ONLY data/processed/train_model_ready.csv,
split chronologically into a training period and a later holdout
period. data/validation.csv is never read here - it has no
posted_rate to evaluate against and is reserved for final predictions,
not for choosing or judging this baseline.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DATA_DIR = Path(__file__).parent.parent / "data"
PROCESSED_DIR = DATA_DIR / "processed"

# train-test.csv spans Jan-Oct 2025 (~304 days). Holding out the last two
# months mirrors the real task: training data ends in October and we must
# predict November (validation.csv) and December, i.e. always forward in time.
HOLDOUT_START = "2025-09-01"


def load_data() -> pd.DataFrame:
    features = pd.read_csv(PROCESSED_DIR / "train_model_ready.csv")
    # feature_engineering.py drops the raw date column once it's encoded
    # numerically, so it's re-attached here (via load_id) purely to decide
    # the chronological split - it is not used as a model input.
    dates = pd.read_csv(DATA_DIR / "train-test.csv", usecols=["load_id", "date"])
    dates["date"] = pd.to_datetime(dates["date"])
    return features.merge(dates, on="load_id", how="left")


def main():
    df = load_data()

    cutoff = pd.Timestamp(HOLDOUT_START)
    train_df = df.loc[df["date"] < cutoff]
    holdout_df = df.loc[df["date"] >= cutoff]

    print("===== CHRONOLOGICAL SPLIT =====")
    print(f"Training range: {train_df['date'].min().date()} to {train_df['date'].max().date()}"
          f"  ({len(train_df)} rows)")
    print(f"Holdout range:  {holdout_df['date'].min().date()} to {holdout_df['date'].max().date()}"
          f"  ({len(holdout_df)} rows)")

    drop_cols = ["load_id", "date", "posted_rate"]
    X_train = train_df.drop(columns=drop_cols)
    y_train = train_df["posted_rate"]
    X_holdout = holdout_df.drop(columns=drop_cols)
    y_holdout = holdout_df["posted_rate"]

    model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    preds = model.predict(X_holdout)

    mae = mean_absolute_error(y_holdout, preds)
    rmse = np.sqrt(mean_squared_error(y_holdout, preds))
    r2 = r2_score(y_holdout, preds)

    print("\n===== HOLDOUT METRICS =====")
    print(f"MAE:  {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R2:   {r2:.4f}")

    print("\n===== INTERPRETATION =====")
    print(f"- On average, predictions are off by about ${mae:.0f} (MAE).")
    print(f"- RMSE (${rmse:.0f}) is noticeably higher than MAE, meaning a smaller")
    print(f"  number of larger errors are inflating the squared-error metric -")
    print(f"  consistent with the right-skewed posted_rate target found in EDA.")
    print(f"- R2 of {r2:.4f} means the model explains about {r2 * 100:.1f}% of the")
    print(f"  variance in posted_rate on unseen, later dates.")
    print(f"- This is a baseline with no hyperparameter tuning - it establishes")
    print(f"  a reference score, not a final result.")


if __name__ == "__main__":
    main()
