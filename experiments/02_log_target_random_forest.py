"""Experiment: same baseline setup, but the model is trained on
log1p(posted_rate) instead of the raw target, then predictions are
converted back with expm1() before scoring.

Everything else is held fixed vs. train_baseline.py (same chronological
split, same RandomForestRegressor settings, no tuning) so the only
variable being tested is the log-target transform.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DATA_DIR = Path(__file__).parent.parent / "data"
PROCESSED_DIR = DATA_DIR / "processed"

HOLDOUT_START = "2025-09-01"

BASELINE_METRICS = {"MAE": 186.29, "RMSE": 686.16, "R2": 0.7978}


def load_data() -> pd.DataFrame:
    features = pd.read_csv(PROCESSED_DIR / "train_model_ready.csv")
    dates = pd.read_csv(DATA_DIR / "train-test.csv", usecols=["load_id", "date"])
    dates["date"] = pd.to_datetime(dates["date"])
    return features.merge(dates, on="load_id", how="left")


def main():
    df = load_data()

    cutoff = pd.Timestamp(HOLDOUT_START)
    train_df = df.loc[df["date"] < cutoff]
    holdout_df = df.loc[df["date"] >= cutoff]

    print("===== CHRONOLOGICAL SPLIT (same as baseline) =====")
    print(f"Training range: {train_df['date'].min().date()} to {train_df['date'].max().date()}"
          f"  ({len(train_df)} rows)")
    print(f"Holdout range:  {holdout_df['date'].min().date()} to {holdout_df['date'].max().date()}"
          f"  ({len(holdout_df)} rows)")

    drop_cols = ["load_id", "date", "posted_rate"]
    X_train = train_df.drop(columns=drop_cols)
    y_train_log = np.log1p(train_df["posted_rate"])
    X_holdout = holdout_df.drop(columns=drop_cols)
    y_holdout = holdout_df["posted_rate"]  # scored on the original $ scale

    model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train_log)

    preds_log = model.predict(X_holdout)
    preds = np.expm1(preds_log)

    mae = mean_absolute_error(y_holdout, preds)
    rmse = np.sqrt(mean_squared_error(y_holdout, preds))
    r2 = r2_score(y_holdout, preds)

    print("\n===== LOG-TARGET HOLDOUT METRICS ($ scale) =====")
    print(f"MAE:  {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R2:   {r2:.4f}")

    print("\n===== COMPARISON VS BASELINE =====")
    print(f"{'Metric':<8}{'Baseline':>12}{'Log-target':>14}{'Delta':>12}")
    for key, label in [("MAE", "MAE"), ("RMSE", "RMSE"), ("R2", "R2")]:
        base_val = BASELINE_METRICS[key]
        new_val = {"MAE": mae, "RMSE": rmse, "R2": r2}[key]
        delta = new_val - base_val
        print(f"{label:<8}{base_val:>12.4f}{new_val:>14.4f}{delta:>+12.4f}")

    print("\n===== INTERPRETATION =====")
    mae_better = mae < BASELINE_METRICS["MAE"]
    rmse_better = rmse < BASELINE_METRICS["RMSE"]
    r2_better = r2 > BASELINE_METRICS["R2"]
    print(f"- MAE {'improved' if mae_better else 'got worse'} vs baseline "
          f"({mae:.2f} vs {BASELINE_METRICS['MAE']:.2f}).")
    print(f"- RMSE {'improved' if rmse_better else 'got worse'} vs baseline "
          f"({rmse:.2f} vs {BASELINE_METRICS['RMSE']:.2f}).")
    print(f"- R2 {'improved' if r2_better else 'got worse'} vs baseline "
          f"({r2:.4f} vs {BASELINE_METRICS['R2']:.4f}).")
    print("- The log transform compresses large posted_rate values before")
    print("  training, which should help most when large-value errors were")
    print("  dominating RMSE. Whether it helps here depends on the table above.")


if __name__ == "__main__":
    main()
