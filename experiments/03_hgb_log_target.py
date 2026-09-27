"""Experiment: HistGradientBoostingRegressor with the same log-target
setup as train_log_target_experiment.py.

Everything is held fixed vs. that experiment (same chronological split,
same model-ready features, same log1p target / expm1 back-transform, no
tuning) so the only variable being tested is the model family
(Random Forest vs. gradient boosting).
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DATA_DIR = Path(__file__).parent.parent / "data"
PROCESSED_DIR = DATA_DIR / "processed"

HOLDOUT_START = "2025-09-01"

RF_LOG_TARGET_METRICS = {"MAE": 169.41, "RMSE": 656.64, "R2": 0.8149}


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

    print("===== CHRONOLOGICAL SPLIT (same as previous experiments) =====")
    print(f"Training range: {train_df['date'].min().date()} to {train_df['date'].max().date()}"
          f"  ({len(train_df)} rows)")
    print(f"Holdout range:  {holdout_df['date'].min().date()} to {holdout_df['date'].max().date()}"
          f"  ({len(holdout_df)} rows)")

    drop_cols = ["load_id", "date", "posted_rate"]
    X_train = train_df.drop(columns=drop_cols)
    y_train_log = np.log1p(train_df["posted_rate"])
    X_holdout = holdout_df.drop(columns=drop_cols)
    y_holdout = holdout_df["posted_rate"]  # scored on the original $ scale

    model = HistGradientBoostingRegressor(random_state=42)
    model.fit(X_train, y_train_log)

    preds_log = model.predict(X_holdout)
    preds = np.expm1(preds_log)

    mae = mean_absolute_error(y_holdout, preds)
    rmse = np.sqrt(mean_squared_error(y_holdout, preds))
    r2 = r2_score(y_holdout, preds)

    print("\n===== HGB LOG-TARGET HOLDOUT METRICS ($ scale) =====")
    print(f"MAE:  {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R2:   {r2:.4f}")

    print("\n===== COMPARISON VS RANDOM FOREST + LOG TARGET =====")
    print(f"{'Metric':<8}{'RF+log':>12}{'HGB+log':>14}{'Delta':>12}")
    for key in ["MAE", "RMSE", "R2"]:
        base_val = RF_LOG_TARGET_METRICS[key]
        new_val = {"MAE": mae, "RMSE": rmse, "R2": r2}[key]
        delta = new_val - base_val
        print(f"{key:<8}{base_val:>12.4f}{new_val:>14.4f}{delta:>+12.4f}")

    print("\n===== INTERPRETATION =====")
    mae_better = mae < RF_LOG_TARGET_METRICS["MAE"]
    rmse_better = rmse < RF_LOG_TARGET_METRICS["RMSE"]
    r2_better = r2 > RF_LOG_TARGET_METRICS["R2"]
    print(f"- MAE {'improved' if mae_better else 'got worse'} vs RF+log "
          f"({mae:.2f} vs {RF_LOG_TARGET_METRICS['MAE']:.2f}).")
    print(f"- RMSE {'improved' if rmse_better else 'got worse'} vs RF+log "
          f"({rmse:.2f} vs {RF_LOG_TARGET_METRICS['RMSE']:.2f}).")
    print(f"- R2 {'improved' if r2_better else 'got worse'} vs RF+log "
          f"({r2:.4f} vs {RF_LOG_TARGET_METRICS['R2']:.4f}).")
    print("- Both are untuned defaults, so this comparison reflects model")
    print("  family fit on this data, not either model's ceiling.")


if __name__ == "__main__":
    main()
