"""Experiment: XGBoost with a reasonable baseline configuration (no
hyperparameter search yet), using the same log1p(posted_rate) target,
model-ready features, and 3 chronological validation windows as
tune_hgb.py, so the comparison against tuned HGB is apples-to-apples.

data/validation.csv is never touched.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

DATA_DIR = Path(__file__).parent.parent / "data"
PROCESSED_DIR = DATA_DIR / "processed"

WINDOWS = [
    {"name": "Window 1 (Jul-Aug)", "train_end": "2025-07-01", "holdout_end": "2025-09-01"},
    {"name": "Window 2 (Aug-Sep)", "train_end": "2025-08-01", "holdout_end": "2025-10-01"},
    {"name": "Window 3 (Sep-Oct)", "train_end": "2025-09-01", "holdout_end": "2025-11-01"},
]

# Reasonable, commonly-used starting values - not tuned.
XGB_PARAMS = {
    "n_estimators": 300,
    "learning_rate": 0.05,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "n_jobs": -1,
}

# Tuned HGB result (modeling/tune_hgb.py), averaged across the same 3 windows.
TUNED_HGB = {"avg_MAE": 119.40, "avg_RMSE": 627.96, "avg_R2": 0.8260}


def load_data() -> pd.DataFrame:
    features = pd.read_csv(PROCESSED_DIR / "train_model_ready.csv")
    dates = pd.read_csv(DATA_DIR / "train-test.csv", usecols=["load_id", "date"])
    dates["date"] = pd.to_datetime(dates["date"])
    return features.merge(dates, on="load_id", how="left")


def main():
    df = load_data()

    window_results = []
    for window in WINDOWS:
        train_end = pd.Timestamp(window["train_end"])
        holdout_end = pd.Timestamp(window["holdout_end"])

        train_df = df.loc[df["date"] < train_end]
        holdout_df = df.loc[(df["date"] >= train_end) & (df["date"] < holdout_end)]

        drop_cols = ["load_id", "date", "posted_rate"]
        X_train = train_df.drop(columns=drop_cols)
        y_train_log = np.log1p(train_df["posted_rate"])
        X_holdout = holdout_df.drop(columns=drop_cols)
        y_holdout = holdout_df["posted_rate"]

        model = XGBRegressor(**XGB_PARAMS)
        model.fit(X_train, y_train_log)
        preds = np.expm1(model.predict(X_holdout))

        result = {
            "window": window["name"],
            "MAE": mean_absolute_error(y_holdout, preds),
            "RMSE": np.sqrt(mean_squared_error(y_holdout, preds)),
            "R2": r2_score(y_holdout, preds),
        }
        window_results.append(result)

        print(f"===== {window['name']} =====")
        print(f"MAE:  {result['MAE']:.2f}")
        print(f"RMSE: {result['RMSE']:.2f}")
        print(f"R2:   {result['R2']:.4f}")
        print()

    avg_mae = float(np.mean([w["MAE"] for w in window_results]))
    avg_rmse = float(np.mean([w["RMSE"] for w in window_results]))
    avg_r2 = float(np.mean([w["R2"] for w in window_results]))

    print("===== AVERAGE ACROSS 3 WINDOWS (XGBoost baseline) =====")
    print(f"MAE:  {avg_mae:.2f}")
    print(f"RMSE: {avg_rmse:.2f}")
    print(f"R2:   {avg_r2:.4f}")

    print("\n===== COMPARISON VS TUNED HGB =====")
    print(f"{'Metric':<8}{'Tuned HGB':>12}{'XGBoost':>12}{'Delta':>12}")
    mae_delta = avg_mae - TUNED_HGB["avg_MAE"]
    rmse_delta = avg_rmse - TUNED_HGB["avg_RMSE"]
    r2_delta = avg_r2 - TUNED_HGB["avg_R2"]
    print(f"{'MAE':<8}{TUNED_HGB['avg_MAE']:>12.2f}{avg_mae:>12.2f}{mae_delta:>+12.2f}")
    print(f"{'RMSE':<8}{TUNED_HGB['avg_RMSE']:>12.2f}{avg_rmse:>12.2f}{rmse_delta:>+12.2f}")
    print(f"{'R2':<8}{TUNED_HGB['avg_R2']:>12.4f}{avg_r2:>12.4f}{r2_delta:>+12.4f}")

    print("\n===== INTERPRETATION =====")
    if avg_mae < TUNED_HGB["avg_MAE"]:
        print(f"- XGBoost (untuned baseline config) already beats tuned HGB on MAE"
              f" ({avg_mae:.2f} vs {TUNED_HGB['avg_MAE']:.2f}) - worth tuning further.")
    else:
        print(f"- XGBoost (untuned baseline config) does not beat tuned HGB on MAE"
              f" ({avg_mae:.2f} vs {TUNED_HGB['avg_MAE']:.2f}) yet - this is an"
              f" apples-to-apples comparison of an UNTUNED XGBoost against a TUNED HGB,"
              f" so it is not conclusive either way without tuning XGBoost too.")


if __name__ == "__main__":
    main()
