"""Experiment: check whether HGB + log-target performance is stable
across different chronological windows, or whether the Sep-Oct result
from train_hgb_log_target_experiment.py was a favorable slice.

Same model, same features, same log1p/expm1 target handling as that
script - only the train/holdout date ranges change across three windows.
data/validation.csv is not touched.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DATA_DIR = Path(__file__).parent.parent / "data"
PROCESSED_DIR = DATA_DIR / "processed"

WINDOWS = [
    {"name": "Window 1", "train_end": "2025-07-01", "holdout_end": "2025-09-01"},
    {"name": "Window 2", "train_end": "2025-08-01", "holdout_end": "2025-10-01"},
    {"name": "Window 3", "train_end": "2025-09-01", "holdout_end": "2025-11-01"},
]


def load_data() -> pd.DataFrame:
    features = pd.read_csv(PROCESSED_DIR / "train_model_ready.csv")
    dates = pd.read_csv(DATA_DIR / "train-test.csv", usecols=["load_id", "date"])
    dates["date"] = pd.to_datetime(dates["date"])
    return features.merge(dates, on="load_id", how="left")


def run_window(df: pd.DataFrame, train_end: str, holdout_end: str) -> dict:
    train_end_ts = pd.Timestamp(train_end)
    holdout_end_ts = pd.Timestamp(holdout_end)

    train_df = df.loc[df["date"] < train_end_ts]
    holdout_df = df.loc[(df["date"] >= train_end_ts) & (df["date"] < holdout_end_ts)]

    drop_cols = ["load_id", "date", "posted_rate"]
    X_train = train_df.drop(columns=drop_cols)
    y_train_log = np.log1p(train_df["posted_rate"])
    X_holdout = holdout_df.drop(columns=drop_cols)
    y_holdout = holdout_df["posted_rate"]

    model = HistGradientBoostingRegressor(random_state=42)
    model.fit(X_train, y_train_log)

    preds = np.expm1(model.predict(X_holdout))

    return {
        "train_range": (train_df["date"].min().date(), train_df["date"].max().date()),
        "holdout_range": (holdout_df["date"].min().date(), holdout_df["date"].max().date()),
        "train_rows": len(train_df),
        "holdout_rows": len(holdout_df),
        "MAE": mean_absolute_error(y_holdout, preds),
        "RMSE": np.sqrt(mean_squared_error(y_holdout, preds)),
        "R2": r2_score(y_holdout, preds),
    }


def main():
    df = load_data()

    results = []
    for window in WINDOWS:
        result = run_window(df, window["train_end"], window["holdout_end"])
        result["name"] = window["name"]
        results.append(result)

        print(f"===== {window['name']} =====")
        print(f"Train range:    {result['train_range'][0]} to {result['train_range'][1]}"
              f"  ({result['train_rows']} rows)")
        print(f"Validate range: {result['holdout_range'][0]} to {result['holdout_range'][1]}"
              f"  ({result['holdout_rows']} rows)")
        print(f"MAE:  {result['MAE']:.2f}")
        print(f"RMSE: {result['RMSE']:.2f}")
        print(f"R2:   {result['R2']:.4f}")
        print()

    print("===== SUMMARY ACROSS WINDOWS =====")
    print(f"{'Window':<10}{'MAE':>10}{'RMSE':>10}{'R2':>10}")
    for r in results:
        print(f"{r['name']:<10}{r['MAE']:>10.2f}{r['RMSE']:>10.2f}{r['R2']:>10.4f}")

    maes = [r["MAE"] for r in results]
    rmses = [r["RMSE"] for r in results]
    r2s = [r["R2"] for r in results]

    print("\n===== STABILITY CHECK =====")
    print(f"MAE  range: {min(maes):.2f} to {max(maes):.2f}  (spread: {max(maes) - min(maes):.2f})")
    print(f"RMSE range: {min(rmses):.2f} to {max(rmses):.2f}  (spread: {max(rmses) - min(rmses):.2f})")
    print(f"R2   range: {min(r2s):.4f} to {max(r2s):.4f}  (spread: {max(r2s) - min(r2s):.4f})")
    print("\nIf metrics vary a lot between windows, the single Sep-Oct result")
    print("from train_hgb_log_target_experiment.py should not be treated as")
    print("a reliable estimate of real-world performance.")


if __name__ == "__main__":
    main()
