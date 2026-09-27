"""Experiment 11: rate-per-mile target engineering.

Tests whether predicting rate_per_mile = posted_rate / distance, then
converting back to predicted_rate = predicted_rate_per_mile * distance,
beats directly predicting posted_rate (our current approach).

Uses the SAME model-ready features (train_model_ready.csv) and the SAME
3 chronological validation windows as every other experiment in this
project. Two models are tested with this target: tuned HGB (same
hyperparameters as the current final model) and a baseline CatBoost.

Leakage/safety notes:
- rate_per_mile is computed only from each row's own posted_rate and
  distance - no other row's data, and no validation-only information,
  is used to build it.
- distance is checked for zero/negative values before dividing (none
  exist in this dataset, but the check is explicit rather than assumed).
- The feature set is unchanged - distance itself is still a legitimate
  input feature at prediction time (it exists in validation.csv too),
  so using it both as a feature and as the target's denominator is not
  leakage.

This script does not modify train_final.py, validation_predictions.csv,
the current final model, or any other experiment file. It only reads
data/processed/train_model_ready.csv and data/train-test.csv.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DATA_DIR = Path(__file__).parent.parent / "data"
PROCESSED_DIR = DATA_DIR / "processed"

WINDOWS = [
    {"name": "Window 1 (Jul-Aug)", "train_end": "2025-07-01", "holdout_end": "2025-09-01"},
    {"name": "Window 2 (Aug-Sep)", "train_end": "2025-08-01", "holdout_end": "2025-10-01"},
    {"name": "Window 3 (Sep-Oct)", "train_end": "2025-09-01", "holdout_end": "2025-11-01"},
]

TUNED_HGB_PARAMS = {
    "learning_rate": 0.05,
    "max_iter": 300,
    "max_leaf_nodes": 15,
    "max_depth": 10,
    "min_samples_leaf": 50,
    "l2_regularization": 1.0,
}

CATBOOST_PARAMS = {
    "iterations": 500,
    "learning_rate": 0.05,
    "depth": 6,
    "loss_function": "RMSE",
    "random_seed": 42,
    "verbose": False,
}

CURRENT_HGB_BASELINE = {
    "Window 1 (Jul-Aug)": {"MAE": 115.32, "RMSE": 624.19, "R2": 0.8247},
    "Window 2 (Aug-Sep)": {"MAE": 114.90, "RMSE": 619.87, "R2": 0.8291},
    "Window 3 (Sep-Oct)": {"MAE": 127.98, "RMSE": 639.83, "R2": 0.8242},
}
CURRENT_HGB_AVG_MAE = 119.40


def load_data() -> pd.DataFrame:
    features = pd.read_csv(PROCESSED_DIR / "train_model_ready.csv")
    dates = pd.read_csv(DATA_DIR / "train-test.csv", usecols=["load_id", "date"])
    dates["date"] = pd.to_datetime(dates["date"])
    df = features.merge(dates, on="load_id", how="left")

    if (df["distance"] <= 0).any():
        raise ValueError("Found non-positive distance values - cannot safely build rate_per_mile.")

    df["rate_per_mile"] = df["posted_rate"] / df["distance"]
    return df


def report_rate_per_mile_outliers(df: pd.DataFrame):
    print("===== RATE-PER-MILE OUTLIER CHECK (full training data) =====")
    quantiles = df["rate_per_mile"].quantile([0, 0.01, 0.25, 0.5, 0.75, 0.99, 1.0])
    print(quantiles.to_string(float_format=lambda x: f"{x:.3f}"))
    extreme = df[df["rate_per_mile"] > df["rate_per_mile"].quantile(0.99)]
    print(f"\nRows above 99th percentile rate_per_mile: {len(extreme)}")
    if len(extreme):
        print("Distance stats for those rows:")
        print(extreme["distance"].describe().to_string(float_format=lambda x: f"{x:.1f}"))
    print()


def evaluate(df: pd.DataFrame, model_name: str, fit_fn) -> list[dict]:
    results = []
    for window in WINDOWS:
        train_end = pd.Timestamp(window["train_end"])
        holdout_end = pd.Timestamp(window["holdout_end"])

        train_df = df.loc[df["date"] < train_end]
        holdout_df = df.loc[(df["date"] >= train_end) & (df["date"] < holdout_end)]

        drop_cols = ["load_id", "date", "posted_rate", "rate_per_mile"]
        X_train = train_df.drop(columns=drop_cols)
        y_train_rpm = train_df["rate_per_mile"]
        X_holdout = holdout_df.drop(columns=drop_cols)
        y_holdout_rate = holdout_df["posted_rate"]
        holdout_distance = holdout_df["distance"]

        model = fit_fn(X_train, y_train_rpm)
        preds_rpm = model.predict(X_holdout)
        preds_rate = preds_rpm * holdout_distance  # convert back to dollars

        results.append({
            "model": model_name,
            "window": window["name"],
            "MAE": mean_absolute_error(y_holdout_rate, preds_rate),
            "RMSE": np.sqrt(mean_squared_error(y_holdout_rate, preds_rate)),
            "R2": r2_score(y_holdout_rate, preds_rate),
        })
    return results


def fit_hgb(X_train, y_train):
    model = HistGradientBoostingRegressor(random_state=42, **TUNED_HGB_PARAMS)
    model.fit(X_train, y_train)
    return model


def fit_catboost(X_train, y_train):
    model = CatBoostRegressor(**CATBOOST_PARAMS)
    model.fit(X_train, y_train)
    return model


def summarize(results: list[dict]) -> dict:
    return {
        "avg_MAE": float(np.mean([r["MAE"] for r in results])),
        "avg_RMSE": float(np.mean([r["RMSE"] for r in results])),
        "avg_R2": float(np.mean([r["R2"] for r in results])),
    }


def print_model_table(model_name: str, results: list[dict]):
    print(f"----- {model_name}: Model | Window | MAE | RMSE | R2 -----")
    for r in results:
        print(f"{r['model']:<22} {r['window']:<20} MAE={r['MAE']:.2f}  RMSE={r['RMSE']:.2f}  R2={r['R2']:.4f}")
    print()


def main():
    df = load_data()
    report_rate_per_mile_outliers(df)

    hgb_results = evaluate(df, "HGB (rate-per-mile)", fit_hgb)
    print_model_table("Experiment A: Rate-per-mile + HGB", hgb_results)

    cat_results = evaluate(df, "CatBoost (rate-per-mile)", fit_catboost)
    print_model_table("Experiment B: Rate-per-mile + CatBoost", cat_results)

    hgb_summary = summarize(hgb_results)
    cat_summary = summarize(cat_results)

    print("===== SUMMARY: Model | Avg MAE | Avg RMSE | Avg R2 =====")
    print(f"{'Model':<25}{'Avg MAE':>10}{'Avg RMSE':>12}{'Avg R2':>10}")
    print(f"{'Current HGB (baseline)':<25}{CURRENT_HGB_AVG_MAE:>10.2f}"
          f"{np.mean([v['RMSE'] for v in CURRENT_HGB_BASELINE.values()]):>12.2f}"
          f"{np.mean([v['R2'] for v in CURRENT_HGB_BASELINE.values()]):>10.4f}")
    print(f"{'HGB (rate-per-mile)':<25}{hgb_summary['avg_MAE']:>10.2f}{hgb_summary['avg_RMSE']:>12.2f}{hgb_summary['avg_R2']:>10.4f}")
    print(f"{'CatBoost (rate-per-mile)':<25}{cat_summary['avg_MAE']:>10.2f}{cat_summary['avg_RMSE']:>12.2f}{cat_summary['avg_R2']:>10.4f}")

    print("\n===== MAE IMPROVEMENT VS CURRENT HGB BASELINE (119.40) =====")
    for name, summary, results in [("HGB (rate-per-mile)", hgb_summary, hgb_results),
                                    ("CatBoost (rate-per-mile)", cat_summary, cat_results)]:
        improvement = CURRENT_HGB_AVG_MAE - summary["avg_MAE"]
        print(f"{name}: MAE improvement = {improvement:+.2f} "
              f"({'improvement' if improvement > 0 else 'worse'})")

        per_window_deltas = []
        for r in results:
            baseline_mae = CURRENT_HGB_BASELINE[r["window"]]["MAE"]
            delta = baseline_mae - r["MAE"]
            per_window_deltas.append(delta)
            print(f"    {r['window']:<20} baseline MAE={baseline_mae:.2f}  "
                  f"new MAE={r['MAE']:.2f}  improvement={delta:+.2f}")
        consistent = all(d > 0 for d in per_window_deltas)
        print(f"    Consistent improvement across all 3 windows: {consistent}")
        print(f"    RECOMMENDATION: {'RATE-PER-MILE IS A CANDIDATE FOR FURTHER TESTING' if consistent else 'KEEP CURRENT MODEL'}\n")


if __name__ == "__main__":
    main()
