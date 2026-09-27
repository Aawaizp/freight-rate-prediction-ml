"""Controlled feature-engineering experiment: test 3 candidate features,
ONE AT A TIME, added to the tuned HGB configuration from tune_hgb.py.

Each candidate is compared against a baseline run (tuned HGB, no extra
feature) computed in this same script so all numbers are exactly
apples-to-apples, using the same 3 chronological windows, log1p target,
and expm1() predictions used throughout this project.

A feature is only kept if it improves MAE consistently across ALL 3
windows, not just on average - a feature that helps one window and
hurts another is not considered a reliable improvement.

data/validation.csv is never touched.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DATA_DIR = Path(__file__).parent.parent / "data"
PROCESSED_DIR = DATA_DIR / "processed"

WINDOWS = [
    {"name": "Window 1 (Jul-Aug)", "train_end": "2025-07-01", "holdout_end": "2025-09-01"},
    {"name": "Window 2 (Aug-Sep)", "train_end": "2025-08-01", "holdout_end": "2025-10-01"},
    {"name": "Window 3 (Sep-Oct)", "train_end": "2025-09-01", "holdout_end": "2025-11-01"},
]

# Best config found in tune_hgb.py.
TUNED_HGB_PARAMS = {
    "learning_rate": 0.05,
    "max_iter": 300,
    "max_leaf_nodes": 15,
    "max_depth": 10,
    "min_samples_leaf": 50,
    "l2_regularization": 1.0,
}


def add_market_index_x_month(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["market_index_x_month"] = df["market_index"] * df["month"]
    return df


def add_quote_signal_x_distance(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["quote_signal_x_distance"] = df["quote_signal"] * df["distance"]
    return df


def add_weight_over_distance(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["weight_over_distance"] = df["weight"] / df["distance"].replace(0, np.nan)
    df["weight_over_distance"] = df["weight_over_distance"].fillna(0.0)
    return df


CANDIDATES = [
    {"name": "market_index_x_month", "builder": add_market_index_x_month},
    {"name": "quote_signal_x_distance", "builder": add_quote_signal_x_distance},
    {"name": "weight_over_distance", "builder": add_weight_over_distance},
]


def load_data() -> pd.DataFrame:
    features = pd.read_csv(PROCESSED_DIR / "train_model_ready.csv")
    dates = pd.read_csv(DATA_DIR / "train-test.csv", usecols=["load_id", "date"])
    dates["date"] = pd.to_datetime(dates["date"])
    return features.merge(dates, on="load_id", how="left")


def evaluate(df: pd.DataFrame) -> list[dict]:
    results = []
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

        model = HistGradientBoostingRegressor(random_state=42, **TUNED_HGB_PARAMS)
        model.fit(X_train, y_train_log)
        preds = np.expm1(model.predict(X_holdout))

        results.append({
            "window": window["name"],
            "MAE": mean_absolute_error(y_holdout, preds),
            "RMSE": np.sqrt(mean_squared_error(y_holdout, preds)),
            "R2": r2_score(y_holdout, preds),
        })
    return results


def summarize(results: list[dict]) -> dict:
    return {
        "avg_MAE": float(np.mean([r["MAE"] for r in results])),
        "avg_RMSE": float(np.mean([r["RMSE"] for r in results])),
        "avg_R2": float(np.mean([r["R2"] for r in results])),
    }


def print_results(label: str, results: list[dict], summary: dict):
    print(f"===== {label} =====")
    for r in results:
        print(f"  {r['window']:<20} MAE={r['MAE']:.2f}  RMSE={r['RMSE']:.2f}  R2={r['R2']:.4f}")
    print(f"  Average              MAE={summary['avg_MAE']:.2f}  RMSE={summary['avg_RMSE']:.2f}"
          f"  R2={summary['avg_R2']:.4f}")
    print()


def main():
    base_df = load_data()

    baseline_results = evaluate(base_df)
    baseline_summary = summarize(baseline_results)
    print_results("BASELINE (tuned HGB, no new feature)", baseline_results, baseline_summary)

    verdicts = []
    for candidate in CANDIDATES:
        candidate_df = candidate["builder"](base_df)
        results = evaluate(candidate_df)
        summary = summarize(results)
        print_results(f"CANDIDATE: {candidate['name']}", results, summary)

        print(f"  ----- {candidate['name']} vs baseline -----")
        per_window_mae_deltas = []
        for base_r, cand_r in zip(baseline_results, results):
            delta = cand_r["MAE"] - base_r["MAE"]
            per_window_mae_deltas.append(delta)
            print(f"  {cand_r['window']:<20} MAE delta: {delta:+.2f}")

        avg_mae_delta = summary["avg_MAE"] - baseline_summary["avg_MAE"]
        avg_rmse_delta = summary["avg_RMSE"] - baseline_summary["avg_RMSE"]
        avg_r2_delta = summary["avg_R2"] - baseline_summary["avg_R2"]
        print(f"  Average MAE delta:  {avg_mae_delta:+.2f}")
        print(f"  Average RMSE delta: {avg_rmse_delta:+.2f}")
        print(f"  Average R2 delta:   {avg_r2_delta:+.4f}")

        consistent_improvement = all(d < 0 for d in per_window_mae_deltas)
        verdict = "KEEP" if consistent_improvement else "DROP"
        if consistent_improvement:
            reason = "MAE improved in all 3 windows - consistent win."
        else:
            worse_windows = sum(1 for d in per_window_mae_deltas if d >= 0)
            reason = f"MAE did not improve in {worse_windows}/3 window(s) - not consistent."
        print(f"  VERDICT: {verdict} - {reason}\n")

        verdicts.append({"name": candidate["name"], "verdict": verdict, "reason": reason,
                          "avg_mae_delta": avg_mae_delta})

    print("===== FINAL SUMMARY =====")
    print(f"Baseline (tuned HGB): MAE={baseline_summary['avg_MAE']:.2f}  "
          f"RMSE={baseline_summary['avg_RMSE']:.2f}  R2={baseline_summary['avg_R2']:.4f}")
    for v in verdicts:
        print(f"  {v['name']:<26} avg MAE delta {v['avg_mae_delta']:+.2f}  -> {v['verdict']} "
              f"({v['reason']})")

    kept = [v["name"] for v in verdicts if v["verdict"] == "KEEP"]
    if kept:
        print(f"\nFeatures to keep going forward: {kept}")
    else:
        print("\nNone of the 3 tested features showed a consistent improvement - "
              "none should be added to the feature-engineering pipeline yet.")


if __name__ == "__main__":
    main()
