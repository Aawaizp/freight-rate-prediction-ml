"""Experiment 12: additional route/geometry features, tested one group
at a time against the current final feature set (train_model_ready.csv),
using the tuned HGB configuration and the same 3 chronological windows
as every other experiment.

Candidate feature groups (built purely from existing coordinate columns
already in train_model_ready.csv - no new data, no leakage):

  Group A - midpoints:
      lat_midpoint = (pickup_lat + delivery_lat) / 2
      lon_midpoint = (pickup_lon + delivery_lon) / 2

  Group B - absolute coordinate differences:
      abs_delta_lat = abs(delivery_lat - pickup_lat)
      abs_delta_lon = abs(delivery_lon - pickup_lon)
      (delta_lat/delta_lon already exist as SIGNED values in the current
      feature set; these are the unsigned magnitude versions, which is a
      different signal - not a duplicate.)

  Group C - coordinate interactions:
      lat_interaction = pickup_lat * delivery_lat
      lon_interaction = pickup_lon * delivery_lon

  Group D - all of the above combined, to see if they help together
      even if individually weak.

distance / haversine_distance is intentionally NOT retested here since
it already exists as route_directness in the current feature set.

Each group is added ON TOP OF the full current feature set and
evaluated independently (baseline + one group at a time) - groups are
not combined pairwise, only individually and then all-together (Group D).

This script does not modify train_final.py, validation_predictions.csv,
preprocessing.py, feature_engineering.py, or the current final model.
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

TUNED_HGB_PARAMS = {
    "learning_rate": 0.05,
    "max_iter": 300,
    "max_leaf_nodes": 15,
    "max_depth": 10,
    "min_samples_leaf": 50,
    "l2_regularization": 1.0,
}

CURRENT_BASELINE = {
    "Window 1 (Jul-Aug)": {"MAE": 115.32, "RMSE": 624.19, "R2": 0.8247},
    "Window 2 (Aug-Sep)": {"MAE": 114.90, "RMSE": 619.87, "R2": 0.8291},
    "Window 3 (Sep-Oct)": {"MAE": 127.98, "RMSE": 639.83, "R2": 0.8242},
}
CURRENT_BASELINE_AVG_MAE = 119.40


def add_midpoints(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["lat_midpoint"] = (df["pickup_lat"] + df["delivery_lat"]) / 2
    df["lon_midpoint"] = (df["pickup_lon"] + df["delivery_lon"]) / 2
    return df


def add_abs_differences(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["abs_delta_lat"] = (df["delivery_lat"] - df["pickup_lat"]).abs()
    df["abs_delta_lon"] = (df["delivery_lon"] - df["pickup_lon"]).abs()
    return df


def add_coordinate_interactions(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["lat_interaction"] = df["pickup_lat"] * df["delivery_lat"]
    df["lon_interaction"] = df["pickup_lon"] * df["delivery_lon"]
    return df


def add_all_groups(df: pd.DataFrame) -> pd.DataFrame:
    return add_coordinate_interactions(add_abs_differences(add_midpoints(df)))


CANDIDATES = [
    {"name": "Group A: lat/lon midpoints", "builder": add_midpoints},
    {"name": "Group B: absolute lat/lon differences", "builder": add_abs_differences},
    {"name": "Group C: coordinate interactions", "builder": add_coordinate_interactions},
    {"name": "Group D: all groups combined", "builder": add_all_groups},
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


def main():
    base_df = load_data()

    print("===== BASELINE (current final feature set) =====")
    for w, m in CURRENT_BASELINE.items():
        print(f"  {w:<20} MAE={m['MAE']:.2f}  RMSE={m['RMSE']:.2f}  R2={m['R2']:.4f}")
    print(f"  Average              MAE={CURRENT_BASELINE_AVG_MAE:.2f}\n")

    all_summaries = {}
    all_results = {}
    for candidate in CANDIDATES:
        candidate_df = candidate["builder"](base_df)
        results = evaluate(candidate_df)
        summary = summarize(results)
        all_summaries[candidate["name"]] = summary
        all_results[candidate["name"]] = results

        print(f"===== {candidate['name']} =====")
        for r in results:
            print(f"  {r['window']:<20} MAE={r['MAE']:.2f}  RMSE={r['RMSE']:.2f}  R2={r['R2']:.4f}")
        print(f"  Average              MAE={summary['avg_MAE']:.2f}  RMSE={summary['avg_RMSE']:.2f}"
              f"  R2={summary['avg_R2']:.4f}\n")

    print("===== SUMMARY: Candidate | Avg MAE | Avg RMSE | Avg R2 | MAE vs baseline =====")
    print(f"{'Candidate':<38}{'Avg MAE':>10}{'Avg RMSE':>12}{'Avg R2':>10}{'Delta MAE':>12}")
    print(f"{'Baseline (current)':<38}{CURRENT_BASELINE_AVG_MAE:>10.2f}"
          f"{np.mean([v['RMSE'] for v in CURRENT_BASELINE.values()]):>12.2f}"
          f"{np.mean([v['R2'] for v in CURRENT_BASELINE.values()]):>10.4f}{'--':>12}")
    for name, summary in all_summaries.items():
        delta = summary["avg_MAE"] - CURRENT_BASELINE_AVG_MAE
        print(f"{name:<38}{summary['avg_MAE']:>10.2f}{summary['avg_RMSE']:>12.2f}"
              f"{summary['avg_R2']:>10.4f}{delta:>+12.2f}")

    print("\n===== PER-WINDOW CONSISTENCY CHECK =====")
    verdicts = {}
    for name, results in all_results.items():
        print(f"--- {name} ---")
        per_window_deltas = []
        for r in results:
            baseline_mae = CURRENT_BASELINE[r["window"]]["MAE"]
            delta = baseline_mae - r["MAE"]  # positive = improvement
            per_window_deltas.append(delta)
            print(f"  {r['window']:<20} baseline MAE={baseline_mae:.2f}  new MAE={r['MAE']:.2f}  "
                  f"change={delta:+.2f}")
        consistent = all(d > 0 for d in per_window_deltas)
        worst_drop = min(per_window_deltas)
        avg_delta = np.mean(per_window_deltas)
        flag = ""
        if not consistent and avg_delta > 0:
            flag = " -- FLAG: improves average but hurts at least one window"
        verdict = "CANDIDATE FOR FURTHER TESTING" if consistent else "KEEP BASELINE"
        print(f"  Consistent improvement across all 3 windows: {consistent}{flag}")
        print(f"  Verdict: {verdict}\n")
        verdicts[name] = verdict

    print("===== FINAL VERDICTS =====")
    for name, verdict in verdicts.items():
        print(f"{name}: {verdict}")


if __name__ == "__main__":
    main()
