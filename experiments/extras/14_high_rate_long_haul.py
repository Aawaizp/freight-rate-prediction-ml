"""Experiment 14: controlled, segment-aware features for the long-haul /
high-rate error segment identified in prior error analysis
(experiments/08_error_analysis.py).

No ensembling - these are single extra FEATURES added to the same tuned
HGB model, evaluated on the same 3 chronological windows.

Candidates:
  A. long_haul_flag = distance >= 2500
  B. high_rate_proxy_flag: a distance-based proxy for "likely high-rate",
     built WITHOUT ever using a holdout row's own target. For each
     window: take that window's TRAINING rows only, find the 99th
     percentile of posted_rate (the training fold's own high-rate
     threshold), find the 10th-percentile distance among those
     high-rate training rows, and use that distance value as a cutoff.
     The resulting flag = distance >= cutoff is then computed for BOTH
     train and holdout rows using only distance (a known predictor for
     every row) - never the holdout target.
  C. Both A and B together.
  D. A plus a log_distance x long_haul_flag interaction term.

LEAKAGE CHECK: the $ threshold and the resulting distance cutoff in B
are computed strictly from that window's training rows. Holdout rows
only ever contribute their own `distance` (already a legitimate model
input) to compute the flag - never their posted_rate.

Segment-level MAE is reported using the ACTUAL posted_rate for holdout
rows (ground truth) purely for evaluation/reporting - this is standard
practice for diagnosing where a model struggles and is not used as a
model input, so it does not leak into training.

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

LONG_HAUL_DISTANCE = 2500
HIGH_RATE_REPORTING_THRESHOLD = 6000  # for reporting only, matches prior error analysis


def load_data() -> pd.DataFrame:
    features = pd.read_csv(PROCESSED_DIR / "train_model_ready.csv")
    dates = pd.read_csv(DATA_DIR / "train-test.csv", usecols=["load_id", "date"])
    dates["date"] = pd.to_datetime(dates["date"])
    return features.merge(dates, on="load_id", how="left")


def compute_high_rate_distance_cutoff(train_df: pd.DataFrame) -> float:
    """Training-fold-only: find the distance level associated with
    high-rate loads, using ONLY that fold's own training targets."""
    rate_threshold = train_df["posted_rate"].quantile(0.99)
    high_rate_rows = train_df[train_df["posted_rate"] >= rate_threshold]
    if len(high_rate_rows) == 0:
        return float(train_df["distance"].max() + 1)  # degenerate fallback: flag nothing
    return float(high_rate_rows["distance"].quantile(0.10))


def build_features(df: pd.DataFrame, distance_cutoff: float) -> pd.DataFrame:
    out = df.copy()
    out["long_haul_flag"] = (out["distance"] >= LONG_HAUL_DISTANCE).astype(int)
    out["high_rate_proxy_flag"] = (out["distance"] >= distance_cutoff).astype(int)
    out["log_distance_x_long_haul"] = out["log_distance"] * out["long_haul_flag"]
    return out


CANDIDATES = {
    "A: long_haul_flag": ["long_haul_flag"],
    "B: high_rate_proxy_flag": ["high_rate_proxy_flag"],
    "C: both flags": ["long_haul_flag", "high_rate_proxy_flag"],
    "D: long_haul_flag + log_distance interaction": ["long_haul_flag", "log_distance_x_long_haul"],
}


def evaluate(df: pd.DataFrame, extra_cols: list[str]) -> list[dict]:
    results = []
    for window in WINDOWS:
        train_end = pd.Timestamp(window["train_end"])
        holdout_end = pd.Timestamp(window["holdout_end"])

        train_df = df.loc[df["date"] < train_end].copy()
        holdout_df = df.loc[(df["date"] >= train_end) & (df["date"] < holdout_end)].copy()

        distance_cutoff = compute_high_rate_distance_cutoff(train_df)
        train_df = build_features(train_df, distance_cutoff)
        holdout_df = build_features(holdout_df, distance_cutoff)

        drop_cols = ["load_id", "date", "posted_rate",
                     "long_haul_flag", "high_rate_proxy_flag", "log_distance_x_long_haul"]
        base_cols = [c for c in train_df.columns if c not in drop_cols]
        use_cols = base_cols + extra_cols

        X_train = train_df[use_cols]
        y_train_log = np.log1p(train_df["posted_rate"])
        X_holdout = holdout_df[use_cols]
        y_holdout = holdout_df["posted_rate"]

        model = HistGradientBoostingRegressor(random_state=42, **TUNED_HGB_PARAMS)
        model.fit(X_train, y_train_log)
        preds = np.expm1(model.predict(X_holdout))

        abs_error = (y_holdout - preds).abs()
        is_long_haul = holdout_df["distance"] >= LONG_HAUL_DISTANCE
        is_high_rate = y_holdout > HIGH_RATE_REPORTING_THRESHOLD

        results.append({
            "window": window["name"],
            "MAE": mean_absolute_error(y_holdout, preds),
            "RMSE": np.sqrt(mean_squared_error(y_holdout, preds)),
            "R2": r2_score(y_holdout, preds),
            "long_haul_MAE": abs_error[is_long_haul].mean() if is_long_haul.any() else float("nan"),
            "non_long_haul_MAE": abs_error[~is_long_haul].mean(),
            "high_rate_MAE": abs_error[is_high_rate].mean() if is_high_rate.any() else float("nan"),
            "distance_cutoff_used": distance_cutoff,
        })
    return results


def summarize(results: list[dict]) -> dict:
    return {
        "avg_MAE": float(np.mean([r["MAE"] for r in results])),
        "avg_RMSE": float(np.mean([r["RMSE"] for r in results])),
        "avg_R2": float(np.mean([r["R2"] for r in results])),
    }


def main():
    df = load_data()

    print("===== BASELINE (current final feature set) =====")
    for w, m in CURRENT_BASELINE.items():
        print(f"  {w:<20} MAE={m['MAE']:.2f}  RMSE={m['RMSE']:.2f}  R2={m['R2']:.4f}")
    print(f"  Average              MAE={CURRENT_BASELINE_AVG_MAE:.2f}\n")

    all_summaries = {}
    all_results = {}
    for name, extra_cols in CANDIDATES.items():
        results = evaluate(df, extra_cols)
        summary = summarize(results)
        all_summaries[name] = summary
        all_results[name] = results

        print(f"===== {name} =====")
        for r in results:
            print(f"  {r['window']:<20} MAE={r['MAE']:.2f}  RMSE={r['RMSE']:.2f}  R2={r['R2']:.4f}  "
                  f"| long-haul MAE={r['long_haul_MAE']:.2f}  non-long-haul MAE={r['non_long_haul_MAE']:.2f}  "
                  f"high-rate MAE={r['high_rate_MAE']:.2f}  (distance cutoff for B/C/D={r['distance_cutoff_used']:.1f})")
        print(f"  Average              MAE={summary['avg_MAE']:.2f}  RMSE={summary['avg_RMSE']:.2f}"
              f"  R2={summary['avg_R2']:.4f}\n")

    print("===== OVERALL MAE COMPARISON =====")
    print(f"{'Model':<45}{'Avg MAE':>10}{'Avg RMSE':>11}{'Avg R2':>9}{'Delta MAE':>12}")
    print(f"{'Baseline (current)':<45}{CURRENT_BASELINE_AVG_MAE:>10.2f}"
          f"{np.mean([v['RMSE'] for v in CURRENT_BASELINE.values()]):>11.2f}"
          f"{np.mean([v['R2'] for v in CURRENT_BASELINE.values()]):>9.4f}{'--':>12}")
    for name, s in all_summaries.items():
        delta = s["avg_MAE"] - CURRENT_BASELINE_AVG_MAE
        print(f"{name:<45}{s['avg_MAE']:>10.2f}{s['avg_RMSE']:>11.2f}{s['avg_R2']:>9.4f}{delta:>+12.2f}")

    print("\n===== PER-WINDOW CONSISTENCY CHECK (overall MAE) =====")
    verdicts = {}
    for name, results in all_results.items():
        print(f"--- {name} ---")
        per_window_deltas = []
        for r in results:
            baseline_mae = CURRENT_BASELINE[r["window"]]["MAE"]
            delta = baseline_mae - r["MAE"]
            per_window_deltas.append(delta)
            print(f"  {r['window']:<20} baseline MAE={baseline_mae:.2f}  new MAE={r['MAE']:.2f}  "
                  f"change={delta:+.2f}")
        consistent = all(d > 0 for d in per_window_deltas)
        avg_delta = np.mean(per_window_deltas)
        material_worse = avg_delta < -1.0
        flag = ""
        if not consistent and avg_delta > 0:
            flag = " -- FLAG: improves average but hurts at least one window"
        if material_worse:
            verdict = "KEEP BASELINE (materially worse overall)"
        elif consistent:
            verdict = "CANDIDATE FOR FURTHER TESTING"
        else:
            verdict = "KEEP BASELINE (not consistent across all windows)"
        print(f"  Consistent improvement across all 3 windows: {consistent}{flag}")
        print(f"  Verdict: {verdict}\n")
        verdicts[name] = verdict

    print("===== FINAL VERDICTS =====")
    for name, verdict in verdicts.items():
        print(f"{name}: {verdict}")


if __name__ == "__main__":
    main()
