"""Experiment 13: route-level historical pricing features.

Tests whether adding historical route statistics (route = pickup +
delivery) improves prediction, using the tuned HGB configuration and
the same 3 chronological windows as every other experiment.

CRITICAL LEAKAGE RULE (enforced per window):
    Window 1: route statistics built ONLY from Jan-Jun training rows,
              applied to Jul-Aug holdout rows.
    Window 2: route statistics built ONLY from Jan-Jul training rows,
              applied to Aug-Sep holdout rows.
    Window 3: route statistics built ONLY from Jan-Aug training rows,
              applied to Sep-Oct holdout rows.
A route's statistics are recomputed independently for each window, using
only that window's own training period - never the holdout period, and
never a later window's data.

Fallback for routes in the holdout period that were never seen in that
window's training data: the training-period GLOBAL mean/median (across
all training routes), not a per-route value. A route_seen flag marks
which rows got a real per-route statistic vs. this fallback.

Known limitation (disclosed, not hidden): for TRAINING rows themselves,
each row's own route statistic is computed from a group average that
includes that row's own historical value (not leave-one-out). This is a
mild in-sample averaging effect, not future/validation leakage - see the
final report section for why this doesn't violate the leakage rule above
but is still worth flagging.

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

CANDIDATES = [
    "A: route_mean_rate",
    "B: route_median_rate",
    "C: route_mean_rate_per_mile",
    "D: route_load_count",
    "E: route_mean_rate + route_load_count",
]


def load_data() -> pd.DataFrame:
    features = pd.read_csv(PROCESSED_DIR / "train_model_ready.csv")
    context = pd.read_csv(DATA_DIR / "train-test.csv",
                           usecols=["load_id", "date", "pickup", "delivery", "distance"])
    context = context.rename(columns={"distance": "raw_distance"})
    context["date"] = pd.to_datetime(context["date"])
    df = features.merge(context, on="load_id", how="left")
    df["route"] = df["pickup"] + "|" + df["delivery"]
    return df


def build_route_stats(train_df: pd.DataFrame) -> dict:
    """Computes route-level stats from a window's training rows only."""
    grouped = train_df.groupby("route")["posted_rate"]
    route_mean_rate = grouped.mean()
    route_median_rate = grouped.median()
    route_load_count = grouped.size()

    rate_per_mile = train_df["posted_rate"] / train_df["raw_distance"]
    route_mean_rpm = rate_per_mile.groupby(train_df["route"]).mean()

    return {
        "route_mean_rate": route_mean_rate,
        "route_median_rate": route_median_rate,
        "route_mean_rate_per_mile": route_mean_rpm,
        "route_load_count": route_load_count,
        "global_mean_rate": train_df["posted_rate"].mean(),
        "global_median_rate": train_df["posted_rate"].median(),
        "global_mean_rpm": rate_per_mile.mean(),
    }


def apply_route_stats(df: pd.DataFrame, stats: dict) -> pd.DataFrame:
    """Applies pre-computed (training-only) route stats to any rows -
    training or holdout - via a simple route-name lookup. Rows whose
    route never appeared in that window's training data fall back to
    the training-period global mean/median."""
    out = df.copy()
    out["route_seen"] = out["route"].isin(stats["route_mean_rate"].index).astype(int)

    out["route_mean_rate"] = out["route"].map(stats["route_mean_rate"]).fillna(stats["global_mean_rate"])
    out["route_median_rate"] = out["route"].map(stats["route_median_rate"]).fillna(stats["global_median_rate"])
    out["route_mean_rate_per_mile"] = out["route"].map(stats["route_mean_rate_per_mile"]).fillna(stats["global_mean_rpm"])
    out["route_load_count"] = out["route"].map(stats["route_load_count"]).fillna(0).astype(int)

    return out


def evaluate(df: pd.DataFrame, feature_cols: list[str]) -> tuple[list[dict], dict]:
    results = []
    unseen_counts = {}
    for window in WINDOWS:
        train_end = pd.Timestamp(window["train_end"])
        holdout_end = pd.Timestamp(window["holdout_end"])

        train_df = df.loc[df["date"] < train_end].copy()
        holdout_df = df.loc[(df["date"] >= train_end) & (df["date"] < holdout_end)].copy()

        # Route stats built ONLY from this window's training rows.
        stats = build_route_stats(train_df)
        train_df = apply_route_stats(train_df, stats)
        holdout_df = apply_route_stats(holdout_df, stats)

        unseen_counts[window["name"]] = int((holdout_df["route_seen"] == 0).sum())

        drop_cols = ["load_id", "date", "posted_rate", "pickup", "delivery", "route", "raw_distance"]
        keep_extra = [c for c in feature_cols if c not in drop_cols]
        base_cols = [c for c in train_df.columns if c not in drop_cols and c not in
                     ["route_seen", "route_mean_rate", "route_median_rate",
                      "route_mean_rate_per_mile", "route_load_count"]]
        use_cols = base_cols + keep_extra

        X_train = train_df[use_cols]
        y_train_log = np.log1p(train_df["posted_rate"])
        X_holdout = holdout_df[use_cols]
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
    return results, unseen_counts


def summarize(results: list[dict]) -> dict:
    return {
        "avg_MAE": float(np.mean([r["MAE"] for r in results])),
        "avg_RMSE": float(np.mean([r["RMSE"] for r in results])),
        "avg_R2": float(np.mean([r["R2"] for r in results])),
    }


def main():
    df = load_data()

    candidate_feature_sets = {
        CANDIDATES[0]: ["route_mean_rate", "route_seen"],
        CANDIDATES[1]: ["route_median_rate", "route_seen"],
        CANDIDATES[2]: ["route_mean_rate_per_mile", "route_seen"],
        CANDIDATES[3]: ["route_load_count", "route_seen"],
        CANDIDATES[4]: ["route_mean_rate", "route_load_count", "route_seen"],
    }

    print("===== BASELINE (current final feature set) =====")
    for w, m in CURRENT_BASELINE.items():
        print(f"  {w:<20} MAE={m['MAE']:.2f}  RMSE={m['RMSE']:.2f}  R2={m['R2']:.4f}")
    print(f"  Average              MAE={CURRENT_BASELINE_AVG_MAE:.2f}\n")

    all_summaries = {}
    all_results = {}
    all_unseen = {}
    for name, feature_cols in candidate_feature_sets.items():
        results, unseen_counts = evaluate(df, feature_cols)
        summary = summarize(results)
        all_summaries[name] = summary
        all_results[name] = results
        all_unseen[name] = unseen_counts

        print(f"===== {name} =====")
        for r in results:
            print(f"  {r['window']:<20} MAE={r['MAE']:.2f}  RMSE={r['RMSE']:.2f}  R2={r['R2']:.4f}")
        print(f"  Average              MAE={summary['avg_MAE']:.2f}  RMSE={summary['avg_RMSE']:.2f}"
              f"  R2={summary['avg_R2']:.4f}")
        print(f"  Unseen routes in holdout: {unseen_counts}\n")

    print("===== SUMMARY: Model | Window 1 | Window 2 | Window 3 | Avg MAE | Avg RMSE | Avg R2 =====")
    print(f"{'Model':<45}{'W1 MAE':>9}{'W2 MAE':>9}{'W3 MAE':>9}{'Avg MAE':>10}{'Avg RMSE':>11}{'Avg R2':>9}")
    base_vals = list(CURRENT_BASELINE.values())
    print(f"{'Baseline (current)':<45}{base_vals[0]['MAE']:>9.2f}{base_vals[1]['MAE']:>9.2f}"
          f"{base_vals[2]['MAE']:>9.2f}{CURRENT_BASELINE_AVG_MAE:>10.2f}"
          f"{np.mean([v['RMSE'] for v in base_vals]):>11.2f}{np.mean([v['R2'] for v in base_vals]):>9.4f}")
    for name, results in all_results.items():
        s = all_summaries[name]
        print(f"{name:<45}{results[0]['MAE']:>9.2f}{results[1]['MAE']:>9.2f}{results[2]['MAE']:>9.2f}"
              f"{s['avg_MAE']:>10.2f}{s['avg_RMSE']:>11.2f}{s['avg_R2']:>9.4f}")

    print("\n===== CHANGE FROM CURRENT $119.40 MAE =====")
    for name, s in all_summaries.items():
        print(f"{name}: {s['avg_MAE'] - CURRENT_BASELINE_AVG_MAE:+.2f}")

    print("\n===== PER-WINDOW CONSISTENCY CHECK =====")
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
        flag = " -- FLAG: improves average but hurts at least one window" if (not consistent and avg_delta > 0) else ""
        verdict = "CANDIDATE FOR FURTHER TESTING" if consistent else "KEEP BASELINE"
        print(f"  Consistent improvement across all 3 windows: {consistent}{flag}")
        print(f"  Verdict: {verdict}\n")
        verdicts[name] = verdict

    print("===== UNSEEN ROUTES PER WINDOW (fallback used) =====")
    for name, counts in all_unseen.items():
        print(f"{name}: {counts}")

    print("\n===== FINAL VERDICTS =====")
    for name, verdict in verdicts.items():
        print(f"{name}: {verdict}")


if __name__ == "__main__":
    main()
