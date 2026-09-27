"""Experiment 4: small local refinement around the current best HGB
config from tune_hgb.py. That search was a random search over a wide
grid, not a local search - this checks whether nudging each parameter
slightly in either direction finds anything better nearby, using the
same 3 chronological windows.

Kept deliberately small (9 candidates including the current best) per
the "no huge search" instruction.

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

CURRENT_BEST = {
    "learning_rate": 0.05,
    "max_iter": 300,
    "max_leaf_nodes": 15,
    "max_depth": 10,
    "min_samples_leaf": 50,
    "l2_regularization": 1.0,
}

# One parameter nudged at a time, small steps in each direction.
CANDIDATES = [
    {"label": "Current best", **CURRENT_BEST},
    {"label": "learning_rate=0.04", **{**CURRENT_BEST, "learning_rate": 0.04}},
    {"label": "learning_rate=0.07", **{**CURRENT_BEST, "learning_rate": 0.07}},
    {"label": "max_iter=250", **{**CURRENT_BEST, "max_iter": 250}},
    {"label": "max_iter=400", **{**CURRENT_BEST, "max_iter": 400}},
    {"label": "max_leaf_nodes=20", **{**CURRENT_BEST, "max_leaf_nodes": 20}},
    {"label": "min_samples_leaf=35", **{**CURRENT_BEST, "min_samples_leaf": 35}},
    {"label": "l2_regularization=0.5", **{**CURRENT_BEST, "l2_regularization": 0.5}},
    {"label": "l2_regularization=2.0", **{**CURRENT_BEST, "l2_regularization": 2.0}},
]


def load_data() -> pd.DataFrame:
    features = pd.read_csv(PROCESSED_DIR / "train_model_ready.csv")
    dates = pd.read_csv(DATA_DIR / "train-test.csv", usecols=["load_id", "date"])
    dates["date"] = pd.to_datetime(dates["date"])
    return features.merge(dates, on="load_id", how="left")


def evaluate(df: pd.DataFrame, params: dict) -> list[dict]:
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

        model = HistGradientBoostingRegressor(random_state=42, **params)
        model.fit(X_train, y_train_log)
        preds = np.expm1(model.predict(X_holdout))

        results.append({
            "window": window["name"],
            "MAE": mean_absolute_error(y_holdout, preds),
            "RMSE": np.sqrt(mean_squared_error(y_holdout, preds)),
            "R2": r2_score(y_holdout, preds),
        })
    return results


def main():
    df = load_data()

    all_results = []
    for candidate in CANDIDATES:
        label = candidate["label"]
        params = {k: v for k, v in candidate.items() if k != "label"}
        results = evaluate(df, params)
        avg_mae = float(np.mean([r["MAE"] for r in results]))
        avg_rmse = float(np.mean([r["RMSE"] for r in results]))
        avg_r2 = float(np.mean([r["R2"] for r in results]))

        print(f"===== {label} =====")
        for r in results:
            print(f"  {r['window']:<20} MAE={r['MAE']:.2f}  RMSE={r['RMSE']:.2f}  R2={r['R2']:.4f}")
        print(f"  Average              MAE={avg_mae:.2f}  RMSE={avg_rmse:.2f}  R2={avg_r2:.4f}\n")

        all_results.append({"label": label, "results": results,
                             "avg_MAE": avg_mae, "avg_RMSE": avg_rmse, "avg_R2": avg_r2})

    baseline = all_results[0]
    print("===== RANKING (by average MAE) =====")
    ranked = sorted(all_results, key=lambda r: r["avg_MAE"])
    print(f"{'Label':<24}{'Avg MAE':>10}{'Avg RMSE':>12}{'Avg R2':>10}")
    for r in ranked:
        print(f"{r['label']:<24}{r['avg_MAE']:>10.2f}{r['avg_RMSE']:>12.2f}{r['avg_R2']:>10.4f}")

    print("\n===== VERDICTS VS CURRENT BEST =====")
    for r in all_results[1:]:
        per_window_deltas = [a["MAE"] - b["MAE"] for a, b in zip(r["results"], baseline["results"])]
        consistent_improvement = all(d < 0 for d in per_window_deltas)
        verdict = "KEEP (replace current best)" if consistent_improvement else "DROP (keep current best)"
        print(f"{r['label']:<24} per-window MAE delta = {[f'{d:+.2f}' for d in per_window_deltas]} -> {verdict}")


if __name__ == "__main__":
    main()
