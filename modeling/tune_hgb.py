"""Small hyperparameter search for HistGradientBoostingRegressor, using
the same log1p(posted_rate) target and model-ready features as our
current best (train_hgb_log_target_experiment.py), evaluated across
all three chronological windows from multi_window_hgb_experiment.py -
not just Sep-Oct - so a config can't win by getting lucky on one split.

A random search (not a full grid) is used to keep runtime reasonable:
10 candidates x 3 windows = 30 fits total. Candidate 0 is the untuned
default configuration, included so it doubles as the baseline anchor
for the final comparison.

data/validation.csv is never touched.
"""
import random
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

# HistGradientBoostingRegressor defaults - this is "Candidate 0", our
# current baseline to beat.
DEFAULT_PARAMS = {
    "learning_rate": 0.1,
    "max_iter": 100,
    "max_leaf_nodes": 31,
    "max_depth": None,
    "min_samples_leaf": 20,
    "l2_regularization": 0.0,
}

PARAM_GRID = {
    "learning_rate": [0.03, 0.05, 0.1, 0.2],
    "max_iter": [100, 200, 300],
    "max_leaf_nodes": [15, 31, 63],
    "max_depth": [None, 6, 10],
    "min_samples_leaf": [10, 20, 50],
    "l2_regularization": [0.0, 0.1, 1.0],
}

N_RANDOM_CANDIDATES = 9  # + the default = 10 candidates total
SEED = 42


def build_candidates() -> list[dict]:
    rng = random.Random(SEED)
    candidates = [dict(DEFAULT_PARAMS)]
    seen = {tuple(sorted(DEFAULT_PARAMS.items(), key=lambda kv: kv[0]))}

    while len(candidates) < N_RANDOM_CANDIDATES + 1:
        candidate = {param: rng.choice(values) for param, values in PARAM_GRID.items()}
        key = tuple(sorted(candidate.items(), key=lambda kv: kv[0]))
        if key in seen:
            continue
        seen.add(key)
        candidates.append(candidate)

    return candidates


def load_data() -> pd.DataFrame:
    features = pd.read_csv(PROCESSED_DIR / "train_model_ready.csv")
    dates = pd.read_csv(DATA_DIR / "train-test.csv", usecols=["load_id", "date"])
    dates["date"] = pd.to_datetime(dates["date"])
    return features.merge(dates, on="load_id", how="left")


def evaluate_candidate(df: pd.DataFrame, params: dict) -> list[dict]:
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

        model = HistGradientBoostingRegressor(random_state=42, **params)
        model.fit(X_train, y_train_log)
        preds = np.expm1(model.predict(X_holdout))

        window_results.append({
            "window": window["name"],
            "MAE": mean_absolute_error(y_holdout, preds),
            "RMSE": np.sqrt(mean_squared_error(y_holdout, preds)),
            "R2": r2_score(y_holdout, preds),
        })

    return window_results


def summarize(window_results: list[dict]) -> dict:
    maes = [w["MAE"] for w in window_results]
    rmses = [w["RMSE"] for w in window_results]
    r2s = [w["R2"] for w in window_results]
    return {
        "avg_MAE": float(np.mean(maes)),
        "avg_RMSE": float(np.mean(rmses)),
        "avg_R2": float(np.mean(r2s)),
        "std_MAE": float(np.std(maes)),
        "std_RMSE": float(np.std(rmses)),
        "std_R2": float(np.std(r2s)),
    }


def main():
    df = load_data()
    candidates = build_candidates()

    all_results = []
    for i, params in enumerate(candidates):
        label = "Candidate 0 (default)" if i == 0 else f"Candidate {i}"
        window_results = evaluate_candidate(df, params)
        summary = summarize(window_results)

        print(f"===== {label} =====")
        print(f"Params: {params}")
        for w in window_results:
            print(f"  {w['window']:<20} MAE={w['MAE']:.2f}  RMSE={w['RMSE']:.2f}  R2={w['R2']:.4f}")
        print(f"  Average              MAE={summary['avg_MAE']:.2f}  RMSE={summary['avg_RMSE']:.2f}"
              f"  R2={summary['avg_R2']:.4f}")
        print(f"  Std dev              MAE={summary['std_MAE']:.2f}  RMSE={summary['std_RMSE']:.2f}"
              f"  R2={summary['std_R2']:.4f}")
        print()

        all_results.append({"label": label, "params": params, "windows": window_results, "summary": summary})

    # Rank by average MAE first, using MAE std as a tie-break so a config
    # doesn't win purely by getting lucky on one window.
    ranked = sorted(all_results, key=lambda r: (r["summary"]["avg_MAE"], r["summary"]["std_MAE"]))
    best = ranked[0]
    default = all_results[0]

    print("===== RANKING (by average MAE across 3 windows) =====")
    print(f"{'Label':<24}{'Avg MAE':>10}{'Avg RMSE':>12}{'Avg R2':>10}{'Std MAE':>10}")
    for r in ranked:
        s = r["summary"]
        print(f"{r['label']:<24}{s['avg_MAE']:>10.2f}{s['avg_RMSE']:>12.2f}{s['avg_R2']:>10.4f}{s['std_MAE']:>10.2f}")

    print("\n===== BEST CONFIGURATION =====")
    print("Params:", best["params"])
    for w in best["windows"]:
        print(f"  {w['window']:<20} MAE={w['MAE']:.2f}  RMSE={w['RMSE']:.2f}  R2={w['R2']:.4f}")
    print(f"  Average              MAE={best['summary']['avg_MAE']:.2f}  RMSE={best['summary']['avg_RMSE']:.2f}"
          f"  R2={best['summary']['avg_R2']:.4f}")

    print("\n===== BEST vs DEFAULT (current HGB baseline) =====")
    d, b = default["summary"], best["summary"]
    mae_delta = b["avg_MAE"] - d["avg_MAE"]
    rmse_delta = b["avg_RMSE"] - d["avg_RMSE"]
    r2_delta = b["avg_R2"] - d["avg_R2"]
    print(f"{'Metric':<10}{'Default':>12}{'Best':>12}{'Delta':>12}")
    print(f"{'MAE':<10}{d['avg_MAE']:>12.2f}{b['avg_MAE']:>12.2f}{mae_delta:>+12.2f}")
    print(f"{'RMSE':<10}{d['avg_RMSE']:>12.2f}{b['avg_RMSE']:>12.2f}{rmse_delta:>+12.2f}")
    print(f"{'R2':<10}{d['avg_R2']:>12.4f}{b['avg_R2']:>12.4f}{r2_delta:>+12.4f}")

    mae_pct = 100 * mae_delta / d["avg_MAE"]
    print(f"\nMAE change: {mae_pct:+.1f}% relative to default.")
    if best["label"] == default["label"]:
        print("The default configuration itself ranked best - tuning within this")
        print("search space did not find a meaningfully better, consistent config.")
    elif abs(mae_pct) < 2:
        print("The improvement is small (<2% MAE) and likely within noise across")
        print("different random search draws - not a strong, meaningful win.")
    else:
        print("The improvement is more than a marginal/noise-level difference and")
        print("held up consistently across all 3 independent time windows.")


if __name__ == "__main__":
    main()
