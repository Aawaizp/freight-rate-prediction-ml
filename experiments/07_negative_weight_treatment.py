"""Experiment 1: compare 3 ways of handling negative weight values,
holding every other feature fixed, using the tuned HGB configuration
and the same 3 chronological validation windows used throughout.

Starts from data/processed/train_model_ready.csv (which already
contains variant A: abs(weight) + weight_was_missing/weight_was_negative
flags, plus all other engineered features) and re-attaches the ORIGINAL
raw weight from data/train-test.csv so variants B and C can be built by
overriding only weight/weight_was_negative/weight_was_missing - every
other feature (route, date, equipment, market_index) stays identical,
isolating this one variable.

Variant A (current/baseline): abs(weight) + weight_was_negative flag,
    missing values (from raw NaN) filled with the training median of
    abs(weight). This reproduces train_model_ready.csv exactly.
Variant B: negative values are treated AS missing (folded into the
    missing bucket) and filled with the training median of the
    remaining non-negative values; weight_was_negative flag is kept
    for reference but weight_was_missing now also covers ex-negatives.
Variant C: negative values are left unchanged (still negative) in the
    weight column; only genuinely missing (NaN) values are filled with
    the training median of non-negative values; weight_was_negative
    flag marks the untouched negative rows.

data/validation.csv is never touched. preprocessing.py, feature_engineering.py,
and train_model_ready.csv itself are not modified.
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

BASELINE_METRICS = {"MAE": 119.40, "RMSE": 627.96, "R2": 0.8260}


def load_base_data() -> pd.DataFrame:
    features = pd.read_csv(PROCESSED_DIR / "train_model_ready.csv")
    raw = pd.read_csv(DATA_DIR / "train-test.csv", usecols=["load_id", "date", "weight"])
    raw = raw.rename(columns={"weight": "raw_weight"})
    raw["date"] = pd.to_datetime(raw["date"])
    return features.merge(raw, on="load_id", how="left")


def apply_variant_a(df: pd.DataFrame) -> pd.DataFrame:
    # Already baked into train_model_ready.csv's weight / weight_was_negative /
    # weight_was_missing columns - no change needed.
    return df


def apply_variant_b(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    raw = out["raw_weight"]
    out["weight_was_negative"] = raw < 0
    folded = raw.where(raw >= 0, np.nan)  # negatives become NaN, joining true NaNs
    out["weight_was_missing"] = folded.isna()
    median_b = folded.median()  # training-only median of non-negative values
    out["weight"] = folded.fillna(median_b)
    return out


def apply_variant_c(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    raw = out["raw_weight"]
    out["weight_was_negative"] = raw < 0
    out["weight_was_missing"] = raw.isna()
    median_c = raw[raw >= 0].median()  # typical positive weight only
    out["weight"] = raw.fillna(median_c)  # negatives are left untouched
    return out


VARIANTS = [
    {"name": "A: abs(weight) + flag (current)", "builder": apply_variant_a},
    {"name": "B: negative -> missing -> median + flag", "builder": apply_variant_b},
    {"name": "C: keep negative unchanged + flag", "builder": apply_variant_c},
]


def evaluate(df: pd.DataFrame) -> list[dict]:
    results = []
    for window in WINDOWS:
        train_end = pd.Timestamp(window["train_end"])
        holdout_end = pd.Timestamp(window["holdout_end"])

        train_df = df.loc[df["date"] < train_end]
        holdout_df = df.loc[(df["date"] >= train_end) & (df["date"] < holdout_end)]

        drop_cols = ["load_id", "date", "posted_rate", "raw_weight"]
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
    base_df = load_base_data()

    all_summaries = {}
    all_window_results = {}
    for variant in VARIANTS:
        variant_df = variant["builder"](base_df)
        results = evaluate(variant_df)
        summary = summarize(results)
        all_summaries[variant["name"]] = summary
        all_window_results[variant["name"]] = results

        print(f"===== {variant['name']} =====")
        for r in results:
            print(f"  {r['window']:<20} MAE={r['MAE']:.2f}  RMSE={r['RMSE']:.2f}  R2={r['R2']:.4f}")
        print(f"  Average              MAE={summary['avg_MAE']:.2f}  RMSE={summary['avg_RMSE']:.2f}"
              f"  R2={summary['avg_R2']:.4f}")
        print()

    print("===== COMPARISON VS BASELINE (119.40 / 627.96 / 0.8260) =====")
    print(f"{'Variant':<40}{'Avg MAE':>10}{'Avg RMSE':>12}{'Avg R2':>10}")
    for name, s in all_summaries.items():
        print(f"{name:<40}{s['avg_MAE']:>10.2f}{s['avg_RMSE']:>12.2f}{s['avg_R2']:>10.4f}")

    print("\n===== VERDICTS =====")
    baseline_name = VARIANTS[0]["name"]
    baseline_results = all_window_results[baseline_name]
    for variant in VARIANTS[1:]:
        name = variant["name"]
        results = all_window_results[name]
        per_window_mae_deltas = [r["MAE"] - b["MAE"] for r, b in zip(results, baseline_results)]
        consistent_improvement = all(d < 0 for d in per_window_mae_deltas)
        verdict = "KEEP (replace A)" if consistent_improvement else "DROP (keep A)"
        print(f"{name}: per-window MAE delta vs A = {[f'{d:+.2f}' for d in per_window_mae_deltas]}"
              f" -> {verdict}")


if __name__ == "__main__":
    main()
