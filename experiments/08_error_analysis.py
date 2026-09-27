"""Experiment 2: error analysis for the current tuned HGB model.

Trains the exact tuned HGB configuration on each of the 3 chronological
windows (same as tune_hgb.py) and pools the holdout predictions from
all 3 windows together, then reports MAE broken down by:
- equipment
- distance band
- month
- negative-weight rows (flagged by the existing pipeline)
- missing-weight rows (flagged by the existing pipeline)
- high-rate loads (posted_rate > 6000, the threshold EDA used)

This only reads existing pipeline output (train_model_ready.csv) plus
raw train-test.csv for row context (equipment/date labels); it does
not modify any existing script and never touches data/validation.csv.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

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

HIGH_RATE_THRESHOLD = 6000
DISTANCE_BINS = [0, 500, 1000, 1500, 2000, 2500, 3500]
DISTANCE_LABELS = ["0-500", "500-1000", "1000-1500", "1500-2000", "2000-2500", "2500+"]


def load_data() -> pd.DataFrame:
    features = pd.read_csv(PROCESSED_DIR / "train_model_ready.csv")
    raw = pd.read_csv(DATA_DIR / "train-test.csv", usecols=["load_id", "date", "equipment", "distance"])
    raw = raw.rename(columns={"distance": "raw_distance"})
    raw["date"] = pd.to_datetime(raw["date"])
    return features.merge(raw, on="load_id", how="left")


def main():
    df = load_data()

    pooled = []
    for window in WINDOWS:
        train_end = pd.Timestamp(window["train_end"])
        holdout_end = pd.Timestamp(window["holdout_end"])

        train_df = df.loc[df["date"] < train_end]
        holdout_df = df.loc[(df["date"] >= train_end) & (df["date"] < holdout_end)]

        drop_cols = ["load_id", "date", "posted_rate", "equipment", "raw_distance"]
        X_train = train_df.drop(columns=drop_cols)
        y_train_log = np.log1p(train_df["posted_rate"])
        X_holdout = holdout_df.drop(columns=drop_cols)
        y_holdout = holdout_df["posted_rate"]

        model = HistGradientBoostingRegressor(random_state=42, **TUNED_HGB_PARAMS)
        model.fit(X_train, y_train_log)
        preds = np.expm1(model.predict(X_holdout))

        chunk = holdout_df.copy()
        chunk["predicted_rate"] = preds
        chunk["abs_error"] = (chunk["posted_rate"] - preds).abs()
        chunk["window"] = window["name"]
        pooled.append(chunk)

        print(f"{window['name']}: {len(chunk)} holdout rows, window MAE = {chunk['abs_error'].mean():.2f}")

    pooled_df = pd.concat(pooled, ignore_index=True)
    print(f"\nTotal pooled holdout rows: {len(pooled_df)}")
    print(f"Overall pooled MAE: {pooled_df['abs_error'].mean():.2f}\n")

    pooled_df["month"] = pooled_df["date"].dt.month
    pooled_df["distance_band"] = pd.cut(pooled_df["raw_distance"], bins=DISTANCE_BINS, labels=DISTANCE_LABELS)
    pooled_df["is_high_rate"] = pooled_df["posted_rate"] > HIGH_RATE_THRESHOLD

    def report_group(label: str, group_col: str):
        print(f"===== MAE BY {label.upper()} =====")
        grouped = pooled_df.groupby(group_col, observed=True)["abs_error"].agg(["mean", "count"])
        grouped = grouped.rename(columns={"mean": "MAE", "count": "n_rows"}).sort_values("MAE", ascending=False)
        print(grouped.to_string(float_format=lambda x: f"{x:.2f}"))
        print()

    report_group("equipment", "equipment")
    report_group("distance band", "distance_band")
    report_group("month", "month")

    print("===== MAE BY WEIGHT FLAGS =====")
    for flag_col, flag_label in [("weight_was_negative", "negative-weight rows"),
                                  ("weight_was_missing", "missing-weight rows")]:
        flagged = pooled_df[pooled_df[flag_col]]
        not_flagged = pooled_df[~pooled_df[flag_col]]
        print(f"{flag_label}: n={len(flagged)}  MAE={flagged['abs_error'].mean():.2f}"
              if len(flagged) else f"{flag_label}: n=0 (none in pooled holdouts)")
        print(f"{'rest of rows':<24}: n={len(not_flagged)}  MAE={not_flagged['abs_error'].mean():.2f}")
        print()

    print("===== MAE BY HIGH-RATE LOADS (posted_rate > 6000) =====")
    high = pooled_df[pooled_df["is_high_rate"]]
    low = pooled_df[~pooled_df["is_high_rate"]]
    print(f"high-rate (>6000): n={len(high)}  MAE={high['abs_error'].mean():.2f}" if len(high)
          else "high-rate (>6000): n=0")
    print(f"rest (<=6000):      n={len(low)}  MAE={low['abs_error'].mean():.2f}")

    print("\n===== SUMMARY =====")
    print("Groups with the highest MAE above indicate where the model struggles most.")
    print("Compare each group's row count too - a high MAE on very few rows carries")
    print("less weight than a high MAE on a large group.")


if __name__ == "__main__":
    main()
