"""December prediction stage.

Uses the SAME final model configuration (tuned HGB from tune_hgb.py,
log1p/expm1 target handling) and the SAME fitted preprocessing +
feature-engineering artifacts used to build validation_predictions.csv,
applied to december-chart-inputs.csv.

IMPORTANT - december-chart-inputs.csv is missing columns the model
needs, and this script has to make explicit, documented assumptions to
fill them:

- pickup_lat/pickup_lon/delivery_lat/delivery_lon are not in the file
  at all. They are looked up from train-test.csv's known city
  coordinates (EDA confirmed each city name maps to one consistent
  lat/lon pair). This works because every city in december-chart-
  inputs.csv already appears in the training data - the script raises
  an error if that's ever not true, rather than guessing.
- market_index is not in the file. It is left blank and filled by the
  pipeline's own already-fitted training median (from
  preprocessing/artifacts/preprocessing_stats.json) - the same
  treatment a genuinely missing market_index gets in train/validation.
- quote_signal is not in the file. Nothing in preprocess.py handles a
  missing quote_signal (it was never missing in train/validation), so
  it is filled here with the training-only quote_signal median as an
  explicit, stated assumption - not something the trained pipeline was
  originally designed to handle.

This script does not retrain the model differently, and does not
modify preprocessing.py, feature_engineering.py, train_final.py, or
validation_predictions.csv. Raw data/train-test.csv, data/validation.csv,
and data/december-chart-inputs.csv are read only, never written to.
"""
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"

sys.path.insert(0, str(PROJECT_ROOT / "preprocessing"))
sys.path.insert(0, str(PROJECT_ROOT / "feature_engineering"))
from preprocess import FreightPreprocessor  # noqa: E402
from feature_engineering import FeatureEngineer  # noqa: E402

# Same config selected in tune_hgb.py and used in train_final.py.
TUNED_HGB_PARAMS = {
    "learning_rate": 0.05,
    "max_iter": 300,
    "max_leaf_nodes": 15,
    "max_depth": 10,
    "min_samples_leaf": 50,
    "l2_regularization": 1.0,
}

DECEMBER_OUTPUT_PATH = DATA_DIR / "december_predictions.csv"
CHART_OUTPUT_PATH = REPORTS_DIR / "december_prediction_chart.png"


def build_city_coordinates(train: pd.DataFrame) -> pd.DataFrame:
    pickups = train[["pickup", "pickup_lat", "pickup_lon"]].rename(
        columns={"pickup": "city", "pickup_lat": "lat", "pickup_lon": "lon"}
    )
    deliveries = train[["delivery", "delivery_lat", "delivery_lon"]].rename(
        columns={"delivery": "city", "delivery_lat": "lat", "delivery_lon": "lon"}
    )
    coords = pd.concat([pickups, deliveries], ignore_index=True).drop_duplicates(subset="city")
    return coords.set_index("city")[["lat", "lon"]]


def attach_coordinates(december: pd.DataFrame, coords: pd.DataFrame) -> pd.DataFrame:
    out = december.copy()
    missing_pickup = sorted(set(out["pickup"]) - set(coords.index))
    missing_delivery = sorted(set(out["delivery"]) - set(coords.index))
    if missing_pickup or missing_delivery:
        raise ValueError(
            f"December cities not found in training city list - "
            f"pickup: {missing_pickup}, delivery: {missing_delivery}. "
            f"Coordinate lookup cannot proceed for these rows."
        )
    out["pickup_lat"] = out["pickup"].map(coords["lat"])
    out["pickup_lon"] = out["pickup"].map(coords["lon"])
    out["delivery_lat"] = out["delivery"].map(coords["lat"])
    out["delivery_lon"] = out["delivery"].map(coords["lon"])
    return out


def run_scorer_if_available() -> str | None:
    score_py = PROJECT_ROOT / "score.py"
    if not score_py.exists():
        return None
    result = subprocess.run(
        [sys.executable, str(score_py),
         "--predictions", str(PROJECT_ROOT / "validation_predictions.csv"),
         "--december-predictions", str(DECEMBER_OUTPUT_PATH)],
        cwd=PROJECT_ROOT, capture_output=True, text=True,
    )
    return f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}\nreturn code: {result.returncode}"


def main():
    train_raw = pd.read_csv(DATA_DIR / "train-test.csv")
    december = pd.read_csv(DATA_DIR / "december-chart-inputs.csv")

    print(f"December rows: {len(december)}")

    coords = build_city_coordinates(train_raw)
    december = attach_coordinates(december, coords)

    december["market_index"] = np.nan  # filled below by the fitted training median

    quote_signal_median = train_raw["quote_signal"].median()
    december["quote_signal"] = quote_signal_median
    print(f"quote_signal filled with training-only median: {quote_signal_median:.5f} (assumption - see docstring)")

    pre = FreightPreprocessor.load()
    fe = FeatureEngineer.load()
    print(f"market_index filled with training-only median: {pre.market_index_median_} (from saved pipeline artifact)")

    december_clean = pre.transform(december)
    december_features = fe.transform(december_clean)

    # Retrain exactly as train_final.py does: same params, same full
    # training set, same log1p target. Not a different model.
    train_features = pd.read_csv(PROCESSED_DIR / "train_model_ready.csv")
    X_train = train_features.drop(columns=["load_id", "posted_rate"])
    y_train_log = np.log1p(train_features["posted_rate"])

    model = HistGradientBoostingRegressor(random_state=42, **TUNED_HGB_PARAMS)
    model.fit(X_train, y_train_log)

    X_december = december_features.drop(columns=["predicted_rate"], errors="ignore")
    X_december = X_december[X_train.columns]  # enforce identical column order

    preds_log = model.predict(X_december)
    predicted_rate = np.expm1(preds_log)

    december_out = pd.read_csv(DATA_DIR / "december-chart-inputs.csv")
    december_out["predicted_rate"] = predicted_rate
    december_out.to_csv(DECEMBER_OUTPUT_PATH, index=False)

    print("\n===== DECEMBER PREDICTIONS =====")
    print(f"Rows: {len(december_out)}")
    print(f"Prediction range: {predicted_rate.min():.2f} to {predicted_rate.max():.2f}")
    print(f"Prediction mean:  {predicted_rate.mean():.2f}")
    print(f"Saved to: {DECEMBER_OUTPUT_PATH}")

    # ----- Chart -----
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    december_out["date"] = pd.to_datetime(december_out["date"])
    plt.figure(figsize=(10, 6))
    for (pickup, delivery), group in december_out.groupby(["pickup", "delivery"]):
        group = group.sort_values("date")
        plt.plot(group["date"], group["predicted_rate"], marker="o", label=f"{pickup} -> {delivery}")
    plt.xlabel("Date")
    plt.ylabel("Predicted Rate ($)")
    plt.title("December 2025 Predicted Freight Rates")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(CHART_OUTPUT_PATH, dpi=150)
    plt.close()
    print(f"Chart saved to: {CHART_OUTPUT_PATH}")

    # ----- score.py -----
    print("\n===== score.py =====")
    scorer_output = run_scorer_if_available()
    if scorer_output is None:
        print("score.py was not found in the project root - it was not included in this "
              "project folder, so the official scorer could not be run. The chart above "
              "was generated directly by this script as a substitute. Add score.py and "
              "re-run this step once it's available.")
    else:
        print(scorer_output)


if __name__ == "__main__":
    main()
