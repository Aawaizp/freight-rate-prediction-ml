"""Command-line demo: predict a freight rate for one load you describe.

Uses the exact same final model configuration, preprocessing, and
feature engineering as train_final.py - this is a convenience wrapper
for demonstrating the model interactively, not a separate model.

Example:
    python modeling/predict_single_load.py \\
        --pickup "Chicago" --delivery "Atlanta" \\
        --distance 720 --equipment "Dry Van" --weight 32000 --date 2025-12-15

If pickup/delivery are cities that appear in data/train-test.csv, their
coordinates are looked up automatically. For a city outside that list,
pass --pickup-lat/--pickup-lon/--delivery-lat/--delivery-lon directly.
market_index and quote_signal default to the training-data median if
not supplied, since they represent conditions that aren't knowable in
advance for a hypothetical future load.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"

sys.path.insert(0, str(PROJECT_ROOT / "preprocessing"))
sys.path.insert(0, str(PROJECT_ROOT / "feature_engineering"))
from preprocess import FreightPreprocessor  # noqa: E402
from feature_engineering import FeatureEngineer  # noqa: E402

TUNED_HGB_PARAMS = {
    "learning_rate": 0.05,
    "max_iter": 300,
    "max_leaf_nodes": 15,
    "max_depth": 10,
    "min_samples_leaf": 50,
    "l2_regularization": 1.0,
}


def lookup_city_coordinates(city: str, train_raw: pd.DataFrame):
    pickups = train_raw[train_raw["pickup"] == city][["pickup_lat", "pickup_lon"]]
    if len(pickups):
        row = pickups.iloc[0]
        return float(row["pickup_lat"]), float(row["pickup_lon"])
    deliveries = train_raw[train_raw["delivery"] == city][["delivery_lat", "delivery_lon"]]
    if len(deliveries):
        row = deliveries.iloc[0]
        return float(row["delivery_lat"]), float(row["delivery_lon"])
    return None


def parse_args():
    parser = argparse.ArgumentParser(description="Predict a freight rate for one load.")
    parser.add_argument("--pickup", required=True)
    parser.add_argument("--delivery", required=True)
    parser.add_argument("--distance", type=float, required=True)
    parser.add_argument("--equipment", required=True, choices=["Dry Van", "Reefer", "Flatbed"])
    parser.add_argument("--weight", type=float, required=True)
    parser.add_argument("--date", required=True, help="YYYY-MM-DD")
    parser.add_argument("--market-index", type=float, default=None)
    parser.add_argument("--quote-signal", type=float, default=None)
    parser.add_argument("--pickup-lat", type=float, default=None)
    parser.add_argument("--pickup-lon", type=float, default=None)
    parser.add_argument("--delivery-lat", type=float, default=None)
    parser.add_argument("--delivery-lon", type=float, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    train_raw = pd.read_csv(DATA_DIR / "train-test.csv")

    pickup_coords = (args.pickup_lat, args.pickup_lon)
    if None in pickup_coords:
        found = lookup_city_coordinates(args.pickup, train_raw)
        if found is None:
            sys.exit(f"'{args.pickup}' not found in training data - pass --pickup-lat/--pickup-lon.")
        pickup_coords = found

    delivery_coords = (args.delivery_lat, args.delivery_lon)
    if None in delivery_coords:
        found = lookup_city_coordinates(args.delivery, train_raw)
        if found is None:
            sys.exit(f"'{args.delivery}' not found in training data - pass --delivery-lat/--delivery-lon.")
        delivery_coords = found

    pre = FreightPreprocessor.load()
    fe = FeatureEngineer.load()

    row = pd.DataFrame([{
        "pickup": args.pickup,
        "delivery": args.delivery,
        "pickup_lat": pickup_coords[0],
        "pickup_lon": pickup_coords[1],
        "delivery_lat": delivery_coords[0],
        "delivery_lon": delivery_coords[1],
        "distance": args.distance,
        "equipment": args.equipment,
        "weight": args.weight,
        "date": args.date,
        "market_index": args.market_index if args.market_index is not None else np.nan,
        "quote_signal": args.quote_signal if args.quote_signal is not None else train_raw["quote_signal"].median(),
    }])

    row_clean = pre.transform(row)
    row_features = fe.transform(row_clean)

    train_features = pd.read_csv(PROCESSED_DIR / "train_model_ready.csv")
    X_train = train_features.drop(columns=["load_id", "posted_rate"])
    y_train_log = np.log1p(train_features["posted_rate"])

    print("Training final model (same config as train_final.py)...")
    model = HistGradientBoostingRegressor(random_state=42, **TUNED_HGB_PARAMS)
    model.fit(X_train, y_train_log)

    X_row = row_features[X_train.columns]
    predicted_rate = float(np.expm1(model.predict(X_row))[0])

    print(f"\n{args.pickup} -> {args.delivery} ({args.distance:g} mi, {args.equipment}, "
          f"{args.weight:g} lb, {args.date})")
    print(f"Predicted rate: ${predicted_rate:,.2f}")


if __name__ == "__main__":
    main()
