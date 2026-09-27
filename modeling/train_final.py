"""Final model: HistGradientBoostingRegressor with the tuned parameters
from tune_hgb.py, trained on ALL 48,000 rows of the cleaned + engineered
training data, then used to predict every row of validation.csv.

Uses the already-generated outputs of the existing preprocessing and
feature-engineering pipeline (data/processed/train_model_ready.csv and
validation_model_ready.csv) - this script does not re-implement or
modify that pipeline, it only consumes its output.

data/validation.csv's posted_rate is never available (by design) and
is never used for training or tuning here - only for producing the
final predictions this stage is meant to generate.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

PROJECT_ROOT = Path(__file__).parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_PATH = PROJECT_ROOT / "validation_predictions.csv"

# Best config found in tune_hgb.py (selected for consistent performance
# across 3 chronological validation windows, not one lucky window).
TUNED_HGB_PARAMS = {
    "learning_rate": 0.05,
    "max_iter": 300,
    "max_leaf_nodes": 15,
    "max_depth": 10,
    "min_samples_leaf": 50,
    "l2_regularization": 1.0,
}


def main():
    train = pd.read_csv(PROCESSED_DIR / "train_model_ready.csv")
    validation = pd.read_csv(PROCESSED_DIR / "validation_model_ready.csv")

    X_train = train.drop(columns=["load_id", "posted_rate"])
    y_train_log = np.log1p(train["posted_rate"])

    validation_load_ids = validation["load_id"]
    X_validation = validation.drop(columns=["load_id"])

    # Sanity check: both feature sets must line up before fitting anything.
    assert list(X_train.columns) == list(X_validation.columns), (
        "Train/validation feature columns do not match - check the "
        "preprocessing/feature-engineering outputs before proceeding."
    )

    print(f"Training on all {len(train)} rows with {X_train.shape[1]} features...")
    model = HistGradientBoostingRegressor(random_state=42, **TUNED_HGB_PARAMS)
    model.fit(X_train, y_train_log)

    print(f"Predicting {len(validation)} validation rows...")
    preds_log = model.predict(X_validation)
    predicted_rate = np.expm1(preds_log)

    output = pd.DataFrame({
        "load_id": validation_load_ids,
        "predicted_rate": predicted_rate,
    })
    output.to_csv(OUTPUT_PATH, index=False)

    # ----- Verification -----
    print("\n===== VERIFICATION =====")
    n_rows = len(output)
    n_unique_ids = output["load_id"].nunique()
    n_missing = output["predicted_rate"].isna().sum()
    n_negative = (output["predicted_rate"] < 0).sum()
    columns_ok = list(output.columns) == ["load_id", "predicted_rate"]
    order_preserved = (output["load_id"].values == validation["load_id"].values).all()

    print(f"Rows: {n_rows} (expected 12000) -> {'OK' if n_rows == 12000 else 'FAIL'}")
    print(f"Unique load_id: {n_unique_ids} (expected 12000) -> "
          f"{'OK' if n_unique_ids == 12000 else 'FAIL'}")
    print(f"Missing predicted_rate: {n_missing} -> {'OK' if n_missing == 0 else 'FAIL'}")
    print(f"Negative predicted_rate: {n_negative} -> {'OK' if n_negative == 0 else 'FAIL'}")
    print(f"Columns exactly ['load_id', 'predicted_rate']: {columns_ok} -> "
          f"{'OK' if columns_ok else 'FAIL'}")
    print(f"load_id order matches validation.csv: {order_preserved} -> "
          f"{'OK' if order_preserved else 'FAIL'}")

    print(f"\nPredicted rate range: {predicted_rate.min():.2f} to {predicted_rate.max():.2f}")
    print(f"Predicted rate mean:  {predicted_rate.mean():.2f}")
    print(f"\nSaved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
