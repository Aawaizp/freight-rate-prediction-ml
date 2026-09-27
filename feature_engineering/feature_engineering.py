"""Feature engineering pipeline for freight rate data.

Takes the cleaned output of preprocessing/ (train_clean.csv,
validation_clean.csv) and turns it into fully numeric, model-ready
tables. Fits on training data only (e.g. which equipment categories
exist) then applies the same transformation to any other split, so
train and validation always end up with identical columns.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ARTIFACT_PATH = Path(__file__).parent / "artifacts" / "feature_engineering_stats.json"


def _slug(text: str) -> str:
    return text.strip().lower().replace(" ", "_")


def _haversine_miles(lat1, lon1, lat2, lon2):
    """Great-circle distance between two coordinates, in miles.

    Uses only lat/lon math, so it works for any pickup/delivery pair
    including cities never seen in training.
    """
    r = 3958.8  # Earth radius in miles
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


class FeatureEngineer:
    """Learns the equipment category list from training data, then
    builds the same numeric feature set for any dataset that shares
    the preprocessed columns.
    """

    def __init__(self):
        self.equipment_categories_ = None

    def fit(self, df: pd.DataFrame) -> "FeatureEngineer":
        self.equipment_categories_ = sorted(df["equipment"].dropna().unique().tolist())
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.equipment_categories_ is None:
            raise RuntimeError("Call fit() (or load()) before transform().")

        out = df.copy()

        # --- 1. One-hot encode equipment ---
        # Categories are fixed from training, so train and validation always
        # produce the same set of columns even if one category were absent
        # from a given split.
        equipment_cat = pd.Categorical(out["equipment"], categories=self.equipment_categories_)
        equipment_dummies = pd.get_dummies(equipment_cat, prefix="equipment")
        equipment_dummies.columns = [_slug(c) for c in equipment_dummies.columns]
        out = pd.concat([out, equipment_dummies], axis=1)

        # --- 2. Route information from coordinates only ---
        # Built purely from lat/lon math, so it is well-defined for any
        # pickup/delivery pair, including the 8 validation cities never
        # seen in training - nothing here depends on knowing the city name.
        out["delta_lat"] = out["delivery_lat"] - out["pickup_lat"]
        out["delta_lon"] = out["delivery_lon"] - out["pickup_lon"]
        out["haversine_distance"] = _haversine_miles(
            out["pickup_lat"], out["pickup_lon"], out["delivery_lat"], out["delivery_lon"]
        )

        # --- 3. Distance-based features (kept minimal on purpose) ---
        out["log_distance"] = np.log1p(out["distance"])
        # How much longer the real route is vs. a straight line (>1 = indirect route).
        out["route_directness"] = out["distance"] / out["haversine_distance"].replace(0, np.nan)
        out["route_directness"] = out["route_directness"].fillna(1.0)

        # Let each equipment type have its own distance slope, since EDA
        # showed rate-vs-distance shifts somewhat by equipment.
        for cat in self.equipment_categories_:
            col = f"distance_x_{_slug(cat)}"
            out[col] = out["distance"] * equipment_dummies[f"equipment_{_slug(cat)}"]

        # Raw text columns are dropped now that their information has been
        # encoded numerically (equipment -> one-hot, date -> preprocessing's
        # month/day_of_week/day_of_year_sin/cos already carried it through).
        out = out.drop(columns=["equipment", "date"], errors="ignore")

        return out

    def save(self, path: Path = ARTIFACT_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump({"equipment_categories": self.equipment_categories_}, f, indent=2)

    @classmethod
    def load(cls, path: Path = ARTIFACT_PATH) -> "FeatureEngineer":
        with open(path) as f:
            stats = json.load(f)
        obj = cls()
        obj.equipment_categories_ = stats["equipment_categories"]
        return obj
