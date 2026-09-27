"""Freight rate preprocessing pipeline.

Fit on training data only, then reuse the exact same statistics (medians)
to transform any other split (validation, December inputs, ...). This
keeps the pipeline leak-free: validation data never influences the
values used to fill its own missing fields.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ARTIFACT_PATH = Path(__file__).parent / "artifacts" / "preprocessing_stats.json"


class FreightPreprocessor:
    """Learns imputation values from training data, then applies the
    same cleaning + feature steps to any dataset that shares its columns.
    """

    def __init__(self):
        self.weight_median_ = None
        self.market_index_median_ = None

    def fit(self, df: pd.DataFrame) -> "FreightPreprocessor":
        # Use abs() before taking the median so the negative-sign
        # errors don't drag the "typical weight" estimate down.
        self.weight_median_ = df["weight"].abs().median()
        self.market_index_median_ = df["market_index"].median()
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.weight_median_ is None:
            raise RuntimeError("Call fit() (or load()) before transform().")

        out = df.copy()

        if "weight" in out.columns:
            # Hypothesis: negative weights are sign-flip data-entry errors,
            # not a distinct category (EDA found no pattern by equipment/month).
            out["weight_was_negative"] = out["weight"] < 0
            out["weight"] = out["weight"].abs()

            out["weight_was_missing"] = out["weight"].isna()
            out["weight"] = out["weight"].fillna(self.weight_median_)

        if "market_index" in out.columns:
            out["market_index_was_missing"] = out["market_index"].isna()
            out["market_index"] = out["market_index"].fillna(self.market_index_median_)

        if "date" in out.columns:
            date = pd.to_datetime(out["date"])
            out["month"] = date.dt.month
            out["day_of_week"] = date.dt.dayofweek
            day_of_year = date.dt.dayofyear
            # Cyclical encoding so Dec 31 and Jan 1 are numerically close,
            # instead of one-hot months the model may never have seen (Nov/Dec).
            out["day_of_year_sin"] = np.sin(2 * np.pi * day_of_year / 365)
            out["day_of_year_cos"] = np.cos(2 * np.pi * day_of_year / 365)

        # Raw city names are dropped: validation has 8 pickup/delivery cities
        # never seen in training, so a name-based encoding would break on them.
        # pickup_lat/lon, delivery_lat/lon and distance already carry the
        # geographic signal and generalize to unseen cities for free.
        out = out.drop(columns=["pickup", "delivery"], errors="ignore")

        # equipment is left as-is (raw category) - encoding it is a
        # modeling-step decision, not a cleaning one.

        return out

    def save(self, path: Path = ARTIFACT_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(
                {
                    "weight_median": self.weight_median_,
                    "market_index_median": self.market_index_median_,
                },
                f,
                indent=2,
            )

    @classmethod
    def load(cls, path: Path = ARTIFACT_PATH) -> "FreightPreprocessor":
        with open(path) as f:
            stats = json.load(f)
        obj = cls()
        obj.weight_median_ = stats["weight_median"]
        obj.market_index_median_ = stats["market_index_median"]
        return obj
