"""Unit tests for feature_engineering/feature_engineering.py.

Run with: pytest tests/
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "feature_engineering"))
from feature_engineering import FeatureEngineer  # noqa: E402


@pytest.fixture
def preprocessed_df():
    # Shape matches what preprocessing/preprocess.py hands off: no raw
    # pickup/delivery names, lat/lon retained, equipment still raw text.
    return pd.DataFrame({
        "pickup_lat": [38.0, 39.0],
        "pickup_lon": [-77.0, -75.0],
        "delivery_lat": [38.2, 40.0],
        "delivery_lon": [-72.7, -87.6],
        "distance": [300.0, 900.0],
        "equipment": ["Dry Van", "Reefer"],
        "weight": [30000.0, 20000.0],
        "date": ["2025-01-01", "2025-06-15"],
    })


def test_fit_learns_equipment_categories(preprocessed_df):
    fe = FeatureEngineer().fit(preprocessed_df)
    assert fe.equipment_categories_ == ["Dry Van", "Reefer"]


def test_equipment_one_hot_columns_created(preprocessed_df):
    fe = FeatureEngineer().fit(preprocessed_df)
    out = fe.transform(preprocessed_df)
    assert "equipment_dry_van" in out.columns
    assert "equipment_reefer" in out.columns
    assert out.loc[0, "equipment_dry_van"] == 1
    assert out.loc[0, "equipment_reefer"] == 0


def test_route_features_are_created(preprocessed_df):
    fe = FeatureEngineer().fit(preprocessed_df)
    out = fe.transform(preprocessed_df)
    for col in ["delta_lat", "delta_lon", "haversine_distance", "log_distance", "route_directness"]:
        assert col in out.columns
    assert (out["haversine_distance"] > 0).all()


def test_haversine_distance_is_never_larger_than_impossible_value(preprocessed_df):
    # Sanity bound: no US domestic lane should exceed ~3000 straight-line miles.
    fe = FeatureEngineer().fit(preprocessed_df)
    out = fe.transform(preprocessed_df)
    assert (out["haversine_distance"] < 3000).all()


def test_raw_equipment_and_date_columns_are_dropped(preprocessed_df):
    fe = FeatureEngineer().fit(preprocessed_df)
    out = fe.transform(preprocessed_df)
    assert "equipment" not in out.columns
    assert "date" not in out.columns


def test_transform_uses_training_categories_for_unseen_equipment(preprocessed_df):
    # Fit only sees Dry Van/Reefer - simulates a category present in
    # validation but not training being handled without crashing.
    fe = FeatureEngineer().fit(preprocessed_df)
    unseen = preprocessed_df.copy()
    unseen.loc[0, "equipment"] = "Flatbed"
    out = fe.transform(unseen)
    # Flatbed was never in the fitted categories, so it should not create
    # a new one-hot column - it should just fail to match any known category.
    assert "equipment_flatbed" not in out.columns
    assert out.loc[0, "equipment_dry_van"] == 0
    assert out.loc[0, "equipment_reefer"] == 0
