"""Unit tests for preprocessing/preprocess.py.

Run with: pytest tests/
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "preprocessing"))
from preprocess import FreightPreprocessor  # noqa: E402


@pytest.fixture
def sample_train_df():
    return pd.DataFrame({
        "weight": [30000.0, -25000.0, np.nan, 40000.0],
        "market_index": [1.0, np.nan, 1.2, 0.9],
        "date": ["2025-01-01", "2025-01-02", "2025-01-03", "2025-01-04"],
        "pickup": ["A", "B", "C", "D"],
        "delivery": ["E", "F", "G", "H"],
    })


def test_fit_computes_medians_from_training_data_only(sample_train_df):
    pre = FreightPreprocessor().fit(sample_train_df)
    # abs(weight) values are 30000, 25000, 40000 (NaN excluded) -> median 30000
    assert pre.weight_median_ == 30000.0
    # market_index values are 1.0, 1.2, 0.9 (NaN excluded) -> median 1.0
    assert pre.market_index_median_ == 1.0


def test_negative_weight_is_converted_and_flagged(sample_train_df):
    pre = FreightPreprocessor().fit(sample_train_df)
    out = pre.transform(sample_train_df)
    assert (out["weight"] >= 0).all(), "weight must never be negative after transform"
    assert out.loc[1, "weight_was_negative"] == True  # noqa: E712
    assert out.loc[0, "weight_was_negative"] == False  # noqa: E712


def test_missing_weight_is_imputed_and_flagged(sample_train_df):
    pre = FreightPreprocessor().fit(sample_train_df)
    out = pre.transform(sample_train_df)
    assert not out["weight"].isna().any(), "no missing weight should remain"
    assert out.loc[2, "weight_was_missing"] == True  # noqa: E712
    assert out.loc[2, "weight"] == pre.weight_median_


def test_missing_market_index_is_imputed_and_flagged(sample_train_df):
    pre = FreightPreprocessor().fit(sample_train_df)
    out = pre.transform(sample_train_df)
    assert not out["market_index"].isna().any()
    assert out.loc[1, "market_index_was_missing"] == True  # noqa: E712
    assert out.loc[1, "market_index"] == pre.market_index_median_


def test_pickup_delivery_city_names_are_dropped(sample_train_df):
    pre = FreightPreprocessor().fit(sample_train_df)
    out = pre.transform(sample_train_df)
    assert "pickup" not in out.columns
    assert "delivery" not in out.columns


def test_date_features_are_created(sample_train_df):
    pre = FreightPreprocessor().fit(sample_train_df)
    out = pre.transform(sample_train_df)
    for col in ["month", "day_of_week", "day_of_year_sin", "day_of_year_cos"]:
        assert col in out.columns


def test_transform_before_fit_raises():
    pre = FreightPreprocessor()
    with pytest.raises(RuntimeError):
        pre.transform(pd.DataFrame({"weight": [1.0]}))


def test_save_and_load_round_trip(tmp_path, sample_train_df):
    pre = FreightPreprocessor().fit(sample_train_df)
    path = tmp_path / "stats.json"
    pre.save(path)

    loaded = FreightPreprocessor.load(path)
    assert loaded.weight_median_ == pre.weight_median_
    assert loaded.market_index_median_ == pre.market_index_median_
