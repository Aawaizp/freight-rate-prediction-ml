"""Sanity checks on the actual submission files, mirroring the checks
score.py performs. These read the already-generated files - they do not
retrain anything - so they run in under a second and catch accidental
corruption of the final deliverables.

Run with: pytest tests/
"""
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).parent.parent
VALIDATION_PREDICTIONS = PROJECT_ROOT / "validation_predictions.csv"
DECEMBER_PREDICTIONS = PROJECT_ROOT / "data" / "december_predictions.csv"

EXPECTED_VALIDATION_ROWS = 12_000
EXPECTED_DECEMBER_ROWS = 31


@pytest.mark.skipif(not VALIDATION_PREDICTIONS.exists(), reason="run modeling/train_final.py first")
def test_validation_predictions_format():
    df = pd.read_csv(VALIDATION_PREDICTIONS)
    assert list(df.columns) == ["load_id", "predicted_rate"]
    assert len(df) == EXPECTED_VALIDATION_ROWS
    assert df["load_id"].is_unique
    assert not df["predicted_rate"].isna().any()
    assert (df["predicted_rate"] > 0).all()


@pytest.mark.skipif(not DECEMBER_PREDICTIONS.exists(), reason="run modeling/predict_december.py first")
def test_december_predictions_format():
    df = pd.read_csv(DECEMBER_PREDICTIONS)
    assert list(df.columns) == ["pickup", "delivery", "distance", "equipment", "weight", "date", "predicted_rate"]
    assert len(df) == EXPECTED_DECEMBER_ROWS
    assert df["pickup"].eq("Lexington").all()
    assert df["delivery"].eq("Fort Wayne").all()
    assert (df["predicted_rate"] > 0).all()
    assert not df["date"].duplicated().any()
