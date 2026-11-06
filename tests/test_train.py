"""Tests for mibici.modeling.train: training by evaluation mode, without MLflow (track=False)."""

import numpy as np
import pandas as pd
import pytest

from mibici.features import add_history_features
from mibici.modeling.models import MODELS
from mibici.modeling.train import check_mode, hyperparameters, run_training


def make_table() -> pd.DataFrame:
    """Two stations with a row every day at 08:00 from April to September 2026."""
    days = pd.date_range("2026-04-01 08:00", "2026-09-30 08:00", freq="D")
    rng = np.random.default_rng(0)
    frames = []
    for station_id in (1, 2):
        frames.append(
            pd.DataFrame(
                {
                    "station_id": station_id,
                    "hour_start": days,
                    "departures": rng.poisson(3, len(days)),
                    "hour": 8,
                    "day_of_week": days.day_of_week,
                    "is_weekend": days.day_of_week >= 5,
                    "month": days.month,
                    "station_age_days": np.arange(len(days)),
                    "is_new_station_cohort": False,
                    "is_extraordinary_event": False,
                }
            )
        )
    # Same columns as processed_v1 plus the history features
    return add_history_features(pd.concat(frames, ignore_index=True))


def test_test_mode_requires_final():
    with pytest.raises(ValueError, match="FINAL=1"):
        check_mode("test", final=False)
    check_mode("test", final=True)  # accepted when confirmed
    check_mode("validation", final=False)


def test_unknown_mode_and_model_are_rejected():
    with pytest.raises(ValueError, match="Unknown evaluation mode"):
        check_mode("june", final=False)
    with pytest.raises(ValueError, match="Unknown model"):
        run_training(make_table(), "xgboost", "validation", track=False)


def test_baseline_run_matches_its_own_reference():
    # Training the baseline: its scores equal the baseline reference on the same rows
    results = run_training(make_table(), "baseline", "validation", track=False)
    assert list(results.index) == ["baseline-validation-01"]
    assert results.loc["baseline-validation-01", "wape"] == pytest.approx(
        results.loc["baseline-validation-01", "baseline_wape"]
    )


def test_one_run_per_grid_combination(monkeypatch):
    # Temporary grid of two values for this test only (monkeypatch restores it afterwards)
    monkeypatch.setitem(MODELS["hgb"], "param_grid", {"model__regressor__max_iter": [5, 10]})
    results = run_training(make_table(), "hgb", "window_june", track=False)
    assert list(results.index) == ["hgb-window_june-01", "hgb-window_june-02"]
    assert list(results["model__regressor__max_iter"]) == [5, 10]


def test_final_test_run_works_when_confirmed():
    results = run_training(make_table(), "baseline", "test", final=True, track=False)
    assert len(results) == 1


def test_hyperparameters_keep_only_model_and_preprocessing():
    from mibici.modeling.models import make_model

    params = hyperparameters(make_model("hgb"))
    assert params["model__regressor__random_state"] == "0"
    # Steps of the common pipeline (column selection) are not hyperparameters of a block
    assert not any(name.startswith("select") for name in params)
