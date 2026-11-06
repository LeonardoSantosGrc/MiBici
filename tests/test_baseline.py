"""Tests for mibici.modeling.baseline: the common four-week baseline (section 7.4)."""

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.exceptions import NotFittedError

from mibici.features import add_history_features
from mibici.modeling.baseline import FourWeekAverageBaseline


def make_features(avg_4_weeks: list[float]) -> pd.DataFrame:
    """Feature table with the average and another column, as the models receive it."""
    return pd.DataFrame({"hour": 8, "avg_4_weeks": avg_4_weeks})


def test_predict_returns_the_four_week_average():
    X = make_features([0.0, 2.5, 7.0])
    baseline = FourWeekAverageBaseline().fit(X, [1, 2, 3])
    np.testing.assert_array_equal(baseline.predict(X), [0.0, 2.5, 7.0])


def test_fit_learns_nothing_from_the_target():
    # Two very different targets give the same predictions
    X = make_features([1.0, 4.0])
    low = FourWeekAverageBaseline().fit(X, [0, 0]).predict(X)
    high = FourWeekAverageBaseline().fit(X, [100, 100]).predict(X)
    np.testing.assert_array_equal(low, high)


def test_predictions_are_not_negative_on_real_features():
    # Built from departures (counts >= 0), so the average can never be negative
    hours = pd.date_range("2026-01-01 08:00", periods=40, freq="D")
    table = add_history_features(
        pd.DataFrame(
            {
                "station_id": 1,
                "hour_start": hours,
                "departures": np.arange(40) % 5,
                "station_age_days": np.arange(40),
            }
        )
    )
    with_history = table[table["history_source"] != "none"]
    X = with_history[["avg_4_weeks"]]
    assert (FourWeekAverageBaseline().fit(X).predict(X) >= 0).all()


def test_new_station_gets_the_fallback_value():
    # Station 2 opens on 2026-01-22: its forecast is the young-station fallback (station 3)
    rows = []
    for station_id, first_day, days, departures in [
        (1, "2025-11-01", 82, 50),
        (2, "2026-01-22", 1, 0),
        (3, "2026-01-10", 13, 6),
    ]:
        rows.append(
            pd.DataFrame(
                {
                    "station_id": station_id,
                    "hour_start": pd.date_range(f"{first_day} 08:00", periods=days, freq="D"),
                    "departures": departures,
                    "station_age_days": np.arange(days),
                }
            )
        )
    table = add_history_features(pd.concat(rows, ignore_index=True))
    new = table[table["station_id"] == 2]
    assert FourWeekAverageBaseline().fit(new[["avg_4_weeks"]]).predict(new[["avg_4_weeks"]]) == [6]


def test_predict_before_fit_fails():
    with pytest.raises(NotFittedError):
        FourWeekAverageBaseline().predict(make_features([1.0]))


def test_missing_average_fails():
    # Without the column, or with empty values, the baseline refuses to predict
    with pytest.raises(ValueError, match="avg_4_weeks"):
        FourWeekAverageBaseline().fit(pd.DataFrame({"hour": [8]}))
    X = make_features([1.0, np.nan])
    with pytest.raises(ValueError, match="missing values"):
        FourWeekAverageBaseline().fit(X).predict(X)


def test_works_with_scikit_learn_tools():
    # clone() copies the parameters, as GridSearchCV and Pipeline do
    copy = clone(FourWeekAverageBaseline(column="avg_4_weeks"))
    assert copy.get_params() == {"column": "avg_4_weeks"}
