"""Tests for mibici.modeling.metrics: common metrics and evaluation by segment (section 7.5)."""

import math

import numpy as np
import pandas as pd
import pytest

from mibici.modeling.metrics import evaluate, flatten_scores, mae, wape


def test_mae_and_wape_on_known_values():
    # Errors 1, 0 and 3: MAE = 4 / 3; total departures 2 + 4 + 0 = 6, so WAPE = 4 / 6
    y_true, y_pred = [2, 4, 0], [1, 4, 3]
    assert mae(y_true, y_pred) == pytest.approx(4 / 3)
    assert wape(y_true, y_pred) == pytest.approx(4 / 6)


def test_perfect_forecast_scores_zero():
    assert mae([0, 3, 5], [0, 3, 5]) == 0
    assert wape([0, 3, 5], [0, 3, 5]) == 0


def test_wape_is_mae_times_rows_over_departures():
    # Same rows: WAPE = MAE * n / sum(y), so both rank models the same way
    rng = np.random.default_rng(0)
    y_true = rng.poisson(2, 200)
    y_pred = rng.uniform(0, 4, 200)
    assert wape(y_true, y_pred) == pytest.approx(mae(y_true, y_pred) * 200 / y_true.sum())


def test_wape_without_departures_is_undefined():
    # All zeros: there is no volume to compare the error with
    assert math.isnan(wape([0, 0, 0], [1, 0, 2]))


def test_empty_input_is_undefined():
    assert math.isnan(mae([], []))
    assert math.isnan(wape([], []))


def test_different_lengths_fail():
    with pytest.raises(ValueError, match="Shapes differ"):
        mae([1, 2], [1])


def make_rows(cohort: list[bool], event: list[bool], departures: list[int]) -> pd.DataFrame:
    """Evaluated rows with the target and the evaluation-only columns."""
    return pd.DataFrame(
        {
            "departures": departures,
            "is_new_station_cohort": cohort,
            "is_extraordinary_event": event,
        }
    )


def test_evaluate_scores_each_segment():
    # Rows 0-1 established, row 2 new cohort; every forecast is 1 departure too high
    rows = make_rows([False, False, True], [False, False, False], [2, 4, 1])
    scores = evaluate(rows, [3, 5, 2])
    assert list(scores.index) == ["all", "established_stations", "new_station_cohort"]
    assert scores.loc["all", "rows"] == 3
    assert scores.loc["established_stations", "wape"] == pytest.approx(2 / 6)
    assert scores.loc["new_station_cohort", "mae"] == pytest.approx(1)


def test_segments_without_rows_are_left_out():
    # No extraordinary events in the period: that segment does not appear
    rows = make_rows([False, False], [False, False], [1, 2])
    assert "extraordinary_event" not in evaluate(rows, [1, 2]).index
    # With an event it appears
    rows = make_rows([False, False], [True, False], [1, 2])
    assert evaluate(rows, [1, 2]).loc["extraordinary_event", "rows"] == 1


def test_evaluate_rejects_a_forecast_of_another_length():
    rows = make_rows([False, False], [False, False], [1, 2])
    with pytest.raises(ValueError, match="predictions"):
        evaluate(rows, [1])


def test_flatten_scores_names_metrics_by_segment():
    rows = make_rows([False, True], [False, False], [2, 2])
    flat = flatten_scores(evaluate(rows, [1, 2]))
    assert flat == {
        "all_mae": 0.5,
        "all_wape": 0.25,
        "established_stations_mae": 1.0,
        "established_stations_wape": 0.5,
        "new_station_cohort_mae": 0.0,
        "new_station_cohort_wape": 0.0,
    }
