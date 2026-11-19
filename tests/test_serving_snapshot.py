"""Tests for the history snapshot and the feature table the API serves from."""

import json

import numpy as np
import pandas as pd
import pytest

from mibici.config import FEATURES
from mibici.features import add_history_features
from mibici.serving.snapshot import (
    build_feature_table,
    build_snapshot,
    check_reference,
    day_rows,
    forecast_range,
    snapshot_info,
)

FIRST_DATE = "2026-07-15"  # first served date of the small table below


def make_processed() -> pd.DataFrame:
    """processed_v1-like table: hours 0 and 8 from June 1 to August 15, 2026.

    Station 1 exists the whole period; station 2 opens on July 20, so its first weeks use
    the young-station fallback of add_history_features.
    """
    hours = pd.date_range("2026-06-01", "2026-08-15 23:00", freq="h")
    hours = hours[hours.hour.isin([0, 8])]
    rng = np.random.default_rng(0)
    frames = []
    for station_id, opening in ((1, "2026-06-01"), (2, "2026-07-20")):
        station_hours = hours[hours >= pd.Timestamp(opening)]
        frames.append(
            pd.DataFrame(
                {
                    "station_id": np.int32(station_id),
                    "hour_start": station_hours,
                    "departures": rng.poisson(3, len(station_hours)).astype("int32"),
                    "hour": station_hours.hour.astype("int32"),
                    "day_of_week": station_hours.day_of_week.astype("int32"),
                    "is_weekend": station_hours.day_of_week >= 5,
                    "month": station_hours.month.astype("int32"),
                    "station_age_days": (
                        station_hours.normalize() - pd.Timestamp(opening)
                    ).days.astype("int32"),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def test_snapshot_features_equal_the_training_features():
    # Train-serving skew check: the API's features must be exactly the ones of training,
    # which were computed on the full table
    processed = make_processed()
    training = add_history_features(processed)
    serving = build_feature_table(build_snapshot(processed, FIRST_DATE))

    keys = ["station_id", "hour_start"]
    expected = training[training["hour_start"] >= FIRST_DATE].sort_values(keys, ignore_index=True)
    served = serving[serving["hour_start"] <= processed["hour_start"].max()]
    pd.testing.assert_frame_equal(
        served[keys + FEATURES + ["history_source"]].reset_index(drop=True),
        expected[keys + FEATURES + ["history_source"]],
    )


def test_next_day_rows_are_built_from_the_last_day():
    processed = make_processed()
    table = build_feature_table(build_snapshot(processed, FIRST_DATE))
    next_day = day_rows(table, "2026-08-16")  # the data ends on August 15

    last_day = processed[processed["hour_start"].dt.normalize() == "2026-08-15"]
    assert len(next_day) == len(last_day)  # same stations and hours
    assert next_day[FEATURES].notna().all().all()
    assert next_day["departures"].isna().all()  # unknown: the day has not happened
    assert (next_day["day_of_week"] == pd.Timestamp("2026-08-16").day_of_week).all()
    # One more day of age for every station
    np.testing.assert_array_equal(
        next_day.sort_values(["station_id", "hour_start"])["station_age_days"].to_numpy(),
        last_day.sort_values(["station_id", "hour_start"])["station_age_days"].to_numpy() + 1,
    )


def test_forecast_range_goes_from_first_date_to_the_day_after_the_data():
    snapshot = build_snapshot(make_processed(), FIRST_DATE)
    first, last = forecast_range(snapshot)
    assert (first, last) == (pd.Timestamp(FIRST_DATE), pd.Timestamp("2026-08-16"))


@pytest.mark.parametrize("date", ["2026-07-14", "2026-08-17", "2026-11-20"])
def test_dates_outside_the_range_are_refused_with_the_range(date):
    table = build_feature_table(build_snapshot(make_processed(), FIRST_DATE))
    with pytest.raises(ValueError, match="from 2026-07-15 to 2026-08-16"):
        day_rows(table, date)


def test_snapshot_needs_enough_history():
    with pytest.raises(ValueError, match="starts after"):
        build_snapshot(make_processed(), "2026-06-10")  # needs data from May 13


def test_reference_is_created_then_checked(tmp_path):
    reference = tmp_path / "snapshot.json"
    snapshot = build_snapshot(make_processed(), FIRST_DATE)
    info = snapshot_info(snapshot)

    assert check_reference(info, reference) is True  # first run writes it
    assert json.loads(reference.read_text()) == info
    assert check_reference(info, reference) is False  # same data: accepted

    # One changed departure gives another digest and is refused
    changed = snapshot.copy()
    changed.loc[0, "departures"] += 1
    with pytest.raises(ValueError, match="differs from the reference"):
        check_reference(snapshot_info(changed), reference)
