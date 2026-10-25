"""Tests for mibici.processed: the PR2 decisions applied to build processed_v1.

Synthetic trips with known answers, one scenario per decision.
"""

import pandas as pd

from mibici.processed import build_processed, implausible_short_cross_trips


def make_trips(rows):
    """Build interim-like trips from (origin, destination, start, duration_s, short_same, short_cross)."""
    trips = pd.DataFrame(
        rows,
        columns=[
            "origin_station_id",
            "destination_station_id",
            "started_at",
            "duration_s",
            "flag_short_same_station",
            "flag_short_cross_station",
        ],
    )
    trips["started_at"] = pd.to_datetime(trips["started_at"])
    trips[["origin_station_id", "destination_station_id"]] = trips[
        ["origin_station_id", "destination_station_id"]
    ].astype("Int32")
    trips["duration_s"] = trips["duration_s"].astype("Int64")
    trips["datetime_format"] = "iso"
    return trips


def normal_trip(start, origin=1, destination=2, duration=600):
    """A regular 10-minute trip between two different stations."""
    return (origin, destination, start, duration, False, False)


def test_only_service_hours_from_window_start_are_kept():
    # A trip in 2021 and one in 2022: rows only from 2022-01-01, and never at 01:00-04:59
    trips = make_trips([normal_trip("2021-12-31 10:00"), normal_trip("2022-01-01 10:00")])

    data = build_processed(trips)
    station = data[data["station_id"] == 1]

    assert station["hour_start"].min() == pd.Timestamp("2022-01-01 00:00")
    assert not station["hour"].isin([1, 2, 3, 4]).any()
    # One day of service: hours 0 and 5-23 = 20 intervals
    assert len(station) == 20


def test_short_same_station_trips_are_not_counted_but_hour_exists():
    trips = make_trips(
        [
            normal_trip("2022-03-01 08:10"),
            # Short trip that returns to the same station: excluded from departures
            (1, 1, "2022-03-01 08:20", 20, True, False),
        ]
    )

    data = build_processed(trips).set_index(["station_id", "hour_start"])

    assert data.loc[(1, pd.Timestamp("2022-03-01 08:00")), "departures"] == 1


def test_implausible_short_cross_trip_rule():
    # Pair 1 -> 2 in 2022: 30 trips of 600 s give a solid reference (median 600 s)
    reference = [normal_trip(f"2022-03-{day:02d} 09:00") for day in range(1, 31)]
    rows = reference + [
        # 0: 2 s, < 10 % of 600 s -> implausible
        (1, 2, "2022-03-31 10:00", 2, False, True),
        # 1: 59 s on pair 3 -> 4 with only 1 reference trip -> kept (not enough reference)
        (3, 4, "2022-03-31 10:00", 59, False, True),
        # 2: 59 s on pair 1 -> 2, just below 10 % of the 600 s median (60 s) -> implausible
        (1, 2, "2022-03-31 11:00", 59, False, True),
    ]
    trips = make_trips(rows)

    flags = implausible_short_cross_trips(trips).tail(3).tolist()

    assert flags == [True, False, True]


def test_short_cross_trips_with_minute_resolution_are_kept():
    reference = [normal_trip(f"2022-03-{day:02d} 09:00") for day in range(1, 31)]
    trips = make_trips(reference + [(1, 2, "2022-03-31 10:00", 0, False, True)])
    # Minute-resolution files cannot measure short durations: separate review, never excluded here
    trips.loc[trips.index[-1], "datetime_format"] = "slash_dayfirst"

    assert not implausible_short_cross_trips(trips).iloc[-1]


def test_station_age_cohort_and_event_marks():
    trips = make_trips(
        [
            # Station 1 first seen in 2022; station 5 first seen in 2026 (new cohort)
            normal_trip("2022-01-01 10:00"),
            normal_trip("2022-01-02 10:00"),
            normal_trip("2026-02-22 10:00", origin=5, destination=6),
        ]
    )

    data = build_processed(trips).set_index(["station_id", "hour_start"])

    assert data.loc[(1, pd.Timestamp("2022-01-02 10:00")), "station_age_days"] == 1
    assert not data.loc[(1, pd.Timestamp("2022-01-02 10:00")), "is_new_station_cohort"]
    assert data.loc[(5, pd.Timestamp("2026-02-22 10:00")), "is_new_station_cohort"]
    # 2026-02-22 is a documented extraordinary event: marked, not removed
    assert data.loc[(5, pd.Timestamp("2026-02-22 10:00")), "is_extraordinary_event"]


def test_output_columns_and_unique_keys():
    data = build_processed(make_trips([normal_trip("2022-01-01 10:00")]))

    assert list(data.columns) == [
        "station_id",
        "hour_start",
        "departures",
        "hour",
        "day_of_week",
        "is_weekend",
        "month",
        "station_age_days",
        "is_new_station_cohort",
        "is_extraordinary_event",
    ]
    assert not data.duplicated(["station_id", "hour_start"]).any()
