"""Tests for mibici.features: station-history features (notebook 3.01, section 7.2).

Small synthetic station x hour tables with known answers. One row per station and day at
08:00, so one week earlier is exactly seven rows back for the same station.
"""

import numpy as np
import pandas as pd

from mibici.features import add_history_features


def make_table(stations: dict[int, tuple[str, list[int]]]) -> pd.DataFrame:
    """Station-hour rows at 08:00 from {station_id: (first day, daily departures)}."""
    frames = []
    for station_id, (first_day, departures) in stations.items():
        # One 08:00 row per day, starting on the station's first day
        days = pd.date_range(f"{first_day} 08:00", periods=len(departures), freq="D")
        frames.append(
            pd.DataFrame(
                {
                    "station_id": station_id,
                    "hour_start": days,
                    "departures": departures,
                    # Age in days since the first row: 0, 1, 2, ...
                    "station_age_days": np.arange(len(departures)),
                }
            )
        )
    # All stations in one table, like processed_v1
    return pd.concat(frames, ignore_index=True)


def row(table: pd.DataFrame, station_id: int, day: str) -> pd.Series:
    """The single row of `station_id` at 08:00 of `day`."""
    match = (table["station_id"] == station_id) & (table["hour_start"] == f"{day} 08:00")
    # squeeze() turns the one-row table into a Series (one value per column)
    return table.loc[match].squeeze()


def test_t_168_is_the_same_station_one_week_earlier():
    # Departures equal to the day number: one week earlier is always 7 less
    table = add_history_features(make_table({1: ("2026-01-01", list(range(21)))}))
    assert row(table, 1, "2026-01-15")["t_168"] == 14 - 7
    assert row(table, 1, "2026-01-21")["t_168"] == 20 - 7


def test_avg_4_weeks_is_the_mean_of_the_four_previous_weeks():
    # Day 28 (2026-01-29): same weekday on days 21, 14, 7 and 0
    table = add_history_features(make_table({1: ("2026-01-01", list(range(29)))}))
    assert row(table, 1, "2026-01-29")["avg_4_weeks"] == np.mean([21, 14, 7, 0])
    assert row(table, 1, "2026-01-29")["history_source"] == "own_history"


def test_avg_4_weeks_skips_missing_weeks():
    # Only two earlier weeks exist for day 14 (days 7 and 0)
    table = add_history_features(make_table({1: ("2026-01-01", list(range(15)))}))
    assert row(table, 1, "2026-01-15")["avg_4_weeks"] == np.mean([7, 0])


def test_features_do_not_look_at_the_future():
    # Changing departures on and after a day must not change that day's features
    base = make_table({1: ("2026-01-01", [3] * 35)})
    changed = base.copy()
    changed.loc[changed["hour_start"] >= "2026-01-29", "departures"] = 100
    before = row(add_history_features(base), 1, "2026-01-29")
    after = row(add_history_features(changed), 1, "2026-01-29")
    assert before[["avg_4_weeks", "t_168"]].equals(after[["avg_4_weeks", "t_168"]])


def test_new_station_falls_back_to_young_stations():
    # Station 2 opens on 2026-01-22; station 3 was young (< 28 days) one week earlier
    table = add_history_features(
        make_table(
            {
                1: ("2025-11-01", [50] * 82),  # old station, very different level
                2: ("2026-01-22", [0] * 1),
                3: ("2026-01-10", [6] * 13),
            }
        )
    )
    new = row(table, 2, "2026-01-22")
    assert new["history_source"] == "young_stations"
    # Station 3 had 6 departures on 2026-01-15 and 2026-01-08 does not exist for it
    assert new["avg_4_weeks"] == 6
    # No own row one week earlier: t_168 takes avg_4_weeks
    assert new["t_168"] == new["avg_4_weeks"]


def test_without_young_stations_falls_back_to_all_stations():
    # Station 2 opens when no other station is young: mean of all stations (only station 1)
    table = add_history_features(
        make_table({1: ("2025-11-01", [10] * 82), 2: ("2026-01-22", [0] * 1)})
    )
    new = row(table, 2, "2026-01-22")
    assert new["history_source"] == "all_stations"
    assert new["avg_4_weeks"] == 10


def test_first_week_of_data_has_no_history():
    table = add_history_features(make_table({1: ("2026-01-01", list(range(10)))}))
    # First seven days: there is no earlier week for any station
    first_week = table["hour_start"] < "2026-01-08"
    assert (table.loc[first_week, "history_source"] == "none").all()
    assert table.loc[first_week, ["avg_4_weeks", "t_168"]].isna().all().all()
    # From day 8 on, both features always have a value
    assert table.loc[~first_week, ["avg_4_weeks", "t_168"]].notna().all().all()


def test_result_does_not_depend_on_row_order():
    table = make_table({1: ("2026-01-01", list(range(35))), 2: ("2026-01-10", [4] * 26)})
    ordered = add_history_features(table)
    # Shuffle the rows, compute again and put them back in the original order
    shuffled = add_history_features(table.sample(frac=1, random_state=0)).sort_index()
    pd.testing.assert_frame_equal(ordered, shuffled)
