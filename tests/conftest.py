"""Fixtures shared by several test files."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def processed_table() -> pd.DataFrame:
    """processed_v1-like table: hours 0 and 8 from June 1 to August 15, 2026.

    Station 1 exists the whole period; station 2 opens on July 20, so its first weeks use
    the young-station fallback of add_history_features. With four weeks of history, the
    first date it can serve is July 15 and the day after the data (D+1) is August 16.
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
