"""Station-history features shared by every Phase 2 model (notebook 3.01, section 7.2).

Both features use only departures from at least one week (168 h) earlier, so they are known
at the 22:00 cutoff of day d for every hour of day d+1.

- ``avg_4_weeks``: mean departures of the same station at the same hour over the previous
  ``HISTORY_WEEKS`` weeks (same weekday; weeks without a row are skipped). A station without
  its own history falls back to the mean of young stations (< ``YOUNG_STATION_DAYS`` days old)
  at those same hours, and then to the mean of all stations.
- ``t_168``: departures of the same station and hour one week earlier. When that row does not
  exist (the station was not open yet) it takes ``avg_4_weeks``, so every model receives the
  same complete table (team decision of 2026-11-05).

``history_source`` records which level produced ``avg_4_weeks``; it is for evaluation, not a
model input. Rows are matched by (station_id, hour_start), never by position, so the result
does not depend on the order of the input rows.
"""

import numpy as np
import pandas as pd

from mibici.config import HISTORY_WEEKS, YOUNG_STATION_DAYS

WEEK = pd.Timedelta(hours=168)


def weekly_mean(weekly_values: list[np.ndarray]) -> np.ndarray:
    """Row-wise mean of several weeks, skipping missing weeks (NaN when all are missing)."""
    stacked = np.vstack(weekly_values)  # one row per week, one column per table row
    available = (~np.isnan(stacked)).sum(axis=0)  # how many weeks have a value
    total = np.nansum(stacked, axis=0)  # sum of the weeks that have a value
    # Divide only where at least one week exists; the rest stay NaN
    return np.divide(total, available, out=np.full(total.shape, np.nan), where=available > 0)


def station_weeks_back(df: pd.DataFrame, weeks: int) -> list[np.ndarray]:
    """Departures of the same station and hour 1..`weeks` weeks earlier (NaN if no row)."""
    by_key = df.set_index(["station_id", "hour_start"])["departures"]
    if not by_key.index.is_unique:
        raise ValueError("Expected one row per (station_id, hour_start).")
    lags = []
    for k in range(1, weeks + 1):
        # Look up (same station, hour_start - k weeks) by key, not by row position
        keys = pd.MultiIndex.from_arrays([df["station_id"], df["hour_start"] - k * WEEK])
        lags.append(by_key.reindex(keys).to_numpy(dtype=float))
    return lags


def group_weeks_back(hourly: pd.Series, hour_start: pd.Series, weeks: int) -> list[np.ndarray]:
    """Values of an hourly group mean 1..`weeks` weeks before each row (NaN if no value)."""
    return [
        hourly.reindex(hour_start - k * WEEK).to_numpy(dtype=float) for k in range(1, weeks + 1)
    ]


def add_history_features(
    df: pd.DataFrame,
    history_weeks: int = HISTORY_WEEKS,
    young_station_days: int = YOUNG_STATION_DAYS,
) -> pd.DataFrame:
    """Return a copy of processed_v1 with ``avg_4_weeks``, ``t_168`` and ``history_source``.

    Compute it on the full table before splitting: the history of a validation or test row
    comes from the weeks before it, which are known at its cutoff.
    """
    own_weeks = station_weeks_back(df, history_weeks)
    own_mean = weekly_mean(own_weeks)

    # Fallback levels: mean departures per hour of young stations and of all stations
    young = df["station_age_days"] < young_station_days
    young_hourly = df[young].groupby("hour_start")["departures"].mean()
    all_hourly = df.groupby("hour_start")["departures"].mean()
    young_mean = weekly_mean(group_weeks_back(young_hourly, df["hour_start"], history_weeks))
    all_mean = weekly_mean(group_weeks_back(all_hourly, df["hour_start"], history_weeks))

    # Own history first, then young stations, then all stations
    has_own, has_young = ~np.isnan(own_mean), ~np.isnan(young_mean)
    avg_4_weeks = np.where(has_own, own_mean, np.where(has_young, young_mean, all_mean))
    source = np.select(
        [has_own, has_young, ~np.isnan(all_mean)],
        ["own_history", "young_stations", "all_stations"],
        default="none",  # only the first week of the data: nothing earlier exists
    )

    # Last week's value; if the station had no row then, use avg_4_weeks
    t_168 = np.where(np.isnan(own_weeks[0]), avg_4_weeks, own_weeks[0])

    return df.assign(
        avg_4_weeks=avg_4_weeks,
        t_168=t_168,
        history_source=pd.Categorical(
            source, categories=["own_history", "young_stations", "all_stations", "none"]
        ),
    )
