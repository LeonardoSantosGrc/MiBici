"""Synthetic drift scenarios of the drift contract, calibrated with the real data.

Every scenario starts from the natural current period (August-September 2026) and has three
levels (mild, moderate, strong). No magnitude is typed by hand: ``calibrate`` derives all of
them from processed_v1, and notebook 4.01 shows each derivation.

| Scenario | What changes | Calibration |
|---|---|---|
| S1 demand shift | departures x factor from the current start, features recomputed | year-over-year change of monthly departures |
| S2 station expansion | share of new-station rows (resampling) | monthly growth of the 2026 cohort |
| S3 data failure | ``t_168`` arrives as 0 for a share of stations | one station, the 2026 cohort size, all stations |
| S4 concept change | departures fall in some hours, inputs unchanged | drop on the extraordinary-event days |

The control is a bootstrap sample of the reference period; it must not raise an alarm.
Every random step uses ``mibici.config.RANDOM_STATE``.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from mibici.config import HISTORY_WEEKS, RANDOM_STATE, SPLITS
from mibici.features import WEEK, add_history_features

LEVELS = ("mild", "moderate", "strong")
# Reference of the input distributions: the latest data the champion saw (validation split)
REFERENCE_PERIOD = SPLITS["validation"]
# Current "production" period: the test split, never seen by the champion
CURRENT_PERIOD = SPLITS["test"]
HISTORY_COLUMNS = ["avg_4_weeks", "t_168", "history_source"]

# S1: quantiles of |year-over-year change| used as mild, moderate and strong
DEMAND_QUANTILES = {"mild": 0.5, "moderate": 0.9, "strong": 1.0}
# S2: months of cohort growth, at the observed monthly pace, added to the current share
EXPANSION_MONTHS = {"mild": 1, "moderate": 3, "strong": 6}
# S4: share of the event-day drop applied to the selected hours
CONCEPT_FRACTIONS = {"mild": 0.25, "moderate": 0.5, "strong": 1.0}


@dataclass(frozen=True)
class Scenario:
    """Rows of one scenario and level, with the magnitude that produced them."""

    name: str
    level: str
    magnitude: float
    description: str
    rows: pd.DataFrame


def period_rows(df: pd.DataFrame, period: tuple[str, str]) -> pd.DataFrame:
    """Rows whose hour_start falls in [start, end)."""
    start, end = period
    return df[(df["hour_start"] >= start) & (df["hour_start"] < end)]


# ---- Calibration: every magnitude comes from the data -------------------------------------


def monthly_departures(df: pd.DataFrame) -> pd.Series:
    """Total departures per calendar month."""
    return df.groupby(df["hour_start"].dt.to_period("M"))["departures"].sum()


def year_over_year_change(df: pd.DataFrame) -> pd.Series:
    """Departures of each month divided by the same month a year earlier, minus 1."""
    monthly = monthly_departures(df)
    previous = monthly.copy()
    previous.index = previous.index + 12  # align each month with the one 12 months later
    return (monthly / previous - 1).dropna()


def demand_factors(df: pd.DataFrame) -> dict[str, float]:
    """S1: 1 + quantile of |year-over-year change| (median, 90th percentile, maximum).

    The shift is an increase: the largest observed change is the growth of September 2026.
    """
    change = year_over_year_change(df).abs()
    return {level: float(1 + change.quantile(q)) for level, q in DEMAND_QUANTILES.items()}


def cohort_share_by_month(df: pd.DataFrame) -> pd.Series:
    """Share of rows of the 2026 new-station cohort in each month."""
    return df.groupby(df["hour_start"].dt.to_period("M"))["is_new_station_cohort"].mean()


def expansion_shares(df: pd.DataFrame, current_period=CURRENT_PERIOD) -> dict[str, float]:
    """S2: current cohort share plus 1, 3 and 6 months of growth at the observed pace.

    The pace is the mean monthly increase of the share, from the first month with cohort
    rows to the last month of the data.
    """
    by_month = cohort_share_by_month(df)
    growing = by_month[by_month > 0]
    pace = (growing.iloc[-1] - growing.iloc[0]) / max(len(growing) - 1, 1)
    current = period_rows(df, current_period)["is_new_station_cohort"].mean()
    return {
        level: float(min(current + months * pace, 1.0))
        for level, months in EXPANSION_MONTHS.items()
    }


def failure_shares(df: pd.DataFrame, current_period=CURRENT_PERIOD) -> dict[str, float]:
    """S3: share of stations whose ``t_168`` arrives as 0.

    One station (an isolated failure), as many stations as the 2026 cohort (the largest
    group the network added at once, a plausible separate feed) and every station.
    """
    current = period_rows(df, current_period)
    stations = current["station_id"].nunique()
    cohort = current.loc[current["is_new_station_cohort"], "station_id"].nunique()
    return {"mild": 1 / stations, "moderate": cohort / stations, "strong": 1.0}


def event_hour_ratios(df: pd.DataFrame) -> pd.Series:
    """Observed / expected departures per hour on the extraordinary-event days.

    The expected value is ``avg_4_weeks``, the four-week history of the same station and hour.
    """
    events = df[df["is_extraordinary_event"]]
    if events.empty:
        raise ValueError("The data has no extraordinary-event rows to calibrate S4.")
    grouped = events.groupby("hour")
    return grouped["departures"].sum() / grouped["avg_4_weeks"].sum()


def concept_hours(ratios: pd.Series, daily_ratio: float) -> list[int]:
    """S4 hours: those whose event-day drop was larger than the drop of the whole day."""
    return sorted(int(h) for h in ratios.index[ratios < daily_ratio])


def concept_factors(df: pd.DataFrame) -> dict[str, dict[int, float]]:
    """S4: per selected hour, 1 - fraction * (1 - event-day ratio of that hour)."""
    ratios = event_hour_ratios(df)
    events = df[df["is_extraordinary_event"]]
    daily = events["departures"].sum() / events["avg_4_weeks"].sum()
    hours = concept_hours(ratios, daily)
    return {
        level: {h: float(1 - fraction * (1 - ratios[h])) for h in hours}
        for level, fraction in CONCEPT_FRACTIONS.items()
    }


def calibrate(df: pd.DataFrame, current_period=CURRENT_PERIOD) -> dict[str, dict]:
    """Magnitudes of S1 to S4 by level, from the modeling table (processed_v1 + features)."""
    return {
        "S1": demand_factors(df),
        "S2": expansion_shares(df, current_period),
        "S3": failure_shares(df, current_period),
        "S4": concept_factors(df),
    }


# ---- Scenario builders --------------------------------------------------------------------


def history_window(df: pd.DataFrame, current_period=CURRENT_PERIOD) -> pd.DataFrame:
    """Rows from HISTORY_WEEKS weeks before the current start to its end.

    Enough to recompute the features of every current row: every lag is at most
    HISTORY_WEEKS weeks, so the result equals the features of the full table.
    """
    start = pd.Timestamp(current_period[0]) - HISTORY_WEEKS * WEEK
    return period_rows(df, (str(start), current_period[1]))


def bootstrap_control(reference: pd.DataFrame, seed: int = RANDOM_STATE) -> pd.DataFrame:
    """Control: rows of the reference drawn with replacement (same size)."""
    return reference.sample(len(reference), replace=True, random_state=seed)


def demand_shift(window: pd.DataFrame, factor: float, current_period=CURRENT_PERIOD):
    """S1: departures x ``factor`` from the current start, then the features recomputed.

    The history of the first weeks still mixes unchanged weeks, as it would in production.
    Departures become expected counts (floats); the metrics accept them.
    """
    base = window.drop(columns=HISTORY_COLUMNS, errors="ignore").copy()
    shifted = base["hour_start"] >= current_period[0]
    base["departures"] = base["departures"].astype(float)
    base.loc[shifted, "departures"] *= factor
    return period_rows(add_history_features(base), current_period).reset_index(drop=True)


def station_expansion(current: pd.DataFrame, share: float, seed: int = RANDOM_STATE):
    """S2: resample the current rows so the new-station cohort has ``share`` of them.

    Same number of rows. Cohort rows are all kept and the missing ones drawn with
    replacement; the other rows are drawn without replacement.
    """
    rng = np.random.default_rng(seed)
    cohort = current.index[current["is_new_station_cohort"].to_numpy(dtype=bool)]
    others = current.index.difference(cohort)
    n_cohort = round(share * len(current))
    if n_cohort < len(cohort) or len(current) - n_cohort > len(others):
        raise ValueError(f"share {share:.3f} is below the current cohort share.")
    extra = rng.choice(cohort, n_cohort - len(cohort), replace=True)
    kept = rng.choice(others, len(current) - n_cohort, replace=False)
    picked = np.concatenate([cohort.to_numpy(), extra, kept])
    return current.loc[picked].sort_values(["station_id", "hour_start"], ignore_index=True)


def data_failure(current: pd.DataFrame, share: float, seed: int = RANDOM_STATE):
    """S3: ``t_168`` = 0 for every row of a random ``share`` of the stations (at least one)."""
    rng = np.random.default_rng(seed)
    stations = np.sort(current["station_id"].unique())
    n = max(1, round(share * len(stations)))
    broken = rng.choice(stations, n, replace=False)
    rows = current.copy()
    rows.loc[rows["station_id"].isin(broken), "t_168"] = 0.0
    return rows.reset_index(drop=True)


def concept_change(current: pd.DataFrame, hour_factors: dict[int, float]):
    """S4: departures of the selected hours x their factor; the inputs stay the same."""
    rows = current.copy()
    factor = rows["hour"].map(hour_factors).fillna(1.0).to_numpy(dtype=float)
    rows["departures"] = rows["departures"].to_numpy(dtype=float) * factor
    return rows.reset_index(drop=True)


def build_scenarios(df: pd.DataFrame, calibration: dict, current_period=CURRENT_PERIOD):
    """Yield every scenario and level (S1 to S4 x mild, moderate, strong) as a Scenario."""
    window = history_window(df, current_period)
    current = period_rows(df, current_period).reset_index(drop=True)
    for level in LEVELS:
        factor = calibration["S1"][level]
        yield Scenario(
            "S1_demand_shift",
            level,
            factor,
            f"departures x {factor:.3f} from {current_period[0]}, features recomputed",
            demand_shift(window, factor, current_period),
        )
    for level in LEVELS:
        share = calibration["S2"][level]
        yield Scenario(
            "S2_station_expansion",
            level,
            share,
            f"new-station cohort = {share:.1%} of the rows",
            station_expansion(current, share),
        )
    for level in LEVELS:
        share = calibration["S3"][level]
        yield Scenario(
            "S3_data_failure",
            level,
            share,
            f"t_168 = 0 for {share:.1%} of the stations",
            data_failure(current, share),
        )
    for level in LEVELS:
        factors = calibration["S4"][level]
        hours = sorted(factors)
        mean_factor = float(np.mean(list(factors.values())))
        yield Scenario(
            "S4_concept_change",
            level,
            mean_factor,
            f"departures of hours {hours[0]}-{hours[-1]} x {mean_factor:.3f} on average",
            concept_change(current, factors),
        )
