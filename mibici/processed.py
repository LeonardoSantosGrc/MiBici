"""Build processed_v1: the station x hour dataset for modelling, after the PR2 decisions.

Derived entirely from interim_v1 (trips). Every rule below comes from a decision
documented with evidence in notebook 1.04, section 8:

- unit: station x hour, departures counted at the origin station (decision 1);
- window: from 2022-01-01, so 2020-2021 are left out (decisions 4-6);
- service hours only: 01:00-04:59 is the scheduled closure (decision 11);
- short same-station trips and implausible short cross-station trips are not
  counted as departures; they stay in interim (decisions 2 and 3);
- extraordinary events and the 2026 station cohort are marked, not removed
  (decisions 7 and 8).
"""

import json

from loguru import logger
import pandas as pd
import typer

from mibici.config import INTERIM_DATA_DIR, PROCESSED_DATA_DIR, REPORTS_DIR
from mibici.station_hour import (
    add_calendar_columns,
    build_hour_grid,
    count_departures,
    station_active_days,
)
from mibici.tracking import check_tracking_server, git_info, log_dataset_run

app = typer.Typer()

DATASET_VERSION = "processed_v1"
TRIPS_DIR = INTERIM_DATA_DIR / "trips"
PROCESSED_PATH = PROCESSED_DATA_DIR / "station_hour_v1.parquet"
PROCESSED_REPORTS_DIR = REPORTS_DIR / "processed"

# --- Decisions (notebook 1.04, section 8) ---
# Decision 0: forecasts for day d+1 use only data available before 22:00 of day d.
# Recorded here as the prediction contract; it constrains Phase 2 lag features.
PREDICTION_CUTOFF_HOUR = 22
# Decisions 4-6: first version of processed_v1 starts in January 2022 (2020-2021 excluded)
WINDOW_START = pd.Timestamp("2022-01-01")
# Decision 11: scheduled closure; service hours are 00:00-00:59 and 05:00-23:59
CLOSED_HOURS = (1, 2, 3, 4)
# Decision 3: a short cross-station trip is implausible when its pair has enough reference
# trips and it lasts less than this fraction of the pair's median duration
SHORT_CROSS_MIN_PAIR_TRIPS = 30
SHORT_CROSS_MAX_MEDIAN_RATIO = 0.10
# Decision 10: fixed cohort of stations whose first observed activity is from 2026
NEW_STATION_COHORT_SINCE = pd.Timestamp("2026-01-01")
# Decision 7: extraordinary, non-recurring events identified by notebook 1.02 (marked, not removed)
EXTRAORDINARY_EVENTS = {
    "2015-10-23": "Hurricane Patricia makes landfall in Jalisco",
    "2026-02-22": "Road blockades in Jalisco",
    "2026-02-23": "Road blockades in Jalisco",
}

TRIP_COLUMNS = [
    "started_at",
    "origin_station_id",
    "destination_station_id",
    "duration_s",
    "datetime_format",
    "flag_short_same_station",
    "flag_short_cross_station",
]
OUTPUT_COLUMNS = [
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


def implausible_short_cross_trips(
    trips: pd.DataFrame,
    min_pair_trips: int = SHORT_CROSS_MIN_PAIR_TRIPS,
    max_median_ratio: float = SHORT_CROSS_MAX_MEDIAN_RATIO,
) -> pd.Series:
    """Flag short cross-station trips far shorter than their origin-destination pair (decision 3).

    Reference: trips with positive duration of the same ordered pair (origin -> destination)
    and year. Trips without enough reference, or recorded with minute resolution, are kept.
    """
    keys = [
        trips["origin_station_id"],
        trips["destination_station_id"],
        trips["started_at"].dt.year,
    ]
    # Only positive durations form the reference; others become missing and are ignored
    reference = trips["duration_s"].where(trips["duration_s"] > 0).astype("float64")
    pair_median = reference.groupby(keys).transform("median")
    pair_trips = reference.groupby(keys).transform("count")

    minute_resolution = trips["datetime_format"] == "slash_dayfirst"
    implausible = (
        trips["flag_short_cross_station"]
        & ~minute_resolution
        & (pair_trips >= min_pair_trips)
        & (trips["duration_s"] < max_median_ratio * pair_median)
    )
    return implausible.fillna(False).astype(bool)


def excluded_trips(trips: pd.DataFrame) -> pd.Series:
    """Trips that are not counted as departures in processed_v1 (decisions 2 and 3)."""
    return trips["flag_short_same_station"].astype(bool) | implausible_short_cross_trips(trips)


def build_processed(trips: pd.DataFrame) -> pd.DataFrame:
    """Station x hour table for modelling, with the PR2 decisions applied."""
    # Active windows come from ALL trips: they describe when a station existed,
    # independently of which trips are later counted as departures
    active_days = station_active_days(trips)
    first_day = active_days.groupby("station_id")["day"].min()

    grid = build_hour_grid(active_days)
    in_window = grid["hour_start"] >= WINDOW_START
    in_service = ~grid["hour_start"].dt.hour.isin(CLOSED_HOURS)
    grid = grid[in_window & in_service]

    # Departures exclude short same-station and implausible short cross-station trips
    counts = count_departures(trips[~excluded_trips(trips).to_numpy()])
    data = grid.merge(counts, on=["station_id", "hour_start"], how="left")
    data["departures"] = data["departures"].fillna(0).astype("int32")
    data["station_id"] = data["station_id"].astype("int32")

    data = add_calendar_columns(data)
    station_first_day = data["station_id"].map(first_day)
    # Days since the first observed activity of the station (not its official opening date)
    data["station_age_days"] = (data["date"] - station_first_day).dt.days.astype("int32")
    data["is_new_station_cohort"] = station_first_day >= NEW_STATION_COHORT_SINCE
    data["is_extraordinary_event"] = data["date"].isin(pd.to_datetime(list(EXTRAORDINARY_EVENTS)))

    return data[OUTPUT_COLUMNS].sort_values(["station_id", "hour_start"]).reset_index(drop=True)


def processed_summary(data: pd.DataFrame, trips: pd.DataFrame) -> dict:
    """Counts that describe processed_v1 and reconcile it with the interim trips."""
    in_window = trips["started_at"] >= WINDOW_START
    excluded = excluded_trips(trips)
    closed = trips["started_at"].dt.hour.isin(CLOSED_HOURS)
    departures = data["departures"]
    return {
        "dataset_version": DATASET_VERSION,
        "n_rows": len(data),
        "n_stations": int(data["station_id"].nunique()),
        "hour_start_min": str(data["hour_start"].min()),
        "hour_start_max": str(data["hour_start"].max()),
        # --- Reconciliation with interim trips in the window ---
        "trips_in_window": int(in_window.sum()),
        "trips_excluded_short_same_station": int(
            (in_window & trips["flag_short_same_station"]).sum()
        ),
        "trips_excluded_short_cross_implausible": int(
            (in_window & implausible_short_cross_trips(trips)).sum()
        ),
        "trips_in_closed_hours_not_counted": int((in_window & ~excluded & closed).sum()),
        "departures_total": int(departures.sum()),
        # --- Target distribution ---
        "zero_share": round(float((departures == 0).mean()), 4),
        "departures_mean": round(float(departures.mean()), 4),
        "departures_p50": float(departures.quantile(0.5)),
        "departures_p90": float(departures.quantile(0.9)),
        "departures_p99": float(departures.quantile(0.99)),
        "departures_max": int(departures.max()),
        # --- Marks for evaluation ---
        "rows_new_station_cohort": int(data["is_new_station_cohort"].sum()),
        "rows_extraordinary_event": int(data["is_extraordinary_event"].sum()),
        "duplicated_keys": int(data.duplicated(["station_id", "hour_start"]).sum()),
    }


def decision_params() -> dict[str, object]:
    """The decisions applied, logged as MLflow parameters."""
    return {
        "source_dataset": "interim_v1",
        "unit": "station_x_hour",
        "prediction_cutoff_hour": PREDICTION_CUTOFF_HOUR,
        "window_start": str(WINDOW_START.date()),
        "closed_hours": "|".join(str(h) for h in CLOSED_HOURS),
        "excluded_from_target": "short_same_station|implausible_short_cross_station",
        "short_cross_min_pair_trips": SHORT_CROSS_MIN_PAIR_TRIPS,
        "short_cross_max_median_ratio": SHORT_CROSS_MAX_MEDIAN_RATIO,
        "new_station_cohort_since": str(NEW_STATION_COHORT_SINCE.date()),
        "extraordinary_events": "|".join(EXTRAORDINARY_EVENTS),
        "capacity_used": False,
    }


@app.command()
def main(
    track: bool = typer.Option(
        True,
        help="Log this dataset version to MLflow (requires `make mlflow`).",
    ),
) -> None:
    """Build data/processed/station_hour_v1.parquet (processed_v1) from interim_v1."""
    # Commit of the code that runs, taken before anything is written
    if track:
        check_tracking_server()
        git = git_info()
        if git["git_dirty"] == "true":
            logger.warning("Uncommitted changes: the MLflow run will be tagged git_dirty=true")

    trips = pd.read_parquet(TRIPS_DIR, columns=TRIP_COLUMNS)
    data = build_processed(trips)

    # Data (outside Git) and small evidence files (versioned)
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    data.to_parquet(PROCESSED_PATH, index=False)

    PROCESSED_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    summary = processed_summary(data, trips)
    schema = {column: str(dtype) for column, dtype in data.dtypes.items()}
    files = {
        "summary.json": summary,
        "schema.json": schema,
        "decisions.json": {**decision_params(), "extraordinary_events": EXTRAORDINARY_EVENTS},
    }
    for name, content in files.items():
        (PROCESSED_REPORTS_DIR / name).write_text(
            json.dumps(content, indent=2, ensure_ascii=False)
        )

    if track:
        metrics = {key: value for key, value in summary.items() if isinstance(value, (int, float))}
        params = decision_params()
        params.update(
            hour_start_min=summary["hour_start_min"], hour_start_max=summary["hour_start_max"]
        )
        log_dataset_run(
            run_name=f"build_{DATASET_VERSION}",
            tags={"pipeline_stage": "processed", "dataset_version": DATASET_VERSION, **git},
            params=params,
            metrics=metrics,
            artifacts=sorted(PROCESSED_REPORTS_DIR.iterdir()),
            datasets={DATASET_VERSION: (data, str(PROCESSED_PATH))},
        )

    logger.success(
        f"processed_v1: {summary['n_rows']:,} rows, {summary['n_stations']} stations, "
        f"{summary['zero_share']:.1%} zeros. Output: {PROCESSED_PATH}"
    )


if __name__ == "__main__":
    app()
