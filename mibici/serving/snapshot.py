"""History snapshot and feature table the API uses to forecast a whole day from its date.

    make serving-snapshot

The model needs ``avg_4_weeks`` and ``t_168``, which come from earlier weeks of departures. The
snapshot keeps the rows of processed_v1 from ``HISTORY_WEEKS`` weeks before
``SERVING_FIRST_DATE`` to the end of the data. ``build_feature_table`` computes the features
with the same ``add_history_features`` used in training (no second implementation), for every
day from ``SERVING_FIRST_DATE`` to the day after the data ends (D+1).

D+1 rows are built from the stations active on the last day of the data: same stations and
hours, one more day of age, calendar columns from the new date and unknown departures. Every
lag is at least one week, so all their features are known at the 22:00 cutoff of the last day.
"""

import json
from pathlib import Path

from loguru import logger
import pandas as pd
import typer

from mibici.config import (
    FEATURES,
    HISTORY_WEEKS,
    SERVING_FIRST_DATE,
    SNAPSHOT_INFO_PATH,
    SNAPSHOT_PATH,
    STATION_HOUR_PATH,
)
from mibici.features import WEEK, add_history_features
from mibici.tracking import dataframe_digest

app = typer.Typer()

# Columns the snapshot keeps: keys, target and the inputs that are not history features
SNAPSHOT_COLUMNS = [
    "station_id",
    "hour_start",
    "departures",
    "hour",
    "day_of_week",
    "is_weekend",
    "month",
    "station_age_days",
]
DAY = pd.Timedelta(days=1)


def build_snapshot(data: pd.DataFrame, first_date: str = SERVING_FIRST_DATE) -> pd.DataFrame:
    """Rows of processed_v1 from ``HISTORY_WEEKS`` weeks before ``first_date`` onwards."""
    # avg_4_weeks of the first served day looks up to HISTORY_WEEKS weeks back
    start = pd.Timestamp(first_date) - HISTORY_WEEKS * WEEK
    if data["hour_start"].min() > start:
        raise ValueError(f"The data starts after {start.date()}; the snapshot needs it.")
    rows = data.loc[data["hour_start"] >= start, SNAPSHOT_COLUMNS]
    return rows.sort_values(["station_id", "hour_start"], ignore_index=True)


def last_data_date(snapshot: pd.DataFrame) -> pd.Timestamp:
    """Last day with observed departures in the snapshot."""
    return snapshot["hour_start"].max().normalize()


def forecast_range(snapshot: pd.DataFrame) -> tuple[pd.Timestamp, pd.Timestamp]:
    """First and last date the API can forecast: from the first served day to D+1."""
    first = snapshot["hour_start"].min().normalize() + HISTORY_WEEKS * WEEK
    return first, last_data_date(snapshot) + DAY


def next_day_rows(snapshot: pd.DataFrame) -> pd.DataFrame:
    """Rows of the day after the data (D+1): stations and hours of the last day, one day on."""
    last = snapshot[snapshot["hour_start"].dt.normalize() == last_data_date(snapshot)]
    rows = last[["station_id", "hour_start", "station_age_days"]].copy()
    rows["hour_start"] = rows["hour_start"] + DAY
    rows["station_age_days"] = rows["station_age_days"] + 1
    # Calendar columns come from the new date, with the same types as processed_v1
    rows["hour"] = rows["hour_start"].dt.hour.astype("int32")
    rows["day_of_week"] = rows["hour_start"].dt.day_of_week.astype("int32")
    rows["is_weekend"] = rows["day_of_week"] >= 5
    rows["month"] = rows["hour_start"].dt.month.astype("int32")
    rows["departures"] = float("nan")  # unknown: the day has not happened yet
    return rows[SNAPSHOT_COLUMNS]


def build_feature_table(snapshot: pd.DataFrame) -> pd.DataFrame:
    """Rows with the contract features of every date the API can forecast, D+1 included."""
    with_next_day = pd.concat([snapshot, next_day_rows(snapshot)], ignore_index=True)
    table = add_history_features(with_next_day)
    first, _ = forecast_range(snapshot)
    table = table[table["hour_start"] >= first].sort_values(
        ["station_id", "hour_start"], ignore_index=True
    )
    # Data contract of the model input: every feature present and known
    missing = table[FEATURES].isna().sum()
    if missing.any() or (table["history_source"] == "none").any():
        raise ValueError(f"Features could not be built: {missing[missing > 0].to_dict()}")
    return table


def day_rows(table: pd.DataFrame, date) -> pd.DataFrame:
    """Rows of one date in the feature table; fails with the range it can forecast."""
    day = pd.Timestamp(date).normalize()
    first, last = table["hour_start"].min().normalize(), table["hour_start"].max().normalize()
    if not first <= day <= last:
        raise ValueError(
            f"Forecasts are available from {first.date()} to {last.date()} (the day after the "
            f"latest data). To forecast {day.date()}, refresh the data up to the day before it."
        )
    return table[table["hour_start"].dt.normalize() == day]


def snapshot_info(snapshot: pd.DataFrame) -> dict:
    """Fingerprint and ranges of a snapshot; same data gives the same result on any machine."""
    first, last = forecast_range(snapshot)
    return {
        "digest": dataframe_digest(snapshot),
        "rows": len(snapshot),
        "stations": int(snapshot["station_id"].nunique()),
        "snapshot_start": str(snapshot["hour_start"].min().date()),
        "last_data_date": str(last_data_date(snapshot).date()),
        "first_forecast_date": str(first.date()),
        "last_forecast_date": str(last.date()),
    }


def check_reference(info: dict, reference_path: Path) -> bool:
    """Compare with the committed reference; write it if missing. True if it was created."""
    if not reference_path.exists():
        reference_path.write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
        return True
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    if reference != info:
        raise ValueError(
            f"This snapshot differs from the reference in {reference_path.name}: "
            f"{reference} vs {info}. Rebuild processed_v1 with make processed."
        )
    return False


@app.command()
def main() -> None:
    """Build the history snapshot from processed_v1 and check it against snapshot.json."""
    # processed_v1 as built in Phase 1; the snapshot keeps base columns only
    snapshot = build_snapshot(pd.read_parquet(STATION_HOUR_PATH))
    # Building the features once proves the snapshot is enough for every served date
    build_feature_table(snapshot)
    SNAPSHOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    snapshot.to_parquet(SNAPSHOT_PATH, index=False)

    info = snapshot_info(snapshot)
    created = check_reference(info, SNAPSHOT_INFO_PATH)
    status = "reference created" if created else "matches the reference"
    logger.success(
        f"Snapshot {info['digest']} ({info['rows']:,} rows, {status}): forecasts from "
        f"{info['first_forecast_date']} to {info['last_forecast_date']} -> {SNAPSHOT_PATH}"
    )


if __name__ == "__main__":
    app()
