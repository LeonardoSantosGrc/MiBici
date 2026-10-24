"""Station x hour base: departures per station and hour, with valid zeros only.

Unit of the business problem: trips started per station and hour. Each row is
one hour in which the station was active; ``departures`` is the number of trips
that started there in that hour (0 if none). Hours before a station's first
trip, after its last trip, or inside a month with no trips at all are not
created, so they never appear as false zeros.

This is a technical base for the EDA and for building processed_v1; no flag
is excluded here.
"""

import numpy as np
import pandas as pd

from mibici.stations import station_month_activity


def station_active_days(trips: pd.DataFrame) -> pd.DataFrame:
    """List each (station, day) in which the station was active.

    Window: from the day of the station's first trip to the day of its last trip
    (as origin or destination). Months with no trips at all inside that window
    are removed completely (inactive periods).
    """
    # Un mismo viaje cuenta para la estación de origen y para la de destino
    day = trips["started_at"].dt.normalize()
    events = pd.concat(
        [
            pd.DataFrame({"station_id": trips["origin_station_id"], "day": day}),
            pd.DataFrame({"station_id": trips["destination_station_id"], "day": day}),
        ]
    ).dropna()

    # --- Bordes por día: primer y último viaje de cada estación ---
    # Por día y no por mes: 285 estaciones abrieron después del día 10 de su primer mes
    bounds = events.groupby("station_id")["day"].agg(first_day="min", last_day="max")

    # Todos los días entre el primero y el último de cada estación (486 estaciones -> bucle corto)
    days = pd.concat(
        [
            pd.DataFrame(
                {"station_id": station, "day": pd.date_range(row.first_day, row.last_day)}
            )
            for station, row in bounds.iterrows()
        ],
        ignore_index=True,
    )

    # --- Huecos por mes: solo se conservan los días de meses con al menos un viaje ---
    days["month"] = days["day"].dt.to_period("M")
    active_months = station_month_activity(trips)
    days = days.merge(active_months, on=["station_id", "month"], how="inner")

    return days.drop(columns="month")


def build_hour_grid(active_days: pd.DataFrame) -> pd.DataFrame:
    """Expand each active (station, day) into its 24 hours."""
    # Cada día se repite 24 veces (index.repeat) y a cada copia se le suma 0, 1, ..., 23 h.
    # np.tile arma la secuencia 0..23 repetida una vez por día, en el mismo orden.
    grid = active_days.loc[active_days.index.repeat(24)].reset_index(drop=True)
    hour_offset = pd.to_timedelta(np.tile(np.arange(24), len(active_days)), unit="h")
    grid["hour_start"] = grid["day"] + hour_offset
    return grid[["station_id", "hour_start"]]


def count_departures(trips: pd.DataFrame) -> pd.DataFrame:
    """Count trips per origin station and start hour (only hours with departures)."""
    # floor("h") lleva cada inicio al comienzo de su hora: 08:47:13 -> 08:00:00
    hour_start = trips["started_at"].dt.floor("h")
    counts = (
        pd.DataFrame({"station_id": trips["origin_station_id"], "hour_start": hour_start})
        .dropna()
        .value_counts()
        .rename("departures")
        .reset_index()
    )
    return counts


def build_station_hour_base(trips: pd.DataFrame) -> pd.DataFrame:
    """Station x hour table with departures, valid zeros and calendar columns."""
    grid = build_hour_grid(station_active_days(trips))
    counts = count_departures(trips)

    # Unión izquierda: toda hora activa se conserva; si no tuvo salidas, queda en 0
    base = grid.merge(counts, on=["station_id", "hour_start"], how="left")
    base["departures"] = base["departures"].fillna(0).astype("int32")
    base["station_id"] = base["station_id"].astype("int32")

    base = add_calendar_columns(base)
    return base.sort_values(["station_id", "hour_start"]).reset_index(drop=True)


def add_calendar_columns(base: pd.DataFrame) -> pd.DataFrame:
    """Add date, hour, day of week, weekend and month derived from hour_start."""
    base["date"] = base["hour_start"].dt.normalize()
    base["hour"] = base["hour_start"].dt.hour
    base["day_of_week"] = base["hour_start"].dt.day_of_week
    base["is_weekend"] = base["day_of_week"].isin([5, 6])
    base["month"] = base["hour_start"].dt.month

    return base
