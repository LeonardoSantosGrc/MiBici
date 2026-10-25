"""Station x hour base: departures per station and hour, with valid zeros only.

Unit of the business problem: trips started per station and hour. Each row is
one hour in which the station was active; ``departures`` is the number of trips
that started there in that hour (0 if none). Hours before a station's first
trip, after its last trip, or inside a month with no trips at all are not
created, so they never appear as false zeros.

This is a technical base for the EDA and for building processed_v1; no flag
is excluded here.
"""

import json

from loguru import logger
import numpy as np
import pandas as pd
import typer

from mibici.config import INTERIM_DATA_DIR, REPORTS_DIR
from mibici.stations import station_month_activity

app = typer.Typer()

# Entrada (viajes de interim) y salidas: la base es intermedia, por eso va a data/interim.
# data/processed queda reservado para processed_v1.
TRIPS_DIR = INTERIM_DATA_DIR / "trips"
STATION_HOUR_PATH = INTERIM_DATA_DIR / "station_hour_base.parquet"
STATION_HOUR_REPORTS_DIR = REPORTS_DIR / "station_hour"

# Solo las columnas que necesita la grilla (lectura ~1 s en lugar de cargar las 21)
TRIP_COLUMNS = ["origin_station_id", "destination_station_id", "started_at"]


def station_active_days(trips: pd.DataFrame) -> pd.DataFrame:
    """List each (station, day) in which the station was active.

    Window: from the day of the station's first trip to the day of its last trip
    (as origin or destination). Months with no trips at all inside that window
    are removed completely (inactive periods).
    """
    # Un mismo viaje cuenta para la estación de origen y para la de destino: una estación
    # que solo recibe bicicletas también está operando ese día.
    # normalize() quita la hora y deja solo el día: 2025-03-01 08:47 -> 2025-03-01 00:00
    day = trips["started_at"].dt.normalize()
    # Se apilan dos tablas (origen y destino) con las mismas columnas: una fila por
    # "evento" de estación. dropna() quita estaciones faltantes.
    events = pd.concat(
        [
            pd.DataFrame({"station_id": trips["origin_station_id"], "day": day}),
            pd.DataFrame({"station_id": trips["destination_station_id"], "day": day}),
        ]
    ).dropna()

    # --- Bordes por día: primer y último viaje de cada estación ---
    # Por día y no por mes: 285 estaciones abrieron después del día 10 de su primer mes.
    # Con bordes por mes, una estación que abrió el 25 tendría ceros falsos del 1 al 24.
    bounds = events.groupby("station_id")["day"].agg(first_day="min", last_day="max")

    # Todos los días entre el primero y el último de cada estación (486 estaciones -> bucle corto).
    # date_range genera la lista de días consecutivos; iterrows recorre la tabla fila por fila
    # y devuelve (station_id, fila con first_day y last_day).
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
    # station_month_activity (del PR1) da las parejas (estación, mes) con actividad.
    # La unión "inner" deja solo los días cuyo (estación, mes) aparece en esa lista:
    # los meses sin viajes desaparecen completos (periodos inactivos).
    days["month"] = days["day"].dt.to_period("M")
    active_months = station_month_activity(trips)
    days = days.merge(active_months, on=["station_id", "month"], how="inner")

    return days.drop(columns="month")


def build_hour_grid(active_days: pd.DataFrame) -> pd.DataFrame:
    """Expand each active (station, day) into its 24 hours."""
    # Cada día se repite 24 veces (index.repeat) y a cada copia se le suma 0, 1, ..., 23 h.
    # np.tile arma la secuencia 0..23 repetida una vez por día, en el mismo orden.
    # Ejemplo con 2 días: filas [d1 x24, d2 x24] + horas [0..23, 0..23]
    grid = active_days.loc[active_days.index.repeat(24)].reset_index(drop=True)
    hour_offset = pd.to_timedelta(np.tile(np.arange(24), len(active_days)), unit="h")
    # Día (a medianoche) + desplazamiento en horas = inicio de cada hora
    grid["hour_start"] = grid["day"] + hour_offset
    return grid[["station_id", "hour_start"]]


def count_departures(trips: pd.DataFrame) -> pd.DataFrame:
    """Count trips per origin station and start hour (only hours with departures)."""
    # Definición común de "salida": viaje contado en origin_station_id a la hora de started_at.
    # floor("h") lleva cada inicio al comienzo de su hora: 08:47:13 -> 08:00:00
    hour_start = trips["started_at"].dt.floor("h")
    # value_counts sobre dos columnas cuenta cuántas veces aparece cada pareja
    # (estación, hora). Solo aparecen las horas CON salidas; los ceros los pone la grilla.
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
    # 1) Todas las horas en que cada estación estaba activa (sin mirar si hubo salidas)
    grid = build_hour_grid(station_active_days(trips))
    # 2) Las horas en que sí hubo salidas, con su conteo
    counts = count_departures(trips)

    # 3) Unión izquierda: toda hora activa se conserva; si no tuvo salidas, queda vacía
    #    y fillna(0) la convierte en cero válido. Con "inner" se perderían todos los ceros
    #    y el promedio de salidas quedaría inflado.
    base = grid.merge(counts, on=["station_id", "hour_start"], how="left")
    base["departures"] = base["departures"].fillna(0).astype("int32")
    base["station_id"] = base["station_id"].astype("int32")

    # 4) Columnas de calendario (conocidas por adelantado: sin fuga temporal)
    base = add_calendar_columns(base)
    return base.sort_values(["station_id", "hour_start"]).reset_index(drop=True)


def add_calendar_columns(base: pd.DataFrame) -> pd.DataFrame:
    """Add date, hour, day of week, weekend and month derived from hour_start."""
    # Todas salen de hour_start: una sola definición para todo el equipo
    base["date"] = base["hour_start"].dt.normalize()  # fecha sin hora (agrupar por día)
    base["hour"] = base["hour_start"].dt.hour  # 0..23 (perfil horario)
    base["day_of_week"] = base["hour_start"].dt.day_of_week  # 0 = lunes ... 6 = domingo
    base["is_weekend"] = base["day_of_week"].isin([5, 6])  # sábado o domingo
    base["month"] = base["hour_start"].dt.month  # 1..12 (estacionalidad)

    return base


def base_summary(base: pd.DataFrame, n_trips: int) -> dict:
    """Counts that describe the base; used to validate it and as evidence."""
    departures = base["departures"]
    return {
        # --- Tamaño y cobertura ---
        "n_rows": len(base),
        "n_stations": int(base["station_id"].nunique()),
        "hour_start_min": str(base["hour_start"].min()),
        "hour_start_max": str(base["hour_start"].max()),
        # --- Validación: toda salida de interim debe caer en la grilla (deben coincidir) ---
        "n_trips_interim": n_trips,
        "departures_total": int(departures.sum()),
        # --- Distribución del target ---
        "zero_share": round(float((departures == 0).mean()), 4),  # proporción de ceros válidos
        "departures_mean": round(float(departures.mean()), 4),
        "departures_p50": float(departures.quantile(0.5)),
        "departures_p90": float(departures.quantile(0.9)),
        "departures_p99": float(departures.quantile(0.99)),
        "departures_max": int(departures.max()),
        # --- Validación: cada (estación, hora) aparece una sola vez (debe ser 0) ---
        "duplicated_keys": int(base.duplicated(["station_id", "hour_start"]).sum()),
    }


@app.command()
def main() -> None:
    """Build data/interim/station_hour_base.parquet from the interim trips."""
    # Lee solo 3 columnas de los viajes de interim (no toca data/raw)
    trips = pd.read_parquet(TRIPS_DIR, columns=TRIP_COLUMNS)
    base = build_station_hour_base(trips)

    # Datos (~96 MB) a data/interim: fuera de Git, se regeneran con `make station-hour`
    base.to_parquet(STATION_HOUR_PATH, index=False)

    # Resumen pequeño como evidencia versionada (los datos quedan fuera de Git)
    STATION_HOUR_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    summary = base_summary(base, n_trips=len(trips))
    (STATION_HOUR_REPORTS_DIR / "summary.json").write_text(json.dumps(summary, indent=2))

    logger.success(
        f"Station x hour base: {summary['n_rows']:,} rows, {summary['n_stations']} stations, "
        f"{summary['zero_share']:.1%} zeros. Output: {STATION_HOUR_PATH}"
    )


if __name__ == "__main__":
    app()
