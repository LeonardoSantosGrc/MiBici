"""Station catalogue and station lifecycle for the interim layer.

The nomenclature file is a single snapshot (2026-09): its name, capacity and
status are not assumed to be valid for earlier years. The lifecycle of each
station is derived from the trips themselves.
"""

from pathlib import Path

import pandas as pd

from mibici.profiling import detect_encoding

# Nomenclatura raw -> interim. El prefijo "snapshot_" recuerda que es una sola foto
# (2026-09): la capacidad y el nombre actuales no son válidos para años anteriores.
NOMENCLATURE_NAMES = {
    "id": "station_id",
    "name": "snapshot_name",
    "latitude": "latitude",
    "longitude": "longitude",
    "dpcapacity": "snapshot_capacity",
}

# Las estaciones retiradas aparecen en la nomenclatura como "ex-(GDL-020) ..."
RETIRED_NAME_PREFIX = "ex-"

# Estación "nueva": primera actividad desde la expansión de 2026 (100 estaciones en el perfilado)
NEW_STATION_SINCE = pd.Period("2026-01", freq="M")

# Sin viajes en los últimos N meses del histórico -> se considera retirada por actividad.
# Coincide con el catálogo: las 14 estaciones "ex-" cumplen esta regla.
INACTIVE_TAIL_MONTHS = 3


def load_nomenclature(path: Path, source_sha256: str) -> pd.DataFrame:
    """Read the station snapshot with interim names, types and lineage."""
    # Igual que con los viajes: todo como texto y sin convertir faltantes al leer
    raw = pd.read_csv(path, encoding=detect_encoding(path), dtype=str, keep_default_na=False)
    stations = raw.rename(columns=NOMENCLATURE_NAMES)

    # --- Tipos ---
    stations["station_id"] = pd.to_numeric(stations["station_id"]).astype("Int32")
    stations["latitude"] = pd.to_numeric(stations["latitude"])
    stations["longitude"] = pd.to_numeric(stations["longitude"])
    stations["snapshot_capacity"] = pd.to_numeric(stations["snapshot_capacity"]).astype("Int16")

    # Retirada según el catálogo: el nombre empieza con "ex-"
    stations["retired_in_snapshot"] = stations["snapshot_name"].str.startswith(RETIRED_NAME_PREFIX)

    # --- Linaje: de qué archivo y versión salió cada estación ---
    stations["snapshot_source_file"] = path.name
    stations["snapshot_sha256"] = source_sha256
    return stations


def station_month_activity(trips: pd.DataFrame) -> pd.DataFrame:
    """List each (station, month) with at least one trip leaving or arriving there."""
    # El mes se toma del inicio del viaje, tanto para la salida como para la llegada.
    # to_period("M") convierte una fecha en su mes: 2024-05-17 08:30 -> 2024-05
    month = trips["started_at"].dt.to_period("M")

    # Una estación está activa en un mes si salió o llegó al menos un viaje
    origin = pd.DataFrame({"station_id": trips["origin_station_id"], "month": month})
    destination = pd.DataFrame({"station_id": trips["destination_station_id"], "month": month})

    # Se apilan salidas y llegadas, y se deja una sola fila por (estación, mes)
    return pd.concat([origin, destination]).dropna().drop_duplicates().reset_index(drop=True)


def build_station_lifecycle(trips: pd.DataFrame, nomenclature: pd.DataFrame) -> pd.DataFrame:
    """Summarise when each station was active and whether it has snapshot metadata."""
    activity = station_month_activity(trips)
    # Último mes del histórico; sirve de referencia para decidir si una estación se retiró
    last_month = activity["month"].max()

    # --- Ciclo de vida por estación (equivale a un GROUP BY de SQL) ---
    # Cada línea es nombre_resultado=(columna, operación)
    lifecycle = activity.groupby("station_id").agg(
        first_active_month=("month", "min"),
        last_active_month=("month", "max"),
        n_active_months=("month", "nunique"),
    )

    # Meses entre la primera y la última actividad, contando ambos extremos.
    # Restar dos meses da un desplazamiento; .n es su número de meses.
    span = (lifecycle["last_active_month"] - lifecycle["first_active_month"]).apply(
        lambda offset: offset.n
    ) + 1
    # Inactivos = meses de vida - meses con viajes (p. ej. estación 124: 56 - 14 = 42)
    lifecycle["n_inactive_months"] = span - lifecycle["n_active_months"]

    # --- Volumen de viajes por estación ---
    # value_counts cuenta viajes por estación; se alinea con lifecycle por station_id
    lifecycle["n_trips_origin"] = trips["origin_station_id"].value_counts()
    lifecycle["n_trips_destination"] = trips["destination_station_id"].value_counts()
    # Una estación que solo aparece como destino no tiene salidas: 0, no faltante
    lifecycle[["n_trips_origin", "n_trips_destination"]] = lifecycle[
        ["n_trips_origin", "n_trips_destination"]
    ].fillna(0)

    # --- Estado de la estación ---
    lifecycle["is_new_station"] = lifecycle["first_active_month"] >= NEW_STATION_SINCE
    lifecycle["is_retired_by_activity"] = lifecycle["last_active_month"] <= (
        last_month - INACTIVE_TAIL_MONTHS
    )

    # --- Cruce con el catálogo ---
    # Unión externa: conserva estaciones con viajes sin metadatos y del catálogo sin viajes.
    # indicator=True agrega "_merge": both, left_only (solo viajes) o right_only (solo catálogo)
    stations = lifecycle.reset_index().merge(
        nomenclature, on="station_id", how="outer", indicator=True
    )
    stations["has_trips"] = stations["_merge"].isin(["both", "left_only"])
    stations["has_metadata"] = stations["_merge"].isin(["both", "right_only"])
    stations = stations.drop(columns="_merge")

    # Tras la unión, la estación sin viajes deja huecos: enteros nullable en vez de float
    count_columns = [
        "n_active_months",
        "n_inactive_months",
        "n_trips_origin",
        "n_trips_destination",
    ]
    stations[count_columns] = stations[count_columns].astype("Int64")

    # Los indicadores que no aplican (estación sin viajes o sin catálogo) quedan en False
    flag_columns = ["is_new_station", "is_retired_by_activity", "retired_in_snapshot"]
    stations[flag_columns] = stations[flag_columns].fillna(False).astype(bool)

    return stations.sort_values("station_id").reset_index(drop=True)


def add_station_metadata_flags(trips: pd.DataFrame, station_ids: pd.Series) -> pd.DataFrame:
    """Flag trips whose origin or destination is missing from the station snapshot."""
    # Un set permite preguntar muy rápido si un id está en el catálogo
    known = set(station_ids.dropna())
    # isin -> True si el id está en el catálogo; ~ lo invierte: True si NO está.
    # Sobre enteros nullable, isin da "boolean"; se pasa a bool como los demás flags.
    for role in ["origin", "destination"]:
        is_known = trips[f"{role}_station_id"].isin(known).astype(bool)
        trips[f"flag_{role}_without_metadata"] = ~is_known
    return trips
