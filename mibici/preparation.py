"""Prepare raw MiBici trips into the interim layer.

Reads ``data/raw`` (never modifies it), normalises names, types and missing
markers, and adds lineage columns. Anomalies are flagged, not removed.
"""

from pathlib import Path

import pandas as pd

from mibici.dataset import MANIFEST_PATH
from mibici.profiling import DATETIME_FORMATS, parse_datetimes, read_raw_trips

# Nombres canónicos de interim (raw -> interim)
COLUMN_NAMES = {
    "Viaje_Id": "trip_id",
    "Usuario_Id": "user_id",
    "Genero": "gender",
    "Año_de_nacimiento": "birth_year",
    "Inicio_del_viaje": "started_at",
    "Fin_del_viaje": "ended_at",
    "Origen_Id": "origin_station_id",
    "Destino_Id": "destination_station_id",
}

# Únicos valores de género válidos; vacíos y marcadores ("NULL", ...) quedan como <NA>
GENDER_VALUES = ["M", "F"]


def load_manifest(path: Path = MANIFEST_PATH) -> dict[str, str]:
    """Map each raw filename to its SHA-256 from the data manifest."""
    manifest = pd.read_csv(path)
    return manifest.set_index("filename")["sha256"].to_dict()


def to_integer(values: pd.Series, dtype: str) -> pd.Series:
    """Convert text to a nullable integer type; empty cells and markers become <NA>."""
    # to_numeric acepta "1990" y "1990.0"; errors="coerce" vuelve NaN lo que no es número
    return pd.to_numeric(values, errors="coerce").astype(dtype)


def normalize_trips(raw: pd.DataFrame) -> pd.DataFrame:
    """Rename columns, unify missing markers and cast to interim types."""
    # Fechas primero: parse_datetimes espera el texto original, con "" en los vacíos
    started_at, _ = parse_datetimes(raw["Inicio_del_viaje"])
    ended_at, _ = parse_datetimes(raw["Fin_del_viaje"])

    # Formato de fecha por fila: "slash_dayfirst" no trae segundos (resolución al minuto)
    is_slash = raw["Inicio_del_viaje"].str.contains("/", regex=False)
    datetime_format = is_slash.map({True: "slash_dayfirst", False: "iso"})

    df = pd.DataFrame(
        {
            "Viaje_Id": to_integer(raw["Viaje_Id"], "Int64"),
            "Usuario_Id": to_integer(raw["Usuario_Id"], "Int64"),
            # Lo que no está en GENDER_VALUES queda como <NA>
            "Genero": pd.Categorical(raw["Genero"], categories=GENDER_VALUES),
            "Año_de_nacimiento": to_integer(raw["Año_de_nacimiento"], "Int16"),
            "Inicio_del_viaje": started_at,
            "Fin_del_viaje": ended_at,
            "Origen_Id": to_integer(raw["Origen_Id"], "Int32"),
            "Destino_Id": to_integer(raw["Destino_Id"], "Int32"),
        }
    )
    df["datetime_format"] = pd.Categorical(datetime_format, categories=list(DATETIME_FORMATS))

    return df.rename(columns=COLUMN_NAMES)


def add_lineage(df: pd.DataFrame, source_file: str, source_sha256: str) -> pd.DataFrame:
    """Add the source filename and its manifest hash to every row."""
    df["source_file"] = pd.Categorical([source_file] * len(df))
    df["source_sha256"] = pd.Categorical([source_sha256] * len(df))
    return df


def prepare_file(path: Path, source_sha256: str) -> pd.DataFrame:
    """Read one raw trips file and return it normalised with lineage."""
    trips = normalize_trips(read_raw_trips(path))
    return add_lineage(trips, path.name, source_sha256)
