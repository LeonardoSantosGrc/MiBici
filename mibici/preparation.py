"""Prepare raw MiBici trips into the interim layer.

Reads ``data/raw`` (never modifies it), normalises names, types and missing
markers, and adds lineage columns. Anomalies are flagged, not removed.
"""

from pathlib import Path

import pandas as pd

from mibici.dataset import MANIFEST_PATH
from mibici.profiling import DATETIME_FORMATS, parse_datetimes, read_raw_trips

# Nombres canónicos de interim (raw -> interim). Las claves son los nombres que deja
# read_raw_trips, iguales en todos los archivos sin importar su encoding.
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

# Un duplicado exacto repite las 8 columnas originales del viaje (sin el linaje)
TRIP_KEY_COLUMNS = list(COLUMN_NAMES.values())

# Umbrales de los flags de calidad. Solo marcan registros; no eliminan ninguno.
# Se registran como parámetros en MLflow para saber con qué reglas se generó interim.
SHORT_TRIP_SECONDS = 60  # viaje "corto": menos de 1 minuto
MAX_TRIP_SECONDS = 24 * 3600  # viaje "largo": más de 24 horas
MIN_BIRTH_YEAR = 1920  # años de nacimiento anteriores se consideran implausibles


def load_manifest(path: Path = MANIFEST_PATH) -> dict[str, str]:
    """Map each raw filename to its SHA-256 from the data manifest."""
    manifest = pd.read_csv(path)
    # filename pasa a ser el índice; de ahí se toma la columna sha256 y se convierte en
    # diccionario: {"datos_abiertos_2014_12.csv": "8c49...", ...}
    return manifest.set_index("filename")["sha256"].to_dict()


def to_integer(values: pd.Series, dtype: str) -> pd.Series:
    """Convert text to a nullable integer type; empty cells and markers become <NA>."""
    # to_numeric acepta "1990" y "1990.0"; errors="coerce" vuelve NaN lo que no es número.
    # Los tipos "Int64"/"Int32"/"Int16" (con I mayúscula) admiten faltantes como <NA>.
    return pd.to_numeric(values, errors="coerce").astype(dtype)


def normalize_trips(raw: pd.DataFrame) -> pd.DataFrame:
    """Rename columns, unify missing markers and cast to interim types."""
    # Fechas primero: parse_datetimes espera el texto original, con "" en los vacíos.
    # Devuelve (fechas, conteo por formato); el conteo no se usa aquí, por eso "_".
    started_at, _ = parse_datetimes(raw["Inicio_del_viaje"])
    ended_at, _ = parse_datetimes(raw["Fin_del_viaje"])

    # Formato de fecha por fila: "slash_dayfirst" no trae segundos (resolución al minuto).
    # Se conserva porque infla los viajes de duración cero en 2023-11, 2024-09 y 2024-10.
    is_slash = raw["Inicio_del_viaje"].str.contains("/", regex=False)
    datetime_format = is_slash.map({True: "slash_dayfirst", False: "iso"})

    # Tabla tipada, todavía con los nombres raw; al final se renombra con COLUMN_NAMES.
    # Ninguna fila se agrega ni se elimina: esta función solo cambia la forma de los datos.
    df = pd.DataFrame(
        {
            "Viaje_Id": to_integer(raw["Viaje_Id"], "Int64"),
            "Usuario_Id": to_integer(raw["Usuario_Id"], "Int64"),
            # Lo que no está en GENDER_VALUES queda como <NA>
            "Genero": pd.Categorical(raw["Genero"], categories=GENDER_VALUES),
            "Año_de_nacimiento": to_integer(raw["Año_de_nacimiento"], "Int16"),
            "Inicio_del_viaje": started_at,
            "Fin_del_viaje": ended_at,
            # Hay menos de 1,000 estaciones: Int32 basta y ocupa la mitad que Int64
            "Origen_Id": to_integer(raw["Origen_Id"], "Int32"),
            "Destino_Id": to_integer(raw["Destino_Id"], "Int32"),
        }
    )
    df["datetime_format"] = pd.Categorical(datetime_format, categories=list(DATETIME_FORMATS))

    return df.rename(columns=COLUMN_NAMES)


def add_lineage(df: pd.DataFrame, source_file: str, source_sha256: str) -> pd.DataFrame:
    """Add the source filename and its manifest hash to every row."""
    # [valor] * len(df) repite el valor una vez por fila. Como category, el texto se guarda
    # una sola vez y cada fila solo guarda un código (~78 MB en vez de ~2.5 GB para el hash).
    df["source_file"] = pd.Categorical([source_file] * len(df))
    df["source_sha256"] = pd.Categorical([source_sha256] * len(df))
    return df


def prepare_file(path: Path, source_sha256: str) -> pd.DataFrame:
    """Read one raw trips file and return it normalised with lineage."""
    # Pipeline de un archivo: leer (texto) -> normalizar (tipos) -> agregar linaje.
    # path.name es solo el nombre del archivo, sin la carpeta.
    trips = normalize_trips(read_raw_trips(path))
    return add_lineage(trips, path.name, source_sha256)


def add_duration(df: pd.DataFrame) -> pd.DataFrame:
    """Add the trip duration in whole seconds (end minus start)."""
    # Restar dos fechas da un lapso de tiempo; total_seconds() lo expresa en segundos
    df["duration_s"] = (df["ended_at"] - df["started_at"]).dt.total_seconds().astype("Int64")
    return df


def add_quality_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Add boolean flags for anomalies found in the profiling; nothing is removed."""
    # Condiciones reutilizadas por varios flags. En pandas: & = y, | = o, ~ = no;
    # cada condición va entre paréntesis.
    duration = df["duration_s"]
    same_station = df["origin_station_id"] == df["destination_station_id"]
    is_short = (duration >= 0) & (duration < SHORT_TRIP_SECONDS)

    # --- Duración ---
    df["flag_duration_negative"] = duration < 0
    df["flag_duration_zero"] = duration == 0
    # Corto y regresa a la misma estación: posible arrepentimiento o bicicleta con falla
    df["flag_short_same_station"] = is_short & same_station
    # Corto entre estaciones distintas: físicamente poco plausible, posible error de registro
    df["flag_short_cross_station"] = is_short & ~same_station
    df["flag_duration_over_24h"] = duration > MAX_TRIP_SECONDS

    # --- Datos demográficos ---
    # Implausible: anterior a MIN_BIRTH_YEAR o posterior al año en que se hizo el viaje
    df["flag_birth_year_implausible"] = (df["birth_year"] < MIN_BIRTH_YEAR) | (
        df["birth_year"] > df["started_at"].dt.year
    )

    # Las comparaciones con <NA> dan <NA>; un faltante no es una anomalía marcada.
    # La lista toma todas las columnas que empiezan con "flag_".
    flag_columns = [column for column in df.columns if column.startswith("flag_")]
    df[flag_columns] = df[flag_columns].fillna(False).astype(bool)

    return df


def drop_exact_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Remove exact duplicate trips, keeping the first; return the data and rows removed."""
    # Única eliminación de interim: es un error inequívoco (las 8 columnas idénticas).
    # keep="first" marca como duplicada cada repetición y conserva la primera aparición.
    is_duplicate = df.duplicated(subset=TRIP_KEY_COLUMNS, keep="first")
    # Se devuelve también cuántas filas se quitaron, para registrarlo en MLflow.
    # Al sumar True/False, True cuenta como 1.
    return df[~is_duplicate].reset_index(drop=True), int(is_duplicate.sum())
