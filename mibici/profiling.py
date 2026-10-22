"""Read-only profiling of the raw MiBici open-data files.

This module only reads files in ``data/raw`` and writes summary tables to
``reports/profiling``. It does not clean, transform or filter any record:
thresholds used here (e.g. trips longer than 24 hours) only count cases so
that cleaning rules can later be decided with evidence.
"""

from collections import Counter
from pathlib import Path
import re

from loguru import logger
import pandas as pd
from tqdm import tqdm
import typer

from mibici.config import RAW_DATA_DIR, REPORTS_DIR

app = typer.Typer()

# Carpeta de salida de las tablas de perfilado (se regeneran con `make profile`)
PROFILING_DIR = REPORTS_DIR / "profiling"

# Nombres canónicos de las 8 columnas de viajes, en el orden en que aparecen en
# todos los archivos. Se usan para renombrar por posición, porque el nombre de
# `Año_de_nacimiento` viene escrito con distintas codificaciones.
TRIP_COLUMNS = [
    "Viaje_Id",
    "Usuario_Id",
    "Genero",
    "Año_de_nacimiento",
    "Inicio_del_viaje",
    "Fin_del_viaje",
    "Origen_Id",
    "Destino_Id",
]
# Columnas que deberían contener solo números enteros
NUMERIC_TRIP_COLUMNS = ["Viaje_Id", "Usuario_Id", "Año_de_nacimiento", "Origen_Id", "Destino_Id"]
DATETIME_COLUMNS = ["Inicio_del_viaje", "Fin_del_viaje"]

# Columnas esperadas en el catálogo de estaciones (nomenclatura)
NOMENCLATURE_COLUMNS = ["id", "name", "latitude", "longitude", "dpcapacity"]

# Formatos de fecha observados en los archivos raw, probados en este orden:
# - "iso": 2024-05-01 08:30:15 (casi todos los archivos)
# - "slash_dayfirst": 01/05/2024 8:30 (2023-11, 2024-09 y 2024-10; sin segundos)
DATETIME_FORMATS = {
    "iso": "%Y-%m-%d %H:%M:%S",
    "slash_dayfirst": "%d/%m/%Y %H:%M",
}

# Rectángulo aproximado del área metropolitana de Guadalajara. Solo sirve para
# señalar coordenadas de estaciones claramente fuera de lugar.
GDL_LAT_RANGE = (20.4, 20.9)
GDL_LON_RANGE = (-103.6, -103.1)

# Nombres de archivo: datos_abiertos_AAAA_MM.csv (con sufijo opcional, p. ej. "-1")
# y nomenclatura_AAAA_MM.csv. Los grupos capturan año, mes y sufijo.
TRIP_FILE_PATTERN = re.compile(r"^datos_abiertos_(\d{4})_(\d{2})(.*)\.csv$")
NOMENCLATURE_PATTERN = re.compile(r"^nomenclatura_(\d{4})_(\d{2})\.csv$")
# Entero sin signo ni decimales, p. ej. "1990"
DIGITS_PATTERN = r"^\d+$"
# Marcadores literales de valor faltante observados en los archivos raw
MISSING_TOKENS = ["NULL", "NA", "null", "N/A", "nan", "NaN"]
# Entero escrito con parte decimal en cero, p. ej. "1990.0"
DECIMAL_ZERO_PATTERN = r"^\d+\.0+$"


def list_raw_files(raw_dir: Path = RAW_DATA_DIR) -> pd.DataFrame:
    """List raw CSV files with their kind and the period stated in the filename."""
    records = []

    for path in sorted(raw_dir.glob("*.csv")):
        trip_match = TRIP_FILE_PATTERN.match(path.name)
        nomenclature_match = NOMENCLATURE_PATTERN.match(path.name)

        # Clasificar el archivo: viajes, nomenclatura o desconocido.
        # El sufijo (p. ej. "-1") marca los nombres irregulares.
        if trip_match:
            kind, match, suffix = "trips", trip_match, trip_match.group(3)
        elif nomenclature_match:
            kind, match, suffix = "nomenclature", nomenclature_match, ""
        else:
            kind, match, suffix = "unknown", None, ""

        records.append(
            {
                "filename": path.name,
                "path": path,
                "kind": kind,
                # Año y mes que declara el nombre del archivo (no el contenido)
                "year": int(match.group(1)) if match else None,
                "month": int(match.group(2)) if match else None,
                "irregular_name": bool(suffix),
                "size_bytes": path.stat().st_size,
            }
        )

    return pd.DataFrame(records)


def detect_encoding(path: Path) -> str:
    """Return 'utf-8' if the whole file decodes as UTF-8, otherwise 'latin-1'."""
    # Se decodifica el archivo completo: un solo byte inválido en cualquier
    # parte indica que no es UTF-8. Latin-1 nunca falla, por eso es el respaldo.
    try:
        path.read_bytes().decode("utf-8")
    except UnicodeDecodeError:
        return "latin-1"

    return "utf-8"


def read_header(path: Path, encoding: str = "latin-1") -> dict[str, str | int | bool]:
    """Describe the raw header line of a CSV without loading the file."""
    # Se lee solo la primera línea en bytes para conservar el final de línea real
    with path.open("rb") as file:
        raw_line = file.readline()

    line_ending = "crlf" if raw_line.endswith(b"\r\n") else "lf"
    text = raw_line.rstrip(b"\r\n").decode(encoding)
    # Nombres de columna sin comillas, para compararlos con TRIP_COLUMNS
    raw_columns = [column.strip('"') for column in text.split(",")]

    return {
        "raw_header": text,
        "n_columns": len(raw_columns),
        "quoted_header": text.startswith('"'),
        "line_ending": line_ending,
        "header_matches_layout": header_matches_trip_layout(raw_columns),
    }


def header_matches_trip_layout(raw_columns: list[str]) -> bool:
    """Check that a header has the expected trip columns in the expected order.

    The birth-year column is compared loosely because its name appears with
    different encodings in the raw files (UTF-8, Latin-1 and a corrupted form).
    """
    if len(raw_columns) != len(TRIP_COLUMNS):
        return False

    for raw, expected in zip(raw_columns, TRIP_COLUMNS):
        if expected == "Año_de_nacimiento":
            # Acepta "Año_de_nacimiento", su versión Latin-1 y "A}äe_nacimiento"
            if not re.match(r"^A.*nacimiento$", raw):
                return False
        elif raw != expected:
            return False

    return True


def read_raw_trips(path: Path, encoding: str | None = None) -> pd.DataFrame:
    """Load a raw trips file as text, keeping empty strings visible.

    Columns are renamed by position to the canonical names in TRIP_COLUMNS so
    that files with differently encoded headers can be compared.
    """
    return pd.read_csv(
        path,
        encoding=encoding or detect_encoding(path),
        # Todo como texto: así se observan los valores tal cual vienen
        # (p. ej. "1990.0" o "NULL") en lugar de que pandas los convierta
        dtype=str,
        # Sin esto, pandas convertiría "", "NA" y "NULL" en NaN y no se podrían
        # distinguir entre sí
        keep_default_na=False,
        # Descarta el encabezado original y usa los nombres canónicos
        header=0,
        names=TRIP_COLUMNS,
    )


def datetime_shapes(values: pd.Series) -> Counter:
    """Count value layouts, replacing every digit with '9'."""
    # "2024-05-01 08:30:15" -> "9999-99-99 99:99:99"; sirve para descubrir
    # formatos de fecha distintos sin conocerlos de antemano
    return Counter(values.str.replace(r"\d", "9", regex=True))


def parse_datetimes(values: pd.Series) -> tuple[pd.Series, dict[str, int]]:
    """Parse datetimes trying each known format; return values and counts per format."""
    parsed = pd.Series(pd.NaT, index=values.index, dtype="datetime64[ns]")
    counts = {}

    for name, fmt in DATETIME_FORMATS.items():
        # Solo se intentan los valores no vacíos que ningún formato anterior pudo leer
        pending = parsed.isna() & values.ne("")
        # errors="coerce": lo que no coincide con el formato queda como NaT
        attempt = pd.to_datetime(values[pending], format=fmt, errors="coerce")
        parsed.loc[attempt.index] = attempt
        # Cuántos valores se leyeron con este formato
        counts[name] = int(attempt.notna().sum())

    return parsed, counts


def slash_order_evidence(values: pd.Series) -> dict[str, int]:
    """Count slash dates whose first or second field can only be a day (> 12)."""
    # Un campo mayor que 12 solo puede ser día. Si el primero lo es a menudo y el
    # segundo nunca, el formato es día/mes/año.
    parts = values[values.str.contains("/", regex=False)].str.extract(r"^(\d+)/(\d+)/")
    parts = parts.dropna().astype(int)

    return {
        "slash_first_field_gt_12": int((parts[0] > 12).sum()),
        "slash_second_field_gt_12": int((parts[1] > 12).sum()),
    }


def value_counts_text(values: pd.Series, top: int = 10) -> str:
    """Summarise the most frequent values as 'value:count|value:count'."""
    # Las celdas vacías se muestran como "<empty>" para que sean visibles en el CSV
    counts = values.replace("", "<empty>").value_counts().head(top)

    return "|".join(f"{value}:{count}" for value, count in counts.items())


def profile_trips(df: pd.DataFrame, year: int, month: int) -> dict[str, str | int | float]:
    """Compute quality indicators for one month of raw trips."""
    profile: dict[str, str | int | float] = {"n_rows": len(df)}

    # --- Faltantes: celdas vacías y marcadores literales ("NULL", "NA", ...) ---
    for column in TRIP_COLUMNS:
        profile[f"empty_{column}"] = int(df[column].eq("").sum())
        profile[f"missing_token_{column}"] = int(df[column].isin(MISSING_TOKENS).sum())

    # --- Columnas numéricas: enteros escritos como decimal y valores no numéricos ---
    for column in NUMERIC_TRIP_COLUMNS:
        values = df[column]
        decimal_text = values.str.match(DECIMAL_ZERO_PATTERN)
        profile[f"decimal_text_{column}"] = int(decimal_text.sum())
        # No numérico = no vacío, no es marcador de faltante, no es entero y no es
        # entero con ".0" (esos casos ya se cuentan aparte)
        profile[f"non_numeric_{column}"] = int(
            (
                values.ne("")
                & ~values.isin(MISSING_TOKENS)
                & ~values.str.match(DIGITS_PATTERN)
                & ~decimal_text
            ).sum()
        )

    # --- Fechas: formatos encontrados y valores que no se pudieron interpretar ---
    parsed = {}
    for column in DATETIME_COLUMNS:
        # Las 5 "formas" de fecha más frecuentes, p. ej. "9999-99-99 99:99:99:23967"
        shapes = datetime_shapes(df[column])
        profile[f"shapes_{column}"] = "|".join(f"{s}:{n}" for s, n in shapes.most_common(5))

        parsed[column], format_counts = parse_datetimes(df[column])
        for name, count in format_counts.items():
            profile[f"format_{name}_{column}"] = count
        profile[f"unparsed_{column}"] = int((parsed[column].isna() & df[column].ne("")).sum())

    # Evidencia de si las fechas con "/" son día/mes o mes/día
    profile.update(slash_order_evidence(df["Inicio_del_viaje"]))

    # --- Cobertura temporal: rango de inicio y viajes fuera del mes del archivo ---
    start, end = parsed["Inicio_del_viaje"], parsed["Fin_del_viaje"]
    profile["start_min"] = str(start.min())
    profile["start_max"] = str(start.max())
    outside_month = start.notna() & ((start.dt.year != year) | (start.dt.month != month))
    profile["start_outside_file_month"] = int(outside_month.sum())

    # --- Duración del viaje en segundos (fin - inicio) ---
    # Los umbrales solo cuentan casos; no se elimina ningún viaje
    duration = (end - start).dt.total_seconds()
    profile["duration_negative"] = int((duration < 0).sum())
    profile["duration_zero"] = int((duration == 0).sum())
    profile["duration_lt_60s"] = int(((duration > 0) & (duration < 60)).sum())
    profile["duration_gt_24h"] = int((duration > 24 * 3600).sum())
    # Percentiles 50, 95 y 99 para comparar la duración típica entre meses
    for quantile in (0.5, 0.95, 0.99):
        profile[f"duration_p{int(quantile * 100)}_s"] = float(duration.quantile(quantile))

    # --- Duplicados e identificadores de viaje ---
    # duplicated() cuenta las repeticiones a partir de la segunda aparición
    profile["duplicated_rows"] = int(df.duplicated().sum())
    profile["duplicated_Viaje_Id"] = int(df["Viaje_Id"].duplicated().sum())
    # Rango de IDs del mes, para revisar la continuidad entre archivos
    trip_ids = pd.to_numeric(df["Viaje_Id"], errors="coerce")
    profile["Viaje_Id_min"] = trip_ids.min()
    profile["Viaje_Id_max"] = trip_ids.max()

    # --- Usuarios y variables demográficas ---
    profile["n_users"] = int(df["Usuario_Id"].nunique())
    profile["values_Genero"] = value_counts_text(df["Genero"])

    # to_numeric acepta "1990" y "1990.0"; vacíos y marcadores quedan como NaN
    birth_year = pd.to_numeric(df["Año_de_nacimiento"], errors="coerce")
    profile["birth_year_min"] = birth_year.min()
    profile["birth_year_max"] = birth_year.max()
    # Años implausibles: anteriores a 1920 o posteriores al año del propio viaje
    profile["birth_year_lt_1920"] = int((birth_year < 1920).sum())
    profile["birth_year_gt_file_year"] = int((birth_year > year).sum())

    # --- Estaciones ---
    profile["n_origin_stations"] = int(df["Origen_Id"].nunique())
    profile["n_destination_stations"] = int(df["Destino_Id"].nunique())
    # Viajes que empiezan y terminan en la misma estación
    profile["same_origin_destination"] = int(df["Origen_Id"].eq(df["Destino_Id"]).sum())

    return profile


def station_counts(df: pd.DataFrame) -> pd.DataFrame:
    """Count trips per station id as origin and as destination."""
    # Tabla larga: una fila por estación y rol (origen = salidas, destino = llegadas)
    origin = df["Origen_Id"].value_counts().rename_axis("station_id").reset_index(name="n_trips")
    origin["role"] = "origin"
    destination = (
        df["Destino_Id"].value_counts().rename_axis("station_id").reset_index(name="n_trips")
    )
    destination["role"] = "destination"

    return pd.concat([origin, destination], ignore_index=True)


def daily_counts(df: pd.DataFrame) -> pd.DataFrame:
    """Count trips per calendar day of the parsed start datetime."""
    # El día se toma de la hora de inicio del viaje
    start, _ = parse_datetimes(df["Inicio_del_viaje"])

    return start.dt.date.value_counts().rename_axis("date").reset_index(name="n_trips")


def profile_nomenclature(path: Path) -> tuple[dict[str, str | int | float], pd.DataFrame]:
    """Compute quality indicators for the station nomenclature file."""
    encoding = detect_encoding(path)
    # Igual que con los viajes: todo como texto y sin convertir faltantes
    df = pd.read_csv(path, encoding=encoding, dtype=str, keep_default_na=False)

    # --- Estructura e identificadores ---
    profile: dict[str, str | int | float] = {
        "filename": path.name,
        "encoding": encoding,
        "columns": "|".join(df.columns),
        "columns_match_expected": list(df.columns) == NOMENCLATURE_COLUMNS,
        "n_rows": len(df),
        "n_unique_id": int(df["id"].nunique()),
        "duplicated_id": int(df["id"].duplicated().sum()),
        "duplicated_name": int(df["name"].duplicated().sum()),
        "non_numeric_id": int((~df["id"].str.match(DIGITS_PATTERN)).sum()),
    }

    for column in NOMENCLATURE_COLUMNS:
        profile[f"empty_{column}"] = int(df[column].eq("").sum())

    # --- Coordenadas fuera del área metropolitana (o no numéricas) ---
    latitude = pd.to_numeric(df["latitude"], errors="coerce")
    longitude = pd.to_numeric(df["longitude"], errors="coerce")
    profile["coordinates_outside_gdl"] = int(
        (~latitude.between(*GDL_LAT_RANGE) | ~longitude.between(*GDL_LON_RANGE)).sum()
    )

    # --- Capacidad de anclajes (solo la foto actual del catálogo) ---
    capacity = pd.to_numeric(df["dpcapacity"], errors="coerce")
    profile["capacity_min"] = capacity.min()
    profile["capacity_max"] = capacity.max()
    profile["capacity_zero_or_missing"] = int((capacity.isna() | (capacity == 0)).sum())

    # --- Municipio según el prefijo del nombre: "(GDL-001) ..." -> "GDL" ---
    # Las estaciones retiradas empiezan con "ex-" y quedan como "<none>"
    prefix = df["name"].str.extract(r"^\(([A-Z]+)-", expand=False).fillna("<none>")
    profile["name_without_prefix"] = int((prefix == "<none>").sum())

    # Catálogo con la columna de municipio añadida, para cruzarlo en el notebook
    stations = df.assign(municipality_prefix=prefix)

    return profile, stations


@app.command()
def main(
    limit: int | None = typer.Option(
        None,
        help="Maximum number of trip files to profile. Useful for testing.",
    ),
) -> None:
    """Profile raw MiBici files and write summary tables to reports/profiling."""
    PROFILING_DIR.mkdir(parents=True, exist_ok=True)

    files = list_raw_files()
    trip_files = files[files["kind"] == "trips"]

    # --limit N procesa solo los primeros N archivos (útil para pruebas rápidas)
    if limit is not None:
        trip_files = trip_files.head(limit)
        logger.info(f"Profiling limit enabled: {limit}")

    profiles, stations, days = [], [], []

    # Un archivo a la vez: solo un mes de viajes está en memoria en cada iteración
    for row in tqdm(trip_files.itertuples(), total=len(trip_files)):
        encoding = detect_encoding(row.path)
        df = read_raw_trips(row.path, encoding=encoding)

        # Identificación del archivo, repetida en cada tabla de salida
        file_info = {"filename": row.filename, "year": row.year, "month": row.month}

        # Una fila por archivo: datos del archivo + encabezado + indicadores de calidad
        profile = {
            **file_info,
            "irregular_name": row.irregular_name,
            "size_bytes": row.size_bytes,
            "encoding": encoding,
        }
        profile.update(read_header(row.path, encoding=encoding))
        profile.update(profile_trips(df, row.year, row.month))
        profiles.append(profile)

        stations.append(station_counts(df).assign(**file_info))
        days.append(daily_counts(df).assign(filename=row.filename))

    # Tablas de viajes, combinando los resultados de todos los archivos
    outputs = {
        "raw_trips_profile.csv": pd.DataFrame(profiles),
        "station_month_counts.csv": pd.concat(stations, ignore_index=True),
        "daily_trip_counts.csv": pd.concat(days, ignore_index=True),
    }

    # Tablas de la nomenclatura (hoy existe un solo archivo)
    for path in files.loc[files["kind"] == "nomenclature", "path"]:
        profile, station_table = profile_nomenclature(path)
        outputs["nomenclature_profile.csv"] = pd.DataFrame([profile])
        outputs["nomenclature_stations.csv"] = station_table

    # Solo se escribe en reports/profiling/; data/raw nunca se modifica
    for filename, table in outputs.items():
        table.to_csv(PROFILING_DIR / filename, index=False)
        logger.info(f"Wrote {filename}: {len(table)} rows")

    logger.success(f"Profiling complete: {len(profiles)} trip files. Output: {PROFILING_DIR}")


if __name__ == "__main__":
    app()
