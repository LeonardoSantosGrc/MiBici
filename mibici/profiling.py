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

PROFILING_DIR = REPORTS_DIR / "profiling"

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
NUMERIC_TRIP_COLUMNS = ["Viaje_Id", "Usuario_Id", "Año_de_nacimiento", "Origen_Id", "Destino_Id"]
DATETIME_COLUMNS = ["Inicio_del_viaje", "Fin_del_viaje"]

NOMENCLATURE_COLUMNS = ["id", "name", "latitude", "longitude", "dpcapacity"]

# Datetime layouts observed in the raw files, tried in this order.
DATETIME_FORMATS = {
    "iso": "%Y-%m-%d %H:%M:%S",
    "slash_dayfirst": "%d/%m/%Y %H:%M",
}

# Approximate bounding box of the Guadalajara metropolitan area, used only to
# flag station coordinates that are clearly misplaced.
GDL_LAT_RANGE = (20.4, 20.9)
GDL_LON_RANGE = (-103.6, -103.1)

TRIP_FILE_PATTERN = re.compile(r"^datos_abiertos_(\d{4})_(\d{2})(.*)\.csv$")
NOMENCLATURE_PATTERN = re.compile(r"^nomenclatura_(\d{4})_(\d{2})\.csv$")
DIGITS_PATTERN = r"^\d+$"
# Literal placeholders for missing values observed in the raw files.
MISSING_TOKENS = ["NULL", "NA", "null", "N/A", "nan", "NaN"]
# Integers written with a zero decimal part, e.g. "1990.0".
DECIMAL_ZERO_PATTERN = r"^\d+\.0+$"


def list_raw_files(raw_dir: Path = RAW_DATA_DIR) -> pd.DataFrame:
    """List raw CSV files with their kind and the period stated in the filename."""
    records = []

    for path in sorted(raw_dir.glob("*.csv")):
        trip_match = TRIP_FILE_PATTERN.match(path.name)
        nomenclature_match = NOMENCLATURE_PATTERN.match(path.name)

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
                "year": int(match.group(1)) if match else None,
                "month": int(match.group(2)) if match else None,
                "irregular_name": bool(suffix),
                "size_bytes": path.stat().st_size,
            }
        )

    return pd.DataFrame(records)


def detect_encoding(path: Path) -> str:
    """Return 'utf-8' if the whole file decodes as UTF-8, otherwise 'latin-1'."""
    try:
        path.read_bytes().decode("utf-8")
    except UnicodeDecodeError:
        return "latin-1"

    return "utf-8"


def read_header(path: Path, encoding: str = "latin-1") -> dict[str, str | int | bool]:
    """Describe the raw header line of a CSV without loading the file."""
    with path.open("rb") as file:
        raw_line = file.readline()

    line_ending = "crlf" if raw_line.endswith(b"\r\n") else "lf"
    text = raw_line.rstrip(b"\r\n").decode(encoding)
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
        dtype=str,
        keep_default_na=False,
        header=0,
        names=TRIP_COLUMNS,
    )


def datetime_shapes(values: pd.Series) -> Counter:
    """Count value layouts, replacing every digit with '9'."""
    return Counter(values.str.replace(r"\d", "9", regex=True))


def parse_datetimes(values: pd.Series) -> tuple[pd.Series, dict[str, int]]:
    """Parse datetimes trying each known format; return values and counts per format."""
    parsed = pd.Series(pd.NaT, index=values.index, dtype="datetime64[ns]")
    counts = {}

    for name, fmt in DATETIME_FORMATS.items():
        pending = parsed.isna() & values.ne("")
        attempt = pd.to_datetime(values[pending], format=fmt, errors="coerce")
        parsed.loc[attempt.index] = attempt
        counts[name] = int(attempt.notna().sum())

    return parsed, counts


def slash_order_evidence(values: pd.Series) -> dict[str, int]:
    """Count slash dates whose first or second field can only be a day (> 12)."""
    parts = values[values.str.contains("/", regex=False)].str.extract(r"^(\d+)/(\d+)/")
    parts = parts.dropna().astype(int)

    return {
        "slash_first_field_gt_12": int((parts[0] > 12).sum()),
        "slash_second_field_gt_12": int((parts[1] > 12).sum()),
    }


def value_counts_text(values: pd.Series, top: int = 10) -> str:
    """Summarise the most frequent values as 'value:count|value:count'."""
    counts = values.replace("", "<empty>").value_counts().head(top)

    return "|".join(f"{value}:{count}" for value, count in counts.items())


def profile_trips(df: pd.DataFrame, year: int, month: int) -> dict[str, str | int | float]:
    """Compute quality indicators for one month of raw trips."""
    profile: dict[str, str | int | float] = {"n_rows": len(df)}

    for column in TRIP_COLUMNS:
        profile[f"empty_{column}"] = int(df[column].eq("").sum())
        profile[f"missing_token_{column}"] = int(df[column].isin(MISSING_TOKENS).sum())

    for column in NUMERIC_TRIP_COLUMNS:
        values = df[column]
        decimal_text = values.str.match(DECIMAL_ZERO_PATTERN)
        profile[f"decimal_text_{column}"] = int(decimal_text.sum())
        profile[f"non_numeric_{column}"] = int(
            (
                values.ne("")
                & ~values.isin(MISSING_TOKENS)
                & ~values.str.match(DIGITS_PATTERN)
                & ~decimal_text
            ).sum()
        )

    parsed = {}
    for column in DATETIME_COLUMNS:
        shapes = datetime_shapes(df[column])
        profile[f"shapes_{column}"] = "|".join(f"{s}:{n}" for s, n in shapes.most_common(5))

        parsed[column], format_counts = parse_datetimes(df[column])
        for name, count in format_counts.items():
            profile[f"format_{name}_{column}"] = count
        profile[f"unparsed_{column}"] = int((parsed[column].isna() & df[column].ne("")).sum())

    profile.update(slash_order_evidence(df["Inicio_del_viaje"]))

    start, end = parsed["Inicio_del_viaje"], parsed["Fin_del_viaje"]
    profile["start_min"] = str(start.min())
    profile["start_max"] = str(start.max())
    outside_month = start.notna() & ((start.dt.year != year) | (start.dt.month != month))
    profile["start_outside_file_month"] = int(outside_month.sum())

    duration = (end - start).dt.total_seconds()
    profile["duration_negative"] = int((duration < 0).sum())
    profile["duration_zero"] = int((duration == 0).sum())
    profile["duration_lt_60s"] = int(((duration > 0) & (duration < 60)).sum())
    profile["duration_gt_24h"] = int((duration > 24 * 3600).sum())
    for quantile in (0.5, 0.95, 0.99):
        profile[f"duration_p{int(quantile * 100)}_s"] = float(duration.quantile(quantile))

    profile["duplicated_rows"] = int(df.duplicated().sum())
    profile["duplicated_Viaje_Id"] = int(df["Viaje_Id"].duplicated().sum())
    trip_ids = pd.to_numeric(df["Viaje_Id"], errors="coerce")
    profile["Viaje_Id_min"] = trip_ids.min()
    profile["Viaje_Id_max"] = trip_ids.max()

    profile["n_users"] = int(df["Usuario_Id"].nunique())
    profile["values_Genero"] = value_counts_text(df["Genero"])

    birth_year = pd.to_numeric(df["Año_de_nacimiento"], errors="coerce")
    profile["birth_year_min"] = birth_year.min()
    profile["birth_year_max"] = birth_year.max()
    profile["birth_year_lt_1920"] = int((birth_year < 1920).sum())
    profile["birth_year_gt_file_year"] = int((birth_year > year).sum())

    profile["n_origin_stations"] = int(df["Origen_Id"].nunique())
    profile["n_destination_stations"] = int(df["Destino_Id"].nunique())
    profile["same_origin_destination"] = int(df["Origen_Id"].eq(df["Destino_Id"]).sum())

    return profile


def station_counts(df: pd.DataFrame) -> pd.DataFrame:
    """Count trips per station id as origin and as destination."""
    origin = df["Origen_Id"].value_counts().rename_axis("station_id").reset_index(name="n_trips")
    origin["role"] = "origin"
    destination = (
        df["Destino_Id"].value_counts().rename_axis("station_id").reset_index(name="n_trips")
    )
    destination["role"] = "destination"

    return pd.concat([origin, destination], ignore_index=True)


def daily_counts(df: pd.DataFrame) -> pd.DataFrame:
    """Count trips per calendar day of the parsed start datetime."""
    start, _ = parse_datetimes(df["Inicio_del_viaje"])

    return start.dt.date.value_counts().rename_axis("date").reset_index(name="n_trips")


def profile_nomenclature(path: Path) -> tuple[dict[str, str | int | float], pd.DataFrame]:
    """Compute quality indicators for the station nomenclature file."""
    encoding = detect_encoding(path)
    df = pd.read_csv(path, encoding=encoding, dtype=str, keep_default_na=False)

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

    latitude = pd.to_numeric(df["latitude"], errors="coerce")
    longitude = pd.to_numeric(df["longitude"], errors="coerce")
    profile["coordinates_outside_gdl"] = int(
        (~latitude.between(*GDL_LAT_RANGE) | ~longitude.between(*GDL_LON_RANGE)).sum()
    )

    capacity = pd.to_numeric(df["dpcapacity"], errors="coerce")
    profile["capacity_min"] = capacity.min()
    profile["capacity_max"] = capacity.max()
    profile["capacity_zero_or_missing"] = int((capacity.isna() | (capacity == 0)).sum())

    prefix = df["name"].str.extract(r"^\(([A-Z]+)-", expand=False).fillna("<none>")
    profile["name_without_prefix"] = int((prefix == "<none>").sum())

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

    if limit is not None:
        trip_files = trip_files.head(limit)
        logger.info(f"Profiling limit enabled: {limit}")

    profiles, stations, days = [], [], []

    for row in tqdm(trip_files.itertuples(), total=len(trip_files)):
        encoding = detect_encoding(row.path)
        df = read_raw_trips(row.path, encoding=encoding)

        file_info = {"filename": row.filename, "year": row.year, "month": row.month}

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

    outputs = {
        "raw_trips_profile.csv": pd.DataFrame(profiles),
        "station_month_counts.csv": pd.concat(stations, ignore_index=True),
        "daily_trip_counts.csv": pd.concat(days, ignore_index=True),
    }

    for path in files.loc[files["kind"] == "nomenclature", "path"]:
        profile, station_table = profile_nomenclature(path)
        outputs["nomenclature_profile.csv"] = pd.DataFrame([profile])
        outputs["nomenclature_stations.csv"] = station_table

    for filename, table in outputs.items():
        table.to_csv(PROFILING_DIR / filename, index=False)
        logger.info(f"Wrote {filename}: {len(table)} rows")

    logger.success(f"Profiling complete: {len(profiles)} trip files. Output: {PROFILING_DIR}")


if __name__ == "__main__":
    app()
