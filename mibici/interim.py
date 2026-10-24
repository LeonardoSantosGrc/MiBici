"""Build the interim dataset (trips + stations) from the raw MiBici files.

Pipeline: raw CSV -> normalise + lineage -> duration + flags -> drop exact
duplicates -> station lifecycle -> Parquet in ``data/interim``. Small summary
files (schema, quality counts, per-file lineage) go to ``reports/interim``.
"""

import hashlib
import json
import shutil

from loguru import logger
import pandas as pd
from tqdm import tqdm
import typer

from mibici.config import INTERIM_DATA_DIR, REPORTS_DIR
from mibici.dataset import MANIFEST_PATH
from mibici.preparation import (
    MAX_TRIP_SECONDS,
    MIN_BIRTH_YEAR,
    SHORT_TRIP_SECONDS,
    add_duration,
    add_quality_flags,
    drop_exact_duplicates,
    load_manifest,
    prepare_file,
)
from mibici.profiling import list_raw_files
from mibici.stations import (
    INACTIVE_TAIL_MONTHS,
    NEW_STATION_SINCE,
    add_station_metadata_flags,
    build_station_lifecycle,
    load_nomenclature,
)
from mibici.tracking import check_tracking_server, git_info, log_dataset_run

app = typer.Typer()

# Salidas pesadas (fuera de Git) y resúmenes pequeños (evidencia versionada)
TRIPS_DIR = INTERIM_DATA_DIR / "trips"
STATIONS_PATH = INTERIM_DATA_DIR / "stations.parquet"
INTERIM_REPORTS_DIR = REPORTS_DIR / "interim"

# Versión lógica del dataset; se registra como tag en MLflow
DATASET_VERSION = "interim_v1"


def read_all_trips(trip_files: pd.DataFrame, manifest: dict[str, str]) -> pd.DataFrame:
    """Prepare every raw trips file and stack them into a single table."""
    # Un archivo a la vez se normaliza; al final se unen (~7 GB en memoria para 39 M filas)
    frames = [
        prepare_file(row.path, manifest[row.filename])
        for row in tqdm(trip_files.itertuples(), total=len(trip_files))
    ]
    trips = pd.concat(frames, ignore_index=True)

    # Al unir archivos, cada uno trae sus propias categorías; se vuelven a unificar
    for column in ["source_file", "source_sha256"]:
        trips[column] = trips[column].astype("category")
    return trips


def file_summary(raw_rows: pd.Series, trips: pd.DataFrame) -> pd.DataFrame:
    """Rows per source file before and after cleaning, with the file hash (lineage)."""
    after = trips.groupby("source_file", observed=True).agg(
        source_sha256=("source_sha256", "first"),
        rows_interim=("trip_id", "size"),
    )
    summary = after.join(raw_rows.rename("rows_raw"), how="outer")
    summary["rows_removed"] = summary["rows_raw"] - summary["rows_interim"]
    return summary.reset_index(names="source_file")


def quality_summary(
    trips: pd.DataFrame, stations: pd.DataFrame, rows_raw: int, duplicates_removed: int
) -> dict:
    """Counts that describe the interim dataset; logged as MLflow metrics."""
    flag_counts = {
        column: int(trips[column].sum()) for column in trips if column.startswith("flag_")
    }
    return {
        "dataset_version": DATASET_VERSION,
        "n_source_files": int(trips["source_file"].nunique()),
        "rows_raw": rows_raw,
        "rows_interim": len(trips),
        "duplicates_removed": duplicates_removed,
        **flag_counts,
        "started_at_min": str(trips["started_at"].min()),
        "started_at_max": str(trips["started_at"].max()),
        "n_stations": len(stations),
        "n_stations_with_trips": int(stations["has_trips"].sum()),
        "n_stations_with_metadata": int(stations["has_metadata"].sum()),
        "n_stations_without_metadata": int((~stations["has_metadata"]).sum()),
        "n_new_stations": int(stations["is_new_station"].sum()),
        "n_retired_by_activity": int(stations["is_retired_by_activity"].sum()),
        "n_retired_in_snapshot": int(stations["retired_in_snapshot"].sum()),
        "n_stations_with_inactive_months": int((stations["n_inactive_months"] > 0).sum()),
    }


def schema_of(df: pd.DataFrame) -> dict[str, str]:
    """Column name -> pandas dtype, to document the final schema."""
    return {column: str(dtype) for column, dtype in df.dtypes.items()}


def write_trips(trips: pd.DataFrame) -> None:
    """Write trips as Parquet partitioned by year, replacing any previous output."""
    # Una escritura particionada agrega archivos; sin borrar antes, repetir el comando
    # duplicaría los viajes. Se borra solo la salida de este paso, nunca data/raw.
    if TRIPS_DIR.exists():
        shutil.rmtree(TRIPS_DIR)
    trips.assign(year=trips["started_at"].dt.year).to_parquet(
        TRIPS_DIR, partition_cols=["year"], index=False
    )


def tracking_params(trip_files: pd.DataFrame, nomenclature_path) -> dict[str, object]:
    """Rules, thresholds and inputs that define how interim_v1 was produced."""
    return {
        # Reglas de limpieza: qué se elimina y qué solo se marca
        "rule_drop": "exact_duplicates",
        "rule_flags": "duration_negative|duration_zero|short_same_station|"
        "short_cross_station|duration_over_24h|birth_year_implausible|"
        "origin_without_metadata|destination_without_metadata",
        # Umbrales de los flags
        "short_trip_seconds": SHORT_TRIP_SECONDS,
        "max_trip_seconds": MAX_TRIP_SECONDS,
        "min_birth_year": MIN_BIRTH_YEAR,
        "new_station_since": str(NEW_STATION_SINCE),
        "inactive_tail_months": INACTIVE_TAIL_MONTHS,
        # Entradas: cuántos archivos y qué versión del manifest (su propio hash)
        "n_trip_files": len(trip_files),
        "nomenclature_file": nomenclature_path.name,
        "manifest_sha256": hashlib.sha256(MANIFEST_PATH.read_bytes()).hexdigest(),
        # Formato de salida
        "output_format": "parquet",
        "partition_cols": "year",
    }


@app.command()
def main(
    limit: int | None = typer.Option(
        None,
        help="Maximum number of trip files to process. Useful for testing.",
    ),
    track: bool = typer.Option(
        True,
        help="Log this dataset version to MLflow (requires `make mlflow`).",
    ),
) -> None:
    """Build data/interim from data/raw and write summary reports."""
    # Antes de procesar: el servidor debe estar arriba y se toma el commit del código
    # que se va a ejecutar (antes de que este mismo comando reescriba reports/interim)
    if track:
        check_tracking_server()
        git = git_info()
        if git["git_dirty"] == "true":
            logger.warning("Uncommitted changes: the MLflow run will be tagged git_dirty=true")

    manifest = load_manifest()
    files = list_raw_files()
    trip_files = files[files["kind"] == "trips"]
    nomenclature_path = files.loc[files["kind"] == "nomenclature", "path"].iloc[0]

    # --limit N procesa solo los primeros N archivos (útil para pruebas rápidas)
    if limit is not None:
        trip_files = trip_files.head(limit)
        logger.info(f"Interim limit enabled: {limit}")

    # --- Viajes: normalizar, calcular duración, marcar anomalías, quitar duplicados ---
    trips = read_all_trips(trip_files, manifest)
    raw_rows = trips["source_file"].value_counts()
    rows_raw = len(trips)
    trips = add_quality_flags(add_duration(trips))
    trips, duplicates_removed = drop_exact_duplicates(trips)
    logger.info(f"Rows raw: {rows_raw:,} | duplicates removed: {duplicates_removed:,}")

    # --- Estaciones: catálogo (foto 2026-09) + ciclo de vida a partir de los viajes ---
    nomenclature = load_nomenclature(nomenclature_path, manifest[nomenclature_path.name])
    stations = build_station_lifecycle(trips, nomenclature)
    trips = add_station_metadata_flags(trips, nomenclature["station_id"])

    # --- Escritura: datos a data/interim (fuera de Git) ---
    INTERIM_DATA_DIR.mkdir(parents=True, exist_ok=True)
    write_trips(trips)
    stations.to_parquet(STATIONS_PATH, index=False)

    # --- Evidencia pequeña a reports/interim (versionada) ---
    INTERIM_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    summary = quality_summary(trips, stations, rows_raw, duplicates_removed)
    schema = {"trips": schema_of(trips), "stations": schema_of(stations)}
    (INTERIM_REPORTS_DIR / "quality_summary.json").write_text(json.dumps(summary, indent=2))
    (INTERIM_REPORTS_DIR / "schema.json").write_text(json.dumps(schema, indent=2))
    file_summary(raw_rows, trips).to_csv(INTERIM_REPORTS_DIR / "file_summary.csv", index=False)

    # --- Trazabilidad: una corrida de MLflow = una versión lógica del dataset ---
    if track:
        # Métricas = valores numéricos del resumen; el rango de fechas va como parámetro
        metrics = {key: value for key, value in summary.items() if isinstance(value, int)}
        params = tracking_params(trip_files, nomenclature_path)
        params.update(
            started_at_min=summary["started_at_min"], started_at_max=summary["started_at_max"]
        )
        log_dataset_run(
            run_name=f"build_{DATASET_VERSION}",
            tags={"pipeline_stage": "interim", "dataset_version": DATASET_VERSION, **git},
            params=params,
            metrics=metrics,
            artifacts=[*sorted(INTERIM_REPORTS_DIR.iterdir()), MANIFEST_PATH],
            datasets={
                "interim_trips": (trips, str(TRIPS_DIR)),
                "interim_stations": (stations, str(STATIONS_PATH)),
            },
        )

    logger.success(
        f"Interim complete: {len(trips):,} trips, {len(stations)} stations. "
        f"Data: {INTERIM_DATA_DIR} | Reports: {INTERIM_REPORTS_DIR}"
    )


if __name__ == "__main__":
    app()
