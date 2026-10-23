"""MLflow helpers shared by the dataset-building steps (interim now, processed later).

Each dataset version is one MLflow run in the ``mibici-datasets`` experiment:
parameters = rules and thresholds, metrics = counts, tags = stage, version and
Git commit, artifacts = small evidence files. Heavy data is never uploaded.
"""

import hashlib
from pathlib import Path
import subprocess

from loguru import logger
import mlflow
from mlflow.data.dataset_source_registry import resolve_dataset_source
from mlflow.data.meta_dataset import MetaDataset
import pandas as pd
import requests

from mibici.config import MLFLOW_TRACKING_URI, PROJ_ROOT

EXPERIMENT_NAME = "mibici-datasets"
TEAM = "tec_tc5044_04"

# Filas usadas para inferir el esquema: los tipos son los mismos en toda la tabla
SCHEMA_SAMPLE_ROWS = 1000


def dataframe_digest(df: pd.DataFrame) -> str:
    """Fingerprint of every row and column: changes if any value in the table changes."""
    # hash_pandas_object da un número por fila (vectorizado: ~4 s para 39 M filas);
    # el MD5 de todos esos números resume la tabla completa en 8 caracteres
    row_hashes = pd.util.hash_pandas_object(df, index=False).values
    return hashlib.md5(row_hashes.tobytes()).hexdigest()[:8]


def describe_dataset(df: pd.DataFrame, source: str, name: str) -> MetaDataset:
    """MLflow dataset entry (name, source, digest, schema) without uploading the data."""
    # mlflow.data.from_pandas recorre la tabla completa para el esquema (~4 min con 39 M
    # filas) y su digest solo usa las primeras 10,000 filas. Aquí el digest cubre todas
    # las filas y el esquema se infiere de una muestra.
    sample = df.head(SCHEMA_SAMPLE_ROWS).copy()
    # MLflow no reconoce los meses (period[M]); en la muestra se describen como texto "2024-05"
    period_columns = [
        column for column in sample if isinstance(sample[column].dtype, pd.PeriodDtype)
    ]
    sample[period_columns] = sample[period_columns].astype(str)
    schema = mlflow.models.infer_signature(sample).inputs
    return MetaDataset(
        source=resolve_dataset_source(source),
        name=name,
        digest=dataframe_digest(df),
        schema=schema,
    )


def git_info() -> dict[str, str]:
    """Current Git commit and whether tracked files have uncommitted changes."""

    def git(*args: str) -> str:
        # Ejecuta un comando git en la raíz del proyecto y devuelve su salida como texto
        result = subprocess.run(
            ["git", *args], cwd=PROJ_ROOT, capture_output=True, text=True, check=True
        )
        return result.stdout.strip()

    # "dirty" = hay cambios sin commit: el código ejecutado no es exactamente el del commit
    dirty = git("status", "--porcelain", "--untracked-files=no") != ""
    return {
        "git_commit": git("rev-parse", "HEAD"),
        "git_branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "git_dirty": str(dirty).lower(),
    }


def check_tracking_server(uri: str = MLFLOW_TRACKING_URI) -> None:
    """Fail fast with a clear message if the MLflow server is not running."""
    # Sin esta revisión, MLflow reintenta varios minutos antes de fallar
    try:
        requests.get(f"{uri}/health", timeout=5).raise_for_status()
    except requests.RequestException as error:
        raise RuntimeError(
            f"MLflow Tracking Server not reachable at {uri}. Start it with `make mlflow`."
        ) from error


def log_dataset_run(
    run_name: str,
    tags: dict[str, str],
    params: dict[str, object],
    metrics: dict[str, float],
    artifacts: list[Path],
    datasets: dict[str, tuple[pd.DataFrame, str]],
) -> str:
    """Log one dataset version to MLflow and return the run id.

    ``datasets`` maps a name to (DataFrame, source path); MLflow stores its schema
    and a digest (fingerprint), not the data itself.
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tags({"team": TEAM, **tags})
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)

        # Datasets como "inputs" de la corrida: aparecen en la UI con su digest y esquema
        for name, (df, source) in datasets.items():
            mlflow.log_input(
                describe_dataset(df, source, name), context=tags.get("pipeline_stage")
            )

        # Solo archivos pequeños de evidencia; nunca los datos raw ni interim
        for path in artifacts:
            mlflow.log_artifact(str(path))

        logger.info(f"MLflow run {run.info.run_id} logged in experiment '{EXPERIMENT_NAME}'")
        return run.info.run_id
