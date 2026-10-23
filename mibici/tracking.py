"""MLflow helpers shared by the dataset-building steps (interim now, processed later).

Each dataset version is one MLflow run in the ``mibici-datasets`` experiment:
parameters = rules and thresholds, metrics = counts, tags = stage, version and
Git commit, artifacts = small evidence files. Heavy data is never uploaded.
"""

from pathlib import Path
import subprocess

from loguru import logger
import mlflow
import pandas as pd
import requests

from mibici.config import MLFLOW_TRACKING_URI, PROJ_ROOT

EXPERIMENT_NAME = "mibici-datasets"
TEAM = "tec_tc5044_04"


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

    ``datasets`` maps a name to (DataFrame, source path); MLflow stores its schema,
    row count and a digest (fingerprint), not the data itself.
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tags({"team": TEAM, **tags})
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)

        # Datasets como "inputs" de la corrida: aparecen en la UI con su digest y esquema
        for name, (df, source) in datasets.items():
            dataset = mlflow.data.from_pandas(df, source=source, name=name)
            mlflow.log_input(dataset, context=tags.get("pipeline_stage"))

        # Solo archivos pequeños de evidencia; nunca los datos raw ni interim
        for path in artifacts:
            mlflow.log_artifact(str(path))

        logger.info(f"MLflow run {run.info.run_id} logged in experiment '{EXPERIMENT_NAME}'")
        return run.info.run_id
