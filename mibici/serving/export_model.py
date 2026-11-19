"""Export the champion from the MLflow Model Registry to a folder the API can load alone.

    make export-model

Resolves ``REGISTERED_MODEL_NAME@CHAMPION_ALIAS``, downloads that version's MLflow model
(``MLmodel``, ``model.pkl``, ``requirements.txt``...) into ``CHAMPION_EXPORT_DIR`` and writes
``metadata.json`` with the version, run, model id and Git commits, so the served model can be
traced back to the Registry. Run it where the Registry lives (the DGX); the exported folder is
committed to Git, so the API and its Docker image do not need the MLflow server.
"""

from datetime import UTC, datetime
import json
from pathlib import Path
import shutil

from loguru import logger
import mlflow
import typer

from mibici.config import (
    CHAMPION_ALIAS,
    CHAMPION_EXPORT_DIR,
    EXPORT_METADATA_FILE,
    FEATURES,
    MLFLOW_TRACKING_URI,
    REGISTERED_MODEL_NAME,
)
from mibici.tracking import check_tracking_server, git_info

app = typer.Typer()


def export_champion(output_dir: Path, alias: str = CHAMPION_ALIAS) -> dict:
    """Download the model behind ``alias`` into ``output_dir`` and return its metadata.

    Uses the current MLflow tracking URI. ``output_dir`` is replaced, so no file of an older
    export is left behind.
    """
    client = mlflow.MlflowClient()
    # Fails with a clear MLflow error if the alias does not exist
    version = client.get_model_version_by_alias(REGISTERED_MODEL_NAME, alias)

    # Start from an empty folder: a stale model.pkl must never survive a new export
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    mlflow.artifacts.download_artifacts(
        artifact_uri=f"models:/{REGISTERED_MODEL_NAME}@{alias}", dst_path=str(output_dir)
    )

    git = git_info()
    metadata = {
        "registered_model": REGISTERED_MODEL_NAME,
        "alias": alias,
        "version": int(version.version),
        # The version's source is "models:/<model id>"
        "model_id": version.source.removeprefix("models:/"),
        "run_id": version.run_id,
        "model": version.tags.get("model"),
        "owner": version.tags.get("owner"),
        "training_git_commit": version.tags.get("git_commit"),
        # Code that ran the export; "true" means it had uncommitted changes
        "export_git_commit": git["git_commit"],
        "export_git_dirty": git["git_dirty"],
        "exported_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "mlflow_version": mlflow.__version__,
        "features": FEATURES,
    }
    # ensure_ascii=False keeps names such as "José" readable in the file
    (output_dir / EXPORT_METADATA_FILE).write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return metadata


@app.command()
def main(
    output_dir: str = typer.Option(str(CHAMPION_EXPORT_DIR), help="Folder for the export."),
) -> None:
    """Export REGISTERED_MODEL_NAME@CHAMPION_ALIAS for serving."""
    check_tracking_server()
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    metadata = export_champion(Path(output_dir))
    logger.success(
        f"Exported {metadata['registered_model']}@{metadata['alias']} -> version "
        f"{metadata['version']} ({metadata['model']}) into {output_dir}"
    )


if __name__ == "__main__":
    app()
