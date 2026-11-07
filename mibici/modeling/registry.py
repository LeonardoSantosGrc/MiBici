"""Assign the champion alias to a registered model version, only if it is a candidate.

    make champion VERSION=3

The version's ``model`` tag (set by ``make train ... REGISTER=1``) says which team model it is.
The command rebuilds the comparison from MLflow (``mibici.modeling.compare``, which never
reads the test split) and refuses unless that model beats the baseline on validation, June
and July (notebook 3.01, section 7.6, rule 3). Choosing among several candidates is the
team's decision, made with the comparison table.
"""

from loguru import logger
import mlflow
import pandas as pd
import typer

from mibici.config import CHAMPION_ALIAS, MLFLOW_TRACKING_URI, REGISTERED_MODEL_NAME
from mibici.modeling.compare import comparison_table, load_runs, official_runs
from mibici.tracking import check_tracking_server

app = typer.Typer()


def check_champion(table: pd.DataFrame, model: str) -> None:
    """Fail unless ``model`` is in the comparison table and is a candidate."""
    if model not in table.index:
        raise ValueError(f"{model!r} has no clean runs to compare; run `make train` first.")
    if not table.loc[model, "candidate"]:
        # Name the periods where it does not beat the baseline
        lost = [
            column.removeprefix("beats_baseline_")
            for column in table.columns
            if column.startswith("beats_baseline_") and not table.loc[model, column]
        ]
        raise ValueError(
            f"{model!r} is not a candidate: it does not beat the baseline in {', '.join(lost)}."
        )


@app.command()
def main(
    version: int = typer.Option(..., help=f"Version of {REGISTERED_MODEL_NAME} to promote."),
    commit: str = typer.Option("", help="Compare only runs of this Git commit."),
) -> None:
    """Set the champion alias on VERSION after checking that its model is a candidate."""
    check_tracking_server()
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = mlflow.MlflowClient()
    # Fails with a clear MLflow error if the version does not exist
    model_version = client.get_model_version(REGISTERED_MODEL_NAME, str(version))
    model = model_version.tags.get("model")
    if model is None:
        raise ValueError(f"Version {version} has no 'model' tag; register it with make train.")

    table = comparison_table(official_runs(load_runs(), commit or None))
    check_champion(table, model)
    client.set_registered_model_alias(REGISTERED_MODEL_NAME, CHAMPION_ALIAS, str(version))
    logger.success(
        f"{REGISTERED_MODEL_NAME}@{CHAMPION_ALIAS} -> version {version} ({model}); "
        f"validation WAPE {table.loc[model, 'validation_wape']:.3f}"
    )


if __name__ == "__main__":
    app()
