"""Train a model of ``mibici/modeling/models.py`` on an evaluation mode and log it to MLflow.

    make train MODEL=hgb MODE=validation

For every combination of the model's ``param_grid`` (one run if the grid is empty): build the
common pipeline, fit it on the training rows of the mode, predict the evaluated rows, score
it by segment and log one MLflow run with the baseline scored on the same rows.

Modes (mibici.modeling.splits): ``validation`` for tuning, ``window_june`` and ``window_july``
for the stability of the best configuration, and ``test`` for the final evaluation, which
also requires ``--final`` so it is never run by accident.

``--register`` (``REGISTER=1``) saves the fitted pipeline and registers it as a new version of
``REGISTERED_MODEL_NAME`` in the MLflow Model Registry. It is only allowed for the final
model (``test`` with ``--final``) and for a single configuration (empty grid); the version
is tagged with the model, owner and commit. The champion alias is set later, with
``make champion``, from the comparison on validation and the windows.
"""

import time

from loguru import logger
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import ParameterGrid
import typer

from mibici.config import (
    FEATURES,
    MLFLOW_TRACKING_URI,
    MODELING_EXPERIMENT_NAME,
    RANDOM_STATE,
    REGISTERED_MODEL_NAME,
    STATION_HOUR_PATH,
    TARGET,
)
from mibici.features import add_history_features
from mibici.modeling.metrics import evaluate, flatten_scores
from mibici.modeling.models import MODELS, make_model
from mibici.modeling.splits import EVALUATION_MODES, train_eval_masks
from mibici.tracking import TEAM, check_tracking_server, describe_dataset, git_info

app = typer.Typer()


def check_mode(mode: str, final: bool, register: bool = False) -> None:
    """Reject unknown modes, the test mode unless final, and registering a non-final model."""
    if mode not in EVALUATION_MODES:
        raise ValueError(f"Unknown evaluation mode {mode!r}; use one of {EVALUATION_MODES}.")
    if mode == "test" and not final:
        raise ValueError(
            "MODE=test is the final evaluation and is used once. Tune with MODE=validation; "
            "add FINAL=1 only for the final run."
        )
    # Only the final model (trained on train + validation) goes to the Model Registry
    if register and not (mode == "test" and final):
        raise ValueError("REGISTER=1 registers the final model: use it with MODE=test FINAL=1.")


def load_modeling_table() -> pd.DataFrame:
    """processed_v1 with the station-history features of the contract."""
    return add_history_features(pd.read_parquet(STATION_HOUR_PATH))


def period_params(df: pd.DataFrame, mask: pd.Series, prefix: str) -> dict[str, object]:
    """First hour, last hour and number of rows of a set of rows, for MLflow."""
    hours = df.loc[mask, "hour_start"]
    return {
        f"{prefix}_start": str(hours.min()),
        f"{prefix}_end": str(hours.max()),
        f"{prefix}_rows": int(mask.sum()),
    }


def hyperparameters(pipeline) -> dict[str, str]:
    """Every parameter of the estimator and of the preprocessing, as text for MLflow."""
    # get_params() lists every step; keep only the parts that a model block can change
    return {
        name: str(value)
        for name, value in pipeline.get_params().items()
        if name.startswith(("model__regressor__", "preprocess__")) or name == "preprocess"
    }


def run_training(
    df: pd.DataFrame,
    model: str,
    mode: str,
    final: bool = False,
    track: bool = True,
    register: bool = False,
) -> pd.DataFrame:
    """Fit every grid combination of ``model`` on ``mode`` and return one row per run."""
    check_mode(mode, final, register)
    if model not in MODELS:
        raise ValueError(f"Unknown model {model!r}; use one of {list(MODELS)}.")
    grid = list(ParameterGrid(MODELS[model]["param_grid"]))  # [{}] when the grid is empty
    # One registered version per model: its block must hold a single configuration
    if register and len(grid) > 1:
        raise ValueError(f"REGISTER=1 needs a single configuration; {model!r} has {len(grid)}.")
    if register and not track:
        raise ValueError("REGISTER=1 needs MLflow tracking (do not use --no-track).")
    train, evaluate_rows = train_eval_masks(df, mode)
    eval_df = df[evaluate_rows]

    # The baseline on the same evaluated rows: the reference every run is compared with
    baseline = make_model("baseline").fit(df[train], df.loc[train, TARGET])
    baseline_scores = evaluate(eval_df, baseline.predict(eval_df))

    # Values shared by every run of this command
    common_params = {
        "model": model,
        "mode": mode,
        "features": ",".join(FEATURES),
        "target": TARGET,
        "random_state": RANDOM_STATE,
        **period_params(df, train, "train"),
        **period_params(df, evaluate_rows, "eval"),
    }
    if track:
        git = git_info()
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(MODELING_EXPERIMENT_NAME)
        # Name, source and fingerprint of the data, logged without uploading it
        dataset = describe_dataset(df, str(STATION_HOUR_PATH), "processed_v1")

    results = []
    for number, grid_params in enumerate(grid, start=1):
        pipeline = make_model(model).set_params(**grid_params)
        start = time.perf_counter()
        pipeline.fit(df[train], df.loc[train, TARGET])
        fit_seconds = time.perf_counter() - start
        scores = evaluate(eval_df, pipeline.predict(eval_df))

        run_name = f"{model}-{mode}-{number:02d}"
        metrics = {
            **flatten_scores(scores),
            # Same segments for the baseline, prefixed so both appear side by side
            **{f"baseline_{k}": v for k, v in flatten_scores(baseline_scores).items()},
            "fit_seconds": fit_seconds,
        }
        if track:
            with mlflow.start_run(run_name=run_name):
                mlflow.set_tags({"team": TEAM, "owner": MODELS[model]["owner"], **git})
                mlflow.log_params({**common_params, **hyperparameters(pipeline)})
                mlflow.log_metrics(metrics)
                mlflow.log_input(dataset, context=mode)
                # Scores by segment of the model and of the baseline, as a small CSV
                table = pd.concat({"model": scores, "baseline": baseline_scores}, names=["source"])
                mlflow.log_text(table.to_csv(), "scores_by_segment.csv")
                if register:
                    register_model(pipeline, model, git["git_commit"])

        results.append(
            {
                "run": run_name,
                **grid_params,
                "wape": scores.loc["all", "wape"],
                "mae": scores.loc["all", "mae"],
                "volume_ratio": scores.loc["all", "volume_ratio"],
                "baseline_wape": baseline_scores.loc["all", "wape"],
                "baseline_mae": baseline_scores.loc["all", "mae"],
                "baseline_volume_ratio": baseline_scores.loc["all", "volume_ratio"],
                "fit_seconds": round(fit_seconds, 1),
            }
        )
        logger.info(f"{run_name}: WAPE {results[-1]['wape']:.3f} ({fit_seconds:.0f} s)")
    return pd.DataFrame(results).set_index("run")


def register_model(pipeline, model: str, git_commit: str) -> int:
    """Save the fitted pipeline in the active run, register it and return its version."""
    # MLflow's default format (skops) refuses types it does not know, such as the common
    # NonNegativeRegressor, the poisson preprocessing or sklearn's tree storage. cloudpickle
    # saves any pipeline; it trusts the file, which is safe here because the team creates and
    # loads its own models (team decision of 2026-11-07)
    info = mlflow.sklearn.log_model(
        pipeline,
        name="model",
        registered_model_name=REGISTERED_MODEL_NAME,
        serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
    )
    version = info.registered_model_version
    # Tags tell which team model, owner and code each version comes from
    client = mlflow.MlflowClient()
    for key, value in {
        "model": model,
        "owner": MODELS[model]["owner"],
        "git_commit": git_commit,
    }.items():
        client.set_model_version_tag(REGISTERED_MODEL_NAME, version, key, value)
    logger.info(f"Registered {REGISTERED_MODEL_NAME} version {version} ({model})")
    return version


@app.command()
def main(
    model: str = typer.Option(..., help=f"Model block of models.py: {', '.join(MODELS)}."),
    mode: str = typer.Option(
        "validation", help=f"Evaluation mode: {', '.join(EVALUATION_MODES)}."
    ),
    final: bool = typer.Option(False, help="Confirm the final evaluation (required by test)."),
    track: bool = typer.Option(True, help="Log the runs to MLflow (requires `make mlflow`)."),
    register: bool = typer.Option(
        False, help="Register the final model in the Model Registry (needs test and --final)."
    ),
) -> None:
    """Train MODEL on MODE for every combination of its grid and log each run to MLflow."""
    # Check the request and the server before the slow part (loading and features)
    check_mode(mode, final, register)
    if track:
        check_tracking_server()
        if git_info()["git_dirty"] == "true":
            logger.warning("Uncommitted changes: the MLflow runs will be tagged git_dirty=true")

    df = load_modeling_table()
    results = run_training(df, model, mode, final=final, track=track, register=register)
    # Summary table: one line per run, against the baseline on the same rows
    logger.success(f"{model} on {mode}:\n{results.round(3).to_string()}")


if __name__ == "__main__":
    app()
