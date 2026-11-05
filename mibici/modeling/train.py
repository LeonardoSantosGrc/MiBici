"""Train a model of ``mibici/modeling/models.py`` on an evaluation mode and log it to MLflow.

    make train MODEL=hgb MODE=validation

For every combination of the model's ``param_grid`` (one run if the grid is empty): build the
common pipeline, fit it on the training rows of the mode, predict the evaluated rows, score
it by segment and log one MLflow run with the baseline scored on the same rows.

Modes (mibici.modeling.splits): ``validation`` for tuning, ``window_june`` and ``window_july``
for the stability of the best configuration, and ``test`` for the final evaluation, which
also requires ``--final`` so it is never run by accident. Trained models are not saved yet;
the best configuration of each model is registered at integration.
"""

import time

from loguru import logger
import mlflow
import pandas as pd
from sklearn.model_selection import ParameterGrid
import typer

from mibici.config import (
    FEATURES,
    MLFLOW_TRACKING_URI,
    MODELING_EXPERIMENT_NAME,
    RANDOM_STATE,
    STATION_HOUR_PATH,
    TARGET,
)
from mibici.features import add_history_features
from mibici.modeling.metrics import evaluate, flatten_scores
from mibici.modeling.models import MODELS, make_model
from mibici.modeling.splits import EVALUATION_MODES, train_eval_masks
from mibici.tracking import TEAM, check_tracking_server, describe_dataset, git_info

app = typer.Typer()


def check_mode(mode: str, final: bool) -> None:
    """Reject unknown modes, and the test mode unless it is the final evaluation."""
    if mode not in EVALUATION_MODES:
        raise ValueError(f"Unknown evaluation mode {mode!r}; use one of {EVALUATION_MODES}.")
    if mode == "test" and not final:
        raise ValueError(
            "MODE=test is the final evaluation and is used once. Tune with MODE=validation; "
            "add FINAL=1 only for the final run."
        )


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
    df: pd.DataFrame, model: str, mode: str, final: bool = False, track: bool = True
) -> pd.DataFrame:
    """Fit every grid combination of ``model`` on ``mode`` and return one row per run."""
    check_mode(mode, final)
    if model not in MODELS:
        raise ValueError(f"Unknown model {model!r}; use one of {list(MODELS)}.")
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
    grid = list(ParameterGrid(MODELS[model]["param_grid"]))  # [{}] when the grid is empty
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
                table = pd.concat({"model": scores, "baseline": baseline_scores})
                mlflow.log_text(table.to_csv(), "scores_by_segment.csv")

        results.append(
            {
                "run": run_name,
                **grid_params,
                "wape": scores.loc["all", "wape"],
                "mae": scores.loc["all", "mae"],
                "baseline_wape": baseline_scores.loc["all", "wape"],
                "baseline_mae": baseline_scores.loc["all", "mae"],
                "fit_seconds": round(fit_seconds, 1),
            }
        )
        logger.info(f"{run_name}: WAPE {results[-1]['wape']:.3f} ({fit_seconds:.0f} s)")
    return pd.DataFrame(results).set_index("run")


@app.command()
def main(
    model: str = typer.Option(..., help=f"Model block of models.py: {', '.join(MODELS)}."),
    mode: str = typer.Option(
        "validation", help=f"Evaluation mode: {', '.join(EVALUATION_MODES)}."
    ),
    final: bool = typer.Option(False, help="Confirm the final evaluation (required by test)."),
    track: bool = typer.Option(True, help="Log the runs to MLflow (requires `make mlflow`)."),
) -> None:
    """Train MODEL on MODE for every combination of its grid and log each run to MLflow."""
    # Check the request and the server before the slow part (loading and features)
    check_mode(mode, final)
    if track:
        check_tracking_server()
        if git_info()["git_dirty"] == "true":
            logger.warning("Uncommitted changes: the MLflow runs will be tagged git_dirty=true")

    df = load_modeling_table()
    results = run_training(df, model, mode, final=final, track=track)
    # Summary table: one line per run, against the baseline on the same rows
    logger.success(f"{model} on {mode}:\n{results.round(3).to_string()}")


if __name__ == "__main__":
    app()
