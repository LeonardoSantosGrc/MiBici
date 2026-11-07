"""Forecast the departures of every station and service hour of one day with a registered model.

    make predict DATE=2026-09-30

Loads ``REGISTERED_MODEL_NAME@CHAMPION_ALIAS`` from the MLflow Model Registry (or any model
URI given with ``--model-uri``), so it does not depend on which algorithm won. The output CSV
has ``station_id``, ``hour_start`` and ``predicted_departures``.

Limitation: the day must exist in processed_v1 (up to 2026-09-30), because its rows say which
stations were active. Forecasting a day after the data would require building those rows.
"""

from pathlib import Path

from loguru import logger
import mlflow
import mlflow.sklearn
import pandas as pd
import typer

from mibici.config import (
    CHAMPION_ALIAS,
    MLFLOW_TRACKING_URI,
    PREDICTIONS_DIR,
    REGISTERED_MODEL_NAME,
)
from mibici.modeling.train import load_modeling_table
from mibici.tracking import check_tracking_server

app = typer.Typer()

DEFAULT_MODEL_URI = f"models:/{REGISTERED_MODEL_NAME}@{CHAMPION_ALIAS}"


def predict_day(data: pd.DataFrame, model, date: str) -> pd.DataFrame:
    """Forecast of every row of ``date`` in a table with the contract features."""
    day = pd.Timestamp(date)
    rows = data[data["hour_start"].dt.normalize() == day]
    if rows.empty:
        first, last = data["hour_start"].min().date(), data["hour_start"].max().date()
        raise ValueError(f"No rows for {day.date()}; the data covers {first} to {last}.")
    # The first week of the data has no history, so its features cannot be computed
    if (rows["history_source"] == "none").any():
        raise ValueError(f"{day.date()} has no earlier week in the data to build features.")
    # The pipeline selects the contract features itself and clips the forecast at 0
    return pd.DataFrame(
        {
            "station_id": rows["station_id"].to_numpy(),
            "hour_start": rows["hour_start"].to_numpy(),
            "predicted_departures": model.predict(rows),
        }
    ).sort_values(["station_id", "hour_start"], ignore_index=True)


@app.command()
def main(
    date: str = typer.Option(..., help="Day to forecast (YYYY-MM-DD), present in processed_v1."),
    model_uri: str = typer.Option(DEFAULT_MODEL_URI, help="MLflow model URI to load."),
    output_dir: str = typer.Option(str(PREDICTIONS_DIR), help="Folder for the CSV."),
) -> None:
    """Forecast DATE with the registered model and save predictions_<DATE>.csv."""
    check_tracking_server()
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    model = mlflow.sklearn.load_model(model_uri)
    predictions = predict_day(load_modeling_table(), model, date)

    output = Path(output_dir) / f"predictions_{pd.Timestamp(date).date()}.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output, index=False)
    logger.success(
        f"{len(predictions):,} forecasts ({predictions['station_id'].nunique()} stations) "
        f"with {model_uri}: {output}"
    )


if __name__ == "__main__":
    app()
