"""FastAPI service that forecasts the departures of every station and hour of one day.

    make api            # http://127.0.0.1:8000/docs

At startup it loads the champion (``MIBICI_MODEL_URI``, by default the folder exported with
``make export-model``) and builds, once, the feature table of every date it can forecast from
the history snapshot (``MIBICI_SNAPSHOT_PATH``, built with ``make serving-snapshot``). A request
only selects the rows of its date and calls the model's own pipeline: the API never
re-implements a feature or a transformation.

Endpoints: ``GET /health`` and ``POST /forecast``. Each forecast writes one log line with its
request id, date, model version, rows and latency.
"""

from contextlib import asynccontextmanager
from datetime import UTC, datetime
import json
import os
from pathlib import Path
import time
import uuid

from fastapi import FastAPI, HTTPException, Request
from loguru import logger
import mlflow.sklearn
import pandas as pd

from mibici.config import CHAMPION_EXPORT_DIR, EXPORT_METADATA_FILE, SNAPSHOT_PATH
from mibici.serving.schemas import (
    ForecastRequest,
    ForecastResponse,
    HealthResponse,
    HourForecast,
    ModelInfo,
    StationForecast,
)
from mibici.serving.snapshot import build_feature_table, day_rows

# Defaults work from a clone of the repository; Docker or a developer can override them
DEFAULT_MODEL_URI = os.getenv("MIBICI_MODEL_URI", str(CHAMPION_EXPORT_DIR))
DEFAULT_SNAPSHOT_PATH = os.getenv("MIBICI_SNAPSHOT_PATH", str(SNAPSHOT_PATH))


def load_model_info(model_uri: str) -> ModelInfo:
    """Registry identity from the export's metadata.json; empty if the URI has none."""
    metadata_path = Path(model_uri) / EXPORT_METADATA_FILE
    if not metadata_path.exists():  # e.g. a models:/ URI loaded straight from the Registry
        return ModelInfo()
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    # Keep only the fields ModelInfo declares
    return ModelInfo(**{key: metadata.get(key) for key in ModelInfo.model_fields})


def to_stations(rows: pd.DataFrame, predictions) -> list[StationForecast]:
    """Group the forecast rows of one day by station, hours in order."""
    forecast = rows[["station_id", "hour_start"]].assign(predicted_departures=predictions)
    stations = []
    for station_id, group in forecast.groupby("station_id", sort=True):
        hours = [
            HourForecast(hour_start=hour.to_pydatetime(), predicted_departures=round(value, 3))
            for hour, value in zip(group["hour_start"], group["predicted_departures"])
        ]
        total = round(float(group["predicted_departures"].sum()), 3)
        stations.append(
            StationForecast(
                station_id=int(station_id), total_predicted_departures=total, hours=hours
            )
        )
    return stations


def create_app(
    model_uri: str = DEFAULT_MODEL_URI, snapshot_path: str = DEFAULT_SNAPSHOT_PATH
) -> FastAPI:
    """Build the API for one model and one history snapshot (tests pass their own)."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Runs once at startup: if the model or the snapshot cannot be loaded, the service
        # does not start (a container fails visibly instead of answering with errors)
        app.state.model = mlflow.sklearn.load_model(model_uri)
        app.state.model_info = load_model_info(model_uri)
        app.state.features = build_feature_table(pd.read_parquet(snapshot_path))
        app.state.next_day = app.state.features["hour_start"].max().normalize()
        logger.info(
            f"Model {model_uri} (version {app.state.model_info.version}) and "
            f"{len(app.state.features):,} feature rows ready"
        )
        yield  # the service answers requests while paused here

    app = FastAPI(
        title="MiBici departures forecast",
        description="Forecast of trips started per station and hour for one day.",
        version="1.0.0",
        lifespan=lifespan,
    )

    @app.get("/health", response_model=HealthResponse)
    def health(request: Request) -> HealthResponse:
        """Whether the model is loaded and which dates can be forecast."""
        state = request.app.state
        if not hasattr(state, "model"):
            raise HTTPException(status_code=503, detail="Model not loaded.")
        features = state.features
        return HealthResponse(
            status="ok",
            model_loaded=True,
            model_version=state.model_info.version,
            first_forecast_date=features["hour_start"].min().date(),
            last_forecast_date=state.next_day.date(),
        )

    @app.post("/forecast", response_model=ForecastResponse)
    def forecast(body: ForecastRequest, request: Request) -> ForecastResponse:
        """Forecast departures of every active station (or one) and service hour of a day."""
        start = time.perf_counter()
        request_id = uuid.uuid4().hex  # random id that links the response and the log line
        state = request.app.state
        if not hasattr(state, "model"):
            raise HTTPException(status_code=503, detail="Model not loaded.")

        try:
            rows = day_rows(state.features, body.date)
        except ValueError as error:  # date outside the range: say which dates are available
            raise HTTPException(status_code=422, detail=str(error)) from error
        if body.station_id is not None:
            rows = rows[rows["station_id"] == body.station_id]
            if rows.empty:
                raise HTTPException(
                    status_code=404,
                    detail=f"Station {body.station_id} is not active on {body.date}.",
                )

        # The registered pipeline selects the features, applies its own preprocessing and
        # clips at 0
        predictions = state.model.predict(rows)
        stations = to_stations(rows, predictions)
        latency_ms = round((time.perf_counter() - start) * 1000, 1)
        logger.info(
            f"forecast request_id={request_id} date={body.date} station={body.station_id} "
            f"model_version={state.model_info.version} rows={len(rows)} latency_ms={latency_ms}"
        )
        return ForecastResponse(
            request_id=request_id,
            generated_at=datetime.now(UTC),
            latency_ms=latency_ms,
            date=body.date,
            is_next_day=pd.Timestamp(body.date) == state.next_day,
            model=state.model_info,
            total_predicted_departures=round(float(predictions.sum()), 3),
            stations=stations,
        )

    return app


# Object uvicorn serves: uvicorn mibici.serving.app:app
app = create_app()
