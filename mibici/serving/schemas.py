"""Request and response contract of the forecasting API (Pydantic models).

FastAPI validates every request against these models (wrong types, unknown fields or invalid
dates get a 422 before any code runs) and uses them to document the API at ``/docs``.
"""

import datetime as dt
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ForecastRequest(BaseModel):
    """Day to forecast and, optionally, a single station."""

    # Unknown fields are refused, so a typo such as "dat" is not silently ignored
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [{"date": "2026-10-01"}, {"date": "2026-09-30", "station_id": 2}]
        },
    )

    date: dt.date = Field(description="Day to forecast (YYYY-MM-DD).")
    station_id: int | None = Field(
        default=None, gt=0, description="Only this station; all active stations if omitted."
    )


class HourForecast(BaseModel):
    """Forecast departures of one station in one service hour."""

    hour_start: dt.datetime = Field(description="Start of the hour, local time (Guadalajara).")
    predicted_departures: float = Field(ge=0)
    observed_departures: float | None = Field(
        default=None,
        ge=0,
        description="Departures that actually happened, for a past day; null for the next day.",
    )


class StationForecast(BaseModel):
    """Forecast of one station for the 20 service hours of the day."""

    station_id: int
    total_predicted_departures: float = Field(ge=0)
    total_observed_departures: float | None = Field(
        default=None, ge=0, description="Observed departures of the day; null for the next day."
    )
    hours: list[HourForecast]


class ModelInfo(BaseModel):
    """Registry identity of the served model (from metadata.json of the export)."""

    registered_model: str | None = None
    version: int | None = None
    model: str | None = None
    run_id: str | None = None
    training_git_commit: str | None = None


class ForecastResponse(BaseModel):
    """Forecast of one day plus the telemetry of the request."""

    request_id: str = Field(description="Unique id of this request, also written to the log.")
    generated_at: dt.datetime = Field(description="When the forecast was made (UTC).")
    latency_ms: float = Field(description="Time spent building the answer, in milliseconds.")
    date: dt.date
    is_next_day: bool = Field(
        description="True for the day after the latest data (a real forecast); False for an "
        "earlier day, whose observed departures exist and can be compared."
    )
    model: ModelInfo
    total_predicted_departures: float = Field(ge=0)
    total_observed_departures: float | None = Field(
        default=None,
        ge=0,
        description="Observed departures of all returned stations and hours; null for the next "
        "day.",
    )
    stations: list[StationForecast]


class ModelDetails(BaseModel):
    """Everything a client needs to know about the served model before calling /predict."""

    # Allow a field named model_uri (Pydantic reserves some names that start with "model_")
    model_config = ConfigDict(protected_namespaces=())

    model_uri: str = Field(description="Where the service loaded the model from.")
    metadata: dict[str, Any] = Field(
        description="Full metadata.json of the exported model (version, run, commits, export "
        "time); empty if the model was loaded straight from a models:/ URI."
    )
    features: list[str] = Field(description="Input features of the model, in pipeline order.")
    service_hours: list[int] = Field(description="Hours of the day the service forecasts.")
    first_forecast_date: dt.date
    last_forecast_date: dt.date = Field(description="The day after the latest data (D+1).")


class ErrorResponse(BaseModel):
    """Body of an unexpected error (500)."""

    detail: str
    request_id: str = Field(description="Same id as the X-Request-ID header and the log line.")


class HealthResponse(BaseModel):
    """Service status and the dates it can forecast."""

    status: str
    model_loaded: bool
    model_version: int | None = None
    first_forecast_date: dt.date | None = None
    last_forecast_date: dt.date | None = None
