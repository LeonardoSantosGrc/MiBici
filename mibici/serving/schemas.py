"""Request and response contract of the forecasting API (Pydantic models).

FastAPI validates every request against these models (wrong types, unknown fields or invalid
dates get a 422 before any code runs) and uses them to document the API at ``/docs``.
"""

import datetime as dt

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


class StationForecast(BaseModel):
    """Forecast of one station for the 20 service hours of the day."""

    station_id: int
    total_predicted_departures: float = Field(ge=0)
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
    stations: list[StationForecast]


class HealthResponse(BaseModel):
    """Service status and the dates it can forecast."""

    status: str
    model_loaded: bool
    model_version: int | None = None
    first_forecast_date: dt.date | None = None
    last_forecast_date: dt.date | None = None
