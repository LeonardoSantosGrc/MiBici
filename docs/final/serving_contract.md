# Serving contract (FastAPI)

**Owner:** Erik Román Cortez · **Branch:** `feat/final-fastapi` · **Code:** `mibici/serving/`

## Purpose

A planner sends the date to rebalance and receives the expected departures of every active
station for the 20 service hours (0 and 5 to 23) of that day.

## How the service works

| Step | When | What |
|---|---|---|
| Load the model | Startup | `mlflow.sklearn.load_model(MIBICI_MODEL_URI)`; default `models/serving/champion` |
| Read the model identity | Startup | `metadata.json` of the export (version, run, model, training commit) |
| Build the feature table | Startup, once | `build_feature_table(history_snapshot.parquet)`: every date from 2026-08-01 to 2026-10-01 |
| Forecast | Each request | Select the rows of the date, `model.predict(rows)` |

If the model or the snapshot cannot be loaded, the service does not start.

| Variable | Default | Use |
|---|---|---|
| `MIBICI_MODEL_URI` | `models/serving/champion` | Folder of an exported model, or a Registry URI such as `models:/mibici-departures@champion` (needs `make mlflow`) |
| `MIBICI_SNAPSHOT_PATH` | `models/serving/history_snapshot.parquet` | History snapshot |

## Endpoints

| Endpoint | Method | Request | Response | Errors |
|---|---|---|---|---|
| `/health` | GET | — | `status`, `model_loaded`, `model_version`, `first_forecast_date`, `last_forecast_date` | 503 if the model is not loaded |
| `/predict` | POST | `{"date": "2026-10-01"}` or `{"date": "2026-10-01", "station_id": 2}` | See below | 422 invalid request or date outside the range; 404 station not active that day; 503 model not loaded |

`/predict` response:

| Field | Meaning |
|---|---|
| `request_id` | Random id (32 hex characters), also written to the log |
| `generated_at` | UTC time of the forecast |
| `latency_ms` | Time spent building the answer |
| `date` | Requested date |
| `is_next_day` | `true` for the day after the latest data (a real forecast); `false` for an earlier day, whose observed departures exist |
| `model` | `registered_model`, `version`, `model`, `run_id`, `training_git_commit` |
| `total_predicted_departures` | Sum over all returned stations and hours |
| `stations[]` | `station_id`, `total_predicted_departures`, `hours[]` with `hour_start` (local time) and `predicted_departures` (rounded to 3 decimals, never negative) |

## Data contract

| Level | Rule | Where |
|---|---|---|
| Request | `date` required, a real calendar date | `ForecastRequest` (Pydantic) |
| Request | `station_id` optional, integer > 0 | `ForecastRequest` |
| Request | No unknown fields (a typo is refused, not ignored) | `extra="forbid"` |
| Freshness | Dates from the first served day to D+1 (the day after the latest data); any other date gets a 422 that states the available range | `day_rows` |
| Model input | The seven features present and known for every row, every row with history | `build_feature_table` |

The data ends on 2026-09-30, so D+1 is 2026-10-01. Forecasting a later date requires
refreshing the data up to the day before it; that is future work, not part of this delivery.

## Telemetry

One log line per forecast:

```text
predict request_id=<id> date=<date> station=<id|None> model_version=3 rows=9340 latency_ms=61.1
```

## Tests already in place (`tests/test_serving_api.py`, 13 tests)

Health with the real champion; health without a model (503); a full past day (every station
and hour, no negative values, consistent total, model identity, telemetry fields); D+1 flag;
the API returns exactly what the registered pipeline predicts; one station; inactive station
(404); date outside the range (422 with the range); five invalid requests (422).

## Left for Erik

| Task | Detail |
|---|---|
| `GET /model-info` | Full `metadata.json`, the feature list and the forecast range |
| Unexpected errors | Exception handler that answers 500 with the `request_id` and logs the error, instead of a bare traceback |
| `X-Request-ID` header | Same id as the body, so a client can quote it without parsing JSON |
| Observed departures for past days (optional) | For `is_next_day=false`, add the observed departures per hour (the snapshot has them) so a planner can compare |
| Tests | One per new behaviour, in `tests/test_serving_api.py` |
| Usage documentation | `curl` examples in this file and in the README |
| Evidence | `/docs` screenshots and one request/response per endpoint in `reports/evidence/final_fastapi/` |

Do not change the request fields, the existing response fields or the error codes above:
Docker and the integration test depend on them. New fields may be added.
