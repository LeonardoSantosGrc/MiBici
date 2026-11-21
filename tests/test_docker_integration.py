"""Live container reproducibility: HTTP results versus the host's exported model.

By design, ordinary `pytest` skips this module. Run only after starting Docker:
    MIBICI_API_URL=http://127.0.0.1:8000 python -m pytest -q tests/test_docker_integration.py
"""

import os
from pathlib import Path

import mlflow.sklearn
import numpy as np
import pandas as pd
import pytest
import requests

from mibici.config import CHAMPION_EXPORT_DIR, SNAPSHOT_PATH
from mibici.serving.snapshot import build_feature_table, day_rows

BASE_URL = os.environ.get("MIBICI_API_URL", "").rstrip("/")
if not BASE_URL:
    pytest.skip("Set MIBICI_API_URL to enable live Docker checks", allow_module_level=True)

DATES_AND_TOTALS = (
    ("2026-09-30", 11795.739),
    ("2026-10-01", 13307.735),
)
ATOL = 0.00101  # The API serializes predictions rounded to 3 decimals.


@pytest.fixture(scope="module")
def host_reference():
    assert Path(CHAMPION_EXPORT_DIR, "MLmodel").is_file(), "Missing exported champion"
    assert SNAPSHOT_PATH.is_file(), "Run make serving-snapshot before testing Docker"
    model = mlflow.sklearn.load_model(str(CHAMPION_EXPORT_DIR))
    table = build_feature_table(pd.read_parquet(SNAPSHOT_PATH))
    result = {}
    for date, expected_total in DATES_AND_TOTALS:
        rows = day_rows(table, date)
        values = np.asarray(model.predict(rows), dtype=float)
        reference = rows[["station_id", "hour_start"]].copy()
        reference["predicted_departures"] = values
        assert abs(float(values.sum()) - expected_total) < 0.002, (
            f"Local reference disagrees with the team contract for {date}"
        )
        result[date] = reference
    return result


def test_health_and_version():
    response = requests.get(f"{BASE_URL}/health", timeout=120)
    response.raise_for_status()
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert str(body["model_version"]) == "3"
    assert body["first_forecast_date"] == "2026-08-01"
    assert body["last_forecast_date"] == "2026-10-01"
    assert len(response.headers["X-Request-ID"]) == 32


def test_model_info():
    response = requests.get(f"{BASE_URL}/model-info", timeout=120)
    response.raise_for_status()
    body = response.json()
    assert body["metadata"]["version"] == 3 or str(body["metadata"]["version"]) == "3"
    assert len(body["features"]) == 7
    assert body["last_forecast_date"] == "2026-10-01"
    assert len(response.headers["X-Request-ID"]) == 32


@pytest.mark.parametrize("date,expected_total", DATES_AND_TOTALS)
def test_container_forecasts_match_host(date, expected_total, host_reference):
    response = requests.post(f"{BASE_URL}/predict", json={"date": date}, timeout=180)
    response.raise_for_status()
    body = response.json()
    assert response.headers["X-Request-ID"] == body["request_id"]
    assert str(body["model"]["version"]) == "3"
    assert abs(body["total_predicted_departures"] - expected_total) <= ATOL
    assert body["is_next_day"] == (date == "2026-10-01")
    if date == "2026-10-01":
        assert body["total_observed_departures"] is None
    else:
        assert body["total_observed_departures"] is not None

    predictions = [
        (station["station_id"], hour["hour_start"], hour["predicted_departures"])
        for station in body["stations"]
        for hour in station["hours"]
    ]
    assert len(predictions) == len(host_reference[date])
    received = pd.DataFrame(
        predictions, columns=["station_id", "hour_start", "predicted_departures"]
    )
    received["hour_start"] = pd.to_datetime(received["hour_start"])
    reference = host_reference[date]
    joined = reference.merge(
        received,
        on=["station_id", "hour_start"],
        how="outer",
        suffixes=("_host", "_container"),
        validate="one_to_one",
        indicator=True,
    )
    assert joined["_merge"].eq("both").all(), "Mismatch in forecast station/hour keys"
    differences = (
        joined["predicted_departures_host"] - joined["predicted_departures_container"]
    ).abs()
    assert differences.max() <= ATOL, f"Max prediction deviation: {differences.max()}"
