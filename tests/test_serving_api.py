"""Tests for the forecasting API with the exported champion and a small history snapshot.

The real champion (models/serving/champion, committed to Git) is loaded, so these tests also
check that the deployable artifact works; the history comes from the processed_table fixture.
"""

import numpy as np
from fastapi.testclient import TestClient
import mlflow.sklearn
import pytest

from mibici.config import CHAMPION_EXPORT_DIR
from mibici.serving.app import create_app
from mibici.serving.snapshot import build_feature_table, build_snapshot, day_rows

FIRST_DATE = "2026-07-15"  # first served date of the processed_table fixture (conftest.py)
NEXT_DAY = "2026-08-16"  # the fixture's data ends on August 15


@pytest.fixture
def snapshot(processed_table):
    return build_snapshot(processed_table, FIRST_DATE)


@pytest.fixture
def client(snapshot, tmp_path):
    """API client with the champion and the small snapshot; `with` runs the startup."""
    path = tmp_path / "history_snapshot.parquet"
    snapshot.to_parquet(path, index=False)
    with TestClient(create_app(str(CHAMPION_EXPORT_DIR), str(path))) as test_client:
        yield test_client


def test_health_reports_the_model_and_the_dates_it_can_forecast(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "model_loaded": True,
        "model_version": 3,
        "first_forecast_date": FIRST_DATE,
        "last_forecast_date": NEXT_DAY,
    }


def test_health_without_a_loaded_model_is_unavailable():
    # Without `with`, the startup does not run and no model is loaded
    response = TestClient(create_app()).get("/health")
    assert response.status_code == 503


def test_forecast_of_a_past_day_has_every_station_and_hour(client):
    response = client.post("/forecast", json={"date": "2026-08-01"})
    assert response.status_code == 200
    body = response.json()

    assert body["date"] == "2026-08-01"
    assert body["is_next_day"] is False
    assert body["model"]["version"] == 3 and body["model"]["model"] == "hgb"
    assert len(body["request_id"]) == 32 and body["latency_ms"] >= 0
    # Two stations with two service hours each (0 and 8), none negative
    assert [station["station_id"] for station in body["stations"]] == [1, 2]
    hours = [hour for station in body["stations"] for hour in station["hours"]]
    assert len(hours) == 4
    assert all(hour["predicted_departures"] >= 0 for hour in hours)
    total = sum(hour["predicted_departures"] for hour in hours)
    assert body["total_predicted_departures"] == pytest.approx(total, abs=0.01)


def test_forecast_of_the_next_day_is_flagged(client):
    response = client.post("/forecast", json={"date": NEXT_DAY})
    assert response.status_code == 200
    assert response.json()["is_next_day"] is True


def test_api_returns_what_the_registered_pipeline_predicts(client, snapshot):
    # Train-serving skew check at the HTTP level: same rows, same model, same numbers
    rows = day_rows(build_feature_table(snapshot), "2026-08-10")
    expected = mlflow.sklearn.load_model(str(CHAMPION_EXPORT_DIR)).predict(rows)

    body = client.post("/forecast", json={"date": "2026-08-10"}).json()
    served = [hour["predicted_departures"] for s in body["stations"] for hour in s["hours"]]
    np.testing.assert_allclose(served, expected, atol=0.001)  # the API rounds to 3 decimals


def test_forecast_of_one_station(client):
    body = client.post("/forecast", json={"date": "2026-08-01", "station_id": 2}).json()
    assert [station["station_id"] for station in body["stations"]] == [2]


def test_station_not_active_that_day_is_not_found(client):
    # Station 2 opens on July 20
    response = client.post("/forecast", json={"date": FIRST_DATE, "station_id": 2})
    assert response.status_code == 404


def test_date_outside_the_range_explains_which_dates_exist(client):
    response = client.post("/forecast", json={"date": "2026-11-20"})
    assert response.status_code == 422
    assert f"from {FIRST_DATE} to {NEXT_DAY}" in response.json()["detail"]


@pytest.mark.parametrize(
    "payload",
    [
        {},  # date missing
        {"date": "2026-02-30"},  # not a real date
        {"date": "tomorrow"},  # not a date
        {"date": "2026-08-01", "station_id": 0},  # station ids are positive
        {"date": "2026-08-01", "station": 1},  # unknown field (typo)
    ],
)
def test_invalid_requests_are_refused(client, payload):
    assert client.post("/forecast", json=payload).status_code == 422
