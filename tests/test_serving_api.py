"""Tests for the forecasting API with the exported champion and a small history snapshot.

The real champion (models/serving/champion, committed to Git) is loaded, so these tests also
check that the deployable artifact works; the history comes from the processed_table fixture.
"""

import re

from fastapi.testclient import TestClient
from loguru import logger
import mlflow.sklearn
import numpy as np
import pytest

from mibici.config import CHAMPION_EXPORT_DIR, FEATURES
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
    response = client.post("/predict", json={"date": "2026-08-01"})
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
    response = client.post("/predict", json={"date": NEXT_DAY})
    assert response.status_code == 200
    assert response.json()["is_next_day"] is True


def test_api_returns_what_the_registered_pipeline_predicts(client, snapshot):
    # Train-serving skew check at the HTTP level: same rows, same model, same numbers
    rows = day_rows(build_feature_table(snapshot), "2026-08-10")
    expected = mlflow.sklearn.load_model(str(CHAMPION_EXPORT_DIR)).predict(rows)

    body = client.post("/predict", json={"date": "2026-08-10"}).json()
    served = [hour["predicted_departures"] for s in body["stations"] for hour in s["hours"]]
    np.testing.assert_allclose(served, expected, atol=0.001)  # the API rounds to 3 decimals


def test_forecast_of_one_station(client):
    body = client.post("/predict", json={"date": "2026-08-01", "station_id": 2}).json()
    assert [station["station_id"] for station in body["stations"]] == [2]


def test_station_not_active_that_day_is_not_found(client):
    # Station 2 opens on July 20
    response = client.post("/predict", json={"date": FIRST_DATE, "station_id": 2})
    assert response.status_code == 404


def test_date_outside_the_range_explains_which_dates_exist(client):
    response = client.post("/predict", json={"date": "2026-11-20"})
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
    assert client.post("/predict", json=payload).status_code == 422


# ---- /model-info -------------------------------------------------------------------------


def test_model_info_describes_the_served_model(client):
    response = client.get("/model-info")
    assert response.status_code == 200
    body = response.json()
    # Full metadata.json of the export, not only the fields of ModelInfo
    assert body["metadata"]["version"] == 3 and body["metadata"]["model"] == "hgb"
    assert "exported_at" in body["metadata"] and "training_git_commit" in body["metadata"]
    assert body["model_uri"] == str(CHAMPION_EXPORT_DIR)
    assert body["features"] == FEATURES
    assert body["service_hours"] == [0, 8]  # the hours of the processed_table fixture
    assert body["first_forecast_date"] == FIRST_DATE
    assert body["last_forecast_date"] == NEXT_DAY


def test_model_info_without_a_loaded_model_is_unavailable():
    response = TestClient(create_app()).get("/model-info")
    assert response.status_code == 503


# ---- X-Request-ID header -----------------------------------------------------------------


def test_forecast_header_and_body_share_the_request_id(client):
    response = client.post("/predict", json={"date": "2026-08-01"})
    assert response.headers["X-Request-ID"] == response.json()["request_id"]


def test_every_response_has_its_own_request_id(client):
    first = client.get("/health").headers["X-Request-ID"]
    second = client.get("/health").headers["X-Request-ID"]
    assert re.fullmatch(r"[0-9a-f]{32}", first)
    assert first != second


@pytest.mark.parametrize(
    ("payload", "status"),
    [
        ({"date": "2026-11-20"}, 422),  # date outside the range
        ({"date": "tomorrow"}, 422),  # invalid request
        ({"date": FIRST_DATE, "station_id": 2}, 404),  # station not active that day
    ],
)
def test_error_responses_also_carry_the_request_id(client, payload, status):
    response = client.post("/predict", json=payload)
    assert response.status_code == status
    assert re.fullmatch(r"[0-9a-f]{32}", response.headers["X-Request-ID"])


# ---- Unexpected errors -------------------------------------------------------------------


class BrokenModel:
    """Stands in for the champion and fails, to simulate a bug during a forecast."""

    def predict(self, rows):
        raise RuntimeError("simulated failure")


def test_unexpected_error_answers_500_with_the_request_id_and_logs_it(client):
    client.app.state.model = BrokenModel()
    messages = []
    sink = logger.add(messages.append, level="ERROR")  # capture what the service logs
    try:
        response = client.post("/predict", json={"date": "2026-08-01"})
    finally:
        logger.remove(sink)

    assert response.status_code == 500
    body = response.json()
    request_id = response.headers["X-Request-ID"]
    assert body["request_id"] == request_id
    assert "simulated failure" not in body["detail"]  # the client never sees the traceback
    # The log has the id and the real error, so the id leads to the cause
    assert any(request_id in message and "simulated failure" in message for message in messages)


# ---- Observed departures -----------------------------------------------------------------


def test_past_day_includes_the_observed_departures(client, snapshot):
    body = client.post("/predict", json={"date": "2026-08-01"}).json()
    day = snapshot[snapshot["hour_start"].dt.normalize() == "2026-08-01"]

    served = [hour["observed_departures"] for s in body["stations"] for hour in s["hours"]]
    expected = day.sort_values(["station_id", "hour_start"])["departures"].tolist()
    assert served == expected
    assert body["total_observed_departures"] == float(day["departures"].sum())
    for station in body["stations"]:
        observed = sum(hour["observed_departures"] for hour in station["hours"])
        assert station["total_observed_departures"] == observed


def test_next_day_has_no_observed_departures(client):
    body = client.post("/predict", json={"date": NEXT_DAY}).json()
    assert body["total_observed_departures"] is None
    for station in body["stations"]:
        assert station["total_observed_departures"] is None
        assert all(hour["observed_departures"] is None for hour in station["hours"])
