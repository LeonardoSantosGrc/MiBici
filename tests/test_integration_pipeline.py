"""End-to-end integration test: trips -> processed table -> features -> model -> API -> metrics.

Every stage calls the project's own code; only the input trips are synthetic, so the test runs
in seconds without the real data, the MLflow server or Docker. It follows the real contract:
splits from mibici.config, history features, the HGB block, an MLflow-format model folder like
the exported champion, and the history snapshot the API serves from.
"""

from fastapi.testclient import TestClient
import mlflow.sklearn
import numpy as np
import pandas as pd
import pytest

from mibici.config import FEATURES, SERVING_FIRST_DATE, TARGET
from mibici.features import add_history_features
from mibici.modeling.metrics import evaluate
from mibici.modeling.models import make_model
from mibici.modeling.splits import train_eval_masks
from mibici.modeling.train import run_training
from mibici.processed import build_processed
from mibici.serving.app import create_app
from mibici.serving.snapshot import build_snapshot

DATA_END = "2026-09-30"
NEXT_DAY = "2026-10-01"


def make_trips() -> pd.DataFrame:
    """Interim-like trips of two stations, every day until September 30, 2026.

    Station 1 opens on 2025-12-01 (established); station 2 on 2026-03-01 (2026 cohort).
    Departures follow a daily pattern with more trips at peak hours. Trips return to their own
    station: processed_v1 dates a station by its first trip in or out, so a trip arriving at
    station 2 would make it older than its opening.
    """
    rng = np.random.default_rng(0)
    rows = []
    for origin, destination, opening in ((1, 1, "2025-12-01"), (2, 2, "2026-03-01")):
        for day in pd.date_range(opening, DATA_END, freq="D"):
            for hour, mean in ((7, 3), (8, 5), (13, 2), (18, 4), (21, 1)):
                for _ in range(rng.poisson(mean)):
                    start = day + pd.Timedelta(hours=hour, minutes=int(rng.integers(60)))
                    rows.append((origin, destination, start, 600, False, False))
    trips = pd.DataFrame(
        rows,
        columns=[
            "origin_station_id",
            "destination_station_id",
            "started_at",
            "duration_s",
            "flag_short_same_station",
            "flag_short_cross_station",
        ],
    )
    trips[["origin_station_id", "destination_station_id"]] = trips[
        ["origin_station_id", "destination_station_id"]
    ].astype("Int32")
    trips["duration_s"] = trips["duration_s"].astype("Int64")
    trips["datetime_format"] = "iso"
    return trips


@pytest.fixture(scope="module")
def processed() -> pd.DataFrame:
    """Stage 1, data preparation: the processed_v1 rules applied to the trips."""
    return build_processed(make_trips())


def test_processed_table_follows_the_contract(processed):
    # One row per station and service hour, valid zeros included, no closed hours
    assert not processed.duplicated(["station_id", "hour_start"]).any()
    assert not processed["hour"].isin([1, 2, 3, 4]).any()
    assert (processed["departures"] == 0).any() and (processed["departures"] > 0).any()
    assert processed["hour_start"].max() == pd.Timestamp(f"{DATA_END} 23:00")


def test_training_predicts_and_scores_on_validation(processed):
    # Stage 2, modeling: the same entry point as `make train`, without MLflow
    table = add_history_features(processed)
    results = run_training(table, "hgb", "validation", track=False)
    run = results.iloc[0]
    for metric in ("wape", "mae", "baseline_wape", "baseline_mae"):
        assert np.isfinite(run[metric]) and run[metric] >= 0


# MLflow's model saving warns about pip internals that this test does not use
@pytest.mark.filterwarnings("ignore::UserWarning")
def test_served_forecasts_equal_the_pipeline_and_are_scored(processed, tmp_path):
    # Stage 3, final model: trained up to July like the champion, saved as an MLflow folder
    table = add_history_features(processed)
    train, test = train_eval_masks(table, "test")
    pipeline = make_model("hgb").fit(table[train], table.loc[train, TARGET])
    model_dir = tmp_path / "champion"
    mlflow.sklearn.save_model(
        pipeline,
        str(model_dir),
        serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
    )

    # Stage 4, serving: history snapshot and API, as `make serving-snapshot` and `make api`
    snapshot_path = tmp_path / "history_snapshot.parquet"
    build_snapshot(processed, SERVING_FIRST_DATE).to_parquet(snapshot_path, index=False)
    with TestClient(create_app(str(model_dir), str(snapshot_path))) as client:
        assert client.get("/health").json()["last_forecast_date"] == NEXT_DAY

        # Every test day through the API, compared with the pipeline on the training features
        served = []
        for day in pd.date_range("2026-08-01", DATA_END, freq="D"):
            body = client.post("/predict", json={"date": str(day.date())}).json()
            served += [
                (station["station_id"], pd.Timestamp(hour["hour_start"]), hour["predicted_departures"])
                for station in body["stations"]
                for hour in station["hours"]
            ]
        next_day = client.post("/predict", json={"date": NEXT_DAY})

    served = pd.DataFrame(served, columns=["station_id", "hour_start", "predicted"])
    rows = table[test].merge(served, on=["station_id", "hour_start"], validate="one_to_one")
    assert len(rows) == test.sum()  # every test row was served exactly once
    expected = pipeline.predict(rows[FEATURES])
    np.testing.assert_allclose(rows["predicted"], expected, atol=0.001)  # API rounds to 3 decimals

    # Stage 5, metrics: WAPE and MAE of the served forecasts, by segment
    scores = evaluate(rows, rows["predicted"])
    assert {"all", "established_stations", "new_station_cohort"} <= set(scores.index)
    assert np.isfinite(scores[["wape", "mae"]].to_numpy()).all()

    # The day after the data is a real forecast: no observed departures, values not negative
    assert next_day.status_code == 200 and next_day.json()["is_next_day"] is True
    assert next_day.json()["total_predicted_departures"] >= 0
