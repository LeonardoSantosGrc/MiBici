"""Tests for the champion check, the forecast of one day and model registration.

No MLflow server and no test split: registration is checked against a temporary MLflow
store inside the test folder (added after the first real registration failed, 2026-11-07).
"""

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import pytest

from mibici.features import add_history_features
from mibici.config import REGISTERED_MODEL_NAME
from mibici.modeling.models import MODELS, make_model
from mibici.modeling.predict import predict_day
from mibici.modeling.registry import check_champion
from mibici.modeling.train import check_mode, register_model


def make_comparison() -> pd.DataFrame:
    """Comparison table: hgb beats the baseline everywhere, random_forest loses in June."""
    return pd.DataFrame(
        {
            "beats_baseline_validation": [True, True],
            "beats_baseline_june": [True, False],
            "beats_baseline_july": [True, True],
            "candidate": [True, False],
        },
        index=pd.Index(["hgb", "random_forest"], name="model"),
    )


def test_champion_must_be_a_candidate():
    table = make_comparison()
    check_champion(table, "hgb")  # accepted
    with pytest.raises(ValueError, match="does not beat the baseline in june"):
        check_champion(table, "random_forest")


def test_champion_needs_compared_runs():
    with pytest.raises(ValueError, match="no clean runs"):
        check_champion(make_comparison(), "poisson")


def test_register_only_the_final_model():
    # Registering is refused for every mode except the confirmed final evaluation
    for mode in ("validation", "window_june", "window_july"):
        with pytest.raises(ValueError, match="REGISTER=1"):
            check_mode(mode, final=False, register=True)
    check_mode("test", final=True, register=True)


def make_table() -> pd.DataFrame:
    """Two stations with rows at 08:00 and 09:00 every day of January 2026."""
    hours = pd.date_range("2026-01-01", "2026-01-31 23:00", freq="h")
    hours = hours[hours.hour.isin([8, 9])]
    frames = []
    for station_id in (1, 2):
        frames.append(
            pd.DataFrame(
                {
                    "station_id": station_id,
                    "hour_start": hours,
                    "departures": station_id * 2,
                    "hour": hours.hour,
                    "day_of_week": hours.day_of_week,
                    "is_weekend": hours.day_of_week >= 5,
                    "month": hours.month,
                    "station_age_days": (hours - hours[0]).days,
                    "is_new_station_cohort": False,
                    "is_extraordinary_event": False,
                }
            )
        )
    return add_history_features(pd.concat(frames, ignore_index=True))


def test_predict_day_returns_every_station_and_hour_of_that_day():
    data = make_table()
    model = make_model("baseline").fit(data, data["departures"])
    forecast = predict_day(data, model, "2026-01-20")
    assert list(forecast.columns) == ["station_id", "hour_start", "predicted_departures"]
    assert len(forecast) == 4  # 2 stations x 2 hours
    assert (forecast["hour_start"].dt.date == pd.Timestamp("2026-01-20").date()).all()
    # Constant departures per station: the four-week average forecasts them exactly
    np.testing.assert_array_equal(forecast["predicted_departures"], [2, 2, 4, 4])


def test_predict_day_outside_the_data_fails():
    data = make_table()
    model = make_model("baseline").fit(data, data["departures"])
    with pytest.raises(ValueError, match="No rows for 2026-03-01"):
        predict_day(data, model, "2026-03-01")


def test_predict_day_without_history_fails():
    data = make_table()
    model = make_model("baseline").fit(data, data["departures"])
    with pytest.raises(ValueError, match="no earlier week"):
        predict_day(data, model, "2026-01-02")


@pytest.fixture
def temporary_mlflow(tmp_path):
    """MLflow store and artifacts in a temporary folder; the real server is never touched."""
    previous = mlflow.get_tracking_uri()
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow.db'}")
    experiment = mlflow.create_experiment("test", artifact_location=(tmp_path / "art").as_uri())
    yield experiment
    mlflow.set_tracking_uri(previous)


# Known warnings that only this small test table or the temporary store produce; this test
# checks saving and loading, not forecast quality:
# - the untuned defaults may not converge on so few rows;
# - max_samples=0.05 of the random_forest block draws only a few rows per tree here
#   (about 566,000 with the real training data);
# - MLflow's SQLite store uses a SQLAlchemy option marked as deprecated (MLflow internals)
@pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")
@pytest.mark.filterwarnings("ignore:Using the fractional value max_samples:UserWarning")
@pytest.mark.filterwarnings("ignore::sqlalchemy.exc.SADeprecationWarning")
def test_every_team_model_can_be_registered_and_loaded(temporary_mlflow):
    # Every block, with its own pieces (common clipping, poisson preprocessing, trees), must
    # survive save -> register -> load and give the same forecast
    data = make_table()
    data = data[data["history_source"] != "none"]  # rows with history, as in training
    for number, name in enumerate(MODELS, start=1):
        pipeline = make_model(name).fit(data, data["departures"])
        with mlflow.start_run(experiment_id=temporary_mlflow):
            version = register_model(pipeline, name, git_commit="abc1234")
        assert int(version) == number
        loaded = mlflow.sklearn.load_model(f"models:/{REGISTERED_MODEL_NAME}/{version}")
        np.testing.assert_allclose(loaded.predict(data), pipeline.predict(data))
        tags = mlflow.MlflowClient().get_model_version(REGISTERED_MODEL_NAME, version).tags
        assert tags == {"model": name, "owner": MODELS[name]["owner"], "git_commit": "abc1234"}
