"""Tests for the export of the champion used by the API (no MLflow server is touched)."""

import json

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import pytest

from mibici.config import CHAMPION_ALIAS, EXPORT_METADATA_FILE, FEATURES, REGISTERED_MODEL_NAME
from mibici.features import add_history_features
from mibici.modeling.models import make_model
from mibici.modeling.train import register_model
from mibici.serving.export_model import export_champion


@pytest.fixture
def temporary_mlflow(tmp_path):
    """MLflow store and artifacts in a temporary folder; the real server is never touched."""
    previous = mlflow.get_tracking_uri()
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow.db'}")
    experiment = mlflow.create_experiment("test", artifact_location=(tmp_path / "art").as_uri())
    yield experiment
    mlflow.set_tracking_uri(previous)


def make_table() -> pd.DataFrame:
    """Two stations with rows at 08:00 every day of January 2026, with history features."""
    hours = pd.date_range("2026-01-01 08:00", "2026-01-31 08:00", freq="D")
    frames = [
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
            }
        )
        for station_id in (1, 2)
    ]
    table = add_history_features(pd.concat(frames, ignore_index=True))
    return table[table["history_source"] != "none"]  # rows with history, as in training


# MLflow's SQLite store uses a SQLAlchemy option marked as deprecated (MLflow internals)
@pytest.mark.filterwarnings("ignore::sqlalchemy.exc.SADeprecationWarning")
def test_export_writes_the_champion_and_its_metadata(temporary_mlflow, tmp_path):
    # Register a small model, give it the champion alias and export it
    data = make_table()
    pipeline = make_model("baseline").fit(data, data["departures"])
    with mlflow.start_run(experiment_id=temporary_mlflow):
        version = register_model(pipeline, "baseline", git_commit="abc1234")
    mlflow.MlflowClient().set_registered_model_alias(REGISTERED_MODEL_NAME, CHAMPION_ALIAS, version)

    output = tmp_path / "serving" / "champion"
    output.mkdir(parents=True)
    (output / "stale.txt").write_text("from an older export")  # must be removed
    metadata = export_champion(output)

    # The folder holds a loadable MLflow model and nothing from the older export
    assert {"MLmodel", "model.pkl", "requirements.txt", EXPORT_METADATA_FILE} <= {
        path.name for path in output.iterdir()
    }
    assert not (output / "stale.txt").exists()
    loaded = mlflow.sklearn.load_model(str(output))
    np.testing.assert_allclose(loaded.predict(data), pipeline.predict(data))

    # metadata.json traces the export back to the Registry version and its training commit
    saved = json.loads((output / EXPORT_METADATA_FILE).read_text(encoding="utf-8"))
    assert saved == metadata
    assert saved["version"] == int(version)
    assert saved["alias"] == CHAMPION_ALIAS
    assert saved["model"] == "baseline"
    assert saved["training_git_commit"] == "abc1234"
    assert saved["model_id"].startswith("m-")
    assert saved["features"] == FEATURES
