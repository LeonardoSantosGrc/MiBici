import os
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

# Load environment variables from .env file if it exists
load_dotenv()

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

MODELS_DIR = PROJ_ROOT / "models"

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# MLflow Tracking Server local (`make mlflow`). Se puede cambiar con la variable de
# entorno MLFLOW_TRACKING_URI (p. ej. en .env) sin tocar el código.
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")

# Modeling contract (notebook 3.01, section 7). Every model reads these values from here;
# changes are proposed to the team first.
STATION_HOUR_PATH = PROCESSED_DATA_DIR / "station_hour_v1.parquet"
KEY_COLUMNS = ["station_id", "hour_start"]  # identify a row; not model inputs
TARGET = "departures"
FEATURES = [
    "hour",
    "day_of_week",
    "is_weekend",
    "month",
    "station_age_days",
    "avg_4_weeks",
    "t_168",
]
EVALUATION_ONLY = ["is_new_station_cohort", "is_extraordinary_event"]

# Station history: same station and hour, 1 to HISTORY_WEEKS weeks earlier (lags >= 168 h are
# known at the 22:00 cutoff of day d for every hour of day d+1)
HISTORY_WEEKS = 4
YOUNG_STATION_DAYS = 28  # fallback group for stations without their own history

# Periods are [start, end): the end date is excluded
SPLITS = {
    "train": ("2022-01-01", "2026-06-01"),
    "validation": ("2026-06-01", "2026-08-01"),
    "test": ("2026-08-01", "2026-10-01"),
}
# Stability windows: train on [train start, train_end), evaluate on eval
WINDOWS = {
    "window_june": {"train_end": "2026-06-01", "eval": ("2026-06-01", "2026-07-01")},
    "window_july": {"train_end": "2026-07-01", "eval": ("2026-07-01", "2026-08-01")},
}

PRIMARY_METRIC = "wape"
SECONDARY_METRIC = "mae"
RANDOM_STATE = 0
MODELING_EXPERIMENT_NAME = "mibici-modeling"

# If tqdm is installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass
