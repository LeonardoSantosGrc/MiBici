"""Models of Phase 2: one block per model (notebook 3.01, section 7.7).

Each team member edits only the block of their own model:

- ``poisson``: José Eduardo De Moraes López
- ``random_forest``: Erik Román Cortez
- ``hgb``: José Emmanuel Vázquez Galán

A block has:

- ``estimator``: the scikit-learn regressor with its hyperparameters;
- ``preprocessing``: an optional transformer for the seven contract features (``None`` passes
  them unchanged);
- ``param_grid``: hyperparameter values to try on validation, as {pipeline parameter: list}.
  Parameters of the estimator are named ``model__regressor__<name>`` and those of the
  preprocessing ``preprocess__<name>``. An empty grid trains the block once as written.

Explore in your own notebook first, then copy the chosen values here so ``make train``
reproduces them. Do not change the features, splits, metrics or the common pipeline.
The blocks start with scikit-learn defaults plus the common seed; nothing is tuned yet.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import PoissonRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from mibici.config import RANDOM_STATE
from mibici.modeling.baseline import FourWeekAverageBaseline
from mibici.modeling.pipeline import build_pipeline

# ---- Preprocessing of the poisson block (José Eduardo, notebook 3.02) ----
# PoissonRegressor is a linear model on the log scale: log(E[departures]) = w·x + b.
# So the calendar enters as categories (one coefficient per hour, weekday and month), the
# hourly profile can differ between weekdays and weekends through an interaction, and the
# station history enters as log(x + offset), which makes the forecast proportional to it.
# Everything is module-level so the fitted pipeline can be pickled (MLflow registry).


class LogOffset(TransformerMixin, BaseEstimator):
    """log(x + offset) of non-negative values; ``offset`` keeps zeros finite.

    A small offset keeps the forecast close to proportional to the history (a station with
    half the history gets about half the forecast); offset=1 is the usual log1p.
    """

    def __init__(self, offset: float = 1.0):
        # scikit-learn rule: __init__ only stores the parameters, without checks
        self.offset = offset

    def fit(self, X, y=None):
        """Nothing to learn; records the number of input columns."""
        self.n_features_in_ = np.asarray(X).shape[1]
        return self

    def transform(self, X):
        """log(X + offset), as floats."""
        return np.log(np.asarray(X, dtype=float) + self.offset)

    def get_feature_names_out(self, input_features=None):
        """Same names as the input columns."""
        return np.asarray(input_features, dtype=object)


def hour_by_day_type(X: pd.DataFrame) -> pd.DataFrame:
    """One category per hour and day type (weekday / weekend): ``hour * 2 + is_weekend``."""
    code = X["hour"].astype(int) * 2 + X["is_weekend"].astype(int)
    return pd.DataFrame({"hour_by_day_type": code}, index=X.index)


def hour_by_weekday(X: pd.DataFrame) -> pd.DataFrame:
    """One category per hour and weekday: ``hour * 10 + day_of_week``."""
    code = X["hour"].astype(int) * 10 + X["day_of_week"].astype(int)
    return pd.DataFrame({"hour_by_weekday": code}, index=X.index)


def _combined_name(transformer, input_features) -> list[str]:
    """Output name of an interaction step, so ``get_feature_names_out`` works end to end."""
    return [transformer.func.__name__]


def _log_scaled(offset: float) -> Pipeline:
    """log(x + offset) followed by standardization, for non-negative skewed values."""
    return Pipeline([("log", LogOffset(offset)), ("scale", StandardScaler())])


def poisson_preprocessing(
    interaction: str | None = "day_type", history_offset: float | None = 0.1
):
    """Encoding of the seven contract features for the Poisson regressor.

    Parameters
    ----------
    interaction
        ``"day_type"`` adds hour × (weekday / weekend), ``"weekday"`` adds hour × weekday and
        ``None`` adds no interaction (one hourly profile for every day).
    history_offset
        ``avg_4_weeks`` and ``t_168`` enter as log(x + history_offset) (multiplicative effect);
        ``None`` only standardizes them (additive on the log scale). Tunable in a grid as
        ``preprocess__history__log__offset``.
    """
    interactions = {
        "day_type": (hour_by_day_type, ["hour", "is_weekend"]),
        "weekday": (hour_by_weekday, ["hour", "day_of_week"]),
    }
    if interaction is not None and interaction not in interactions:
        raise ValueError(f"Unknown interaction {interaction!r}; use {list(interactions)} or None.")
    history = StandardScaler() if history_offset is None else _log_scaled(history_offset)
    onehot = OneHotEncoder(handle_unknown="ignore")
    transformers = [
        ("calendar", onehot, ["hour", "day_of_week", "month"]),
        ("history", history, ["avg_4_weeks", "t_168"]),
        ("age", _log_scaled(1.0), ["station_age_days"]),
        ("weekend", "passthrough", ["is_weekend"]),
    ]
    if interaction is not None:
        combine, columns = interactions[interaction]
        combined = FunctionTransformer(combine, feature_names_out=_combined_name)
        step = Pipeline([("combine", combined), ("onehot", clone(onehot))])
        transformers.append((f"hour_x_{interaction}", step, columns))
    # sparse_threshold=1.0 keeps the one-hot output sparse (≈10.8 M training rows)
    return ColumnTransformer(transformers, remainder="drop", sparse_threshold=1.0)


MODELS = {
    # Common reference (notebook 3.01, section 5): not tuned, nobody edits it
    "baseline": {
        "owner": "Equipo 04",
        "estimator": FourWeekAverageBaseline(),
        "preprocessing": None,
        "param_grid": {},
    },
    # ---- José Eduardo De Moraes López ----
    "poisson": {
        "owner": "José Eduardo De Moraes López",
        # Best on validation (notebook 3.02, section 6): hour × weekday, log(history + 0.1)
        "estimator": PoissonRegressor(alpha=1e-6, max_iter=1000),
        "preprocessing": poisson_preprocessing(interaction="weekday", history_offset=0.1),
        # Grid of notebook 3.02, section 6 (15 runs, seconds each)
        "param_grid": {
            "preprocess__history__log__offset": [0.01, 0.05, 0.1, 0.3, 1.0],
            "model__regressor__alpha": [1e-6, 1e-4, 1e-2],
        },
    },
    # ---- Erik Román Cortez ----
    "random_forest": {
        "owner": "Erik Román Cortez",
        # n_jobs=-1 uses every CPU core; it changes the speed, not the result
        "estimator": RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1),
        "preprocessing": None,
        "param_grid": {},
    },
    # ---- José Emmanuel Vázquez Galán ----
    "hgb": {
        "owner": "José Emmanuel Vázquez Galán",
        "estimator": HistGradientBoostingRegressor(random_state=RANDOM_STATE),
        "preprocessing": None,
        "param_grid": {},
    },
}


def make_model(name: str) -> Pipeline:
    """New, unfitted common pipeline for the model ``name`` (a key of ``MODELS``)."""
    if name not in MODELS:
        raise ValueError(f"Unknown model {name!r}; use one of {list(MODELS)}.")
    block = MODELS[name]
    # clone() returns fresh copies, so fitting one pipeline never changes the blocks above
    return clone(build_pipeline(block["estimator"], block["preprocessing"]))
