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
from sklearn.base import clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import PoissonRegressor
from sklearn.pipeline import Pipeline

from mibici.config import RANDOM_STATE
from mibici.modeling.baseline import FourWeekAverageBaseline
from mibici.modeling.pipeline import build_pipeline

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
        "estimator": PoissonRegressor(),
        "preprocessing": None,
        "param_grid": {},
    },
    # ---- Erik Román Cortez ----
    "random_forest": {
        "owner": "Erik Román Cortez",
        # WAPE rewards the median, but a tree leaf predicts the mean: with skewed counts the
        # first grid (no transform) over-predicted 66% of validation rows and lost to the
        # baseline. Training on log(1 + departures) pulls the leaf values toward the median;
        # predictions come back in departures with exp(x) - 1.
        # n_jobs=6 instead of -1: same result, less heat on the laptop
        "estimator": TransformedTargetRegressor(
            regressor=RandomForestRegressor(
                n_estimators=100,
                max_samples=0.05,
                random_state=RANDOM_STATE,
                n_jobs=6,
            ),
            func=np.log1p,
            inverse_func=np.expm1,
        ),
        "preprocessing": None,  # trees use the seven features without encoding
        # 2 x 2 x 2 factorial around the log-target forest:
        # - max_depth: does extra complexity help now?
        # - max_features: 0.5 forces some splits away from avg_4_weeks (97.8% importance)
        # - min_samples_leaf: smoothing of the leaf values
        "param_grid": {
            "model__regressor__regressor__max_depth": [12, 18],
            "model__regressor__regressor__max_features": [1.0, 0.5],
            "model__regressor__regressor__min_samples_leaf": [50, 200],
        },
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
