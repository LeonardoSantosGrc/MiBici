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

from sklearn.base import clone
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
