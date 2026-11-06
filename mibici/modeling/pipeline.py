"""Common scikit-learn pipeline of the modeling contract (notebook 3.01, section 7).

Every model goes through the same steps; only the estimator and its own preprocessing change:

    select the FEATURES of mibici.config -> preprocessing -> estimator -> clip at 0

This file is shared architecture: team members do not edit it. Each model's estimator,
preprocessing and hyperparameters live in ``mibici/modeling/models.py``.
"""

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

from mibici.config import FEATURES


class NonNegativeRegressor(RegressorMixin, BaseEstimator):
    """Wrap a regressor and clip its predictions at 0 (departures cannot be negative).

    A scikit-learn ``Pipeline`` only transforms inputs, so the clipping is done by wrapping
    the estimator. Its hyperparameters are reached as ``regressor__<name>``.
    """

    def __init__(self, regressor):
        # scikit-learn rule: __init__ only stores the parameters, without checks
        self.regressor = regressor

    def fit(self, X, y):
        """Fit a copy of the wrapped regressor (the original object is left untouched)."""
        self.regressor_ = clone(self.regressor).fit(X, y)
        return self

    def predict(self, X) -> np.ndarray:
        """Predictions of the wrapped regressor, with negative values set to 0."""
        # Fails with a clear message if predict is called before fit
        check_is_fitted(self)
        return np.clip(self.regressor_.predict(X), 0, None)


def build_pipeline(estimator, preprocessing=None) -> Pipeline:
    """Common pipeline: select the contract features, preprocess, estimate, clip at 0.

    Parameters
    ----------
    estimator
        Any scikit-learn regressor.
    preprocessing
        Optional transformer applied to the selected features (e.g. one-hot encoding for a
        linear model). ``None`` passes the features unchanged.
    """
    # Keep only the contract features, always in the same order; every other column
    # (station_id, hour_start, history_source, evaluation flags) is dropped
    select = ColumnTransformer(
        [("features", "passthrough", FEATURES)],
        remainder="drop",
        verbose_feature_names_out=False,  # keep the original column names
    ).set_output(transform="pandas")  # output a DataFrame, so later steps see column names
    return Pipeline(
        [
            ("select", select),
            # "passthrough" is scikit-learn's name for a step that changes nothing
            ("preprocess", preprocessing if preprocessing is not None else "passthrough"),
            ("model", NonNegativeRegressor(estimator)),
        ]
    )
