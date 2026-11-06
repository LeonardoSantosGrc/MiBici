"""Common baseline of the modeling contract (notebook 3.01, sections 5 and 7.4).

Forecast = mean departures of the same station, weekday and hour over the last four weeks,
with the young-station and all-station fallbacks. That value is already the ``avg_4_weeks``
feature (``mibici.features``), so the baseline learns nothing in ``fit`` and returns that
column in ``predict``. It is written as a scikit-learn estimator so it goes through the same
pipeline, evaluation and MLflow logging as the three team models.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_is_fitted


class FourWeekAverageBaseline(RegressorMixin, BaseEstimator):
    """Predict the four-week average of the same station and hour (``avg_4_weeks``).

    Parameters
    ----------
    column : str
        Feature that holds the four-week average.
    """

    def __init__(self, column: str = "avg_4_weeks"):
        # scikit-learn rule: __init__ only stores the parameters, without checks
        self.column = column

    def fit(self, X: pd.DataFrame, y=None):
        """Check that the average is among the features; nothing is learned."""
        self._check_column(X)
        # Attributes ending in "_" mark the estimator as fitted (check_is_fitted looks for them)
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = X.shape[1]
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return the four-week average of every row."""
        # Fails with a clear message if predict is called before fit
        check_is_fitted(self)
        self._check_column(X)
        forecast = X[self.column].to_numpy(dtype=float)
        # Every row used for modeling has history; a NaN means the features were not built
        if np.isnan(forecast).any():
            raise ValueError(f"{self.column!r} has missing values; build the features first.")
        return forecast

    def _check_column(self, X) -> None:
        """Require a table with column names that includes the average."""
        if not isinstance(X, pd.DataFrame) or self.column not in X.columns:
            raise ValueError(f"X must be a DataFrame with the column {self.column!r}.")
