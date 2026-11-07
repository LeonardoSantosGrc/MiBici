"""Tests for mibici.modeling.pipeline and mibici.modeling.models: the common pipeline."""

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyRegressor
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import ParameterGrid
from sklearn.preprocessing import StandardScaler

from mibici.config import FEATURES, RANDOM_STATE
from mibici.modeling.models import MODELS, make_model
from mibici.modeling.pipeline import NonNegativeRegressor, build_pipeline


def make_rows(n: int = 300) -> tuple[pd.DataFrame, np.ndarray]:
    """Rows with the contract features plus the other processed_v1 columns, and a target."""
    rng = np.random.default_rng(0)
    rows = pd.DataFrame(
        {
            "station_id": rng.integers(1, 20, n),
            "hour_start": pd.date_range("2026-01-01", periods=n, freq="h"),
            "hour": rng.integers(0, 24, n),
            "day_of_week": rng.integers(0, 7, n),
            "is_weekend": rng.integers(0, 2, n).astype(bool),
            "month": rng.integers(1, 13, n),
            "station_age_days": rng.integers(0, 1000, n),
            "avg_4_weeks": rng.uniform(0, 5, n),
            "t_168": rng.integers(0, 8, n).astype(float),
            "is_new_station_cohort": False,
            "history_source": "own_history",
        }
    )
    # Target related to the history, so the models have something to learn
    y = rng.poisson(rows["avg_4_weeks"] + 0.1)
    return rows, y


def test_only_the_contract_features_reach_the_estimator():
    rows, y = make_rows()
    pipeline = build_pipeline(LinearRegression()).fit(rows, y)
    # The fitted estimator saw exactly the seven features, in the contract order
    fitted = pipeline.named_steps["model"].regressor_
    assert list(fitted.feature_names_in_) == FEATURES


def test_extra_columns_and_column_order_do_not_change_predictions():
    rows, y = make_rows()
    pipeline = build_pipeline(LinearRegression()).fit(rows, y)
    # Reverse the column order and add an unrelated column
    reordered = rows[rows.columns[::-1]].assign(unrelated=123)
    np.testing.assert_array_equal(pipeline.predict(rows), pipeline.predict(reordered))


def test_missing_feature_fails():
    rows, y = make_rows()
    with pytest.raises((KeyError, ValueError)):
        build_pipeline(LinearRegression()).fit(rows.drop(columns="t_168"), y)


def test_negative_predictions_are_clipped_at_zero():
    rows, y = make_rows()
    # An estimator that always predicts -2 departures
    pipeline = build_pipeline(DummyRegressor(strategy="constant", constant=-2)).fit(rows, y)
    assert (pipeline.predict(rows) == 0).all()


def test_predictions_follow_the_row_order():
    rows, y = make_rows()
    pipeline = build_pipeline(LinearRegression()).fit(rows, y)
    # Shuffling the rows shuffles the predictions the same way
    order = np.random.default_rng(1).permutation(len(rows))
    np.testing.assert_allclose(pipeline.predict(rows)[order], pipeline.predict(rows.iloc[order]))


def test_preprocessing_step_is_applied():
    rows, y = make_rows()
    pipeline = build_pipeline(LinearRegression(), preprocessing=StandardScaler()).fit(rows, y)
    # After scaling, the features reaching the estimator have mean 0
    scaled = pipeline[:-1].transform(rows)
    np.testing.assert_allclose(np.asarray(scaled).mean(axis=0), 0, atol=1e-9)


def test_wrapper_fits_a_copy_and_requires_fit():
    regressor = LinearRegression()
    wrapper = NonNegativeRegressor(regressor)
    with pytest.raises(NotFittedError):
        wrapper.predict(np.zeros((1, 2)))
    wrapper.fit(np.arange(6).reshape(3, 2), [0, 1, 2])
    # The original object stays unfitted; the fitted copy is regressor_
    assert not hasattr(regressor, "coef_")
    assert hasattr(wrapper.regressor_, "coef_")


# Untuned defaults may not converge on this small table; this test only checks fit/predict
@pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")
@pytest.mark.parametrize("name", list(MODELS))
def test_every_model_fits_and_predicts_non_negative_values(name):
    rows, y = make_rows()
    pipeline = make_model(name).fit(rows, y)
    predictions = pipeline.predict(rows)
    assert predictions.shape == (len(rows),)
    assert (predictions >= 0).all()


@pytest.mark.parametrize("name", list(MODELS))
def test_every_grid_uses_valid_parameter_names(name):
    # Every combination of the grid can be set on the model's pipeline
    for params in ParameterGrid(MODELS[name]["param_grid"]):
        make_model(name).set_params(**params)


def test_models_use_the_common_seed():
    for name in ("random_forest", "hgb"):
        # A block may wrap its estimator (e.g. TransformedTargetRegressor), so check every
        # random_state of the block, including the nested ones (regressor__random_state)
        params = MODELS[name]["estimator"].get_params()
        seeds = [value for key, value in params.items() if key.endswith("random_state")]
        assert seeds and all(seed == RANDOM_STATE for seed in seeds)


def test_make_model_returns_independent_pipelines():
    rows, y = make_rows()
    first = make_model("hgb").fit(rows, y)
    second = make_model("hgb")
    # Fitting one pipeline leaves the other (and the block in MODELS) unfitted
    with pytest.raises(NotFittedError):
        second.predict(rows)
    assert first is not second


def test_unknown_model_is_rejected():
    with pytest.raises(ValueError, match="Unknown model"):
        make_model("xgboost")
