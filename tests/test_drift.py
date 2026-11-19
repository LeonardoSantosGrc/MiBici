"""Tests for the drift statistics, scenarios, alert rules and drift evaluation."""

import numpy as np
import pandas as pd
import pytest

from mibici.config import FEATURES, TARGET
from mibici.features import add_history_features
from mibici.modeling.models import make_model
from mibici.monitoring.drift import (
    NO_ACTION,
    PSI_MODERATE,
    PSI_STRONG,
    data_contract_alert,
    data_contract_violations,
    data_drift_alert,
    evaluate_alerts,
    feature_drift,
    performance_alert,
    performance_without_drift_alert,
    propose_action,
    psi_categorical,
    psi_numeric,
    psi_status,
    quantile_edges,
    wape_tolerance,
)
from mibici.monitoring.evaluation import CONTROL, NATURAL, run_drift
from mibici.monitoring.scenarios import (
    LEVELS,
    bootstrap_control,
    build_scenarios,
    concept_change,
    concept_factors,
    data_failure,
    demand_shift,
    history_window,
    period_rows,
    station_expansion,
)

# Periods of the processed_table fixture (conftest.py: June 1 to August 15, 2026)
REFERENCE = ("2026-06-29", "2026-07-15")
CURRENT = ("2026-07-15", "2026-08-16")


@pytest.fixture
def modeling_table(processed_table) -> pd.DataFrame:
    """Fixture with the history features and the evaluation-only flags.

    Station 2 (opens July 20) plays the new-station cohort; July 1 is an event day where
    departures fell to half.
    """
    table = processed_table.assign(
        is_new_station_cohort=processed_table["station_id"] == 2,
        is_extraordinary_event=processed_table["hour_start"].dt.normalize() == "2026-07-01",
    )
    table = add_history_features(table)
    event = table["is_extraordinary_event"]
    table.loc[event, "departures"] = (table.loc[event, "avg_4_weeks"] / 2).round().astype("int32")
    return add_history_features(table)


@pytest.fixture
def calibration() -> dict:
    """Small magnitudes for the fixture (the real ones come from scenarios.calibrate)."""
    return {
        "S1": {"mild": 1.1, "moderate": 1.5, "strong": 3.0},
        "S2": {"mild": 0.5, "moderate": 0.6, "strong": 0.7},
        "S3": {"mild": 0.5, "moderate": 0.5, "strong": 1.0},
        "S4": {level: {8: f} for level, f in zip(LEVELS, (0.9, 0.7, 0.5))},
    }


def drift_table(feature: str, psi: float, alarm: bool = True) -> pd.DataFrame:
    """One-feature drift table built to sit on a chosen side of a threshold."""
    return pd.DataFrame({"feature": [feature], "psi": [psi], "alarm": [alarm]})


# ---- PSI and KS -----------------------------------------------------------------------------


def test_psi_of_a_sample_against_itself_is_zero():
    sample = np.random.default_rng(0).normal(size=5_000)
    assert psi_numeric(sample, sample) == pytest.approx(0.0)
    assert psi_categorical([1, 2, 2, 3], [1, 2, 2, 3]) == pytest.approx(0.0)


def test_psi_of_a_clearly_shifted_sample_is_strong():
    rng = np.random.default_rng(0)
    reference = rng.normal(0, 1, 5_000)
    assert psi_numeric(reference, rng.normal(1.5, 1, 5_000)) > PSI_STRONG
    assert psi_categorical(rng.integers(0, 7, 5_000), np.full(5_000, 3)) > PSI_STRONG


def test_quantile_edges_merge_ties_and_cover_every_value():
    # Half zeros: the repeated zero quantiles collapse into one edge
    edges = quantile_edges(np.r_[np.zeros(50), np.arange(1, 51)])
    assert edges[0] == -np.inf and edges[-1] == np.inf
    assert len(np.unique(edges)) == len(edges)
    # A constant reference still gives a valid single bin
    assert psi_numeric(np.ones(10), np.ones(10)) == pytest.approx(0.0)


@pytest.mark.parametrize(
    ("psi", "status"), [(0.05, "stable"), (0.1, "moderate"), (0.25, "moderate"), (0.3, "strong")]
)
def test_psi_status_uses_the_industry_thresholds(psi, status):
    assert psi_status(psi) == status


def test_feature_drift_reports_every_feature_and_the_prediction(modeling_table):
    reference = period_rows(modeling_table, REFERENCE)
    current = period_rows(modeling_table, CURRENT)
    table = feature_drift(reference, current, np.ones(len(reference)), np.ones(len(current)))
    assert table["feature"].tolist() == [*FEATURES, "prediction"]
    # month and station_age_days are reported but never alarmed
    assert not table.set_index("feature").loc[["month", "station_age_days"], "alarm"].any()
    assert table.loc[table["kind"] == "categorical", "ks_d"].isna().all()


# ---- Scenarios ------------------------------------------------------------------------------


def test_scenarios_with_the_same_seed_are_identical(modeling_table, calibration):
    first = list(build_scenarios(modeling_table, calibration, CURRENT))
    second = list(build_scenarios(modeling_table, calibration, CURRENT))
    assert [(s.name, s.level) for s in first] == [(s.name, s.level) for s in second]
    assert len(first) == 4 * len(LEVELS)
    for a, b in zip(first, second):
        pd.testing.assert_frame_equal(a.rows, b.rows)


def test_control_bootstrap_does_not_drift():
    # A large synthetic reference: a bootstrap of it must look the same (PSI < 0.1)
    rng = np.random.default_rng(1)
    n = 20_000
    reference = pd.DataFrame(
        {
            "hour": rng.choice([0, *range(5, 24)], n),
            "day_of_week": rng.integers(0, 7, n),
            "month": rng.choice([6, 7], n),
            "station_age_days": rng.integers(0, 3000, n),
            "avg_4_weeks": rng.gamma(1.0, 2.0, n),
            "t_168": rng.poisson(2, n).astype(float),
        }
    )
    reference["is_weekend"] = reference["day_of_week"] >= 5
    drift = feature_drift(reference, bootstrap_control(reference))
    assert (drift["psi"] < PSI_MODERATE).all()
    assert not data_drift_alert(drift)


def test_demand_shift_of_one_keeps_the_training_features(modeling_table):
    # The history window is enough to recompute the current features exactly
    shifted = demand_shift(history_window(modeling_table, CURRENT), 1.0, CURRENT)
    expected = period_rows(modeling_table, CURRENT).reset_index(drop=True)
    np.testing.assert_allclose(shifted["avg_4_weeks"], expected["avg_4_weeks"])
    np.testing.assert_allclose(shifted["t_168"], expected["t_168"])


def test_strong_demand_shift_drifts_the_four_week_average(modeling_table, calibration):
    reference = period_rows(modeling_table, REFERENCE)
    window = history_window(modeling_table, CURRENT)
    shifted = demand_shift(window, calibration["S1"]["strong"], CURRENT)
    assert psi_numeric(reference["avg_4_weeks"], shifted["avg_4_weeks"]) > PSI_STRONG


def test_station_expansion_reaches_the_share_with_the_same_rows(modeling_table):
    current = period_rows(modeling_table, CURRENT).reset_index(drop=True)
    expanded = station_expansion(current, 0.7)
    assert len(expanded) == len(current)
    assert expanded["is_new_station_cohort"].mean() == pytest.approx(0.7, abs=0.01)
    with pytest.raises(ValueError, match="below the current cohort share"):
        station_expansion(current, 0.0)


def test_data_failure_zeroes_t168_of_the_chosen_stations_only(modeling_table):
    current = period_rows(modeling_table, CURRENT).reset_index(drop=True)
    broken = data_failure(current, 0.5)  # one of the two stations
    zeroed = broken.groupby("station_id")["t_168"].apply(lambda s: (s == 0).all())
    assert zeroed.sum() == 1
    pd.testing.assert_frame_equal(broken.drop(columns="t_168"), current.drop(columns="t_168"))


def test_concept_change_moves_the_target_but_not_the_inputs(modeling_table):
    current = period_rows(modeling_table, CURRENT).reset_index(drop=True)
    changed = concept_change(current, {8: 0.5})
    pd.testing.assert_frame_equal(changed[FEATURES], current[FEATURES])
    at_8 = current["hour"] == 8
    np.testing.assert_allclose(changed.loc[at_8, TARGET], current.loc[at_8, TARGET] * 0.5)
    np.testing.assert_allclose(changed.loc[~at_8, TARGET], current.loc[~at_8, TARGET])


def test_concept_factors_come_from_the_event_day_drop(modeling_table):
    factors = concept_factors(modeling_table)
    # Departures were set to half of avg_4_weeks on the event day: the strong level is ~0.5
    assert set(factors) == set(LEVELS)
    for hour, factor in factors["strong"].items():
        assert factor == pytest.approx(0.5, abs=0.15)
        assert factors["mild"][hour] > factors["moderate"][hour] > factor


# ---- Alert rules ----------------------------------------------------------------------------


def test_data_contract_catches_missing_negative_and_out_of_domain_values(modeling_table):
    rows = period_rows(modeling_table, CURRENT).head(4).copy()
    assert not data_contract_alert(rows)
    rows.loc[rows.index[0], "hour"] = 3  # no service at 03:00
    rows.loc[rows.index[1], "avg_4_weeks"] = -1.0
    rows.loc[rows.index[2], "t_168"] = np.nan
    violations = data_contract_violations(rows)
    assert violations[["hour", "avg_4_weeks", "t_168"]].tolist() == [1, 1, 1]
    assert data_contract_alert(rows)
    assert data_contract_alert(rows.drop(columns="month"))  # a missing feature


def test_data_drift_alert_fires_above_strong_and_ignores_excluded_features():
    assert data_drift_alert(drift_table("avg_4_weeks", 0.3))
    assert not data_drift_alert(drift_table("avg_4_weeks", 0.2))
    assert not data_drift_alert(drift_table("month", 5.0, alarm=False))


def test_performance_alert_fires_when_the_champion_stops_beating_the_baseline():
    assert performance_alert(0.63, 0.62)
    assert performance_alert(0.62, 0.62)
    assert not performance_alert(0.61, 0.62)


def test_performance_without_drift_needs_a_rise_beyond_tolerance_and_no_drift():
    stable = drift_table("avg_4_weeks", 0.05)
    assert performance_without_drift_alert(0.65, 0.6066, stable, tolerance=0.01)
    assert not performance_without_drift_alert(0.61, 0.6066, stable, tolerance=0.01)
    assert not performance_without_drift_alert(0.65, 0.6066, drift_table("t_168", 0.15))


def test_alerts_stay_quiet_on_the_control(modeling_table):
    reference = period_rows(modeling_table, REFERENCE)
    control = bootstrap_control(reference)
    drift = feature_drift(reference, control)
    alerts = evaluate_alerts(control, drift, None, None, None)
    assert not any(v for k, v in alerts.items() if k.endswith("_alert"))
    assert alerts["action"] == NO_ACTION


def test_proposed_action_follows_the_most_urgent_alert():
    assert propose_action({"data_drift_alert": True, "data_contract_alert": True}) == (
        "review the feature pipeline"
    )
    assert propose_action({"data_drift_alert": True}).startswith("investigate")
    assert propose_action({}) == NO_ACTION


def test_wape_tolerance_is_positive_and_reproducible(modeling_table):
    rows = period_rows(modeling_table, CURRENT)
    prediction = rows["avg_4_weeks"]
    tolerance = wape_tolerance(rows, prediction, n_boot=200)
    assert tolerance > 0
    assert tolerance == wape_tolerance(rows, prediction, n_boot=200)


# ---- Full evaluation ------------------------------------------------------------------------


def test_performance_table_is_computable_for_every_scenario_model_and_segment(
    modeling_table, calibration
):
    reference = period_rows(modeling_table, REFERENCE)
    champion = make_model("baseline").fit(reference, reference[TARGET])
    tables = run_drift(modeling_table, champion, calibration, REFERENCE, CURRENT)

    performance = tables["performance_by_segment"]
    assert set(performance["model"]) == {"champion", "baseline"}
    assert performance[["wape", "mae", "volume_ratio"]].notna().all().all()
    # 12 scenario levels + the natural period; the control has no performance
    assert performance.groupby(["scenario", "level"]).ngroups == 13
    assert CONTROL not in set(performance["scenario"])
    natural = performance[performance["scenario"] == NATURAL]
    assert (natural["wape_change"] == 0).all()

    alerts = tables["alerts"].set_index("scenario")
    assert alerts.loc[CONTROL, "action"] == NO_ACTION
    assert set(tables["drift_by_feature"]["scenario"]) == set(alerts.index)
