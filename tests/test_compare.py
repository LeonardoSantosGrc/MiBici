"""Tests for mibici.modeling.compare: comparison table built from MLflow runs (no server)."""

import pandas as pd

from mibici.modeling.compare import comparison_table, official_runs


def make_run(model, mode, wape, baseline_wape=0.70, dirty="false", start=1, commit="abc1234"):
    """One run as mlflow.search_runs returns it (only the columns the comparison reads)."""
    return {
        "params.model": model,
        "params.mode": mode,
        "metrics.all_wape": wape,
        "metrics.all_mae": wape * 1.4,
        "metrics.baseline_all_wape": baseline_wape,
        "metrics.new_station_cohort_wape": 1.0,
        "metrics.all_volume_ratio": 0.9,
        "metrics.new_station_cohort_volume_ratio": 0.8,
        "metrics.fit_seconds": 40.0,
        "tags.git_dirty": dirty,
        "tags.git_commit": commit,
        "start_time": pd.Timestamp("2026-11-07") + pd.Timedelta(minutes=start),
    }


def all_modes(model, wapes, **kwargs):
    """Runs of a model in validation, June and July with the given WAPE values."""
    modes = ["validation", "window_june", "window_july"]
    return [make_run(model, mode, wape, **kwargs) for mode, wape in zip(modes, wapes)]


def test_latest_clean_run_is_used():
    runs = pd.DataFrame(
        [
            make_run("hgb", "validation", 0.66, start=1),
            make_run("hgb", "validation", 0.65, start=2),  # newer: this one is used
            make_run("hgb", "validation", 0.60, start=3, dirty="true"),  # uncommitted code
        ]
    )
    official = official_runs(runs)
    assert len(official) == 1
    assert official.loc[0, "metrics.all_wape"] == 0.65


def test_test_mode_is_never_read():
    runs = pd.DataFrame([make_run("hgb", "validation", 0.66), make_run("hgb", "test", 0.50)])
    assert list(official_runs(runs)["params.mode"]) == ["validation"]


def test_runs_can_be_restricted_to_one_commit():
    runs = pd.DataFrame(
        [
            make_run("hgb", "validation", 0.66, commit="aaa1111", start=1),
            make_run("hgb", "validation", 0.65, commit="bbb2222", start=2),
        ]
    )
    assert official_runs(runs, commit="aaa")["metrics.all_wape"].tolist() == [0.66]


def test_candidate_must_beat_the_baseline_in_all_three_periods():
    runs = pd.DataFrame(
        all_modes("hgb", [0.66, 0.68, 0.63])  # beats 0.70 everywhere
        + all_modes("random_forest", [0.69, 0.72, 0.65])  # loses in June
    )
    table = comparison_table(official_runs(runs))
    assert table.loc["hgb", "candidate"]
    assert not table.loc["random_forest", "candidate"]
    assert not table.loc["random_forest", "beats_baseline_june"]


def test_a_missing_period_is_not_a_candidate():
    runs = pd.DataFrame(all_modes("poisson", [0.60, 0.61])[:2])  # no July run
    table = comparison_table(official_runs(runs))
    assert not table.loc["poisson", "candidate"]
    assert pd.isna(table.loc["poisson"].get("july_wape"))


def test_equal_to_the_baseline_does_not_beat_it():
    # The baseline's own runs tie with themselves, so it is never a candidate
    runs = pd.DataFrame(all_modes("baseline", [0.70, 0.70, 0.70]))
    assert not comparison_table(official_runs(runs)).loc["baseline", "candidate"]


def test_candidates_come_first_ordered_by_validation_wape():
    runs = pd.DataFrame(
        all_modes("hgb", [0.66, 0.68, 0.63])
        + all_modes("poisson", [0.64, 0.66, 0.62])
        + all_modes("baseline", [0.70, 0.70, 0.70])
    )
    table = comparison_table(official_runs(runs))
    assert list(table.index) == ["poisson", "hgb", "baseline"]
    assert table.loc["hgb", "volume_ratio"] == 0.9
