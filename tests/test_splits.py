"""Tests for mibici.modeling.splits: temporal splits of the modeling contract (section 7.3)."""

import pandas as pd
import pytest

from mibici.config import SPLITS, WINDOWS
from mibici.modeling.splits import (
    EVALUATION_MODES,
    label_splits,
    split_summary,
    train_eval_masks,
)


def make_hours(start: str, end: str) -> pd.DataFrame:
    """One station with a row every hour in [start, end), all with history except the first."""
    hours = pd.date_range(start, end, freq="h", inclusive="left")
    df = pd.DataFrame(
        {
            "station_id": 1,
            "hour_start": hours,
            "departures": 1,
            "is_new_station_cohort": False,
            "history_source": "own_history",
        }
    )
    df.loc[0, "history_source"] = "none"  # first row of the data: nothing earlier
    return df


def test_config_splits_are_ordered_and_contiguous():
    periods = list(SPLITS.values())
    for (_, end), (next_start, _) in zip(periods, periods[1:]):
        assert end == next_start  # no gap and no overlap
    assert all(start < end for start, end in periods)


def test_config_windows_are_months_inside_validation():
    validation_start, validation_end = SPLITS["validation"]
    for window in WINDOWS.values():
        eval_start, eval_end = window["eval"]
        assert window["train_end"] == eval_start  # train right up to the evaluated month
        assert validation_start <= eval_start < eval_end <= validation_end


def test_boundary_hours_go_to_the_right_split():
    df = make_hours("2026-05-31 23:00", "2026-08-01 01:00")
    labels = label_splits(df).set_axis(df["hour_start"])
    assert labels[pd.Timestamp("2026-05-31 23:00")] == "train"
    assert labels[pd.Timestamp("2026-06-01 00:00")] == "validation"
    assert labels[pd.Timestamp("2026-07-31 23:00")] == "validation"
    assert labels[pd.Timestamp("2026-08-01 00:00")] == "test"


def test_every_row_in_the_data_range_has_exactly_one_split():
    df = make_hours(SPLITS["train"][1], SPLITS["test"][1])
    assert label_splits(df).notna().all()


@pytest.mark.parametrize("mode", EVALUATION_MODES)
def test_training_ends_before_evaluation_starts(mode):
    df = make_hours("2026-05-01", "2026-10-01")
    train, evaluate = train_eval_masks(df, mode)
    assert train.any() and evaluate.any()
    assert df.loc[train, "hour_start"].max() < df.loc[evaluate, "hour_start"].min()


@pytest.mark.parametrize("mode", [m for m in EVALUATION_MODES if m != "test"])
def test_test_rows_are_reserved_for_the_test_mode(mode):
    df = make_hours("2026-05-01", "2026-10-01")
    in_test = label_splits(df) == "test"
    train, evaluate = train_eval_masks(df, mode)
    assert not (train & in_test).any()
    assert not (evaluate & in_test).any()


def test_test_mode_trains_on_train_and_validation():
    df = make_hours("2026-05-01", "2026-10-01")
    train, evaluate = train_eval_masks(df, "test")
    labels = label_splits(df)
    assert set(labels[train]) == {"train", "validation"}
    assert set(labels[evaluate]) == {"test"}


def test_rows_without_history_are_not_used_to_train():
    df = make_hours("2026-05-01", "2026-07-01")
    train, _ = train_eval_masks(df, "validation")
    assert not train[df["history_source"] == "none"].any()


def test_unknown_mode_is_rejected():
    df = make_hours("2026-05-01", "2026-07-01")
    with pytest.raises(ValueError, match="Unknown evaluation mode"):
        train_eval_masks(df, "tuning_on_test")


def test_summary_counts_rows_without_history():
    df = make_hours("2026-05-01", "2026-10-01")
    summary = split_summary(df)
    assert summary.loc["train", "rows_without_history"] == 1
    assert summary["rows"].sum() == len(df) + summary.loc[list(WINDOWS), "rows"].sum()
