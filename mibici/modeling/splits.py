"""Temporal splits and evaluation modes of the modeling contract (notebook 3.01, section 7.3).

Every period is [start, end) on ``hour_start`` and comes from ``mibici.config``. Rows are
never moved between periods: a station appears in a split only for the hours it had rows.

Evaluation modes (what the model is trained on -> what it is evaluated on):

- ``validation``: train -> validation (hyperparameter tuning);
- ``window_june`` / ``window_july``: train up to the window -> that month (stability);
- ``test``: train + validation -> test (final evaluation, used once).

Training rows must have station history (``history_source != "none"``): only the first week
of the data lacks it, because nothing earlier exists (team decision of 2026-11-05).
"""

import pandas as pd

from mibici.config import SPLITS, WINDOWS

EVALUATION_MODES = ("validation", *WINDOWS, "test")


def period_mask(df: pd.DataFrame, start: str, end: str) -> pd.Series:
    """Rows whose hour_start falls in [start, end)."""
    return (df["hour_start"] >= start) & (df["hour_start"] < end)


def label_splits(df: pd.DataFrame) -> pd.Series:
    """Name of the split (train, validation or test) of every row; NA outside all of them."""
    labels = pd.Series(pd.NA, index=df.index, dtype="string")
    for name, (start, end) in SPLITS.items():
        labels[period_mask(df, start, end)] = name
    return labels


def train_eval_masks(df: pd.DataFrame, mode: str) -> tuple[pd.Series, pd.Series]:
    """Boolean masks of the training rows and the evaluation rows of an evaluation mode."""
    if "history_source" not in df:
        raise ValueError("Add the history features first (mibici.features.add_history_features).")
    train_start = SPLITS["train"][0]
    if mode == "validation":
        train_end, eval_period = SPLITS["train"][1], SPLITS["validation"]
    elif mode in WINDOWS:
        train_end, eval_period = WINDOWS[mode]["train_end"], WINDOWS[mode]["eval"]
    elif mode == "test":
        train_end, eval_period = SPLITS["validation"][1], SPLITS["test"]
    else:
        raise ValueError(f"Unknown evaluation mode {mode!r}; use one of {EVALUATION_MODES}.")
    # Rows with no earlier week at all have no features and cannot be used to train
    train = period_mask(df, train_start, train_end) & (df["history_source"] != "none")
    evaluate = period_mask(df, *eval_period)
    return train, evaluate


def split_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Size and composition of train, validation, test and the windows' evaluation months."""
    periods = {name: period_mask(df, *dates) for name, dates in SPLITS.items()}
    periods |= {name: period_mask(df, *window["eval"]) for name, window in WINDOWS.items()}
    train_stations = set(df.loc[periods["train"], "station_id"])
    rows = []
    for name, mask in periods.items():
        part = df[mask]
        rows.append(
            {
                "period": name,
                "start": part["hour_start"].min(),
                "end": part["hour_start"].max(),
                "rows": len(part),
                "stations": part["station_id"].nunique(),
                "zero_share": (part["departures"] == 0).mean(),
                "new_cohort_stations": part.loc[
                    part["is_new_station_cohort"], "station_id"
                ].nunique(),
                "rows_unseen_in_train": (~part["station_id"].isin(train_stations)).mean(),
                "rows_without_history": int((part["history_source"] == "none").sum()),
            }
        )
    return pd.DataFrame(rows).set_index("period")
