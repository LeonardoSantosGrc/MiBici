"""Common metrics of the modeling contract (notebook 3.01, sections 6 and 7.5).

- WAPE (primary): total absolute error as a share of total departures.
- MAE (secondary): mean absolute error, in departures per station-hour.

Within the same rows both rank models the same way (WAPE = MAE * n / sum(y)). Every model is
scored with ``evaluate`` so all results are comparable: same metrics, same segments.
"""

import numpy as np
import pandas as pd

from mibici.config import TARGET


def mae(y_true, y_pred) -> float:
    """Mean absolute error; NaN when there are no rows."""
    y_true, y_pred = _as_arrays(y_true, y_pred)
    # An empty segment has no error to average
    if y_true.size == 0:
        return float("nan")
    return float(np.abs(y_true - y_pred).mean())


def wape(y_true, y_pred) -> float:
    """Total absolute error divided by total departures; NaN when there were no departures."""
    y_true, y_pred = _as_arrays(y_true, y_pred)
    total = y_true.sum()
    # With zero departures there is no volume to compare the error with: undefined
    if total == 0:
        return float("nan")
    return float(np.abs(y_true - y_pred).sum() / total)


def evaluate(rows: pd.DataFrame, y_pred) -> pd.DataFrame:
    """MAE and WAPE of a forecast by segment, one row per segment.

    ``rows`` are the evaluated rows of processed_v1 (target and evaluation-only columns) and
    ``y_pred`` the forecast for those rows, in the same order. Segments: all rows, established
    stations, the 2026 new-station cohort and, when the period has any, extraordinary events.
    A segment with no rows is left out.
    """
    y_true = rows[TARGET].to_numpy(dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    # The forecast must have exactly one value per evaluated row
    if len(y_pred) != len(y_true):
        raise ValueError(f"Got {len(y_pred)} predictions for {len(y_true)} rows.")
    new_cohort = rows["is_new_station_cohort"].to_numpy(dtype=bool)
    segments = {
        "all": np.ones(len(rows), dtype=bool),
        "established_stations": ~new_cohort,
        "new_station_cohort": new_cohort,
        "extraordinary_event": rows["is_extraordinary_event"].to_numpy(dtype=bool),
    }
    scores = []
    for name, mask in segments.items():
        # Skip segments absent from this period (e.g. no events in validation)
        if not mask.any():
            continue
        scores.append(
            {
                "segment": name,
                "rows": int(mask.sum()),
                "departures": int(y_true[mask].sum()),
                "mae": mae(y_true[mask], y_pred[mask]),
                "wape": wape(y_true[mask], y_pred[mask]),
            }
        )
    return pd.DataFrame(scores).set_index("segment")


def flatten_scores(scores: pd.DataFrame) -> dict[str, float]:
    """``evaluate`` table as {"<segment>_<metric>": value}, the format MLflow logs."""
    return {
        f"{segment}_{metric}": float(scores.loc[segment, metric])
        for segment in scores.index
        for metric in ("mae", "wape")
    }


def _as_arrays(y_true, y_pred) -> tuple[np.ndarray, np.ndarray]:
    """Both inputs as float arrays of the same length (accepts lists, arrays and Series)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shapes differ: {y_true.shape} and {y_pred.shape}.")
    return y_true, y_pred
