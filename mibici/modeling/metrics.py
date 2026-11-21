"""Common metrics of the modeling contract (notebook 3.01, sections 6 and 7.5).

- WAPE (primary): total absolute error as a share of total departures.
- MAE (secondary): mean absolute error, in departures per station-hour.

Within the same rows both rank models the same way (WAPE = MAE * n / sum(y)). Every model is
scored with ``evaluate`` so all results are comparable: same metrics, same segments.

``evaluate`` also reports the volume ratio (forecast departures / observed departures), a
diagnostic of systematic under- or over-forecasting: WAPE can favour a model that forecasts
far fewer bicycles than leave. It informs the final choice; it is not a candidate rule
(team decision of 2026-11-07).
"""

import numpy as np
import pandas as pd

from mibici.config import RANDOM_STATE, TARGET


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


def volume_ratio(y_true, y_pred) -> float:
    """Forecast departures divided by observed departures; NaN when there were none.

    1.0 means the total volume is right; 0.8 means 20 % fewer bicycles were forecast.
    """
    y_true, y_pred = _as_arrays(y_true, y_pred)
    total = y_true.sum()
    # Same rule as WAPE: without observed departures the ratio is undefined
    if total == 0:
        return float("nan")
    return float(y_pred.sum() / total)


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
                "volume_ratio": volume_ratio(y_true[mask], y_pred[mask]),
            }
        )
    return pd.DataFrame(scores).set_index("segment")


def paired_bootstrap_wape_difference(
    days,
    y_true,
    pred_a,
    pred_b,
    n_boot: int = 2000,
    seed: int = RANDOM_STATE,
) -> dict[str, float]:
    """How sure we can be that forecast A has a lower WAPE than forecast B on the same rows.

    Whole days are resampled with replacement (the hours of a day share weather, events and
    errors), and both forecasts are scored on the same resampled days, so a hard day counts
    against both: the comparison is paired. Returns the observed difference WAPE(A) - WAPE(B),
    its 95 % percentile interval and the share of resamples where A has the lower WAPE.
    """
    y_true, pred_a = _as_arrays(y_true, pred_a)
    _, pred_b = _as_arrays(y_true, pred_b)
    days = pd.to_datetime(np.asarray(days)).normalize()
    if len(days) != len(y_true):
        raise ValueError(f"Got {len(days)} days for {len(y_true)} rows.")

    # Per day: absolute error of each forecast and observed departures
    daily = (
        pd.DataFrame(
            {
                "day": days,
                "error_a": np.abs(y_true - pred_a),
                "error_b": np.abs(y_true - pred_b),
                "departures": y_true,
            }
        )
        .groupby("day")[["error_a", "error_b", "departures"]]
        .sum()
        .to_numpy()
    )
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, len(daily), size=(n_boot, len(daily)))  # resampled day indices
    sums = daily[picks].sum(axis=1)  # (n_boot, 3): errors of A and B, departures
    differences = (sums[:, 0] - sums[:, 1]) / sums[:, 2]
    observed = (daily[:, 0].sum() - daily[:, 1].sum()) / daily[:, 2].sum()
    low, high = np.quantile(differences, [0.025, 0.975])
    return {
        "difference": float(observed),
        "ci_low": float(low),
        "ci_high": float(high),
        "share_a_better": float((differences < 0).mean()),
        "days": len(daily),
        "n_boot": int(n_boot),
    }


def flatten_scores(scores: pd.DataFrame) -> dict[str, float]:
    """``evaluate`` table as {"<segment>_<metric>": value}, the format MLflow logs."""
    return {
        f"{segment}_{metric}": float(scores.loc[segment, metric])
        for segment in scores.index
        for metric in ("mae", "wape", "volume_ratio")
    }


def _as_arrays(y_true, y_pred) -> tuple[np.ndarray, np.ndarray]:
    """Both inputs as float arrays of the same length (accepts lists, arrays and Series)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shapes differ: {y_true.shape} and {y_pred.shape}.")
    return y_true, y_pred
