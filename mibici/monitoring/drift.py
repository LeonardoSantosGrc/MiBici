"""Data-drift statistics and alert rules of the drift contract (docs/final/drift_contract.md).

Data drift asks whether the model inputs look different from a reference period; it needs no
observed departures. Performance drift asks whether the model errs more; it does. Both are
kept apart: a drift alarm does not prove a performance loss, and a loss can come without drift.

- PSI (population stability index) = sum (a_i - e_i) * ln(a_i / e_i), with e_i and a_i the
  shares of reference and current rows in bin i. Numeric features use 10 quantile bins of the
  reference; categorical features use one bin per category. Thresholds: < 0.1 stable,
  0.1 to 0.25 moderate, > 0.25 strong.
- KS statistic D = max |F_ref(x) - F_cur(x)| for numeric features, reported as a magnitude.
  Its p-value is not used: with about 560,000 rows any tiny difference is significant.

The alert rules turn a scenario's drift table and scores into the four signals of the
contract and one proposed action.
"""

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from mibici.config import FEATURES, RANDOM_STATE

PSI_BINS = 10
# Share given to an empty bin, so ln(a / e) stays finite
PSI_EPSILON = 1e-4
PSI_MODERATE = 0.1
PSI_STRONG = 0.25
# Resamples of the days of a period for the tolerance of its WAPE
BOOTSTRAP_SAMPLES = 1000

NUMERIC_FEATURES = ["avg_4_weeks", "t_168", "station_age_days"]
CATEGORICAL_FEATURES = ["hour", "day_of_week", "is_weekend", "month"]
PREDICTION = "prediction"  # name of the model output in the drift table
# Reported but never alarmed: both grow with the calendar, so any later period drifts.
# - month: June-July and August-September share no month (contract).
# - station_age_days: every station is two months older in the current period (PSI 1.45
#   between the reference and the natural current period); proposed to the team, see the PR.
ALARM_EXCLUDED = ("month", "station_age_days")

# Domain of the data contract, the same rules as mibici.serving.snapshot.build_feature_table
SERVICE_HOURS = (0, *range(5, 24))
DAYS_OF_WEEK = tuple(range(7))

ACTIONS = {
    "data_contract_alert": "review the feature pipeline",
    "performance_alert": "retrain with recent data",
    "performance_without_drift_alert": "review the model inputs (add weather, events)",
    "data_drift_alert": "investigate and confirm with performance",
}
NO_ACTION = "none"


def quantile_edges(reference, bins: int = PSI_BINS) -> np.ndarray:
    """Bin edges from the reference quantiles; repeated edges (many zeros) are merged.

    The outer edges are -inf and +inf, so every current value falls in some bin.
    """
    reference = np.asarray(reference, dtype=float)
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    # A constant reference gives one edge: keep a single bin around it
    inner = edges[1:-1] if len(edges) > 2 else edges[:0]
    return np.concatenate([[-np.inf], inner, [np.inf]])


def _psi_from_shares(expected: np.ndarray, actual: np.ndarray) -> float:
    """PSI of two share vectors over the same bins, with empty bins set to PSI_EPSILON."""
    expected = np.clip(expected, PSI_EPSILON, None)
    actual = np.clip(actual, PSI_EPSILON, None)
    return float(np.sum((actual - expected) * np.log(actual / expected)))


def psi_numeric(reference, current, bins: int = PSI_BINS) -> float:
    """PSI of a numeric variable over quantile bins of the reference."""
    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)
    edges = quantile_edges(reference, bins)
    # np.histogram closes the last bin; with +inf as last edge that changes nothing.
    # Bins are [edge_i, edge_i+1): a value equal to an edge goes to the upper bin
    expected = np.histogram(reference, edges)[0] / len(reference)
    actual = np.histogram(current, edges)[0] / len(current)
    return _psi_from_shares(expected, actual)


def psi_categorical(reference, current) -> float:
    """PSI over the categories present in either sample."""
    reference, current = pd.Series(reference), pd.Series(current)
    categories = pd.Index(reference.unique()).union(pd.Index(current.unique()))
    expected = reference.value_counts(normalize=True).reindex(categories, fill_value=0)
    actual = current.value_counts(normalize=True).reindex(categories, fill_value=0)
    return _psi_from_shares(expected.to_numpy(dtype=float), actual.to_numpy(dtype=float))


def ks_statistic(reference, current) -> float:
    """Kolmogorov-Smirnov statistic D (largest gap between both empirical CDFs)."""
    return float(ks_2samp(np.asarray(reference, float), np.asarray(current, float)).statistic)


def psi_status(psi: float) -> str:
    """stable (< 0.1), moderate (0.1 to 0.25) or strong (> 0.25)."""
    if psi > PSI_STRONG:
        return "strong"
    if psi >= PSI_MODERATE:
        return "moderate"
    return "stable"


def feature_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    reference_prediction=None,
    current_prediction=None,
) -> pd.DataFrame:
    """PSI, KS D and status of every contract feature (and the prediction, if given).

    One row per feature; ``alarm`` says whether the feature counts for the data-drift alert.
    """
    rows = []
    for feature in FEATURES:
        numeric = feature in NUMERIC_FEATURES
        ref, cur = reference[feature], current[feature]
        rows.append(
            {
                "feature": feature,
                "kind": "numeric" if numeric else "categorical",
                "psi": psi_numeric(ref, cur) if numeric else psi_categorical(ref, cur),
                "ks_d": ks_statistic(ref, cur) if numeric else np.nan,
            }
        )
    if reference_prediction is not None and current_prediction is not None:
        rows.append(
            {
                "feature": PREDICTION,
                "kind": "numeric",
                "psi": psi_numeric(reference_prediction, current_prediction),
                "ks_d": ks_statistic(reference_prediction, current_prediction),
            }
        )
    table = pd.DataFrame(rows)
    table["status"] = table["psi"].map(psi_status)
    table["alarm"] = ~table["feature"].isin(ALARM_EXCLUDED)
    return table


def data_contract_violations(rows: pd.DataFrame) -> pd.Series:
    """Number of rows that break the data contract, per feature.

    A feature breaks it when it is missing, negative, or outside its domain (hours 0 and 5
    to 23, day_of_week 0 to 6). A missing column counts every row.
    """
    counts = {}
    for feature in FEATURES:
        if feature not in rows:
            counts[feature] = len(rows)
            continue
        values = pd.to_numeric(rows[feature], errors="coerce")
        bad = values.isna() | (values < 0)
        if feature == "hour":
            bad |= ~values.isin(SERVICE_HOURS)
        elif feature == "day_of_week":
            bad |= ~values.isin(DAYS_OF_WEEK)
        counts[feature] = int(bad.sum())
    return pd.Series(counts, name="violations")


def data_contract_alert(rows: pd.DataFrame) -> bool:
    """True if any monitored feature is missing, negative or outside its domain."""
    return bool(data_contract_violations(rows).any())


def data_drift_alert(drift: pd.DataFrame) -> bool:
    """True if any alarmed feature, or the prediction, has PSI > 0.25."""
    alarmed = drift[drift["alarm"]]
    return bool((alarmed["psi"] > PSI_STRONG).any())


def performance_alert(champion_wape: float, baseline_wape: float) -> bool:
    """True if the champion no longer beats the baseline on the same rows."""
    return bool(champion_wape >= baseline_wape)


def wape_tolerance(
    rows: pd.DataFrame,
    prediction,
    n_boot: int = BOOTSTRAP_SAMPLES,
    seed: int = RANDOM_STATE,
) -> float:
    """How much the WAPE of a period can rise by chance: bootstrap of its days.

    Days are drawn with replacement ``n_boot`` times; the tolerance is the 97.5th percentile
    of the resampled WAPE minus the WAPE of the period. Whole days are resampled because the
    hours of a day share their weather, events and errors.
    """
    error = np.abs(rows["departures"].to_numpy(dtype=float) - np.asarray(prediction, float))
    days = rows["hour_start"].dt.normalize().to_numpy()
    daily = pd.DataFrame({"day": days, "error": error, "departures": rows["departures"]})
    daily = daily.groupby("day")[["error", "departures"]].sum().to_numpy()
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, len(daily), size=(n_boot, len(daily)))
    sampled = daily[picks].sum(axis=1)  # (n_boot, 2): total error and total departures
    point = daily[:, 0].sum() / daily[:, 1].sum()
    return float(np.quantile(sampled[:, 0] / sampled[:, 1], 0.975) - point)


def performance_without_drift_alert(
    champion_wape: float, natural_wape: float, drift: pd.DataFrame, tolerance: float = 0.0
) -> bool:
    """True if the champion's WAPE rises more than ``tolerance`` while every PSI < 0.1.

    ``tolerance`` (``wape_tolerance`` of the natural period) keeps a rise that chance alone
    explains from raising the alert.
    """
    alarmed = drift[drift["alarm"]]
    no_drift = bool((alarmed["psi"] < PSI_MODERATE).all())
    return bool(champion_wape - natural_wape > tolerance and no_drift)


def propose_action(alerts: dict[str, bool]) -> str:
    """Action of the most urgent alert that fired; "none" when nothing fired.

    Order: a broken input first (scores on broken data mean nothing), then the loss of the
    advantage over the baseline, then a loss without drift, then drift alone.
    """
    for alert, action in ACTIONS.items():
        if alerts.get(alert, False):
            return action
    return NO_ACTION


def evaluate_alerts(
    rows: pd.DataFrame,
    drift: pd.DataFrame,
    champion_wape: float | None,
    baseline_wape: float | None,
    natural_wape: float | None,
    tolerance: float = 0.0,
) -> dict[str, object]:
    """The four signals of the contract and the proposed action for one scenario and level.

    Performance signals need observed departures: pass ``None`` scores (the control, which
    is a sample of training data) and they stay False.
    """
    scored = champion_wape is not None and baseline_wape is not None
    alerts = {
        "data_contract_alert": data_contract_alert(rows),
        "data_drift_alert": data_drift_alert(drift),
        "performance_alert": scored and performance_alert(champion_wape, baseline_wape),
        "performance_without_drift_alert": scored
        and natural_wape is not None
        and performance_without_drift_alert(champion_wape, natural_wape, drift, tolerance),
    }
    return {**alerts, "action": propose_action(alerts)}
