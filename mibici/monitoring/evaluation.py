"""Run the drift evaluation of the champion and write its evidence tables and figures.

    make drift

For the control, the natural current period and every scenario and level of
``mibici.monitoring.scenarios``:

1. data drift of the seven features and of the prediction against the reference period
   (June-July 2026), with PSI and KS (``mibici.monitoring.drift``);
2. performance of the champion and of the baseline on the same rows, by segment, and the
   change against the natural current period (August-September 2026, champion WAPE 0.6066);
3. the four alert signals of the contract and the proposed action.

Outputs in ``reports/evidence/final_drift/``: ``drift_by_feature.csv``,
``performance_by_segment.csv``, ``alerts.csv``, ``calibration.csv``, ``run_info.json`` and
the figures. Drift does not use FastAPI, Docker or the MLflow server.
"""

import json
from pathlib import Path

from loguru import logger
from matplotlib.figure import Figure
import mlflow.sklearn
import pandas as pd
import typer

from mibici.config import CHAMPION_EXPORT_DIR, RANDOM_STATE, REPORTS_DIR, TARGET
from mibici.modeling.metrics import evaluate
from mibici.modeling.models import make_model
from mibici.modeling.train import load_modeling_table
from mibici.monitoring.drift import evaluate_alerts, feature_drift, wape_tolerance
from mibici.monitoring.scenarios import (
    CURRENT_PERIOD,
    REFERENCE_PERIOD,
    bootstrap_control,
    build_scenarios,
    calibrate,
    period_rows,
)
from mibici.tracking import git_info

app = typer.Typer()

DRIFT_EVIDENCE_DIR = REPORTS_DIR / "evidence" / "final_drift"
CONTROL, NATURAL = "control", "natural"
# Action each scenario is expected to trigger (contract, alert criteria table)
EXPECTED_ACTION = {
    CONTROL: "none",
    NATURAL: "none",
    "S1_demand_shift": "retrain with recent data",
    "S2_station_expansion": "retrain with recent data",
    "S3_data_failure": "review the feature pipeline",
    "S4_concept_change": "review the model inputs (add weather, events)",
}
METRICS = ["wape", "mae", "volume_ratio"]


def score_models(rows: pd.DataFrame, predictions: dict[str, object]) -> pd.DataFrame:
    """``evaluate`` of every model's forecast on the same rows, stacked by model."""
    tables = {model: evaluate(rows, forecast) for model, forecast in predictions.items()}
    scores = pd.concat(tables, names=["model"]).reset_index()
    return scores[["model", "segment", "rows", "departures", *METRICS]]


def add_changes(scores: pd.DataFrame, natural: pd.DataFrame) -> pd.DataFrame:
    """Add ``<metric>_change`` = metric minus the natural period's, same model and segment."""
    reference = natural.set_index(["model", "segment"])[METRICS]
    keyed = scores.set_index(["model", "segment"])
    changes = keyed[METRICS] - reference.reindex(keyed.index)
    return scores.assign(
        **{f"{m}_change": changes[m].to_numpy() for m in METRICS},
    )


def run_drift(
    df: pd.DataFrame,
    champion,
    calibration: dict | None = None,
    reference_period=REFERENCE_PERIOD,
    current_period=CURRENT_PERIOD,
) -> dict[str, pd.DataFrame]:
    """Drift, performance and alert tables of the control, the natural period and S1-S4.

    ``df`` is the modeling table (processed_v1 with the history features) and ``champion``
    any fitted pipeline with ``predict``. The baseline is ``make_model("baseline")``.
    """
    calibration = calibration if calibration is not None else calibrate(df, current_period)
    reference = period_rows(df, reference_period).reset_index(drop=True)
    natural = period_rows(df, current_period).reset_index(drop=True)
    # The baseline learns nothing; fit only checks its column
    baseline = make_model("baseline").fit(reference, reference[TARGET])
    reference_prediction = champion.predict(reference)

    # (scenario, level, magnitude, description, rows, scored?)
    # The control is a sample of training data: drift only, no performance
    control = bootstrap_control(reference, RANDOM_STATE)
    cases = [
        (CONTROL, "none", float("nan"), "bootstrap sample of the reference", control, False),
        (NATURAL, "none", float("nan"), "current period without changes", natural, True),
    ]
    for s in build_scenarios(df, calibration, current_period):
        cases.append((s.name, s.level, s.magnitude, s.description, s.rows, True))

    drift_tables, performance_tables, alert_rows = [], [], []
    natural_scores, natural_wape, tolerance = None, None, 0.0
    for scenario, level, magnitude, description, rows, scored in cases:
        prediction = champion.predict(rows)
        drift = feature_drift(reference, rows, reference_prediction, prediction)
        drift_tables.append(drift.assign(scenario=scenario, level=level))

        champion_wape = baseline_wape = None
        if scored:
            scores = score_models(
                rows, {"champion": prediction, "baseline": baseline.predict(rows)}
            )
            if scenario == NATURAL:
                natural_scores = scores
                natural_wape = float(
                    scores.query("model == 'champion' and segment == 'all'")["wape"].iloc[0]
                )
                tolerance = wape_tolerance(rows, prediction)
            scores = add_changes(scores, natural_scores)
            performance_tables.append(scores.assign(scenario=scenario, level=level))
            overall = scores[scores["segment"] == "all"].set_index("model")["wape"]
            champion_wape, baseline_wape = overall["champion"], overall["baseline"]

        alerts = evaluate_alerts(
            rows, drift, champion_wape, baseline_wape, natural_wape, tolerance
        )
        expected = EXPECTED_ACTION[scenario]
        alert_rows.append(
            {
                "scenario": scenario,
                "level": level,
                "magnitude": magnitude,
                "description": description,
                "max_alarm_psi": float(drift.loc[drift["alarm"], "psi"].max()),
                "champion_wape": champion_wape,
                "baseline_wape": baseline_wape,
                "wape_tolerance": tolerance if scored else None,
                **alerts,
                "expected_action": expected,
                "matches_expected": alerts["action"] == expected,
            }
        )
        logger.info(f"{scenario} {level}: {alerts['action']}")

    first = ["scenario", "level"]
    drift_table = pd.concat(drift_tables, ignore_index=True)
    performance = pd.concat(performance_tables, ignore_index=True)
    return {
        "drift_by_feature": drift_table[first + [c for c in drift_table if c not in first]],
        "performance_by_segment": performance[first + [c for c in performance if c not in first]],
        "alerts": pd.DataFrame(alert_rows),
        "calibration": calibration_table(calibration),
    }


def calibration_table(calibration: dict) -> pd.DataFrame:
    """Magnitudes of every scenario and level as rows (S4 has one factor per hour)."""
    rows = []
    for scenario, levels in calibration.items():
        for level, value in levels.items():
            if isinstance(value, dict):
                rows += [
                    {"scenario": scenario, "level": level, "hour": h, "magnitude": v}
                    for h, v in value.items()
                ]
            else:
                rows.append(
                    {"scenario": scenario, "level": level, "hour": pd.NA, "magnitude": value}
                )
    return pd.DataFrame(rows)


def case_label(table: pd.DataFrame) -> pd.Series:
    """'S1_demand_shift · mild' style label of each row (control and natural keep their name)."""
    return table["scenario"].where(
        table["level"] == "none", table["scenario"] + " · " + table["level"]
    )


def plot_psi_heatmap(drift: pd.DataFrame, path: Path) -> None:
    """PSI of every feature (columns) in every scenario and level (rows)."""
    table = drift.assign(case=case_label(drift))
    order = list(dict.fromkeys(table["case"]))
    matrix = table.pivot(index="case", columns="feature", values="psi").loc[order]
    matrix = matrix[list(dict.fromkeys(drift["feature"]))]
    fig = Figure(figsize=(10, 7))
    ax = fig.subplots()
    image = ax.imshow(matrix.clip(upper=1.0), cmap="Reds", vmin=0, vmax=1.0, aspect="auto")
    ax.set_xticks(range(matrix.shape[1]), matrix.columns, rotation=35, ha="right")
    ax.set_yticks(range(matrix.shape[0]), matrix.index)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix.iat[i, j]
            ax.text(
                j,
                i,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=7,
                color="white" if value > 0.5 else "black",
            )
    fig.colorbar(image, ax=ax, label="PSI (colour capped at 1.0)")
    ax.set_title("4.01a — PSI against June-July 2026 (> 0.25 strong; month and age not alarmed)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def plot_wape(performance: pd.DataFrame, natural_wape: float, path: Path) -> None:
    """WAPE of the champion and the baseline (all rows) in every scenario and level."""
    table = performance[performance["segment"] == "all"].assign(case=case_label(performance))
    order = list(dict.fromkeys(table["case"]))
    wide = table.pivot(index="case", columns="model", values="wape").loc[order]
    fig = Figure(figsize=(10, 6))
    ax = fig.subplots()
    wide[["champion", "baseline"]].plot.barh(ax=ax, color=["#1f77b4", "#bbbbbb"])
    ax.axvline(
        natural_wape,
        color="black",
        linestyle="--",
        linewidth=0.8,
        label=f"champion, natural period ({natural_wape:.4f})",
    )
    ax.invert_yaxis()
    ax.set_xlabel("WAPE (lower is better)")
    ax.set_ylabel("")
    ax.legend(loc="center right")
    ax.set_title("4.01b — Champion vs baseline on the same rows")
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def write_outputs(tables: dict[str, pd.DataFrame], output_dir: Path, info: dict) -> None:
    """CSV tables, figures and run_info.json in ``output_dir``."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(output_dir / f"{name}.csv", index=False)
    natural = (
        tables["performance_by_segment"]
        .query("scenario == @NATURAL and model == 'champion' and segment == 'all'")["wape"]
        .iloc[0]
    )
    plot_psi_heatmap(tables["drift_by_feature"], output_dir / "4.01a-psi-heatmap.png")
    plot_wape(tables["performance_by_segment"], natural, output_dir / "4.01b-wape-by-scenario.png")
    (output_dir / "run_info.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")


@app.command()
def main(output_dir: Path = DRIFT_EVIDENCE_DIR):
    """Evaluate data drift and performance drift of the champion on S1-S4 and write evidence."""
    git = git_info()
    if git["git_dirty"] == "true":
        logger.warning("Uncommitted changes: run_info.json will record git_dirty=true")
    df = load_modeling_table()
    champion = mlflow.sklearn.load_model(str(CHAMPION_EXPORT_DIR))
    tables = run_drift(df, champion)
    info = {
        **git,
        "model": str(CHAMPION_EXPORT_DIR.relative_to(CHAMPION_EXPORT_DIR.parents[2])),
        "reference_period": list(REFERENCE_PERIOD),
        "current_period": list(CURRENT_PERIOD),
        "random_state": RANDOM_STATE,
    }
    write_outputs(tables, output_dir, info)
    summary = tables["alerts"][
        [
            "scenario",
            "level",
            "max_alarm_psi",
            "champion_wape",
            "baseline_wape",
            "action",
            "matches_expected",
        ]
    ]
    logger.success(f"Drift evidence in {output_dir}:\n{summary.round(4).to_string()}")


if __name__ == "__main__":
    app()
