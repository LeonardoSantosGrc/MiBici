"""Compare the team models from their MLflow runs (notebook 3.01, section 7.6).

    make compare

Reads the ``mibici-modeling`` experiment and builds one row per model with its WAPE and MAE
on validation, June and July, whether it beats the baseline on the same rows in each period,
and whether it is a **candidate** (beats the baseline on all three). It also shows the
2026 new-station cohort and the volume ratio (forecast / observed departures), which inform
the final choice but are not candidate rules.

For every model and mode only the latest run with committed code (``git_dirty=false``) is
used, optionally restricted to one Git commit. The test split is never read here: the
champion is chosen without it.
"""

from pathlib import Path

from loguru import logger
import mlflow
import pandas as pd
import typer

from mibici.config import MLFLOW_TRACKING_URI, MODELING_EXPERIMENT_NAME, REPORTS_DIR
from mibici.tracking import check_tracking_server

app = typer.Typer()

# Evaluation modes compared, with the short label used in the column names
PERIODS = {"validation": "validation", "window_june": "june", "window_july": "july"}
COMPARISON_PATH = REPORTS_DIR / "modeling" / "model_comparison.csv"


def load_runs() -> pd.DataFrame:
    """Every finished run of the modeling experiment (one row per run, MLflow columns)."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    return mlflow.search_runs(
        experiment_names=[MODELING_EXPERIMENT_NAME],
        filter_string="attributes.status = 'FINISHED'",
    )


def official_runs(runs: pd.DataFrame, commit: str | None = None) -> pd.DataFrame:
    """Latest run with committed code for every model and compared mode."""
    keep = runs["tags.git_dirty"].eq("false") & runs["params.mode"].isin(PERIODS)
    # Optionally only runs of one commit (a full hash or its first characters)
    if commit:
        keep &= runs["tags.git_commit"].str.startswith(commit)
    # Sort by start time so tail(1) keeps the most recent run of each (model, mode)
    latest = runs[keep].sort_values("start_time").groupby(["params.model", "params.mode"]).tail(1)
    return latest.reset_index(drop=True)


def comparison_table(runs: pd.DataFrame) -> pd.DataFrame:
    """One row per model: scores by period, baseline comparison and candidate status."""
    rows = []
    for model, group in runs.groupby("params.model"):
        by_mode = group.set_index("params.mode")
        row = {"model": model}
        for mode, label in PERIODS.items():
            # A period without a run leaves its columns empty and the model is not a candidate
            if mode not in by_mode.index:
                row[f"beats_baseline_{label}"] = False
                continue
            run = by_mode.loc[mode]
            row[f"{label}_wape"] = run["metrics.all_wape"]
            row[f"{label}_mae"] = run["metrics.all_mae"]
            # Strictly lower WAPE than the baseline scored on the same rows of the same run
            row[f"beats_baseline_{label}"] = bool(
                run["metrics.all_wape"] < run["metrics.baseline_all_wape"]
            )
        if "validation" in by_mode.index:
            run = by_mode.loc["validation"]
            # .get(): runs logged before a metric existed simply show it empty
            row["baseline_validation_wape"] = run.get("metrics.baseline_all_wape")
            row["new_cohort_wape"] = run.get("metrics.new_station_cohort_wape")
            row["volume_ratio"] = run.get("metrics.all_volume_ratio")
            row["new_cohort_volume_ratio"] = run.get("metrics.new_station_cohort_volume_ratio")
            row["fit_seconds"] = run.get("metrics.fit_seconds")
            row["git_commit"] = str(run.get("tags.git_commit", ""))[:7]
        # Candidate rule of the contract (section 7.6, rule 3)
        row["candidate"] = all(row[f"beats_baseline_{label}"] for label in PERIODS.values())
        rows.append(row)
    table = pd.DataFrame(rows).set_index("model")
    # Candidates first, then by validation WAPE (lower is better)
    return table.sort_values(["candidate", "validation_wape"], ascending=[False, True])


@app.command()
def main(
    commit: str = typer.Option("", help="Only runs of this Git commit (full or short hash)."),
    output: str = typer.Option(str(COMPARISON_PATH), help="CSV file for the comparison table."),
) -> None:
    """Build the model comparison table from MLflow and save it as CSV."""
    check_tracking_server()
    table = comparison_table(official_runs(load_runs(), commit or None))
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output)
    # Transposed (one column per model) so every row fits in the terminal
    logger.success(
        f"Model comparison ({len(table)} models) saved to {output}:\n"
        f"{table.round(3).T.to_string()}"
    )


if __name__ == "__main__":
    app()
