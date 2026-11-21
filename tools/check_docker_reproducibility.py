"""Write *measured* host-vs-container reproducibility evidence, never hardcoded results.

Usage:
  MIBICI_API_URL=http://127.0.0.1:8000 python tools/check_docker_reproducibility.py \
    --environment local-docker --machine MY-PC --architecture linux/amd64
"""

import argparse
import csv
import os
from pathlib import Path

import mlflow.sklearn
import numpy as np
import pandas as pd
import requests

from mibici.config import CHAMPION_EXPORT_DIR, SNAPSHOT_PATH
from mibici.serving.snapshot import build_feature_table, day_rows

DATES = ("2026-09-30", "2026-10-01")
OUTPUT = Path("reports/evidence/final_docker/reproducibility.csv")
COLUMNS = [
    "environment",
    "machine",
    "architecture",
    "total_2026_09_30",
    "total_2026_10_01",
    "max_abs_diff_vs_host",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--environment", required=True)
    parser.add_argument("--machine", required=True)
    parser.add_argument("--architecture", required=True)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    url = os.getenv("MIBICI_API_URL", "").rstrip("/")
    if not url:
        parser.error("MIBICI_API_URL is required")
    if not SNAPSHOT_PATH.is_file():
        parser.error("Missing snapshot. Run make serving-snapshot SNAPSHOT_ARGS='--no-track'")

    model = mlflow.sklearn.load_model(str(CHAMPION_EXPORT_DIR))
    features = build_feature_table(pd.read_parquet(SNAPSHOT_PATH))
    values = {}
    max_abs = 0.0
    for date in DATES:
        rows = day_rows(features, date)
        host_values = np.asarray(model.predict(rows), dtype=float)
        response = requests.post(url + "/predict", json={"date": date}, timeout=180)
        response.raise_for_status()
        payload = response.json()
        reported = float(payload["total_predicted_departures"])
        if abs(reported - float(host_values.sum())) > 0.00101:
            raise SystemExit(f"Total predictions disagree for {date}")
        remote = pd.DataFrame(
            [
                (station["station_id"], hour["hour_start"], hour["predicted_departures"])
                for station in payload["stations"]
                for hour in station["hours"]
            ],
            columns=["station_id", "hour_start", "remote"],
        )
        remote["hour_start"] = pd.to_datetime(remote["hour_start"])
        expected = rows[["station_id", "hour_start"]].assign(host=host_values)
        joined = expected.merge(
            remote,
            on=["station_id", "hour_start"],
            how="outer",
            indicator=True,
            validate="one_to_one",
        )
        if not joined["_merge"].eq("both").all():
            raise SystemExit(f"Forecast keys disagree for {date}")
        maximum = float((joined["host"] - joined["remote"]).abs().max())
        if maximum > 0.00101:
            raise SystemExit(f"Per-hour predictions differ for {date}: {maximum}")
        max_abs = max(max_abs, maximum)
        values[date] = reported

    row = {
        "environment": args.environment,
        "machine": args.machine,
        "architecture": args.architecture,
        "total_2026_09_30": f"{values[DATES[0]]:.3f}",
        "total_2026_10_01": f"{values[DATES[1]]:.3f}",
        "max_abs_diff_vs_host": f"{max_abs:.6f}",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    existing = []
    if args.output.exists():
        with args.output.open(newline="", encoding="utf-8") as handle:
            existing = list(csv.DictReader(handle))
        existing = [
            item for item in existing
            if (item["environment"], item["machine"], item["architecture"])
            != (row["environment"], row["machine"], row["architecture"])
        ]
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows([*existing, row])
    print("Measured and verified:", row)
    print("Evidence:", args.output)


if __name__ == "__main__":
    main()
