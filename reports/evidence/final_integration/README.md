# Final integration evidence (technical close)

End-to-end validation of `main` after merging the three final-phase pull requests, run on the
DGX (Linux, `linux/arm64`) on 2026-11-21.

## Technical close

| Item | Value |
|---|---|
| Integrated `main` | `86b87d6` (PR #19 common base, #20 FastAPI, #21 drift, #22 Docker) |
| Tests | 173 passed, 1 skipped (the live Docker test, which runs only with `MIBICI_API_URL`) |
| Lint | Ruff: all checks passed |
| Served model | `models:/mibici-departures/3` (alias `champion`), HistGradientBoostingRegressor, run `3a0e96c6c8c14ec5a8755d675634b1ee` |
| History snapshot | Digest `66bc9009`, versioned in MLflow as `build_serving_snapshot_v1` |
| Docker image | [`joseemmanuelvg/mibici-api`](https://hub.docker.com/r/joseemmanuelvg/mibici-api), tags `1.0.0` and `latest` (same digest), `linux/amd64` and `linux/arm64` |
| Reference forecasts | 2026-09-30: 11795.739 departures; 2026-10-01 (D+1): 13307.735 |

## Pull request reviews

| PR | Owner | Component | Review |
|---|---|---|---|
| #20 | Erik Román Cortez | FastAPI | Contract kept; `/model-info`, `X-Request-ID`, 500 handler with request id and observed departures verified against the real champion |
| #21 | José Eduardo De Moraes López | Data drift and performance drift | `make drift` reproduces the committed tables and figures exactly on another machine |
| #22 | José Emmanuel Vázquez Galán | Docker and DockerHub | Image builds on the DGX; local and published images pass the integration test (4/4) |

## Files

| File | What it shows |
|---|---|
| `01_mlflow_serving_snapshot.png` | MLflow run `build_serving_snapshot_v1` (experiment `mibici-datasets`) with the snapshot parquet and `snapshot.json` as artifacts |
| `02_github_contributors.png` | GitHub *Insights → Contributors* (last 3 months, commits to `main` excluding merges): the four team members |
| `03_api_checks.txt` | HTTP checks of the API served with `make api` from `main`: status codes, totals, telemetry lines |
| `04_openapi.json` | OpenAPI 3.1 schema served by the API (`/health`, `/model-info`, `/predict`) |
| `05_docker_checks.txt` | Image built from `main`, healthy container, live integration test, published image digests and the reproducibility table |
| `06_drift_reproduction.txt` | `make drift` on `main`: identical outputs and the alert summary |
| `07_tests_and_lint.txt` | Final test suite and lint of the integration branch |

## Reproducibility

The same model and snapshot give the same forecasts in every environment measured with
`tools/check_docker_reproducibility.py` (`reports/evidence/final_docker/reproducibility.csv`):

| Environment | Machine | Architecture | Max abs difference vs host |
|---|---|---|---|
| Local image | WSL2 PC | linux/amd64 | 0.000499 |
| API from the conda environment | DGX | linux/arm64 | 0.000499 |
| Image built from `main` | DGX | linux/arm64 | 0.000499 |
| Published image pulled from DockerHub | DGX | linux/arm64 | 0.000499 |

A difference of 0.000499 means identical values: the API rounds each forecast to 3 decimals.
