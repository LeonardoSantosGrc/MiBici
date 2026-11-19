# Entrega Final: common base

The final phase adds serving, portability and monitoring to the model selected in Phase 2.
It does not retrain or retune models, change the champion or rebuild `processed_v1`.

```text
SERVING                                            MONITORING (independent)
MLflow Registry  mibici-departures@champion (v3)   processed_v1
      | make export-model (DGX only)                    |
      v                                            reference: Jun-Jul 2026
models/serving/champion/  (in Git)                 current:   Aug-Sep 2026 + synthetic scenarios
models/serving/history_snapshot.parquet            (features recomputed with features.py)
      | (make serving-snapshot, outside Git)            |
      v                                                 v
mibici/serving/app.py (FastAPI)  <-- same pipeline -->  champion.predict
      |                                                 |
      v                                                 v
Docker (uvicorn :8000)  <-- HTTP from the host     data drift (PSI, KS) + performance (WAPE, MAE by segment)
```

Serving and drift share only the model and the seven contract features
(`mibici.config.FEATURES`). Drift does not depend on FastAPI or Docker.

## Contracts

| Contract | Owner | Branch |
|---|---|---|
| [Serving (FastAPI)](serving_contract.md) | Erik Román Cortez | `feat/final-fastapi` |
| [Docker](docker_contract.md) | José Emmanuel Vázquez Galán | `feat/final-docker` |
| [Drift](drift_contract.md) | José Eduardo De Moraes López | `feat/final-drift` |
| Common base, integration and report | Leonardo Santos García | `feat/final-common-base` |

## What already exists

| Piece | Location | Command |
|---|---|---|
| Champion export (version 3, `hgb`) and `metadata.json` | `models/serving/champion/` (in Git) | `make export-model` (only where the Registry lives) |
| History snapshot (Jul 4 to Sep 30, 2026) | `models/serving/history_snapshot.parquet` (outside Git) | `make serving-snapshot` |
| Snapshot reference digest | `models/serving/snapshot.json` (in Git) | Written by the first `make serving-snapshot` |
| Feature table and D+1 rows | `mibici/serving/snapshot.py` | — |
| API: `GET /health`, `POST /forecast` | `mibici/serving/app.py`, `schemas.py` | `make api` |
| Tests | `tests/test_serving_export.py`, `test_serving_snapshot.py`, `test_serving_api.py` | `make test` |

## Getting started on a team machine

```bash
git switch main && git pull
git switch -c feat/final-<topic>
make requirements          # environment with the pinned model versions
make serving-snapshot      # needs data/processed/station_hour_v1.parquet; must print "matches the reference"
make test
make api                   # http://127.0.0.1:8000/docs
```

If `make serving-snapshot` reports that the snapshot differs from the reference, rebuild
`processed_v1` (`make interim`, `make station-hour`, `make processed`) before going on.

## Rules

| Rule | Reason |
|---|---|
| Do not edit `models/serving/champion/` by hand | It is the exported Registry version; its `metadata.json` traces it to the training run |
| Do not change the pinned versions in `environment.yml` | The champion was saved with them and the Registry database uses the MLflow 3.17.2 schema |
| Do not re-implement features or transformations | The API and drift use `add_history_features` and the registered pipeline, so serving equals training |
| New logic lives in `mibici/` with tests; notebooks read, validate and explain | Same principle as Phases 1 and 2 |
| Code, comments, commits and PR texts in English | Project language rules |
| Evidence goes to `reports/evidence/final_<topic>/` | One folder per component |

## Evidence folders

| Folder | Content |
|---|---|
| `reports/evidence/final_fastapi/` | `/docs` screenshots, request and response examples |
| `reports/evidence/final_docker/` | Build output, `docker ps` with `healthy`, requests from the host |
| `reports/evidence/final_drift/` | Drift and performance tables (CSV) and figures |
| `reports/evidence/final_integration/` | End-to-end run on the DGX |
