# Docker contract

**Owner:** José Emmanuel Vázquez Galán · **Branch:** `feat/final-docker`

## Purpose

Run the forecasting API in a container that works the same on the DGX (ARM64) and on x86
machines, without the MLflow server and without the project data.

## Image

| Item | Contract |
|---|---|
| Base image | `python:3.12-slim` (published for ARM64 and x86) |
| Working directory | `/app` |
| Dependencies | `pip install -r requirements-serving.txt` (new file at the repository root) |
| Project package | The pickle references `mibici.modeling.pipeline`, so `mibici/` must be importable: copy it to `/app/mibici` and set `PYTHONPATH=/app` (or `pip install --no-deps .`, which also needs `pyproject.toml` and `README.md`) |
| Files to copy | `mibici/`, `models/serving/champion/`, `models/serving/history_snapshot.parquet` (and `pyproject.toml`, `README.md` if the package is installed) |
| Never copy | `data/`, `mlflow/`, `notebooks/`, `reports/`, `.git/`, `.env` (list them in `.dockerignore`) |
| Environment | `MIBICI_MODEL_URI=/app/models/serving/champion`, `MIBICI_SNAPSHOT_PATH=/app/models/serving/history_snapshot.parquet` |
| Command | `uvicorn mibici.serving.app:app --host 0.0.0.0 --port 8000` |
| Port | `8000` (`EXPOSE 8000`) |
| Health check | `HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"` (the slim image has no `curl`) |

The history snapshot is not in Git: run `make serving-snapshot SNAPSHOT_ARGS="--no-track"`
before building and check that it prints `matches the reference`.

## Dependencies (`requirements-serving.txt`)

Pin the versions the champion was saved with (from `models/serving/champion/requirements.txt`)
plus what the API imports:

| Package | Version | Why |
|---|---|---|
| `scikit-learn` | `1.9.1` | Unpickles the pipeline |
| `pandas` | `3.1.0` | Rows and features |
| `numpy` | `2.5.4` | Arrays |
| `scipy` | `1.18.1` | scikit-learn dependency |
| `cloudpickle` | `3.1.2` | Format of `model.pkl` |
| `mlflow` | `3.17.2` | `mlflow.sklearn.load_model`; trying `mlflow-skinny==3.17.2` to make the image smaller is welcome if the model still loads |
| `pyarrow` | Same as the conda environment | Reads the snapshot parquet |
| `fastapi`, `uvicorn` | Same as the conda environment | The service |
| `loguru`, `python-dotenv`, `tqdm`, `typer`, `requests` | Same as the conda environment | Imported by `mibici.config`, `mibici.tracking` and `mibici.serving.snapshot` |

Get the environment versions with `conda list -n mibici`.

## Make targets

```text
make docker-build    # docker build -t mibici-api:1.0.0 .
make docker-run      # docker run --rm -p 8000:8000 --name mibici-api mibici-api:1.0.0
```

## Publishing on DockerHub (required by the assignment)

The final PDF must link to the published image. Publish it from your DockerHub account, so
anyone can run the service with one command and without the repository.

| Item | Contract |
|---|---|
| Repository | `<your-dockerhub-user>/mibici-api`, public |
| Tags | `1.0.0` (champion version 3, snapshot `66bc9009`; never overwritten) and `latest` (points to the newest version) |
| Architectures | `linux/amd64` and `linux/arm64` under the same tag, built with `docker buildx` (most machines are x86; the DGX is ARM64) |
| Login | `docker login` with a DockerHub access token, not the account password; never commit the token |
| Version rule | A new champion or snapshot gets a new tag (`1.1.0`, …); published tags are not replaced |

```bash
docker buildx build --platform linux/amd64,linux/arm64 \
  -t <your-dockerhub-user>/mibici-api:1.0.0 -t <your-dockerhub-user>/mibici-api:latest --push .
docker run --rm -p 8000:8000 <your-dockerhub-user>/mibici-api:1.0.0   # check the published image
```

Write the image URL and its tags in the README (Docker section) and in the PR.

## Checks

| Check | Expected |
|---|---|
| Build | Succeeds on the DGX (ARM64) and on your machine |
| Start | `docker ps` shows the container as `healthy` |
| `GET /health` from the host | `model_version` = 3, forecast range 2026-08-01 to 2026-10-01 |
| `POST /predict {"date": "2026-09-30"}` from the host | `total_predicted_departures` = 11795.739 (same as `make predict`) |
| `POST /predict {"date": "2026-10-01"}` from the host | `total_predicted_departures` = 13307.735 |
| Absolute path in `MLmodel` | The file stores a DGX path (`artifact_path`); confirm the model loads from `/app/models/serving/champion` anyway |
| Image size and build time | Reported in the PR |
| Published image | `docker run` of `<your-dockerhub-user>/mibici-api:1.0.0` on a machine that did not build it gives the same totals |

## Integration test

`tests/test_docker_integration.py`: calls a running container at the URL given by the
environment variable `MIBICI_API_URL` (for example `http://127.0.0.1:8000`) and checks:

| Check | Expected |
|---|---|
| `/health` | Status `ok`, `model_version` = 3, forecast range 2026-08-01 to 2026-10-01 |
| `/predict` for 2026-09-30 and 2026-10-01 | Every station and hour equals the forecast computed on the host from `models/serving/champion` and the local snapshot (`build_feature_table` + `model.predict`): maximum absolute difference ≤ 0.001 (the API rounds to 3 decimals) |
| Totals | 11795.739 and 13307.735 |

It is skipped when the variable is not set, so `make test` keeps working without Docker.

## Reproducibility in a clean environment (required by the assignment)

The assignment asks to show that the same model gives the same results in an environment
different from the one where it was trained. The container is that environment: no conda,
only `requirements-serving.txt`, the code, the exported model and the snapshot. Training used
fixed seeds (`RANDOM_STATE = 0` in `mibici/config.py`); inference is deterministic.

Run the integration test against each environment and report the results in the PR and in
`reports/evidence/final_docker/reproducibility.csv`, with the columns `environment`,
`machine`, `architecture`, `total_2026_09_30`, `total_2026_10_01` and `max_abs_diff_vs_host`:

| Environment (one row each) | Reference values |
|---|---|
| Conda environment on the DGX, ARM64 | The reference: 11795.739, 13307.735, difference 0 |
| Local image `mibici-api:1.0.0` on the DGX, ARM64 | Must match the reference |
| Local image `mibici-api:1.0.0` on your machine, x86 | Must match the reference |
| Published image pulled from DockerHub on a machine that did not build it | Must match the reference |

Leave a row out only if that machine is not available, and say so in the PR.

## Evidence

`reports/evidence/final_docker/`: build output, `docker images` (size), `docker ps` with
`healthy`, the requests from the host, `reproducibility.csv`, and the DockerHub page showing
both tags and architectures.
