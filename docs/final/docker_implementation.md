# Final delivery — Docker serving and portability (José Emmanuel)

This section implements the existing contract in `docs/final/docker_contract.md`.
It only packages the champion and API already built by the team. It does not
retrain HGB, change the champion, change FastAPI, or modify the drift pipeline.

## 1. Clean Git handoff (preserve local and ignored data)

From `~/MiBici`, review local changes before switching. These commands do not
delete untracked data, logs or local MLflow databases:

```bash
cd ~/MiBici
git branch --show-current
git status --short --untracked-files=all
git status --ignored --short -- models/serving/history_snapshot.parquet data/ mlflow/
git fetch origin --prune

git switch main
git pull --ff-only origin main
git switch -c feat/final-docker
```

If `git switch` refuses to proceed because local files conflict, stop. Do not use
`git reset --hard`, `git clean`, or `git stash -u`. Back up any conflicting local
file separately and reconcile it before retrying. A branch stores committed code,
not the ignored files; untracked non-conflicting data stay on disk.

Confirm the branch contains FastAPI PR #20 (`mibici/serving/app.py`, `/model-info`).

## 2. Prepare the exact serving snapshot

The snapshot is intentionally not in Git. A local machine must regenerate it
using the exact processed dataset, and the result must match the tracked digest:

```bash
make serving-snapshot SNAPSHOT_ARGS="--no-track"
cat models/serving/snapshot.json
ls -lh models/serving/history_snapshot.parquet
```

Expected fingerprint: `66bc9009`, 814240 rows, 470 stations,
forecast range 2026-08-01 through 2026-10-01.
The build must report `matches the reference`.
If processed data are missing, regenerate them using the repo README; do not copy
an arbitrary parquet from another run.

The champion `models/serving/champion/` is already exported and committed.
Its MLflow Registry URI is `models:/mibici-departures/3`, whose exported model
is loaded from a local path **without running MLflow** in the container.

## 3. Lock the Python dependencies from the tested Conda environment

```bash
conda activate mibici
python tools/prepare_serving_requirements.py
cat requirements-serving.txt
```

The script fails if the active environment disagrees with the versions recorded
in `models/serving/champion/requirements.txt`. It locks the API's additional
direct dependencies as found in the verified environment. Commit the resulting
`requirements-serving.txt` with the Docker changes. Indirect dependency
resolution remains delegated to pip; for stricter supply-chain reproducibility,
create and review a transitive lock file in a separate PR.

## 4. Local image and health checks

```bash
make docker-build
make docker-run
```

`make docker-run` is foreground and starts Uvicorn; use a second terminal for:

```bash
docker ps --filter name=mibici-api
curl -fsS http://127.0.0.1:8000/health
curl -fsS http://127.0.0.1:8000/model-info
curl -fsS -X POST http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' -d '{"date":"2026-09-30"}'
curl -fsS -X POST http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' -d '{"date":"2026-10-01"}'
```

For large JSON responses, redirect the `curl` output to untracked files and
inspect the totals with Python rather than committing the full payloads.
Expected totals: 11795.739 (2026-09-30) and 13307.735 (2026-10-01).
Both responses must have `X-Request-ID` and the model must be version 3.

## 5. Live integration and reproducibility checks

Run with the container running and the same snapshot/model available to the host:

```bash
MIBICI_API_URL=http://127.0.0.1:8000 python -m pytest -q tests/test_docker_integration.py
MIBICI_API_URL=http://127.0.0.1:8000 python tools/check_docker_reproducibility.py \
  --environment local-image --machine my-pc --architecture linux/amd64
```

The test compares every station/hour with `model.predict()` on the host and
permits a maximum difference of 0.00101 for API rounding. It also checks the
recorded totals and model identity. Without `MIBICI_API_URL`, pytest skips the
live integration suite, so the regular test command works without Docker.

The evidence CSV is generated only from measured results. Repeat on the DGX
using `--architecture linux/arm64` and again on a different machine after
pulling from DockerHub. Do not fill untested rows with assumed results.

## 6. Publish a multi-architecture image to DockerHub

Use an access token interactively, **never** put a credential in the repo.
Substitute your DockerHub username; confirm Buildx works before publishing:

```bash
docker login -u <dockerhub-user>
docker buildx version
docker buildx create --name mibici-multiarch --use --bootstrap

docker buildx build --platform linux/amd64,linux/arm64 \
  --tag <dockerhub-user>/mibici-api:1.0.0 \
  --tag <dockerhub-user>/mibici-api:latest \
  --push .

docker buildx imagetools inspect <dockerhub-user>/mibici-api:1.0.0
```

Buildx requires an appropriate builder, cross-platform emulation or native
workers, a working network and sufficient resources. A published tag is not
proof of correctness: pull it on an independent machine and rerun the tests.
Do not overwrite version `1.0.0` after publication. Bump the version whenever
the exported champion or the snapshot changes. Record your real image link
and the inspect output in the PR and top-level README.

## 7. Evidence and acceptance criteria

Store only factual evidence in `reports/evidence/final_docker/`:

- Build log with successful image ID, relevant warnings and measured duration.
- `docker image ls` size, and `docker ps` showing `(healthy)`.
- Host requests `/health`, `/model-info`, `/predict` for both days, and a 422 error.
- `pytest` integration output, `reproducibility.csv` from real hosts, and tags/architectures from DockerHub.
- Note any unavailable environment explicitly instead of inventing a result.

Acceptance: Docker starts without MLflow or `data/`, the image's model and snapshot
match the committed references, healthcheck passes, predictions agree with the host,
error responses preserve `X-Request-ID`, and two architectures exist under both tags.

## 8. Integration responsibilities

FastAPI remains owned by Erik; drift remains owned by José Eduardo; shared
infrastructure and final report by Leonardo. This PR only adds deployment,
portability and integration verification. Any serving contract change must be
proposed to the team first.
