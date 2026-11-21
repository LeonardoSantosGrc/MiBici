# Docker serving evidence

Evidence of the containerized service (José Emmanuel Vázquez Galán), produced by running the
builds and tests described in [`docs/final/docker_implementation.md`](../../../docs/final/docker_implementation.md).
The image is published as [`joseemmanuelvg/mibici-api`](https://hub.docker.com/r/joseemmanuelvg/mibici-api)
with tags `1.0.0` and `latest` for `linux/amd64` and `linux/arm64`.

## Screenshots

| File | What it shows |
|---|---|
| `01_docker_build.png` | Successful build of `mibici-api:1.0.0` (155 s on the WSL machine) |
| `02_docker_image.png` | Image size (1.29 GB on disk, 292 MB content) and architecture (`amd64`) |
| `03_docker_healthy.png` | Container running with status `healthy` |
| `04_api_health_model.png` | `/health` and `/model-info` of the container: model version 3 and the forecast range |
| `05_predict_reference.png` | `/predict` totals for 2026-09-30 and 2026-10-01 (11795.739 and 13307.735) and matching `X-Request-ID` header and body |
| `06_integration_tests.png` | `tests/test_docker_integration.py` against the running container: 4 passed |
| `07_reproducibility.png` | Output of `tools/check_docker_reproducibility.py` (host vs container) |
| `08_dockerhub_tags.png` | DockerHub repository with the `1.0.0` and `latest` tags |
| `09_multiarch_manifest.png` | Multi-architecture manifest (`linux/amd64` and `linux/arm64`) |
| `10_published_image.png` | The image pulled from DockerHub: digest, container `healthy`, `/health` and `/model-info` |

## Text outputs

| File | What it shows |
|---|---|
| `local_service_checks.txt` | Local image (size, architecture), container status (`healthy`), `/health` and `/model-info` |
| `prediction_totals.txt` | Totals of both reference dates, model version and matching `X-Request-ID` header and body |
| `integration_tests.txt` | Result of the live integration test (4 passed) |
| `multiarch_manifest.txt` | `docker buildx imagetools inspect` of both tags |
| `published_image_validation.txt` | Pulled image: digest, container status, health, model info and integration test |
| `reproducibility_local.txt` | Log of the reproducibility measurement on the WSL machine |
| `reproducibility.csv` | Measured host-vs-container results, one row per environment (columns below) |

`reproducibility.csv` columns: `environment`, `machine`, `architecture`, `total_2026_09_30`,
`total_2026_10_01`, `max_abs_diff_vs_host`. Every row comes from running
`tools/check_docker_reproducibility.py`: the WSL row from this PR and the DGX rows from the
final integration. A maximum difference of 0.000499 means identical values: the API rounds
each forecast to 3 decimals.

Do not commit temporary model data, full API payloads, authentication tokens or made-up results.
