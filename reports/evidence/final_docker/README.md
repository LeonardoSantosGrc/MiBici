# Docker serving evidence

Add real evidence only after executing the builds/tests. Recommended names:

- `01_build.log` — build summary and duration
- `02_image_size.txt` — `docker image ls mibici-api:1.0.0`
- `03_container_health.txt` — `docker ps` (healthy) + `/health`
- `04_model_info.txt` — `/model-info` and model version
- `05_predict_2026_09_30.txt` — prediction totals for past day
- `06_predict_2026_10_01.txt` — D+1 totals
- `07_integration_tests.txt` — live pytest output
- `08_dockerhub_manifest.txt` — multi-arch manifest of both image tags
- `reproducibility.csv` — created by `tools/check_docker_reproducibility.py`

Do not commit temporary model data, API full payloads, auth tokens or made-up results.
