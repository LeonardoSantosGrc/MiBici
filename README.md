# MiBici

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

MLOps project for analyzing, preparing, versioning, and modeling MiBici open mobility data.

## Quick start (team)

Data is not versioned in Git: everything under `data/` and the derived artifacts in
`reports/profiling/` are regenerated with the commands below.

```bash
git clone git@github.com:LeonardoSantosGrc/MiBici.git
cd MiBici

make setup              # create or update the conda environment and register the "Python (mibici)" kernel
conda activate mibici

make test               # pytest
make lint               # ruff

make data               # download the official CSV files to data/raw/ and build references/data_manifest.csv
make profile            # profile data/raw/ (read-only) and write reports/profiling/

make mlflow             # (another terminal) local MLflow at http://127.0.0.1:5000
make interim            # data/raw -> data/interim (Parquet) + reports/interim/ + MLflow run
make station-hour       # data/interim -> data/interim/station_hour_base.parquet + reports/station_hour/
make processed          # data/interim -> data/processed/station_hour_v1.parquet + reports/processed/ + MLflow run

make train MODEL=baseline                   # train + evaluate on validation + MLflow run
make train MODEL=hgb MODE=window_june       # another model and evaluation mode (see "Phase 2: modeling")
```

Quick checks with a few files:

```bash
make data DATA_ARGS="--limit 1"
make profile PROFILE_ARGS="--limit 2"
make interim INTERIM_ARGS="--limit 3 --no-track"   # without logging to MLflow
make processed PROCESSED_ARGS="--no-track"         # without logging to MLflow
make train MODEL=baseline TRAIN_ARGS="--no-track"  # without logging to MLflow
```

If a download is interrupted, just run `make data` again: complete files are skipped and
incomplete ones (`.part`) are not considered valid.

### Data preparation (interim layer)

`make interim` reads the 142 trip CSV files and the station nomenclature without modifying
them and produces:

| Output | Content |
|---|---|
| `data/interim/trips/year=YYYY/` | Normalized trips (names, types, missing values), with duration, quality flags (`flag_*`) and lineage (`source_file`, `source_sha256`) |
| `data/interim/stations.parquet` | Lifecycle per station (first/last active month, inactive months, new, retired) joined with the 2026-09 nomenclature (`snapshot_*`) |
| `reports/interim/` | `quality_summary.json`, `schema.json` and `file_summary.csv` (rows per file before and after) |

Rules: the only removal is exact duplicates; all other anomalies are only flagged, and how
to treat them is decided with evidence from the EDA. Station capacity comes from a single
snapshot (2026-09) and is not used as historical data.

### Station × hour base (`make station-hour`)

`data/interim/station_hour_base.parquet`: one row per station and hour in which the station
was active, with the number of departures (trips starting at that station in that hour) and
calendar columns. An hour without departures only exists (with value 0) if the station was
operating: no rows are created before its first trip, after its last trip, or in months
without trips. It is a technical base for the EDA; it applies no modeling decisions.

### Modeling dataset: `processed_v1` (`make processed`)

`data/processed/station_hour_v1.parquet` applies the decisions documented with evidence in
`notebooks/1.04-lsg-target-integration.ipynb` (section 8):

- station × hour unit, from 2022-01-01 and only during service hours (00:00–00:59 and 05:00–23:59);
- short trips that return to the same station, and short cross-station trips that are
  extraordinarily short relative to the typical duration of their origin–destination pair,
  are not counted as departures (they remain in interim);
- columns: `station_id`, `hour_start`, `departures` (target), `hour`, `day_of_week`,
  `is_weekend`, `month`, `station_age_days`, `is_new_station_cohort`, `is_extraordinary_event`.

The decisions are constants in `mibici/processed.py`; notebook 1.04 checks that the code and
the documented decisions match. `reports/processed/` contains the summary, the schema and the
applied decisions.

### Phase 2: modeling

#### Modeling contract

The single reference for training and comparing models is section 7 of
`notebooks/3.01-lsg-modeling-contract.ipynb`. Its values live in `mibici/config.py`, and the
notebook checks that both match (section 7.10).

| Element | Value |
|---|---|
| Problem | Departures per station and hour for the 20 service hours of day d+1, using only data available before 22:00 of day d; one global model for all stations |
| Target | `departures` |
| Features | `hour`, `day_of_week`, `is_weekend`, `month`, `station_age_days`, `avg_4_weeks`, `t_168` |
| Not features | `station_id` and `hour_start` (keys); `is_new_station_cohort` and `is_extraordinary_event` (evaluation only) |
| Splits | Train 2022-01-01 → 2026-05-31 · Validation 2026-06-01 → 2026-07-31 · Test 2026-08-01 → 2026-09-30 |
| Baseline | Mean of the same station, weekday and hour over the last 4 weeks, with young-station and all-station fallbacks (`avg_4_weeks`) |
| Metrics | WAPE (primary) and MAE (secondary), by segment: all rows, established stations, 2026 new-station cohort, extraordinary events (when present) |
| Predictions | Clipped at 0 by the common pipeline |
| Seed | `random_state = 0` |

History features use lags of at least one week, so they are known at the cutoff. `t_168`
takes `avg_4_weeks` when the station had no row one week earlier, and the first week of the
data (no earlier week at all) is not used to train (notebook 3.01, section 7.9).

#### How the pieces fit

```text
config.py      contract values: target, features, dates, seed
   ↓
features.py    add_history_features(): avg_4_weeks, t_168, history_source
   ↓
splits.py      train_eval_masks(data, mode): training rows and evaluated rows
   ↓
models.py      make_model(name): common pipeline with that model's block
   │             pipeline.py: select the 7 features → preprocessing → estimator → clip at 0
   ↓
metrics.py     evaluate(rows, predictions): MAE and WAPE by segment
   ↓
train.py       make train: one MLflow run per grid combination, with the baseline on the same rows
```

#### Evaluation modes

| `MODE` | Trains on | Evaluates on | Use |
|---|---|---|---|
| `validation` (default) | Train | Validation (Jun–Jul 2026) | Hyperparameter tuning |
| `window_june` | Train | June 2026 | Stability of the best configuration |
| `window_july` | Train + June 2026 | July 2026 | Stability of the best configuration |
| `test` | Train + validation | Test (Aug–Sep 2026) | Final evaluation, **once**; requires `FINAL=1` |

```bash
make mlflow                                  # another terminal, keep it open
make train MODEL=hgb                         # MODE=validation by default
make train MODEL=hgb MODE=window_june
make train MODEL=hgb MODE=window_july
make train MODEL=hgb MODE=test FINAL=1       # final evaluation only; refused without FINAL=1
```

`MODEL` is a key of `MODELS` in `mibici/modeling/models.py`: `baseline`, `poisson`,
`random_forest` or `hgb`. A model is a candidate only if it beats the baseline WAPE on
validation and on both windows (notebook 3.01, section 7.6).

#### Workflow for each model owner

| Model | Owner | Block in `models.py` | Notebook |
|---|---|---|---|
| `PoissonRegressor` | José Eduardo | `poisson` | `3.02-jedm-poisson.ipynb` |
| `RandomForestRegressor` | Erik | `random_forest` | `3.03-erc-random-forest.ipynb` |
| `HistGradientBoostingRegressor` | José Emmanuel | `hgb` | `3.04-jevg-hgb.ipynb` |

1. Explore in your notebook with the common pieces, always evaluating on validation:

   ```python
   from mibici.config import TARGET
   from mibici.modeling.metrics import evaluate
   from mibici.modeling.models import make_model
   from mibici.modeling.splits import train_eval_masks
   from mibici.modeling.train import load_modeling_table

   data = load_modeling_table()  # processed_v1 + contract features
   train, evaluated = train_eval_masks(data, "validation")

   model = make_model("hgb").set_params(model__regressor__max_depth=8)
   model.fit(data[train], data.loc[train, TARGET])
   evaluate(data[evaluated], model.predict(data[evaluated]))  # MAE and WAPE by segment
   ```

2. Copy the chosen values to your block in `mibici/modeling/models.py`: the `estimator`,
   an optional `preprocessing` transformer, and the `param_grid` to try. Parameter names
   follow the pipeline: `model__regressor__<name>` for the estimator and
   `preprocess__<name>` for the preprocessing, e.g.
   `{"model__regressor__max_depth": [6, 10], "model__regressor__learning_rate": [0.05, 0.1]}`.
   An empty grid trains the block once as written.
3. Commit, then run `make train MODEL=<your block>` (validation) and, for your best
   configuration, the two windows. Runs made with uncommitted changes are tagged
   `git_dirty=true`.

| You may change | Do not change |
|---|---|
| Your own block in `mibici/modeling/models.py` | `mibici/config.py` (target, features, dates, seed) |
| Your own notebook | `features.py`, `splits.py`, `baseline.py`, `metrics.py`, `pipeline.py`, `train.py` |
| | Notebook 3.01 and the `baseline` block |
| | The test split: never use `MODE=test` to choose anything |

Propose any change to the shared pieces to the team first. Each member's MLflow is local, so
runs are not shared: what is shared is your block in `models.py`. At integration, the final
runs of all models are reproduced from Git on a single machine and MLflow.

### MLflow

Local state in `mlflow/` (outside Git): `backend/mlflow.db` (SQLite: experiments, parameters,
metrics, tags) and `artifacts/` (evidence files). Each dataset version is a run in the
`mibici-datasets` experiment (`build_interim_v1`, `build_processed_v1`) with the rules and
thresholds (parameters), quality counts (metrics), the Git commit (tags), the reports from
`reports/interim/` or `reports/processed/` (artifacts), and the dataset digest and schema
(inputs). Data is not uploaded to MLflow.

Each `make train` run goes to the `mibici-modeling` experiment, named
`<model>-<mode>-<NN>` (one per grid combination):

| Part | Content |
|---|---|
| Parameters | Model, mode, features, target, seed, first and last hour and rows of the training and evaluated sets, every hyperparameter of the estimator and of the preprocessing |
| Metrics | `<segment>_wape` and `<segment>_mae` of the model and `baseline_<segment>_*` of the baseline on the same rows, plus `fit_seconds` |
| Tags | `owner`, `team`, `git_commit`, `git_branch`, `git_dirty` |
| Inputs and artifacts | `processed_v1` digest and schema; `scores_by_segment.csv` |

Trained models are not saved by `make train`; the best configuration of each model is
registered at integration.

### What is versioned

| Path | Git | How it is produced |
|---|---|---|
| `data/raw/`, `data/interim/`, `data/processed/`, `data/external/` | No | `make data` (raw); later pipeline steps |
| `references/data_manifest.csv` | Yes | `make data` (name, URL, size and SHA-256 per file) |
| `reports/profiling/raw_trips_profile.csv`, `nomenclature_profile.csv` | Yes | `make profile` (summaries used as evidence) |
| `reports/profiling/station_month_counts.csv`, `daily_trip_counts.csv`, `nomenclature_stations.csv` | No | `make profile` |
| `reports/interim/` | Yes | `make interim` (quality summaries, schema and per-file lineage) |
| `reports/station_hour/` | Yes | `make station-hour` (base validation summary) |
| `reports/processed/` | Yes | `make processed` (summary, schema and decisions of `processed_v1`) |
| `reports/1.02-erc-atypical-days.csv` | Yes | Notebook 1.02 (atypical days, reused by notebook 1.04) |
| `reports/evidence/` | Yes | Manual screenshots for the deliverable (e.g. MLflow) |
| `mlflow/` | No | `make mlflow` (local Tracking Server state) |
| `reports/figures/` | Yes | By running the notebooks |
| `notebooks/` | Yes, with outputs | Can be re-run to regenerate them |

Do not commit `.env`, credentials, tokens or temporary files.

### Notebooks

#### Naming a notebook

We follow the Cookiecutter Data Science convention: `PHASE.NUM-initials-description.ipynb`

| Part | Meaning | Values |
|---|---|---|
| `PHASE` | Analysis stage | `0` exploration · `1` cleaning and features · `2` visualization · `3` modeling · `4` publication |
| `NUM` | Order within the phase, two digits | `01`, `02`, … |
| `initials` | Notebook owner | e.g. `lsg` |
| `description` | What it does, in English, lowercase, hyphen-separated | e.g. `profile-raw-data` |

Examples: `0.02-lsg-profile-raw-data.ipynb`, `1.01-abc-clean-trips.ipynb`, `3.01-abc-baseline-model.ipynb`.

Team rules:

- Before creating one, check the last `NUM` of that phase so it is not repeated.
- Logic repeated across notebooks moves to `mibici/` and is imported (see `mibici/profiling.py`).
- Notebooks are committed **with their outputs**; anyone can re-run them.
- Code, comments and commit messages are in English; notebook explanations (markdown) are in Spanish.
- Avoid two people editing the same notebook at the same time: `.ipynb` conflicts are hard to resolve.

#### Current notebooks

- `0.01-lsg-inspect-mibici-links.ipynb`: inspection of the official source and link validation.
- `0.02-lsg-profile-raw-data.ipynb`: diagnosis of the raw history. Requires `make data` and
  `make profile` before running:

  ```bash
  jupyter nbconvert --to notebook --execute --inplace notebooks/0.02-lsg-profile-raw-data.ipynb
  ```

- PR2 EDA (require `make interim`):
  - `1.01-jedm-temporal-patterns.ipynb` (José Eduardo): patterns by hour and day of week.
  - `1.02-erc-seasonality-anomalies.ipynb` (Erik): seasonality, atypical days and 2020–2021.
  - `1.03-jevg-station-expansion.ipynb` (José Emmanuel): new 2026 stations versus established ones.
- `1.04-lsg-target-integration.ipynb` (Leonardo): validates the station × hour base, integrates
  the three EDA notebooks (cell-level citations, their files and cross-checks against the base),
  documents the decisions and validates `processed_v1`. Requires `make station-hour`,
  `make processed` and, for the MLflow section, `make mlflow`.
- `3.01-lsg-modeling-contract.ipynb` (Leonardo): audits `processed_v1`, chooses the
  validation strategy, baseline and metrics with evidence, and closes with the modeling
  contract (section 7). Imports the logic from `mibici/`. Requires `make processed`.
- Reserved for Phase 2 models: `3.02-jedm-poisson`, `3.03-erc-random-forest`,
  `3.04-jevg-hgb` (see "Workflow for each model owner").

## Project Organization

```
├── Makefile           <- Makefile with convenience commands like `make data`, `make processed` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── environment.yml    <- Conda environment (conda-forge) used to reproduce the project: `make setup`
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is PHASE.NUM, the creator's
│                         initials, and a short `-` delimited description, e.g.
│                         `0.02-lsg-profile-raw-data` (see "Naming a notebook").
│
├── pyproject.toml     <- Package metadata for mibici and Ruff configuration (lint and format)
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   ├── evidence       <- Manual screenshots used as evidence for the deliverable (MLflow)
│   ├── figures        <- Generated graphics and figures to be used in reporting
│   ├── interim        <- Interim quality summaries generated by `make interim`
│   ├── processed      <- processed_v1 summary, schema and decisions (`make processed`)
│   ├── profiling      <- Raw-data profiling tables generated by `make profile`
│   └── station_hour   <- Station x hour base summary generated by `make station-hour`
│
├── tests              <- Pytest suite (`make test`)
│
└── mibici   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes mibici a Python module
    │
    ├── config.py               <- Paths, MLflow URI and the modeling contract values
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── profiling.py            <- Read-only profiling of the raw data
    │
    ├── preparation.py          <- Trip normalisation, lineage, duration, flags, duplicates
    │
    ├── stations.py             <- Station snapshot, lifecycle and metadata flags
    │
    ├── interim.py              <- `make interim`: builds data/interim and reports/interim
    │
    ├── station_hour.py         <- `make station-hour`: station x hour base with valid zeros
    │
    ├── processed.py            <- `make processed`: processed_v1 with the PR2 decisions
    │
    ├── tracking.py             <- MLflow helpers (dataset runs, digest, Git info)
    │
    ├── features.py             <- Station-history features: avg_4_weeks, t_168 (with fallbacks)
    │
    ├── modeling
    │   ├── __init__.py
    │   ├── splits.py           <- Train / validation / test splits and evaluation modes
    │   ├── baseline.py         <- Four-week average baseline as a scikit-learn estimator
    │   ├── metrics.py          <- MAE, WAPE and evaluation by segment
    │   ├── pipeline.py         <- Common pipeline: feature selection, preprocessing, estimator, clip at 0
    │   ├── models.py           <- One block per model (estimator, preprocessing, grid); owners edit theirs
    │   ├── train.py            <- `make train`: train, evaluate and log to MLflow
    │   └── predict.py          <- Code to run model inference with trained models (template)
    │
    └── plots.py                <- Code to create visualizations
```

--------
