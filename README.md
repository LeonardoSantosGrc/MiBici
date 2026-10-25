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
```

Quick checks with a few files:

```bash
make data DATA_ARGS="--limit 1"
make profile PROFILE_ARGS="--limit 2"
make interim INTERIM_ARGS="--limit 3 --no-track"   # without logging to MLflow
make processed PROCESSED_ARGS="--no-track"         # without logging to MLflow
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
  impossible given the typical duration of their origin–destination pair, are not counted
  as departures (they remain in interim);
- columns: `station_id`, `hour_start`, `departures` (target), `hour`, `day_of_week`,
  `is_weekend`, `month`, `station_age_days`, `is_new_station_cohort`, `is_extraordinary_event`.

The decisions are constants in `mibici/processed.py`; notebook 1.04 checks that the code and
the documented decisions match. `reports/processed/` contains the summary, the schema and the
applied decisions.

### MLflow

Local state in `mlflow/` (outside Git): `backend/mlflow.db` (SQLite: experiments, parameters,
metrics, tags) and `artifacts/` (evidence files). Each dataset version is a run in the
`mibici-datasets` experiment (`build_interim_v1`, `build_processed_v1`) with the rules and
thresholds (parameters), quality counts (metrics), the Git commit (tags), the reports from
`reports/interim/` or `reports/processed/` (artifacts), and the dataset digest and schema
(inputs). Data is not uploaded to MLflow.

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

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is PHASE.NUM, the creator's
│                         initials, and a short `-` delimited description, e.g.
│                         `0.02-lsg-profile-raw-data` (see "Naming a notebook").
│
├── pyproject.toml     <- Project configuration file with package metadata for
│                         mibici and configuration for tools like black
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
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── mibici   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes mibici a Python module
    │
    ├── config.py               <- Store useful variables and configuration
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
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling
    │   ├── __init__.py
    │   ├── predict.py          <- Code to run model inference with trained models
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

--------
