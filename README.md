# MiBici

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

MLOps project for analyzing, preparing, versioning, and modeling MiBici open mobility data.

## Uso rápido (equipo)

Los datos no se versionan en Git: todo lo que está en `data/` y los artefactos
derivados de `reports/profiling/` se regeneran con los comandos siguientes.

```bash
git clone git@github.com:LeonardoSantosGrc/MiBici.git
cd MiBici

make setup              # crea o actualiza el entorno conda y registra el kernel "Python (mibici)"
conda activate mibici

make test               # pytest
make lint               # ruff

make data               # descarga los CSV oficiales a data/raw/ y genera references/data_manifest.csv
make profile            # perfila data/raw/ (solo lectura) y escribe reports/profiling/

make mlflow             # (otra terminal) MLflow local en http://127.0.0.1:5000
make interim            # data/raw -> data/interim (Parquet) + reports/interim/ + corrida en MLflow
make station-hour       # data/interim -> data/interim/station_hour_base.parquet + reports/station_hour/
make processed          # data/interim -> data/processed/station_hour_v1.parquet + reports/processed/ + corrida en MLflow
```

Pruebas controladas con pocos archivos:

```bash
make data DATA_ARGS="--limit 1"
make profile PROFILE_ARGS="--limit 2"
make interim INTERIM_ARGS="--limit 3 --no-track"   # sin registrar en MLflow
make processed PROCESSED_ARGS="--no-track"         # sin registrar en MLflow
```

Si la descarga se interrumpe, basta con repetir `make data`: los archivos completos se
omiten y los incompletos (`.part`) no se consideran válidos.

### Preparación de datos (capa interim)

`make interim` lee los 142 CSV de viajes y la nomenclatura sin modificarlos y genera:

| Salida | Contenido |
|---|---|
| `data/interim/trips/year=AAAA/` | Viajes normalizados (nombres, tipos, faltantes), con duración, flags de calidad (`flag_*`) y linaje (`source_file`, `source_sha256`) |
| `data/interim/stations.parquet` | Ciclo de vida por estación (primer/último mes activo, meses inactivos, nueva, retirada) cruzado con la nomenclatura 2026-09 (`snapshot_*`) |
| `reports/interim/` | `quality_summary.json`, `schema.json` y `file_summary.csv` (filas por archivo antes y después) |

Reglas: la única eliminación son los duplicados exactos; el resto de anomalías solo se
marcan con flags y su tratamiento se decide con evidencia del EDA. La capacidad de la
nomenclatura es una sola foto (2026-09) y no se usa como dato histórico.

### Base estación × hora (`make station-hour`)

`data/interim/station_hour_base.parquet`: una fila por estación y hora en que la estación
estaba activa, con el número de salidas (viajes que inician en esa estación y hora) y
columnas de calendario. Una hora sin salidas solo existe (con valor 0) si la estación
operaba: no se crean filas antes del primer viaje, después del último ni en meses sin
viajes. Es una base técnica para el EDA; no aplica decisiones de modelado.

### Dataset para modelar: `processed_v1` (`make processed`)

`data/processed/station_hour_v1.parquet` aplica las decisiones documentadas con evidencia
en `notebooks/1.04-lsg-target-integration.ipynb` (sección 8):

- unidad estación × hora, desde 2022-01-01 y solo en horario de servicio (00:00–00:59 y 05:00–23:59);
- no cuentan como salidas los viajes cortos que regresan a la misma estación ni los cortos
  entre estaciones que son imposibles frente a la duración típica de su pareja (siguen en interim);
- columnas: `station_id`, `hour_start`, `departures` (target), `hour`, `day_of_week`,
  `is_weekend`, `month`, `station_age_days`, `is_new_station_cohort`, `is_extraordinary_event`.

Las decisiones son constantes de `mibici/processed.py`; el notebook 1.04 verifica que el
código y lo documentado coincidan. `reports/processed/` contiene el resumen, el esquema y
las decisiones aplicadas.

### MLflow

Estado local en `mlflow/` (fuera de Git): `backend/mlflow.db` (SQLite: experimentos,
parámetros, métricas, tags) y `artifacts/` (archivos de evidencia). Cada versión del
dataset es una corrida en el experimento `mibici-datasets` (`build_interim_v1`,
`build_processed_v1`) con las reglas y umbrales (parámetros), los conteos de calidad
(métricas), el commit de Git (tags), los reportes de `reports/interim/` o
`reports/processed/` (artefactos) y el digest y esquema de los datasets (inputs). Los
datos no se suben a MLflow.

### Qué se versiona

| Ruta | Git | Cómo se obtiene |
|---|---|---|
| `data/raw/`, `data/interim/`, `data/processed/`, `data/external/` | No | `make data` (raw); pasos posteriores del pipeline |
| `references/data_manifest.csv` | Sí | `make data` (nombre, URL, tamaño y SHA-256 por archivo) |
| `reports/profiling/raw_trips_profile.csv`, `nomenclature_profile.csv` | Sí | `make profile` (resúmenes usados como evidencia) |
| `reports/profiling/station_month_counts.csv`, `daily_trip_counts.csv`, `nomenclature_stations.csv` | No | `make profile` |
| `reports/interim/` | Sí | `make interim` (resúmenes de calidad, esquema y linaje por archivo) |
| `reports/station_hour/` | Sí | `make station-hour` (resumen de validación de la base) |
| `reports/processed/` | Sí | `make processed` (resumen, esquema y decisiones de `processed_v1`) |
| `reports/1.02-erc-atypical-days.csv` | Sí | Notebook 1.02 (días atípicos, reutilizado por el 1.04) |
| `reports/evidence/` | Sí | Capturas manuales para la entrega (p. ej. MLflow) |
| `mlflow/` | No | `make mlflow` (estado local del Tracking Server) |
| `reports/figures/` | Sí | Ejecutando los notebooks |
| `notebooks/` | Sí, con resultados | Pueden re-ejecutarse para regenerarlos |

No subir `.env`, credenciales, tokens ni archivos temporales.

### Notebooks

#### Cómo nombrar un notebook

Seguimos la convención de Cookiecutter Data Science: `FASE.NUM-iniciales-descripcion.ipynb`

| Parte | Qué indica | Valores |
|---|---|---|
| `FASE` | Etapa del análisis | `0` exploración · `1` limpieza y features · `2` visualización · `3` modelado · `4` publicación |
| `NUM` | Orden dentro de la fase, con dos dígitos | `01`, `02`, … |
| `iniciales` | Responsable del notebook | p. ej. `lsg` |
| `descripcion` | Qué hace, en inglés, minúsculas y separado con guiones | p. ej. `profile-raw-data` |

Ejemplos: `0.02-lsg-profile-raw-data.ipynb`, `1.01-abc-clean-trips.ipynb`, `3.01-abc-baseline-model.ipynb`.

Reglas del equipo:

- Antes de crear uno, revisar el último `NUM` de esa fase para no repetirlo.
- La lógica que se repita entre notebooks se mueve a `mibici/` y se importa (ver `mibici/profiling.py`).
- Los notebooks se suben **con sus resultados**; cualquiera puede volver a ejecutarlos.
- Evitar que dos personas editen el mismo notebook a la vez: los conflictos en `.ipynb` son difíciles de resolver.

#### Notebooks actuales

- `0.01-lsg-inspect-mibici-links.ipynb`: inspección de la fuente oficial y validación de enlaces.
- `0.02-lsg-profile-raw-data.ipynb`: diagnóstico del histórico raw. Requiere `make data` y
  `make profile` antes de ejecutarse:

  ```bash
  jupyter nbconvert --to notebook --execute --inplace notebooks/0.02-lsg-profile-raw-data.ipynb
  ```

- EDA del PR2 (requieren `make interim`):
  - `1.01-jedm-temporal-patterns.ipynb` (José Eduardo): patrones por hora y día de la semana.
  - `1.02-erc-seasonality-anomalies.ipynb` (Erik): estacionalidad, días atípicos y 2020–2021.
  - `1.03-jevg-station-expansion.ipynb` (José Emmanuel): estaciones nuevas de 2026 frente a consolidadas.
- `1.04-lsg-target-integration.ipynb` (Leonardo): valida la base estación × hora, integra
  los tres EDA (citas por celda, sus archivos y verificación contra la base), documenta
  las decisiones y valida `processed_v1`. Requiere `make station-hour`, `make processed`
  y, para la sección de MLflow, `make mlflow`.

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
│                         `0.02-lsg-profile-raw-data` (see "Cómo nombrar un notebook").
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         mibici and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   ├── figures        <- Generated graphics and figures to be used in reporting
│   └── profiling      <- Raw-data profiling tables generated by `make profile`
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

