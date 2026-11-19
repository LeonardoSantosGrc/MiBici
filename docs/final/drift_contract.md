# Drift contract (data drift and performance drift)

**Owner:** José Eduardo De Moraes López · **Branch:** `feat/final-drift`

## Purpose

Show that the team can detect when the inputs of the model change (data drift) and when the
model forecasts worse (performance drift), using synthetic scenarios built from real data.

| Concept | Question | Needs observed departures? |
|---|---|---|
| Data drift | Do the inputs look different? P(X current) ≠ P(X reference) | No: it can be measured the same day |
| Performance drift | Does the model err more? (higher WAPE, MAE) | Yes |

Detecting data drift does not prove a performance loss, and a performance loss can happen
without data drift. The report must keep both apart.

## Inputs

| Input | Source |
|---|---|
| Data | `data/processed/station_hour_v1.parquet` with features from `add_history_features` (`mibici.modeling.train.load_modeling_table`) |
| Model | `mlflow.sklearn.load_model("models/serving/champion")` |
| Metrics | `mibici.modeling.metrics.evaluate` (segments: all, established stations, new-station cohort) |

Drift does not use FastAPI or Docker.

## Periods

| Role | Period | Reason |
|---|---|---|
| Reference for data drift | 2026-06-01 to 2026-07-31 | Latest data the champion saw; two months, like the current period |
| Current (natural) | 2026-08-01 to 2026-09-30 | The "production" period. Its WAPE (0.6066) is the performance reference |
| Control | Bootstrap sample of the reference | Must not raise an alarm |

The champion (version 3) was trained up to 2026-07-31, so June and July are training data for
it. Use them only as the distribution reference. Performance drift is measured against the
unperturbed current period, never against June and July.

## Features monitored

| Features | Type | Metric |
|---|---|---|
| `avg_4_weeks`, `t_168`, `station_age_days` | Numeric | PSI with 10 quantile bins of the reference, and the KS statistic D |
| `hour`, `day_of_week`, `is_weekend` | Categorical | PSI over categories |
| `month` | Calendar | Excluded from alarms: June-July and August-September share no month, so it would always drift |
| Model prediction | Numeric | PSI (monitoring without observed departures) |

PSI = Σ (a_i − e_i) · ln(a_i / e_i), with e_i and a_i the shares of the reference and current
rows in bin i (add a small constant to empty bins). Industry thresholds: < 0.1 stable,
0.1 to 0.25 moderate, > 0.25 strong. The KS statistic D = max |F_ref(x) − F_cur(x)| is reported
as a magnitude; its p-value is not used, because with about 560,000 rows any tiny difference
is significant.

## Scenarios

Each scenario has three levels (mild, moderate, strong). Every magnitude must be derived from
the real data and its derivation shown in the notebook; no arbitrary values.

| Scenario | What changes | How | Calibration source | What it shows |
|---|---|---|---|---|
| S1 Demand shift | Departures from 2026-08-01 on, then features recomputed with `add_history_features` | Multiply `departures` by a factor | Observed year-over-year change of monthly departures | Data drift and performance loss together |
| S2 Station expansion | Share of new-station rows in the current period | Resample current rows to raise the share of `is_new_station_cohort` | Observed growth of the 2026 cohort | Drift concentrated in one segment; read it with the segment table |
| S3 Data failure | `t_168` arrives as 0 for some stations | Set `t_168 = 0` on a share of stations (features only) | Share of stations affected, from one station to a large group | A broken input that the data contract or drift must catch |
| S4 Concept change | Observed departures fall in some hours, inputs unchanged | Multiply the target of selected hours after the features are computed | Observed drop on extraordinary-event days (`is_extraordinary_event`) | No data drift, yet performance falls |

Use `mibici.config.RANDOM_STATE` for every random step.

## Performance against the baseline (required by the assignment)

Every scenario and level is scored for two models on the same rows:

| Model | How |
|---|---|
| Champion | `mlflow.sklearn.load_model("models/serving/champion").predict(rows)` |
| Baseline | `make_model("baseline")` (`FourWeekAverageBaseline`, it returns `avg_4_weeks`), the reference of the candidate rule in Phase 2 |

This shows whether the champion still beats the baseline under drift. Example: in S3 the
champion reads the broken `t_168`, while the baseline only uses `avg_4_weeks`.

## Alert criteria and proposed action (required by the assignment)

| Signal | Threshold | Why this threshold | Proposed action |
|---|---|---|---|
| Data contract | Any monitored feature missing, negative or outside its domain (hours 0 and 5 to 23, `day_of_week` 0 to 6) | Same rules as `build_feature_table` | **Review the feature pipeline** before scoring (expected in S3) |
| Data drift | PSI > 0.25 on any monitored feature, or on the prediction | Industry convention for a strong shift | **Investigate**: confirm with performance once observed departures arrive |
| Performance | Champion WAPE ≥ baseline WAPE on the same rows | The champion was promoted because it beat the baseline (Phase 2 candidate rule; on the test period 0.6066 vs 0.6227). Losing that advantage removes the reason it is in production | **Retrain** with recent data (expected in S1 and S2) |
| Performance without data drift | Champion WAPE rises against the natural current period (0.6066) while every PSI < 0.1 | The inputs did not change, so the relation between inputs and departures did | **Review the model inputs**: add information the model lacks, such as weather or events (expected in S4) |

Report each scenario and level with the alerts it raised and the action, and compare them
with the expected column above: a scenario that does not raise its expected alert is a
finding, not a failure to hide.

## Code and outputs

| Piece | Location |
|---|---|
| PSI and KS functions | `mibici/monitoring/drift.py` |
| Scenario builders | `mibici/monitoring/scenarios.py` |
| Command | `make drift` |
| Notebook (reads, validates, explains) | `notebooks/4.01-jedm-drift-evaluation.ipynb` |
| Drift table | `reports/evidence/final_drift/drift_by_feature.csv`: scenario, level, feature, psi, ks_d, status |
| Performance table | `reports/evidence/final_drift/performance_by_segment.csv`: scenario, level, model (champion or baseline), segment, wape, mae, volume_ratio, and the change against the natural current period |
| Alert table | `reports/evidence/final_drift/alerts.csv`: scenario, level, data_contract_alert, data_drift_alert, performance_alert, action |
| Figures | `reports/evidence/final_drift/` |

## Tests (`tests/test_drift.py`)

| Test | Expected |
|---|---|
| PSI of a sample against itself | 0 |
| PSI of a clearly shifted sample | > 0.25 |
| Scenarios with the same seed | Identical output |
| Control (bootstrap of the reference) | PSI < 0.1 for every feature |
| Strong S1 | PSI > 0.25 on `avg_4_weeks` |
| Performance table | WAPE and MAE computable for every scenario, model and segment |
| Alert rules | Each rule fires on a small table built to cross its threshold and stays quiet on the control |

Tests use small synthetic tables, like the rest of the test suite.
