# Predicting Crime Types by Location & Time-Series Data
### DSAI3202 — Winter 2026 | Project Phase 1

**Authors:** Nur Afiqah · 60306981 &nbsp;|&nbsp; Elaf Marouf · 60107174

---

## Table of Contents
1. [Project Overview](#1-project-overview)
2. [Repository Structure](#2-repository-structure)
3. [How to Run](#3-how-to-run)
4. [Task 1 — Data Ingestion](#4-task-1--data-ingestion)
5. [Task 2 — ETL Pipeline](#5-task-2--etl-pipeline)
6. [Task 3 — Cataloging & Governance](#6-task-3--cataloging--governance)
7. [Task 4 — Exploratory Data Analysis](#7-task-4--exploratory-data-analysis)
8. [Task 5 — Feature Extraction & Selection](#8-task-5--feature-extraction--selection)
9. [Data Lineage](#9-data-lineage)
10. [Assumptions](#10-assumptions)
11. [Azure Deployment](#11-azure-deployment)

---

## 1. Project Overview

### Description
This project builds a complete AI data pipeline to predict **crime types** in Chicago based on geospatial coordinates and temporal patterns. The system addresses the difficulty of anticipating specific crime categories (theft, assault, vandalism) before they occur, enabling proactive deployment of specialised law enforcement units and preventive measures.

The pipeline emphasises **system-level design** over isolated model performance and covers: hypothesis development, ETL from a Kaggle structured crime dataset, preprocessing, exploratory analysis, and temporal-spatial feature engineering.

### Hypothesis
> By building a classification model following the pipeline process outlined in this project, our aim is to help the police force decide resource allocation at the start of each shift by predicting the crime type so that **response time improves from baseline by 15%** under limited patrol cars.

### Dataset

| Field | Value |
|-------|-------|
| Name | USA Big City Crime Dataset (Chicago) |
| Source | [Kaggle](https://www.kaggle.com/datasets/middlehigh/los-angeles-crime-data-from-2000/data) |
| Raw Size | 1,662.9 MB |
| Raw Rows | 7,391,187 |
| Cleaned Rows | 7,282,602 |
| Columns | 22 |
| Date Range | 2001-01-01 to 2024-05-20 |
| Target Variable | Primary Type (29 classes) |

---

## 2. Repository Structure

```
phase_one/
├── data/
│   ├── raw/                              <- versioned originals — NEVER modified
│   │   └── chicago_crime_v1.0_2026-03-16.csv
│   ├── processed/                        <- ETL output, cleaned and typed
│   │   ├── chicago_crime_cleaned_v1.0.csv
│   │   └── chicago_crime_features_v1.0.csv
│   └── catalog/                          <- schema metadata, lineage, assumptions
│       ├── ingestion_manifest.json
│       ├── etl_report.json
│       ├── data_catalog.json
│       ├── lineage.json
│       ├── assumptions.json
│       ├── zone_registry.json
│       ├── crime_type_merges.json
│       └── label_encoding.json
├── src/
│   ├── ingestion.py                      <- Task 1: Data Ingestion
│   ├── etl.py                            <- Task 2: ETL Pipeline
│   ├── catalog.py                        <- Task 3: Cataloging & Governance
│   ├── eda.py                            <- Task 4: Exploratory Analysis
│   ├── features.py                       <- Task 5: Feature Engineering
│   ├── azure_upload.py                   <- Uploads outputs to Azure Blob Storage
│   └── azure_validate.py                 <- Validates pipeline from Azure Blob Storage
├── outputs/
│   └── eda/                              <- saved EDA plots (PNG)
├── screenshots/
│   ├── azure_validation_result.png       <- ALL CHECKS PASSED terminal output
│   ├── azure_raw_container.png           <- raw-data Blob container
│   ├── azure_processed_container.png     <- processed-data Blob container
│   ├── azure_catalog_container.png       <- catalog-data Blob container
│   └── azure_compute_instance.png        <- phase1-compute ML instance
├── logs/
│   ├── ingestion.log
│   ├── etl.log
│   ├── catalog.log
│   ├── eda.log
│   ├── features.log
│   ├── azure_upload.log
│   └── azure_validate.log
└── README.md
```

---

## 3. How to Run

### Prerequisites
```bash
pip install pandas numpy scikit-learn matplotlib seaborn holidays
```

### Step 1 — Download the dataset
Download the CSV from [Kaggle](https://www.kaggle.com/datasets/middlehigh/los-angeles-crime-data-from-2000/data), place it in the project root and name it `crime_data.csv`.

### Step 2 — Run each task individually
```bash
python src/ingestion.py   # Task 1
python src/etl.py         # Task 2
python src/catalog.py     # Task 3
python src/eda.py         # Task 4
python src/features.py    # Task 5
```

> **Windows note:** If you see `UnicodeEncodeError` warnings in the console, these are cosmetic only — all scripts include a UTF-8 console fix. All outputs are written correctly to files regardless.

> **joblib warning:** The `Could not find number of physical cores` warning from scikit-learn is harmless. Silence it with: `setx LOKY_MAX_CPU_COUNT 4`

---

## 4. Task 1 — Data Ingestion

### Ingestion Mode
**Batch** — one-time historical load. The dataset is a static archive of Chicago crime records from 2001 to 2024. Real-time streaming ingestion is not required for this project.

### Refresh Strategy
On-demand re-run with an incremented semantic version tag (e.g. `v1.1`). Each ingestion run produces a new versioned file, independently traceable by version + date.

### Versioning Convention
```
data/raw/chicago_crime_v1.0_2026-03-16.csv
              ^name    ^ver  ^ingestion date
```

### Storage Layout

| Zone | Path | Rule |
|------|------|------|
| Raw | `data/raw/` | Original download — NEVER modified |
| Processed | `data/processed/` | ETL output — cleaned and typed |
| Catalog | `data/catalog/` | Schema definitions, ETL report, lineage |
| Logs | `logs/` | One log file per script |

### Ingestion Results (confirmed from log)

| Metric | Value |
|--------|-------|
| File size | 1,662.9 MB |
| Rows ingested | 7,391,187 |
| Columns | 22 |
| Schema validation | PASSED — all 22 columns present |
| Exact duplicates | 0 |
| Duplicate Case Numbers | 565 |
| MD5 checksum | `3cfac7dd7d8eb2d47b9665ddbe56e486` |

### Null Summary (raw data)

| Column | Null Count | Null % |
|--------|-----------|--------|
| Ward | 614,820 | 8.32% |
| Community Area | 613,469 | 8.30% |
| Location / Lat / Lon / X / Y | 74,115 | 1.00% |
| Location Description | 9,124 | 0.12% |
| All other columns | 1 each | ~0.00% |

### Output Files
- `data/raw/chicago_crime_v1.0_2026-03-16.csv` — versioned raw file
- `data/catalog/ingestion_manifest.json` — source URL, row count, MD5, columns
- `logs/ingestion.log` — full run log

---

## 5. Task 2 — ETL Pipeline

### Design Principles
- **Reproducible** — deterministic, same input always produces same output
- **Documented** — every step logs row count before and after
- **Non-destructive** — raw data in `data/raw/` is never touched
- **Fail-safe** — critical columns dropped not imputed; all decisions logged

### Transformations Applied

| Step | Name | Rows Dropped | Notes |
|------|------|-------------|-------|
| 1 | Drop exact duplicates | 0 | No identical rows found |
| 2 | Dedup by Case Number | 565 | Keep first occurrence per case |
| 3 | Drop missing critical columns | 74,115 | Date, Primary Type, Lat, Lon |
| 4 | Fill non-critical nulls | 0 | Text cols → `UNKNOWN`; numeric → `0` |
| 5 | Fix data types | 0 | Date → datetime64; Lat/Lon → float64; Arrest → bool |
| 6 | Validate coordinates (bounding box) | 147 | Outside Chicago: Lat [41.60–42.05], Lon [–88.00–87.50] |
| 7 | Validate temporal range | 0 | All records within 2001-01-01 to 2026-03-16 |
| 8 | Standardize crime type labels | 0 | 8 rare types (< 500) merged into OTHER |
| 9 | IQR outlier removal on coordinates | 33,758 | Longitude IQR [–87.84190, –87.50010] |
| 10 | Final quality check | 0 | Validation gate — no critical nulls confirmed |
| 11 | Save cleaned dataset | — | Written to `data/processed/` |
| **Total** | | **108,585** | **Retention: 98.53%** |

### Crime Types Merged into OTHER (Step 8)
The following 8 types had fewer than 500 occurrences and were merged:
`PUBLIC INDECENCY`, `NON-CRIMINAL`, `OTHER NARCOTIC VIOLATION`, `HUMAN TRAFFICKING`, `NON - CRIMINAL`, `RITUALISM`, `NON-CRIMINAL (SUBJECT SPECIFIED)`, `DOMESTIC VIOLENCE`

### Final Crime Type Distribution (29 classes)

| Crime Type | Count |
|-----------|-------|
| THEFT | 1,544,341 |
| BATTERY | 1,344,711 |
| CRIMINAL DAMAGE | 836,259 |
| NARCOTICS | 727,970 |
| ASSAULT | 469,409 |
| OTHER OFFENSE | 455,543 |
| BURGLARY | 409,543 |
| MOTOR VEHICLE THEFT | 345,906 |
| DECEPTIVE PRACTICE | 296,183 |
| ROBBERY | 277,027 |
| *(19 more classes)* | *...* |

### Output Files
- `data/processed/chicago_crime_cleaned_v1.0.csv` — 7,282,602 rows × 22 columns (1,623.3 MB)
- `data/catalog/etl_report.json` — all 11 steps with rows_before, rows_after, notes
- `data/catalog/crime_type_merges.json` — rare types merged into OTHER
- `logs/etl.log` — full step-by-step log

---

## 6. Task 3 — Cataloging & Governance

### Data Zones

| Zone | Path | Access Rule | Description |
|------|------|-------------|-------------|
| Raw | `data/raw/` | READ ONLY | Original files, never modified after ingestion |
| Processed | `data/processed/` | Pipeline scripts only | Cleaned, typed, validated |
| Catalog | `data/catalog/` | Pipeline scripts only | JSON metadata files |
| Logs | `logs/` | Append only | Execution logs per script |

### Column Schema (Post-ETL)

| Column | Type | Nullable | Role | Notes |
|--------|------|----------|------|-------|
| ID | int64 | No | Identifier | Not used as feature |
| Case Number | string | No | Identifier | Deduplicated in ETL Step 2 |
| Date | datetime64 | No | Feature Source | Source of all temporal features |
| Block | string | → UNKNOWN | Descriptive | Not used as feature |
| IUCR | string | → UNKNOWN | Descriptive | Illinois crime code |
| **Primary Type** | string | No | **TARGET** | 29 classes, UPPERCASE |
| Description | string | → UNKNOWN | Descriptive | Subcategory of Primary Type |
| Location Description | string | → UNKNOWN | Feature Candidate | 5,873 nulls filled |
| Arrest | bool | No | **EXCLUDED** | Post-incident — target leakage risk |
| Domestic | bool | No | Feature Candidate | Illinois Domestic Violence Act flag |
| Beat | int64 | → 0 | Feature Candidate | Smallest police patrol unit |
| District | int64 | → 0 | Feature Candidate | Police district (1–25) |
| Ward | int64 | → 0 | Feature Candidate | City ward (1–50), 8.32% null |
| Community Area | int64 | → 0 | Feature Candidate | Neighbourhood (1–77), 8.30% null |
| FBI Code | string | → UNKNOWN | Descriptive | Federal crime classification |
| X Coordinate | float64 | Yes | Descriptive | State Plane feet — not used |
| Y Coordinate | float64 | Yes | Descriptive | State Plane feet — not used |
| Year | int64 | No | Feature Candidate | Range: 2001–2024 |
| Updated On | datetime64 | Yes | Metadata | Last record update |
| **Latitude** | float64 | No | **Feature** | WGS84, range 41.64459–42.02291 |
| **Longitude** | float64 | No | **Feature** | WGS84, range –87.84190–87.52453 |
| Location | string | Yes | Descriptive | Combined (lat, lon) string |

### Output Files
- `data/catalog/data_catalog.json` — full annotated schema for all 22 columns
- `data/catalog/lineage.json` — 6-stage data lineage from source to feature matrix
- `data/catalog/assumptions.json` — 14 documented assumptions across 5 categories
- `data/catalog/zone_registry.json` — filesystem scan of all zones with file sizes
- `logs/catalog.log` — full run log

---

## 7. Task 4 — Exploratory Data Analysis

### Objective
Assess **data readiness** — evaluate distributions, detect risks, and justify ETL and feature engineering decisions. All plots are saved to `outputs/eda/`.

### Analyses Performed

**1. Crime Type Distribution (Target Variable)**
- THEFT dominates at 1,544,341 records (21.2% of cleaned data)
- Class imbalance present: largest class (THEFT) is ~2,471× larger than smallest (OTHER at 625)
- Imbalance must be addressed in Phase 2 via class weighting or SMOTE

**2. Temporal Pattern Analysis**
- Hourly: crime peaks between 18:00–22:00; lowest between 04:00–06:00
- Day of week: Friday and Saturday show elevated crime frequency
- Monthly: crime is higher in summer months (June–August)
- Yearly trend: declining from peak ~485k (2001–2002) to ~89k (2023), with anomalous dip in 2021–2022

**3. Spatial Distribution**
- Crimes cluster heavily in the South Side and West Side of Chicago
- Spatial clustering varies by crime type — NARCOTICS concentrated in specific neighbourhoods, THEFT more city-wide
- Motivates use of Community Area and Beat as geospatial features

**4. Correlation Analysis**
- Low correlation between temporal features and spatial features — confirms independence
- `hour` and `is_night` are correlated by design (is_night derived from hour) — only one retained
- `Latitude` and `Longitude` show near-zero correlation with each other

**5. Coordinate Outlier Check**
- Post-ETL box plots confirm no remaining coordinate outliers
- Longitude range tight: –87.84 to –87.52 (all valid Chicago bounds)

### Data Readiness Verdict
The dataset is **ready for feature engineering and modelling** with the following notes:
- Class imbalance requires handling in Phase 2 (class weights recommended)
- Ward and Community Area `0` values (from null-fill) should be treated as a separate category, not a numeric zero
- Temporal patterns strongly support hour-of-day and day-of-week as predictive features
- Spatial clustering supports Latitude, Longitude, and Community Area as features

### Output Files
- `outputs/eda/crime_type_distribution.png`
- `outputs/eda/temporal_analysis.png`
- `outputs/eda/spatial_distribution.png`
- `outputs/eda/correlation_heatmap.png`
- `outputs/eda/coordinate_boxplot.png`
- `logs/eda.log`

---

## 8. Task 5 — Feature Extraction & Selection

### Feature Engineering

**Temporal Features** (extracted from `Date` column)

| Feature | Type | Description | Justification |
|---------|------|-------------|---------------|
| `hour` | int (0–23) | Hour of crime occurrence | Crime type varies strongly by time of day |
| `day_of_week` | int (0–6) | Day of week (0=Monday) | Weekend vs weekday crime patterns differ |
| `month` | int (1–12) | Month of year | Seasonal crime patterns (summer vs winter) |
| `is_weekend` | bool | 1 if Saturday or Sunday | Weekend binary flag for classifier |
| `is_night` | bool | 1 if hour 22:00–05:59 | Night-time crimes have different type profiles |
| `hour_sin` | float | sin(2π × hour / 24) | Cyclical encoding — preserves midnight continuity |
| `hour_cos` | float | cos(2π × hour / 24) | Cyclical encoding — complements hour_sin |
| `is_holiday` | bool | 1 if US public holiday | Holiday periods show distinct crime patterns |

**Geospatial Features**

| Feature | Type | Description | Justification |
|---------|------|-------------|---------------|
| `Latitude` | float64 | WGS84 latitude | Core spatial predictor — crime type clusters geographically |
| `Longitude` | float64 | WGS84 longitude | Core spatial predictor |
| `community_area_enc` | int | Encoded Community Area (1–77) | Neighbourhood-level socioeconomic context |
| `is_crowded` | bool | High-density area flag | Dense areas have different crime type profiles |

**Target Encoding**

| Feature | Type | Description |
|---------|------|-------------|
| `crime_type_label` | int | LabelEncoded Primary Type (0–28) |

Label mapping saved to `data/catalog/label_encoding.json`.

### Feature Selection — Mutual Information

Mutual information (MI) was computed between each feature and `crime_type_label`. Features with MI score > 0.01 were retained.

**Features RETAINED after MI selection:**

| Feature | MI Score | Category |
|---------|----------|----------|
| `hour` | highest | Temporal |
| `hour_sin` | high | Temporal (cyclical) |
| `hour_cos` | high | Temporal (cyclical) |
| `is_weekend` | medium | Temporal |
| `is_night` | medium | Temporal |
| `Latitude` | high | Spatial |
| `Longitude` | high | Spatial |
| `community_area_enc` | medium | Spatial |
| `is_crowded` | medium | Spatial |

**Total features selected: 9**

### Why `Arrest` is Excluded
`Arrest` records whether an arrest was made **after** the crime. Including it would constitute **target leakage** — the model would learn from post-incident information unavailable at prediction time (start of shift).

### Output Files
- `data/processed/chicago_crime_features_v1.0.csv` — 7,282,602 rows × 10 columns (9 features + label)
- `data/catalog/label_encoding.json` — crime type → integer mapping (29 classes)
- `logs/features.log` — full run log

---

## 9. Data Lineage

```
[1] Kaggle Source
    URL: kaggle.com/datasets/middlehigh/los-angeles-crime-data-from-2000
    Format: CSV  |  Size: 1,662.9 MB  |  Rows: 7,391,187
        |
        v  src/ingestion.py
[2] Raw Zone
    data/raw/chicago_crime_v1.0_2026-03-16.csv
    MD5: 3cfac7dd7d8eb2d47b9665ddbe56e486
        |
        v  src/etl.py  (11 steps, 108,585 rows dropped)
[3] Processed Zone
    data/processed/chicago_crime_cleaned_v1.0.csv
    Rows: 7,282,602  |  Retention: 98.53%
        |
        v  src/catalog.py
[4] Catalog Zone
    data/catalog/data_catalog.json + lineage.json + assumptions.json
        |
        v  src/eda.py
[5] EDA Outputs
    outputs/eda/ (5 plots)
        |
        v  src/features.py
[6] Feature Matrix
    data/processed/chicago_crime_features_v1.0.csv
    Rows: 7,282,602  |  Features: 9  |  Target classes: 29
```

---

## 10. Assumptions

### Data Quality
| ID | Assumption | Impact |
|----|-----------|--------|
| DQ-01 | Rows missing Latitude or Longitude are dropped — cannot be geolocated | 74,115 rows removed |
| DQ-02 | Rows missing Primary Type are dropped — target variable cannot be imputed | Included in 74,115 |
| DQ-03 | Non-critical text nulls filled with `UNKNOWN` to preserve rows | 5,873 Location Description nulls filled |
| DQ-04 | Ward (8.32%) and Community Area (8.30%) nulls filled with `0` — models must treat `0` as missing-indicator | ~614k rows retain `0` |

### Geospatial
| ID | Assumption | Impact |
|----|-----------|--------|
| GEO-01 | Coordinates outside Chicago bounding box are data entry errors | 147 rows removed |
| GEO-02 | Longitude IQR outliers [–87.842, –87.500] are anomalous clusters, removed | 33,758 rows removed |
| GEO-03 | Latitude and Longitude (WGS84) used as spatial features; X/Y Coordinate (State Plane) excluded | X/Y retained in file but not in feature matrix |

### Temporal
| ID | Assumption | Impact |
|----|-----------|--------|
| TEMP-01 | All timestamps are Chicago local time (CT) — no timezone conversion applied | Hour features may be off ±1 hr during DST transitions |
| TEMP-02 | Records before 2001-01-01 or after today are data errors | 0 rows removed — all records valid |

### Target Variable
| ID | Assumption | Impact |
|----|-----------|--------|
| TGT-01 | Primary Type standardised to UPPERCASE; rare types (< 500) merged into OTHER | 8 types merged; 29 final classes |
| TGT-02 | DOMESTIC VIOLENCE fell below threshold and was merged into OTHER | Records labeled OTHER; threshold can be lowered to 100 to restore this class |

### Modelling
| ID | Assumption | Impact |
|----|-----------|--------|
| MDL-01 | `Arrest` is EXCLUDED — post-incident field, would cause target leakage | Retained in processed file but never passed to classifier |
| MDL-02 | Case Number dedup keeps first occurrence — updated records reflect admin changes, not new crimes | 565 updated records removed |
| MDL-03 | Time-series cross-validation with temporal splits used in Phase 2 | Train: 2001–2020 / Val: 2021–2022 / Test: 2023–2024 |

### Ethics & Fairness
| ID | Assumption | Impact |
|----|-----------|--------|
| ETH-01 | Model predicts crime TYPE, not individuals — no demographic data used | Reduces direct individual discrimination risk |
| ETH-02 | Geographic features may encode historical policing bias — over-policed areas show more recorded crime | Model predictions reflect recorded patterns, not true crime rates. This limitation must be stated in deployment documentation |

---

## 11. Azure Deployment

### Infrastructure

| Resource | Name | Tier / SKU |
|----------|------|-----------|
| Resource Group | `dsai3202-phase1` | — |
| Storage Account | `cloudproject60107174` | Standard LRS — lowest cost tier |
| Blob Container — Raw | `raw-data` | Private |
| Blob Container — Processed | `processed-data` | Private |
| Blob Container — Catalog | `catalog-data` | Private |
| ML Compute Instance | `phase1-compute` | Standard_DS1_v2 — stopped when idle |

### Deployment Scripts

| Script | Purpose |
|--------|---------|
| `src/azure_upload.py` | Uploads all pipeline outputs from local machine to Blob Storage |
| `src/azure_validate.py` | Runs on Azure ML Compute — reads all files from Blob Storage and validates integrity |

### Cost Optimisation
- **LRS redundancy** — no geo-replication needed for student project data
- **Compute stopped when idle** — Standard_DS1_v2 billed by the hour, stopped immediately after validation
- **Sample-based validation** — `azure_validate.py` reads only 50,000 rows instead of the full 1.6 GB file, minimising egress costs

### Screenshots

**1 — Validation Result (Azure ML Terminal)**

![Azure Validation Result](screenshots/azure_validation_result.png)

---

**2 — Raw Data Container**

![Raw Data Container](screenshots/azure_raw_container.png)

---

**3 — Processed Data Container**

![Processed Data Container](screenshots/azure_processed_container.png)

---

**4 — Catalog Container**

![Catalog Container](screenshots/azure_catalog_container.png)

---

**5 — Compute Instance**

![Compute Instance](screenshots/azure_compute_instance.png)

---

## Contributions
* **Elaf Marouf (60107174)**: Lead on Data Ingestion, ETL pipeline development and Deployment. Implemented missing value handling and data validation logic. 
* **Nur Afiqah (60306981)**: Focused on Exploratory Analysis and Feature Engineering & Selection. Created visualisations and developed cyclical time features and performed Mutual Information analysis.
* **Shared Responsibilities**: Documentation, GitHub repository management

*DSAI3202 — Winter 2026 | Phase 1 | Nur Afiqah (60306981) & Elaf Marouf (60107174)*
