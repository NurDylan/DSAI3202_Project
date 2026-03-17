"""
=============================================================
DSAI3202 - Phase 1  |  Task 3: Data Cataloging & Governance
Project: Predicting Crime Types by Location & Time-Series Data
Authors: Nur Afiqah (60306981), Elaf Marouf (60107174)
=============================================================

PURPOSE:
    Register the dataset schema, data types, lineage, and
    assumptions in a structured data catalog. Reflects the
    raw and processed zone separation established in Tasks 1-2.

OUTPUTS:
    data/catalog/data_catalog.json       <- full schema catalog
    data/catalog/lineage.json            <- data lineage record
    data/catalog/assumptions.json        <- all project assumptions
    data/catalog/zone_registry.json      <- raw / processed zones
    logs/catalog.log                     <- catalog run log
"""

import os
import sys
import json
import logging
import datetime
import glob
import pandas as pd

# ---------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------

DATASET_VERSION = "v1.0"
RAW_DIR         = os.path.join("data", "raw")
PROCESSED_DIR   = os.path.join("data", "processed")
CATALOG_DIR     = os.path.join("data", "catalog")
LOG_DIR         = "logs"

CLEANED_FILENAME = f"chicago_crime_cleaned_{DATASET_VERSION}.csv"
TODAY            = datetime.date.today().isoformat()

# ---------------------------------------------------------------
# SETUP - directories & logging
# ---------------------------------------------------------------

for d in [CATALOG_DIR, LOG_DIR]:
    os.makedirs(d, exist_ok=True)

LOG_FILE = os.path.join(LOG_DIR, "catalog.log")

# Windows UTF-8 fix (same as ingestion/etl)
if sys.platform == "win32":
    _stream = open(sys.stdout.fileno(), mode="w", encoding="utf-8", buffering=1)
else:
    _stream = sys.stdout

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(stream=_stream),
    ]
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------
# STEP 1 - LOAD CLEANED DATASET FOR SCHEMA INFERENCE
# ---------------------------------------------------------------

def load_cleaned_data() -> pd.DataFrame:
    """
    Load the processed dataset produced by etl.py.
    Used to infer actual dtypes and compute per-column stats.
    """
    path = os.path.join(PROCESSED_DIR, CLEANED_FILENAME)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Cleaned dataset not found: {path}\n"
            "Run etl.py first before running catalog.py."
        )
    log.info(f"Loading cleaned dataset: {path}")
    # Load a sample for fast stat computation (full file is ~1.6 GB)
    df = pd.read_csv(path, low_memory=False, nrows=100_000)
    log.info(f"Loaded {len(df):,} rows (sample) for schema inference.")
    return df


# ---------------------------------------------------------------
# STEP 2 - BUILD SCHEMA CATALOG
# ---------------------------------------------------------------

# Full column definitions - hand-annotated with domain knowledge
# aligned to what ETL actually produced (confirmed from ETL log)
COLUMN_DEFINITIONS = [
    {
        "name"        : "ID",
        "dtype"       : "int64",
        "nullable"    : False,
        "zone"        : "both",
        "role"        : "identifier",
        "description" : "Unique numeric identifier for each crime record assigned by Chicago PD.",
        "notes"       : "Not used as a model feature."
    },
    {
        "name"        : "Case Number",
        "dtype"       : "string",
        "nullable"    : False,
        "zone"        : "both",
        "role"        : "identifier",
        "description" : "Chicago Police Department case number (e.g. JG423596).",
        "notes"       : "Deduplicated in ETL Step 2 - first occurrence kept."
    },
    {
        "name"        : "Date",
        "dtype"       : "datetime64[ns]",
        "nullable"    : False,
        "zone"        : "processed",
        "role"        : "feature_source",
        "description" : "Timestamp of crime occurrence. Raw format: MM/DD/YYYY HH:MM:SS AM/PM (Chicago CT).",
        "notes"       : "Parsed to datetime64 in ETL Step 5. Source of all temporal features in Task 5."
    },
    {
        "name"        : "Block",
        "dtype"       : "string",
        "nullable"    : True,
        "zone"        : "both",
        "role"        : "descriptive",
        "description" : "Anonymized block-level address of the crime (e.g. 073XX S JEFFERY AVE).",
        "notes"       : "Nulls filled with UNKNOWN in ETL Step 4. Not used as a model feature."
    },
    {
        "name"        : "IUCR",
        "dtype"       : "string",
        "nullable"    : True,
        "zone"        : "both",
        "role"        : "descriptive",
        "description" : "Illinois Uniform Crime Reporting code - 4-character crime classification code.",
        "notes"       : "Nulls filled with UNKNOWN in ETL Step 4."
    },
    {
        "name"        : "Primary Type",
        "dtype"       : "string",
        "nullable"    : False,
        "zone"        : "both",
        "role"        : "target",
        "description" : "High-level crime category (e.g. THEFT, BATTERY, ASSAULT). This is the classification TARGET variable.",
        "notes"       : (
            "Standardized to UPPERCASE in ETL Step 5. "
            "8 rare types (< 500 occurrences) merged into OTHER in ETL Step 8: "
            "PUBLIC INDECENCY, NON-CRIMINAL, OTHER NARCOTIC VIOLATION, "
            "HUMAN TRAFFICKING, NON - CRIMINAL, RITUALISM, "
            "NON-CRIMINAL (SUBJECT SPECIFIED), DOMESTIC VIOLENCE. "
            "Final unique count: 29 classes."
        )
    },
    {
        "name"        : "Description",
        "dtype"       : "string",
        "nullable"    : True,
        "zone"        : "both",
        "role"        : "descriptive",
        "description" : "Subcategory description of the crime under Primary Type (e.g. SIMPLE, AGGRAVATED).",
        "notes"       : "Nulls filled with UNKNOWN in ETL Step 4."
    },
    {
        "name"        : "Location Description",
        "dtype"       : "string",
        "nullable"    : True,
        "zone"        : "both",
        "role"        : "descriptive",
        "description" : "Type of location where crime occurred (e.g. STREET, RESIDENCE, APARTMENT).",
        "notes"       : "5,873 nulls filled with UNKNOWN in ETL Step 4. Candidate feature in Task 5."
    },
    {
        "name"        : "Arrest",
        "dtype"       : "bool",
        "nullable"    : False,
        "zone"        : "both",
        "role"        : "excluded",
        "description" : "True if an arrest was made in connection with this crime.",
        "notes"       : (
            "Cast to bool in ETL Step 5. "
            "EXCLUDED from model features - post-incident field that would "
            "introduce target leakage into the classifier."
        )
    },
    {
        "name"        : "Domestic",
        "dtype"       : "bool",
        "nullable"    : False,
        "zone"        : "both",
        "role"        : "feature_candidate",
        "description" : "True if the crime was classified as a domestic incident under the Illinois Domestic Violence Act.",
        "notes"       : "Cast to bool in ETL Step 5. Candidate feature in Task 5."
    },
    {
        "name"        : "Beat",
        "dtype"       : "int64",
        "nullable"    : False,
        "zone"        : "both",
        "role"        : "feature_candidate",
        "description" : "Chicago Police Department patrol beat - the smallest geographic unit of policing.",
        "notes"       : "Nulls filled with 0 in ETL Step 4. Candidate geospatial feature in Task 5."
    },
    {
        "name"        : "District",
        "dtype"       : "int64",
        "nullable"    : False,
        "zone"        : "both",
        "role"        : "feature_candidate",
        "description" : "Chicago Police Department district number (1-25) grouping multiple beats.",
        "notes"       : "48 nulls filled with 0 in ETL Step 4."
    },
    {
        "name"        : "Ward",
        "dtype"       : "int64",
        "nullable"    : False,
        "zone"        : "both",
        "role"        : "feature_candidate",
        "description" : "Chicago City Council ward number (1-50). Administrative/political subdivision.",
        "notes"       : "614,820 nulls (8.32%) filled with 0 in ETL Step 4. High null rate documented."
    },
    {
        "name"        : "Community Area",
        "dtype"       : "int64",
        "nullable"    : False,
        "zone"        : "both",
        "role"        : "feature_candidate",
        "description" : "Chicago community area number (1-77). Stable socioeconomic neighbourhood boundaries.",
        "notes"       : (
            "613,469 nulls (8.30%) filled with 0 in ETL Step 4. "
            "Key geospatial feature candidate - encodes neighbourhood context."
        )
    },
    {
        "name"        : "FBI Code",
        "dtype"       : "string",
        "nullable"    : True,
        "zone"        : "both",
        "role"        : "descriptive",
        "description" : "FBI Uniform Crime Reporting classification code for the offense.",
        "notes"       : "Nulls filled with UNKNOWN in ETL Step 4."
    },
    {
        "name"        : "X Coordinate",
        "dtype"       : "float64",
        "nullable"    : True,
        "zone"        : "both",
        "role"        : "descriptive",
        "description" : "Illinois State Plane East coordinate in feet (NAD 1927 projection).",
        "notes"       : "74,115 nulls (1.00%). Not used as feature - Latitude/Longitude used instead."
    },
    {
        "name"        : "Y Coordinate",
        "dtype"       : "float64",
        "nullable"    : True,
        "zone"        : "both",
        "role"        : "descriptive",
        "description" : "Illinois State Plane North coordinate in feet (NAD 1927 projection).",
        "notes"       : "74,115 nulls (1.00%). Not used as feature - Latitude/Longitude used instead."
    },
    {
        "name"        : "Year",
        "dtype"       : "int64",
        "nullable"    : False,
        "zone"        : "both",
        "role"        : "feature_candidate",
        "description" : "Year the crime occurred (2001-2024). Extracted from Date by Chicago PD system.",
        "notes"       : "Redundant with Date column but pre-extracted. Range confirmed: 2001-2024."
    },
    {
        "name"        : "Updated On",
        "dtype"       : "datetime64[ns]",
        "nullable"    : True,
        "zone"        : "both",
        "role"        : "metadata",
        "description" : "Timestamp of the last update to this record in the Chicago data portal.",
        "notes"       : "Not used as a feature. Parsed to datetime64 in ETL Step 5."
    },
    {
        "name"        : "Latitude",
        "dtype"       : "float64",
        "nullable"    : False,
        "zone"        : "both",
        "role"        : "feature",
        "description" : "WGS84 decimal latitude of crime location. Valid range after ETL: 41.64459 to 42.02291.",
        "notes"       : (
            "74,115 rows dropped in ETL Step 3 (missing). "
            "Bounding box applied: [41.60, 42.05]. "
            "IQR filter applied: [41.56198, 42.11372] - 0 additional removed. "
            "Core spatial feature for prediction."
        )
    },
    {
        "name"        : "Longitude",
        "dtype"       : "float64",
        "nullable"    : False,
        "zone"        : "both",
        "role"        : "feature",
        "description" : "WGS84 decimal longitude of crime location. Valid range after ETL: -87.84190 to -87.52453.",
        "notes"       : (
            "IQR filter applied: [-87.84190, -87.50010] - 33,758 outliers removed in ETL Step 9. "
            "Core spatial feature for prediction."
        )
    },
    {
        "name"        : "Location",
        "dtype"       : "string",
        "nullable"    : True,
        "zone"        : "both",
        "role"        : "descriptive",
        "description" : "Combined string representation of coordinates in format (latitude, longitude).",
        "notes"       : "74,115 nulls (1.00%). Redundant with Latitude/Longitude columns - not used as feature."
    },
]


def build_schema_catalog(df: pd.DataFrame) -> dict:
    """
    Build the full data catalog JSON combining hand-annotated
    definitions with auto-inferred statistics from the cleaned data.
    """
    log.info("Building schema catalog...")

    enriched_columns = []
    for col_def in COLUMN_DEFINITIONS:
        col = col_def["name"]
        entry = dict(col_def)

        # Auto-enrich with stats if column exists in sample
        if col in df.columns:
            entry["null_count_sample"]  = int(df[col].isnull().sum())
            entry["unique_count_sample"] = int(df[col].nunique())
            entry["inferred_dtype"]     = str(df[col].dtype)

            # For numeric columns add min/max from sample
            if pd.api.types.is_numeric_dtype(df[col]):
                entry["sample_min"] = float(df[col].min()) if not df[col].isnull().all() else None
                entry["sample_max"] = float(df[col].max()) if not df[col].isnull().all() else None

        enriched_columns.append(entry)
        log.info(f"  Cataloged column: {col} ({col_def['dtype']}) [{col_def['role']}]")

    catalog = {
        "catalog_version"   : "1.0",
        "created_date"      : TODAY,
        "dataset_name"      : "Chicago Crime Dataset",
        "dataset_version"   : DATASET_VERSION,
        "source_url"        : "https://www.kaggle.com/datasets/middlehigh/los-angeles-crime-data-from-2000/data",
        "project"           : "DSAI3202 - Predicting Crime Types by Location and Time-Series Data",
        "authors"           : ["Nur Afiqah - 60306981", "Elaf Marouf - 60107174"],
        "raw_row_count"     : 7_391_187,
        "cleaned_row_count" : 7_282_602,
        "column_count"      : 22,
        "target_variable"   : "Primary Type",
        "target_classes"    : 29,
        "date_range"        : {"start": "2001-01-01", "end": "2024-05-20"},
        "column_roles"      : {
            "target"           : ["Primary Type"],
            "features"         : ["Latitude", "Longitude"],
            "feature_candidates": ["Date", "Community Area", "Beat", "District",
                                   "Ward", "Domestic", "Location Description", "Year"],
            "excluded"         : ["Arrest"],
            "identifiers"      : ["ID", "Case Number"],
            "descriptive"      : ["Block", "IUCR", "Description", "FBI Code",
                                  "X Coordinate", "Y Coordinate", "Location"],
            "metadata"         : ["Updated On"],
        },
        "columns"           : enriched_columns,
    }

    return catalog


def save_schema_catalog(catalog: dict):
    out = os.path.join(CATALOG_DIR, "data_catalog.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(catalog, f, indent=2, ensure_ascii=False)
    log.info(f"Schema catalog saved -> {out}")
    return out


# ---------------------------------------------------------------
# STEP 3 - BUILD DATA LINEAGE
# ---------------------------------------------------------------

def build_lineage() -> dict:
    """
    Document the complete data lineage from source to feature matrix.
    Describes every transformation that data passes through.
    """
    log.info("Building data lineage record...")

    lineage = {
        "lineage_version" : "1.0",
        "created_date"    : TODAY,
        "description"     : (
            "End-to-end data lineage for the Chicago Crime "
            "Classification Pipeline - DSAI3202 Phase 1."
        ),
        "stages": [
            {
                "stage"       : 1,
                "name"        : "Source",
                "script"      : None,
                "input"       : "Kaggle dataset (external)",
                "output"      : None,
                "description" : (
                    "USA Big City Crime Dataset downloaded from Kaggle. "
                    "1.74 GB CSV, 7,391,187 rows, 22 columns. "
                    "URL: https://www.kaggle.com/datasets/middlehigh/"
                    "los-angeles-crime-data-from-2000/data"
                ),
                "row_count"   : 7_391_187,
            },
            {
                "stage"       : 2,
                "name"        : "Raw Ingestion",
                "script"      : "src/ingestion.py",
                "input"       : "crime_data.csv (project root)",
                "output"      : "data/raw/chicago_crime_v1.0_2026-03-16.csv",
                "description" : (
                    "Batch ingestion: file copied to versioned raw zone. "
                    "MD5 checksum computed (3cfac7dd7d8eb2d47b9665ddbe56e486). "
                    "Schema validated - all 22 columns present. "
                    "Ingestion manifest written to data/catalog/ingestion_manifest.json."
                ),
                "row_count"   : 7_391_187,
                "md5_checksum": "3cfac7dd7d8eb2d47b9665ddbe56e486",
            },
            {
                "stage"       : 3,
                "name"        : "ETL - Clean & Transform",
                "script"      : "src/etl.py",
                "input"       : "data/raw/chicago_crime_v1.0_2026-03-16.csv",
                "output"      : "data/processed/chicago_crime_cleaned_v1.0.csv",
                "description" : (
                    "11-step ETL pipeline. Key operations: "
                    "565 Case Number duplicates removed; "
                    "74,115 rows dropped (missing Lat/Lon/Date/Primary Type); "
                    "5,873 Location Description nulls filled with UNKNOWN; "
                    "Date parsed to datetime64 (format MM/DD/YYYY HH:MM:SS AM/PM); "
                    "147 out-of-bounds coordinates removed (Chicago bounding box); "
                    "8 rare crime types merged into OTHER (threshold=500); "
                    "33,758 Longitude outliers removed (IQR 1.5x). "
                    "Final retention: 98.53%."
                ),
                "row_count"        : 7_282_602,
                "rows_dropped"     : 108_585,
                "retention_rate_%"  : 98.53,
            },
            {
                "stage"       : 4,
                "name"        : "Data Cataloging",
                "script"      : "src/catalog.py",
                "input"       : "data/processed/chicago_crime_cleaned_v1.0.csv",
                "output"      : "data/catalog/data_catalog.json",
                "description" : (
                    "Schema registration, lineage documentation, "
                    "and assumption recording. No row-level transformations."
                ),
                "row_count"   : 7_282_602,
            },
            {
                "stage"       : 5,
                "name"        : "Exploratory Data Analysis",
                "script"      : "src/eda.py",
                "input"       : "data/processed/chicago_crime_cleaned_v1.0.csv",
                "output"      : "outputs/eda/ (plots)",
                "description" : "EDA plots and data readiness assessment. No transformations.",
                "row_count"   : 7_282_602,
            },
            {
                "stage"       : 6,
                "name"        : "Feature Engineering",
                "script"      : "src/features.py",
                "input"       : "data/processed/chicago_crime_cleaned_v1.0.csv",
                "output"      : "data/processed/chicago_crime_features_v1.0.csv",
                "description" : (
                    "Temporal features (hour, day_of_week, month, is_weekend, is_night, "
                    "hour_sin, hour_cos), geospatial features, label encoding. "
                    "Mutual information feature selection applied."
                ),
                "row_count"   : 7_282_602,
            },
        ]
    }

    out = os.path.join(CATALOG_DIR, "lineage.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(lineage, f, indent=2, ensure_ascii=False)
    log.info(f"Lineage record saved -> {out}")
    return lineage


# ---------------------------------------------------------------
# STEP 4 - DOCUMENT ASSUMPTIONS
# ---------------------------------------------------------------

def build_assumptions() -> dict:
    """
    Record all assumptions made across the pipeline.
    Grouped by category for clarity.
    """
    log.info("Recording project assumptions...")

    assumptions = {
        "version"       : "1.0",
        "created_date"  : TODAY,
        "project"       : "DSAI3202 - Crime Type Prediction",
        "categories": {

            "data_quality": [
                {
                    "id"         : "DQ-01",
                    "assumption" : "Rows missing Latitude or Longitude cannot be geolocated and are dropped.",
                    "impact"     : "74,115 rows removed (1.00% of raw data).",
                    "etl_step"   : "Step 3 - drop_missing_critical"
                },
                {
                    "id"         : "DQ-02",
                    "assumption" : "Rows missing Primary Type are dropped - the target variable cannot be imputed.",
                    "impact"     : "Included in the 74,115 dropped rows. Fabricating labels would invalidate the classifier.",
                    "etl_step"   : "Step 3 - drop_missing_critical"
                },
                {
                    "id"         : "DQ-03",
                    "assumption" : "The single row missing Date is dropped - temporal features cannot be engineered without it.",
                    "impact"     : "1 row removed.",
                    "etl_step"   : "Step 3 - drop_missing_critical"
                },
                {
                    "id"         : "DQ-04",
                    "assumption" : "Non-critical text columns (Description, Location Description, Block, FBI Code, IUCR) with nulls are filled with UNKNOWN to preserve the row.",
                    "impact"     : "5,873 Location Description nulls filled. Row count unchanged.",
                    "etl_step"   : "Step 4 - fill_non_critical_nulls"
                },
                {
                    "id"         : "DQ-05",
                    "assumption" : "Ward (8.32% null) and Community Area (8.30% null) are filled with 0. These nulls are concentrated in older records (pre-2003) and are not random.",
                    "impact"     : "~614k rows retain value 0 for Ward/Community Area. Models must treat 0 as a missing-indicator category.",
                    "etl_step"   : "Step 4 - fill_non_critical_nulls"
                },
            ],

            "geospatial": [
                {
                    "id"         : "GEO-01",
                    "assumption" : "Any coordinate outside Chicago bounding box (Lat 41.60-42.05, Lon -88.00 to -87.50) is a data entry error.",
                    "impact"     : "147 rows removed in ETL Step 6.",
                    "etl_step"   : "Step 6 - validate_coordinates"
                },
                {
                    "id"         : "GEO-02",
                    "assumption" : "Longitude IQR outliers [-87.84190, -87.50010] represent anomalous coordinate clusters within Chicago bounds and are removed.",
                    "impact"     : "33,758 rows removed in ETL Step 9. Latitude had 0 outliers by IQR.",
                    "etl_step"   : "Step 9 - remove_coordinate_outliers"
                },
                {
                    "id"         : "GEO-03",
                    "assumption" : "Latitude and Longitude (WGS84) are used as spatial features. X/Y Coordinate (Illinois State Plane feet) are not used as features.",
                    "impact"     : "X/Y Coordinate columns retained in processed file but excluded from feature matrix.",
                    "etl_step"   : "Task 5 - feature engineering"
                },
            ],

            "temporal": [
                {
                    "id"         : "TEMP-01",
                    "assumption" : "All timestamps are in Chicago local time (CT, UTC-6 / UTC-5 DST). No timezone conversion applied.",
                    "impact"     : "Hour-of-day features may be off by 1 hour during DST transitions (minor).",
                    "etl_step"   : "Step 5 - fix_data_types"
                },
                {
                    "id"         : "TEMP-02",
                    "assumption" : "Records before 2001-01-01 or after today are data entry errors and are dropped.",
                    "impact"     : "0 rows removed - all records fell within 2001-2024.",
                    "etl_step"   : "Step 7 - validate_temporal_range"
                },
            ],

            "target_variable": [
                {
                    "id"         : "TGT-01",
                    "assumption" : "Primary Type is the classification target. It is standardized to UPPERCASE and rare types are merged.",
                    "impact"     : "8 rare types (< 500 occurrences) merged into OTHER. Final: 29 classes.",
                    "etl_step"   : "Step 8 - standardize_crime_types"
                },
                {
                    "id"         : "TGT-02",
                    "assumption" : "DOMESTIC VIOLENCE fell below the 500-row threshold and was merged into OTHER. While meaningful, it had insufficient samples for reliable classification.",
                    "impact"     : "DOMESTIC VIOLENCE records are now labeled OTHER. Threshold can be lowered to 100 to preserve this class if needed.",
                    "etl_step"   : "Step 8 - standardize_crime_types"
                },
            ],

            "modelling": [
                {
                    "id"         : "MDL-01",
                    "assumption" : "Arrest is EXCLUDED from model features. It is a post-incident field - knowing whether an arrest was made would constitute target leakage for a pre-incident prediction system.",
                    "impact"     : "Arrest column retained in processed file but must not be passed to the classifier.",
                    "etl_step"   : "Task 5 - feature engineering"
                },
                {
                    "id"         : "MDL-02",
                    "assumption" : "Case Number deduplication keeps the first occurrence. Updated records (e.g. arrest status changes) reflect administrative updates, not new crimes.",
                    "impact"     : "565 updated records removed. First-reported state of each crime is the prediction target.",
                    "etl_step"   : "Step 2 - deduplicate_case_number"
                },
                {
                    "id"         : "MDL-03",
                    "assumption" : "Time-series cross-validation with temporal splits will be used in Phase 2 to prevent future data leaking into training.",
                    "impact"     : "Train on years 2001-2020, validate on 2021-2022, test on 2023-2024 (subject to Phase 2 design).",
                    "etl_step"   : "Phase 2 - modelling"
                },
            ],

            "ethics_and_fairness": [
                {
                    "id"         : "ETH-01",
                    "assumption" : "The model predicts crime TYPE, not individuals. No demographic or personal data is used as a feature.",
                    "impact"     : "Reduces direct individual discrimination risk.",
                    "etl_step"   : "All phases"
                },
                {
                    "id"         : "ETH-02",
                    "assumption" : "Geographic features (Latitude, Longitude, Community Area) may encode historical policing bias. Over-policed areas appear to have more recorded crime.",
                    "impact"     : "Model predictions reflect recorded crime patterns, not necessarily true crime rates. This limitation must be stated in deployment documentation.",
                    "etl_step"   : "Task 5 - feature engineering"
                },
            ],
        }
    }

    out = os.path.join(CATALOG_DIR, "assumptions.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(assumptions, f, indent=2, ensure_ascii=False)
    log.info(f"Assumptions saved -> {out}")
    return assumptions


# ---------------------------------------------------------------
# STEP 5 - DOCUMENT DATA ZONES
# ---------------------------------------------------------------

def build_zone_registry() -> dict:
    """
    Register all data zones and the files that exist in each.
    Scans the actual filesystem to list current files.
    """
    log.info("Building zone registry...")

    def scan_zone(path):
        if not os.path.exists(path):
            return []
        files = []
        for f in os.listdir(path):
            full = os.path.join(path, f)
            if os.path.isfile(full):
                files.append({
                    "filename"  : f,
                    "size_mb"   : round(os.path.getsize(full) / (1024**2), 2),
                    "modified"  : datetime.datetime.fromtimestamp(
                        os.path.getmtime(full)).isoformat()
                })
        return files

    registry = {
        "version"      : "1.0",
        "created_date" : TODAY,
        "zones": {
            "raw": {
                "path"        : "data/raw/",
                "description" : (
                    "Original downloaded files. NEVER modified after ingestion. "
                    "Versioned by dataset name + semantic version + ingestion date."
                ),
                "access_rule" : "READ ONLY",
                "files"       : scan_zone(RAW_DIR),
            },
            "processed": {
                "path"        : "data/processed/",
                "description" : (
                    "ETL output files. Cleaned, typed, and validated. "
                    "Ready for EDA and feature engineering."
                ),
                "access_rule" : "READ/WRITE by pipeline scripts only",
                "files"       : scan_zone(PROCESSED_DIR),
            },
            "catalog": {
                "path"        : "data/catalog/",
                "description" : (
                    "Metadata files: schema catalog, ETL report, lineage, "
                    "assumptions, label encodings. JSON format."
                ),
                "access_rule" : "READ/WRITE by pipeline scripts only",
                "files"       : scan_zone(CATALOG_DIR),
            },
            "logs": {
                "path"        : "logs/",
                "description" : "Pipeline execution logs. One log file per script.",
                "access_rule" : "READ ONLY (append by pipeline scripts)",
                "files"       : scan_zone(LOG_DIR),
            },
        }
    }

    out = os.path.join(CATALOG_DIR, "zone_registry.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2, ensure_ascii=False)
    log.info(f"Zone registry saved -> {out}")
    return registry


# ---------------------------------------------------------------
# STEP 6 - PRINT SUMMARY
# ---------------------------------------------------------------

def print_catalog_summary(catalog: dict, lineage: dict, assumptions: dict):
    log.info("=" * 60)
    log.info("TASK 3 - CATALOG SUMMARY")
    log.info("=" * 60)
    log.info(f"  Columns cataloged     : {len(catalog['columns'])}")
    log.info(f"  Target variable       : {catalog['target_variable']} ({catalog['target_classes']} classes)")
    log.info(f"  Feature columns       : {len(catalog['column_roles']['features'])}")
    log.info(f"  Feature candidates    : {len(catalog['column_roles']['feature_candidates'])}")
    log.info(f"  Excluded columns      : {catalog['column_roles']['excluded']}")
    log.info(f"  Lineage stages        : {len(lineage['stages'])}")

    total_assumptions = sum(
        len(v) for v in assumptions["categories"].values()
    )
    log.info(f"  Total assumptions     : {total_assumptions}")
    log.info(f"  Assumption categories : {list(assumptions['categories'].keys())}")
    log.info("=" * 60)
    log.info("OUTPUT FILES:")
    log.info(f"  data/catalog/data_catalog.json")
    log.info(f"  data/catalog/lineage.json")
    log.info(f"  data/catalog/assumptions.json")
    log.info(f"  data/catalog/zone_registry.json")
    log.info(f"  logs/catalog.log")
    log.info("=" * 60)


# ---------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------

def run_catalog():
    log.info("=" * 60)
    log.info("TASK 3 - DATA CATALOGING & GOVERNANCE STARTED")
    log.info("=" * 60)

    # Step 1: Load sample of cleaned data for schema inference
    df = load_cleaned_data()

    # Step 2: Build and save schema catalog
    catalog = build_schema_catalog(df)
    save_schema_catalog(catalog)

    # Step 3: Build and save data lineage
    lineage = build_lineage()

    # Step 4: Document all assumptions
    assumptions = build_assumptions()

    # Step 5: Register data zones
    build_zone_registry()

    # Step 6: Print summary
    print_catalog_summary(catalog, lineage, assumptions)

    log.info("TASK 3 - CATALOGING COMPLETE")
    return catalog


if __name__ == "__main__":
    run_catalog()