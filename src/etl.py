"""
=============================================================
DSAI3202 – Phase 1  |  Task 2: ETL Pipeline
Project: Predicting Crime Types by Location & Time-Series Data
Authors: Nur Afiqah (60306981), Elaf Marouf (60107174)
=============================================================

PURPOSE:
    Clean, validate, and transform the raw Chicago Crime dataset
    into an analysis-ready form for EDA and feature engineering.

All transformations are:
    - Reproducible (deterministic, no random state without seed)
    - Documented  (each step logs before/after row counts)
    - Traceable   (transformation log saved to logs/etl.log)

INPUT  : data/raw/chicago_crime_v1.0_<date>.csv
OUTPUT : data/processed/chicago_crime_cleaned_v1.0.csv
         data/catalog/etl_report.json
"""

import os
import glob
import logging
import json
import datetime
import pandas as pd
import numpy as np

# ──────────────────────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────────────────────

DATASET_VERSION = "v1.0"
RAW_DIR         = os.path.join("data", "raw")
PROCESSED_DIR   = os.path.join("data", "processed")
CATALOG_DIR     = os.path.join("data", "catalog")
LOG_DIR         = "logs"

# Chicago geographic bounding box (WGS84)
# Any coordinate outside this box is a data error
LAT_MIN, LAT_MAX  =  41.60,  42.05
LON_MIN, LON_MAX  = -88.00, -87.50

# Crime types with fewer occurrences than this threshold
# will be merged into the "OTHER" category
RARE_TYPE_THRESHOLD = 500

# Columns critical for prediction — rows missing ANY of these are dropped
CRITICAL_COLUMNS = ["Date", "Primary Type", "Latitude", "Longitude"]

# Non-critical columns — missing values filled with "UNKNOWN"
FILL_UNKNOWN_COLUMNS = [
    "Description", "Location Description",
    "Block", "FBI Code", "IUCR"
]

# Output file name
CLEANED_FILENAME = f"chicago_crime_cleaned_{DATASET_VERSION}.csv"

# ──────────────────────────────────────────────────────────────
# SETUP
# ──────────────────────────────────────────────────────────────

for d in [PROCESSED_DIR, CATALOG_DIR, LOG_DIR]:
    os.makedirs(d, exist_ok=True)

TODAY    = datetime.date.today().isoformat()
LOG_FILE = os.path.join(LOG_DIR, "etl.log")

import sys

# Fix 1: Force UTF-8 on Windows console (prevents UnicodeEncodeError
# for special characters like arrows, checkmarks, box-drawing lines)
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

# ──────────────────────────────────────────────────────────────
# ETL REPORT  –  tracks every transformation
# ──────────────────────────────────────────────────────────────

etl_report = {
    "run_date"       : TODAY,
    "version"        : DATASET_VERSION,
    "transformations": []
}

def record_step(name: str, rows_before: int, rows_after: int, notes: str = ""):
    dropped = rows_before - rows_after
    entry = {
        "step"         : name,
        "rows_before"  : rows_before,
        "rows_after"   : rows_after,
        "rows_dropped" : dropped,
        "notes"        : notes
    }
    etl_report["transformations"].append(entry)
    log.info(
        f"[{name}]  {rows_before:,} -> {rows_after:,}  "
        f"(dropped {dropped:,} rows)  {notes}"
    )


# ──────────────────────────────────────────────────────────────
# STEP 0 — LOAD RAW DATA
# ──────────────────────────────────────────────────────────────

def load_raw_data() -> pd.DataFrame:
    """
    Find the most recent versioned raw file and load it.
    Falls back to any CSV in the raw directory.
    """
    pattern = os.path.join(RAW_DIR, f"chicago_crime_{DATASET_VERSION}_*.csv")
    matches = sorted(glob.glob(pattern))

    if not matches:
        # Fallback: look for any CSV in raw/
        matches = sorted(glob.glob(os.path.join(RAW_DIR, "*.csv")))

    if not matches:
        raise FileNotFoundError(
            f"No raw CSV found in {RAW_DIR}.\n"
            "Run ingestion.py first, or place your CSV in data/raw/"
        )

    raw_path = matches[-1]   # latest version
    log.info(f"Loading raw file: {raw_path}")

    df = pd.read_csv(raw_path, low_memory=False, on_bad_lines="warn")
    log.info(f"Loaded {df.shape[0]:,} rows × {df.shape[1]} columns")
    etl_report["input_file"]       = raw_path
    etl_report["rows_raw"]         = int(df.shape[0])
    etl_report["columns_raw"]      = int(df.shape[1])
    return df


# ──────────────────────────────────────────────────────────────
# STEP 1 — DROP EXACT DUPLICATES
# ──────────────────────────────────────────────────────────────

def drop_exact_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.drop_duplicates()
    record_step("1_drop_exact_duplicates", before, len(df),
                "Remove rows that are identical across all columns.")
    return df


# ──────────────────────────────────────────────────────────────
# STEP 2 — DEDUPLICATE BY CASE NUMBER
# ──────────────────────────────────────────────────────────────

def deduplicate_case_number(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only the first occurrence of each Case Number.
    Chicago crime records are updated over time (e.g., arrest status
    changes), so later duplicates may have updated fields — we keep
    the first entry for consistency and note this assumption.
    """
    before = len(df)
    if "Case Number" in df.columns:
        df = df.drop_duplicates(subset=["Case Number"], keep="first")
        record_step("2_dedup_case_number", before, len(df),
                    "Keep first occurrence per Case Number (crime records may be updated).")
    else:
        log.warning("'Case Number' column not found — skipping deduplication by case.")
    return df


# ──────────────────────────────────────────────────────────────
# STEP 3 — DROP ROWS MISSING CRITICAL COLUMNS
# ──────────────────────────────────────────────────────────────

def drop_missing_critical(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop any row missing Date, Primary Type, Latitude, or Longitude.
    These are the columns used directly in feature engineering.
    Imputing them would introduce bias or fabricate geospatial targets.
    """
    before = len(df)
    df = df.dropna(subset=CRITICAL_COLUMNS)
    record_step("3_drop_missing_critical", before, len(df),
                f"Drop rows with null in: {CRITICAL_COLUMNS}.")
    return df


# ──────────────────────────────────────────────────────────────
# STEP 4 — FILL NON-CRITICAL MISSING VALUES
# ──────────────────────────────────────────────────────────────

def fill_non_critical_nulls(df: pd.DataFrame) -> pd.DataFrame:
    """
    For columns that are not used as features (descriptive only),
    fill nulls with 'UNKNOWN' to preserve the row and avoid
    downstream KeyErrors.
    """
    before = len(df)
    for col in FILL_UNKNOWN_COLUMNS:
        if col in df.columns:
            null_count = df[col].isnull().sum()
            df[col] = df[col].fillna("UNKNOWN")
            if null_count > 0:
                log.info(f"  Filled {null_count:,} nulls in '{col}' with 'UNKNOWN'")

    # Fill numeric nulls in non-critical columns with 0
    for col in ["Beat", "District", "Ward", "Community Area"]:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    record_step("4_fill_non_critical_nulls", before, len(df),
                "Non-critical text cols → 'UNKNOWN'; numeric admin cols → 0.")
    return df


# ──────────────────────────────────────────────────────────────
# STEP 5 — FIX DATA TYPES
# ──────────────────────────────────────────────────────────────

def fix_data_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parse and cast all columns to their correct types.

    Date format in Chicago data: '%m/%d/%Y %I:%M:%S %p'
    Example: '01/15/2023 11:45:00 PM'
    """
    before = len(df)

    # ── 5a: Parse Date column ──────────────────────────────────
    log.info("  Parsing 'Date' column to datetime...")
    df["Date"] = pd.to_datetime(
        df["Date"],
        format="%m/%d/%Y %I:%M:%S %p",
        errors="coerce"          # unparseable → NaT
    )
    nat_count = df["Date"].isnull().sum()
    if nat_count > 0:
        log.warning(f"  {nat_count:,} rows failed datetime parsing → will be dropped in next step.")
        df = df.dropna(subset=["Date"])
        log.info(f"  Dropped {nat_count:,} rows with unparseable dates.")

    # ── 5b: Parse UpdatedOn column (if present) ───────────────
    if "Updated On" in df.columns:
        # Fix 2: explicit format silences UserWarning about per-element parsing
        df["Updated On"] = pd.to_datetime(df["Updated On"], format="%m/%d/%Y %I:%M:%S %p", errors="coerce")

    # ── 5c: Numeric columns ───────────────────────────────────
    for col in ["Latitude", "Longitude", "X Coordinate", "Y Coordinate"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in ["Beat", "District", "Ward", "Community Area", "Year"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    # ── 5d: Boolean columns ───────────────────────────────────
    for col in ["Arrest", "Domestic"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.upper().map(
                {"TRUE": True, "FALSE": False}
            )

    # ── 5e: String cleanup ────────────────────────────────────
    for col in ["Primary Type", "Description", "Location Description",
                "Block", "FBI Code", "IUCR"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.upper()

    record_step("5_fix_data_types", before, len(df),
                "Date → datetime64; Lat/Lon → float64; Arrest/Domestic → bool; strings standardized.")
    return df


# ──────────────────────────────────────────────────────────────
# STEP 6 — VALIDATE GEOSPATIAL COORDINATES
# ──────────────────────────────────────────────────────────────

def validate_coordinates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter to Chicago's geographic bounding box.
    Records outside this range are data entry errors or non-Chicago entries.

    Bounding box:
        Latitude  : 41.60 – 42.05
        Longitude : -88.00 – -87.50
    """
    before = len(df)

    # Drop rows where Lat/Lon became NaN after type coercion (Step 5)
    df = df.dropna(subset=["Latitude", "Longitude"])
    after_nan = len(df)
    if before - after_nan > 0:
        log.info(f"  Dropped {before - after_nan:,} rows with NaN coordinates after type coercion.")

    # Apply bounding box filter
    in_bounds = (
        df["Latitude"].between(LAT_MIN, LAT_MAX) &
        df["Longitude"].between(LON_MIN, LON_MAX)
    )
    df = df[in_bounds]

    record_step("6_validate_coordinates", before, len(df),
                f"Bounding box: Lat [{LAT_MIN}, {LAT_MAX}], Lon [{LON_MIN}, {LON_MAX}].")
    return df


# ──────────────────────────────────────────────────────────────
# STEP 7 — VALIDATE TEMPORAL RANGE
# ──────────────────────────────────────────────────────────────

def validate_temporal_range(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only records within the expected date range for Chicago data.
    Records before 2001 or after today are likely data entry errors.
    """
    before  = len(df)
    min_date = pd.Timestamp("2001-01-01")
    max_date = pd.Timestamp(TODAY)

    df = df[df["Date"].between(min_date, max_date)]
    record_step("7_validate_temporal_range", before, len(df),
                f"Date range: {min_date.date()} to {max_date.date()}.")
    return df


# ──────────────────────────────────────────────────────────────
# STEP 8 — STANDARDIZE CRIME TYPE LABELS
# ──────────────────────────────────────────────────────────────

def standardize_crime_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge rare crime types (< RARE_TYPE_THRESHOLD occurrences) into 'OTHER'.
    This prevents extreme class imbalance from causing issues in modelling.

    The 'Primary Type' column has already been uppercased in Step 5.
    """
    before = len(df)

    type_counts   = df["Primary Type"].value_counts()
    rare_types    = type_counts[type_counts < RARE_TYPE_THRESHOLD].index.tolist()

    if rare_types:
        log.info(f"  Merging {len(rare_types)} rare crime types into 'OTHER': {rare_types}")
        df["Primary Type"] = df["Primary Type"].replace(
            {t: "OTHER" for t in rare_types}
        )
    else:
        log.info(f"  No rare crime types below threshold {RARE_TYPE_THRESHOLD}.")

    log.info(f"  Final unique crime types: {df['Primary Type'].nunique()}")
    log.info(f"  Crime type distribution:\n{df['Primary Type'].value_counts().to_string()}")

    # Save the mapping to catalog
    mapping_path = os.path.join(CATALOG_DIR, "crime_type_merges.json")
    with open(mapping_path, "w") as f:
        json.dump({"merged_into_OTHER": rare_types, "threshold": RARE_TYPE_THRESHOLD}, f, indent=2)
    log.info(f"  Crime type merge mapping saved -> {mapping_path}")

    record_step("8_standardize_crime_types", before, len(df),
                f"Merged {len(rare_types)} rare types into 'OTHER' (threshold={RARE_TYPE_THRESHOLD}).")
    return df


# ──────────────────────────────────────────────────────────────
# STEP 9 — REMOVE COORDINATE OUTLIERS (IQR METHOD)
# ──────────────────────────────────────────────────────────────

def remove_coordinate_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply IQR-based outlier removal to Latitude and Longitude.
    After the bounding box filter, this catches any remaining
    clusters of anomalous coordinates within Chicago's bounds.
    """
    before = len(df)

    for col in ["Latitude", "Longitude"]:
        Q1  = df[col].quantile(0.25)
        Q3  = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lo  = Q1 - 1.5 * IQR
        hi  = Q3 + 1.5 * IQR
        out_count = (~df[col].between(lo, hi)).sum()
        df = df[df[col].between(lo, hi)]
        log.info(f"  IQR filter on '{col}': [{lo:.5f}, {hi:.5f}] - removed {out_count:,} outliers")

    record_step("9_remove_coordinate_outliers", before, len(df),
                "IQR (1.5×) outlier removal on Latitude and Longitude.")
    return df


# ──────────────────────────────────────────────────────────────
# STEP 10 — FINAL QUALITY CHECK
# ──────────────────────────────────────────────────────────────

def final_quality_check(df: pd.DataFrame) -> pd.DataFrame:
    """
    Confirm no nulls remain in critical columns.
    Log a final summary of the cleaned dataset.
    """
    log.info("-" * 50)
    log.info("FINAL QUALITY CHECK")
    log.info(f"  Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")

    critical_nulls = df[CRITICAL_COLUMNS].isnull().sum()
    if critical_nulls.sum() > 0:
        log.error(f"  CRITICAL NULLS REMAINING:\n{critical_nulls[critical_nulls > 0]}")
    else:
        log.info("  [OK] No nulls in critical columns.")

    log.info(f"  Date range: {df['Date'].min()} to {df['Date'].max()}")
    log.info(f"  Latitude range: {df['Latitude'].min():.5f} to {df['Latitude'].max():.5f}")
    log.info(f"  Longitude range: {df['Longitude'].min():.5f} to {df['Longitude'].max():.5f}")
    log.info(f"  Unique crime types: {df['Primary Type'].nunique()}")
    log.info(f"  Arrest rate: {df['Arrest'].mean():.1%}")
    log.info("-" * 50)

    return df


# ──────────────────────────────────────────────────────────────
# STEP 11 — SAVE CLEANED DATASET
# ──────────────────────────────────────────────────────────────

def save_cleaned_dataset(df: pd.DataFrame) -> str:
    """Save the cleaned DataFrame to the processed zone."""
    out_path = os.path.join(PROCESSED_DIR, CLEANED_FILENAME)
    df.to_csv(out_path, index=False)
    size_mb = os.path.getsize(out_path) / (1024 ** 2)
    log.info(f"Cleaned dataset saved -> {out_path}  ({size_mb:.1f} MB)")
    return out_path


# ──────────────────────────────────────────────────────────────
# SAVE ETL REPORT
# ──────────────────────────────────────────────────────────────

def save_etl_report(df: pd.DataFrame, out_path: str):
    etl_report["output_file"]   = out_path
    etl_report["rows_cleaned"]  = int(df.shape[0])
    etl_report["columns_final"] = int(df.shape[1])
    etl_report["rows_total_dropped"] = (
        etl_report["rows_raw"] - etl_report["rows_cleaned"]
    )
    etl_report["retention_rate_%"] = round(
        etl_report["rows_cleaned"] / etl_report["rows_raw"] * 100, 2
    )

    report_path = os.path.join(CATALOG_DIR, "etl_report.json")
    with open(report_path, "w") as f:
        json.dump(etl_report, f, indent=2, default=str)
    log.info(f"ETL report saved -> {report_path}")


# ──────────────────────────────────────────────────────────────
# MAIN ETL PIPELINE
# ──────────────────────────────────────────────────────────────

def run_etl():
    log.info("=" * 60)
    log.info("TASK 2 — ETL PIPELINE STARTED")
    log.info("=" * 60)

    df = load_raw_data()

    # Apply each transformation in order
    df = drop_exact_duplicates(df)
    df = deduplicate_case_number(df)
    df = drop_missing_critical(df)
    df = fill_non_critical_nulls(df)
    df = fix_data_types(df)
    df = validate_coordinates(df)
    df = validate_temporal_range(df)
    df = standardize_crime_types(df)
    df = remove_coordinate_outliers(df)
    df = final_quality_check(df)

    out_path = save_cleaned_dataset(df)
    save_etl_report(df, out_path)

    log.info("=" * 60)
    log.info("TASK 2 — ETL COMPLETE")
    log.info(f"  Input rows      : {etl_report['rows_raw']:,}")
    log.info(f"  Output rows     : {etl_report['rows_cleaned']:,}")
    log.info(f"  Rows dropped    : {etl_report['rows_total_dropped']:,}")
    log.info(f"  Retention rate  : {etl_report['retention_rate_%']}%")
    log.info(f"  Output file     : {out_path}")
    log.info("=" * 60)

    return df


if __name__ == "__main__":
    run_etl()