"""
=============================================================
DSAI3202 – Phase 1  |  Task 1: Data Ingestion
Project: Predicting Crime Types by Location & Time-Series Data
Authors: Nur Afiqah (60306981), Elaf Marouf (60107174)
=============================================================

PURPOSE:
    Ingest the raw Chicago Crime dataset from Kaggle, version it,
    validate its structure, and persist it to the raw data zone.

INGESTION MODE:   Batch (one-time historical load)
DATA FORMAT:      CSV
REFRESH STRATEGY: On-demand re-run with new version tag
STORAGE LAYOUT:
    data/
        raw/          <- versioned original files, NEVER modified
        processed/    <- ETL output (written by etl.py)
        catalog/      <- schema & metadata JSON files
    logs/             <- ingestion and pipeline logs
"""

import os
import shutil
import logging
import datetime
import hashlib
import json
import pandas as pd

# ──────────────────────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────────────────────

# ➤ UPDATE THIS PATH to wherever you placed the downloaded CSV
RAW_CSV_SOURCE = "data/raw/crime_data.csv"

DATASET_VERSION  = "v1.0"
DATASET_NAME     = "chicago_crime"
KAGGLE_URL       = "https://www.kaggle.com/datasets/middlehigh/los-angeles-crime-data-from-2000/data"

RAW_DIR      = os.path.join("data", "raw")
CATALOG_DIR  = os.path.join("data", "catalog")
LOG_DIR      = "logs"

# Expected columns from the Chicago Crime dataset (all 22)
EXPECTED_COLUMNS = [
    "ID", "Case Number", "Date", "Block", "IUCR",
    "Primary Type", "Description", "Location Description",
    "Arrest", "Domestic", "Beat", "District", "Ward",
    "Community Area", "FBI Code", "X Coordinate", "Y Coordinate",
    "Year", "Updated On", "Latitude", "Longitude", "Location"
]

# ──────────────────────────────────────────────────────────────
# SETUP  –  directories & logging
# ──────────────────────────────────────────────────────────────

for d in [RAW_DIR, CATALOG_DIR, LOG_DIR]:
    os.makedirs(d, exist_ok=True)

TODAY     = datetime.date.today().isoformat()
LOG_FILE  = os.path.join(LOG_DIR, "ingestion.log")

import sys

# Force UTF-8 on Windows console to handle special characters (e.g. arrows)
if sys.platform == 'win32':
    _stream = open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1)
else:
    _stream = sys.stdout

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  [%(levelname)s]  %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, encoding='utf-8'),
        logging.StreamHandler(stream=_stream),
    ]
)
log = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ──────────────────────────────────────────────────────────────

def compute_md5(filepath: str) -> str:
    """Compute MD5 checksum of a file for integrity verification."""
    h = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def version_raw_file(source_path: str) -> str:
    """
    Copy the source CSV into the raw zone with a version + date tag.
    Returns the versioned destination path.
    Convention: data/raw/<name>_<version>_<date>.csv
    """
    filename   = f"{DATASET_NAME}_{DATASET_VERSION}_{TODAY}.csv"
    dest_path  = os.path.join(RAW_DIR, filename)

    if os.path.exists(dest_path):
        log.warning(f"Versioned file already exists: {dest_path} — skipping copy.")
        return dest_path

    shutil.copy2(source_path, dest_path)
    log.info(f"Raw file versioned → {dest_path}")
    return dest_path


def validate_schema(df: pd.DataFrame) -> bool:
    """
    Confirm all expected columns are present.
    Logs any missing or unexpected columns.
    """
    present   = set(df.columns)
    expected  = set(EXPECTED_COLUMNS)
    missing   = expected - present
    extra     = present - expected

    if missing:
        log.warning(f"Missing expected columns: {sorted(missing)}")
    if extra:
        log.info(f"Extra columns found (not in spec): {sorted(extra)}")

    return len(missing) == 0


def write_ingestion_manifest(versioned_path: str, df: pd.DataFrame, checksum: str):
    """
    Write a JSON manifest capturing ingestion metadata.
    Stored in data/catalog/ingestion_manifest.json
    """
    manifest = {
        "dataset_name"      : DATASET_NAME,
        "version"           : DATASET_VERSION,
        "ingestion_date"    : TODAY,
        "ingestion_mode"    : "batch",
        "source_url"        : KAGGLE_URL,
        "raw_file_path"     : versioned_path,
        "file_size_mb"      : round(os.path.getsize(versioned_path) / (1024 ** 2), 2),
        "row_count"         : int(df.shape[0]),
        "column_count"      : int(df.shape[1]),
        "columns"           : list(df.columns),
        "md5_checksum"      : checksum,
        "refresh_strategy"  : "on-demand re-run with incremented version tag",
        "storage_layout"    : {
            "raw_zone"       : "data/raw/",
            "processed_zone" : "data/processed/",
            "catalog_zone"   : "data/catalog/"
        }
    }

    out = os.path.join(CATALOG_DIR, "ingestion_manifest.json")
    with open(out, "w") as f:
        json.dump(manifest, f, indent=2)
    log.info(f"Ingestion manifest written → {out}")


# ──────────────────────────────────────────────────────────────
# MAIN INGESTION PIPELINE
# ──────────────────────────────────────────────────────────────

def run_ingestion():
    log.info("=" * 60)
    log.info("TASK 1 — DATA INGESTION STARTED")
    log.info("=" * 60)

    # ── Step 1: Verify source file exists ──────────────────────
    if not os.path.exists(RAW_CSV_SOURCE):
        log.error(
            f"Source file not found: '{RAW_CSV_SOURCE}'\n"
            f"  Download from: {KAGGLE_URL}\n"
            f"  Then set RAW_CSV_SOURCE at the top of this script."
        )
        raise FileNotFoundError(f"Source CSV not found: {RAW_CSV_SOURCE}")

    log.info(f"Source file found: {RAW_CSV_SOURCE}")
    file_size_mb = os.path.getsize(RAW_CSV_SOURCE) / (1024 ** 2)
    log.info(f"File size: {file_size_mb:.1f} MB")

    # ── Step 2: Compute checksum before copying ─────────────────
    log.info("Computing MD5 checksum for integrity verification...")
    checksum = compute_md5(RAW_CSV_SOURCE)
    log.info(f"MD5 checksum: {checksum}")

    # ── Step 3: Version and copy to raw zone ────────────────────
    log.info("Versioning raw file into data/raw/ ...")
    versioned_path = version_raw_file(RAW_CSV_SOURCE)

    # ── Step 4: Load a sample to validate structure ─────────────
    log.info("Loading dataset to validate schema and log statistics...")
    df = pd.read_csv(
        versioned_path,
        low_memory=False,       # prevent dtype inference warnings on large files
        on_bad_lines="warn"     # log but skip malformed rows
    )

    log.info(f"Loaded:  {df.shape[0]:,} rows  x  {df.shape[1]} columns")

    # ── Step 5: Schema validation ───────────────────────────────
    log.info("Validating column schema...")
    schema_ok = validate_schema(df)
    if schema_ok:
        log.info("Schema validation PASSED — all expected columns present.")
    else:
        log.warning("Schema validation WARNING — see missing columns above.")

    # ── Step 6: Null summary ────────────────────────────────────
    log.info("Null value summary per column:")
    null_counts = df.isnull().sum()
    null_pct    = (null_counts / len(df) * 100).round(2)
    null_summary = pd.DataFrame({"null_count": null_counts, "null_pct_%": null_pct})
    null_summary = null_summary[null_summary["null_count"] > 0].sort_values("null_count", ascending=False)
    log.info(f"\n{null_summary.to_string()}")

    # ── Step 7: Duplicate check ─────────────────────────────────
    dup_count = df.duplicated().sum()
    log.info(f"Exact duplicate rows: {dup_count:,}")

    case_dup  = df.duplicated(subset=["Case Number"]).sum()
    log.info(f"Duplicate Case Numbers: {case_dup:,}")

    # ── Step 8: Date range check ────────────────────────────────
    sample_dates = df["Date"].dropna().head(5).tolist()
    log.info(f"Date column sample values: {sample_dates}")
    log.info("Year distribution (from 'Year' column):")
    if "Year" in df.columns:
        log.info(f"\n{df['Year'].value_counts().sort_index().to_string()}")

    # ── Step 9: Target variable quick check ─────────────────────
    log.info(f"Unique crime types (Primary Type): {df['Primary Type'].nunique()}")
    log.info("Top 10 crime types:")
    log.info(f"\n{df['Primary Type'].value_counts().head(10).to_string()}")

    # ── Step 10: Write manifest ─────────────────────────────────
    write_ingestion_manifest(versioned_path, df, checksum)

    log.info("=" * 60)
    log.info("TASK 1 — INGESTION COMPLETE")
    log.info(f"  Versioned raw file : {versioned_path}")
    log.info(f"  Rows ingested      : {df.shape[0]:,}")
    log.info(f"  Columns            : {df.shape[1]}")
    log.info(f"  Log file           : {LOG_FILE}")
    log.info("=" * 60)

    return versioned_path, df


if __name__ == "__main__":
    run_ingestion()