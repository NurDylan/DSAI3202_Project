"""
=============================================================
DSAI3202 - Phase 1  |  Azure Deployment Validation
Project: Predicting Crime Types by Location & Time-Series Data
Authors: Nur Afiqah (60306981), Elaf Marouf (60107174)
=============================================================

PURPOSE:
    Validates the Phase 1 pipeline by reading directly from
    Azure Blob Storage. Proves that:
      1. All pipeline outputs exist in Blob Storage
      2. Cleaned data is readable and correct from the cloud
      3. Feature matrix is accessible from Azure
      4. Catalog metadata is intact

    Run this script on Azure ML Compute to prove cloud deployment.
    No local files needed - everything is read from Blob Storage.

USAGE:
    Set environment variable before running:
        set AZURE_STORAGE_CONNECTION_STRING=<your_connection_string>

    Then run:
        python src/azure_validate.py
"""

import os
import sys
import io
import json
import logging
import datetime
import pandas as pd
from azure.storage.blob import BlobServiceClient

# ---------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------

CONN_STR = os.getenv("AZURE_STORAGE_CONNECTION_STRING")

CONTAINERS = {
    "raw"       : "raw-data",
    "processed" : "processed-data",
    "catalog"   : "catalog-data",
}

EXPECTED_FILES = {
    "raw-data": [
        "chicago_crime_v1.0_2026-03-16.csv",
    ],
    "processed-data": [
        "chicago_crime_cleaned_v1.0.csv",
        "chicago_crime_features_v1.0.csv",
    ],
    "catalog-data": [
        "ingestion_manifest.json",
        "etl_report.json",
        "data_catalog.json",
        "lineage.json",
        "assumptions.json",
        "zone_registry.json",
        "crime_type_merges.json",
        "label_encoding.json",
    ],
}

# How many rows to sample from Blob for validation (avoid loading 1.6GB)
SAMPLE_ROWS = 50_000

TODAY    = datetime.date.today().isoformat()
LOG_DIR  = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

# ---------------------------------------------------------------
# LOGGING
# ---------------------------------------------------------------

LOG_FILE = os.path.join(LOG_DIR, "azure_validate.log")

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
# HELPERS
# ---------------------------------------------------------------

def get_client() -> BlobServiceClient:
    if not CONN_STR:
        raise EnvironmentError(
            "AZURE_STORAGE_CONNECTION_STRING is not set.\n"
            "Set it with:  set AZURE_STORAGE_CONNECTION_STRING=<your_connection_string>"
        )
    return BlobServiceClient.from_connection_string(CONN_STR)


def list_blobs(client: BlobServiceClient, container: str) -> list:
    """Return list of (name, size_mb) tuples for all blobs in container."""
    container_client = client.get_container_client(container)
    blobs = []
    for blob in container_client.list_blobs():
        blobs.append({
            "name"    : blob.name,
            "size_mb" : round(blob.size / (1024 ** 2), 2),
            "modified": blob.last_modified.isoformat() if blob.last_modified else None,
        })
    return blobs


def download_blob_text(client: BlobServiceClient, container: str, blob_name: str) -> str:
    """Download a blob and return its content as a string."""
    blob_client = client.get_blob_client(container=container, blob=blob_name)
    data = blob_client.download_blob().readall()
    return data.decode("utf-8")


def download_blob_df(client: BlobServiceClient, container: str,
                     blob_name: str, nrows: int = None) -> pd.DataFrame:
    """Stream a CSV blob into a Pandas DataFrame."""
    blob_client = client.get_blob_client(container=container, blob=blob_name)
    stream = blob_client.download_blob()
    data   = stream.readall()
    return pd.read_csv(io.BytesIO(data), low_memory=False, nrows=nrows)


# ---------------------------------------------------------------
# VALIDATION STEPS
# ---------------------------------------------------------------

def validate_containers(client: BlobServiceClient) -> bool:
    """Check all required containers exist."""
    log.info("=" * 60)
    log.info("STEP 1 - Validating Azure Blob containers...")
    log.info("=" * 60)

    existing = [c["name"] for c in client.list_containers()]
    all_ok   = True

    for key, name in CONTAINERS.items():
        if name in existing:
            log.info(f"  [OK] Container exists: {name}")
        else:
            log.error(f"  [MISSING] Container not found: {name}")
            all_ok = False

    return all_ok


def validate_files(client: BlobServiceClient) -> dict:
    """Check all expected files exist in each container and log their sizes."""
    log.info("=" * 60)
    log.info("STEP 2 - Validating files in Blob Storage...")
    log.info("=" * 60)

    results = {}

    for container, expected in EXPECTED_FILES.items():
        blobs       = list_blobs(client, container)
        blob_names  = {b["name"] for b in blobs}
        blob_lookup = {b["name"]: b for b in blobs}

        log.info(f"\n  Container: {container}")
        container_ok = True

        for filename in expected:
            if filename in blob_names:
                size = blob_lookup[filename]["size_mb"]
                mod  = blob_lookup[filename]["modified"]
                log.info(f"    [OK] {filename} ({size} MB) — last modified: {mod}")
            else:
                log.error(f"    [MISSING] {filename}")
                container_ok = False

        results[container] = container_ok

    return results


def validate_cleaned_data(client: BlobServiceClient) -> dict:
    """
    Read a sample of the cleaned dataset from Blob Storage.
    Validates schema, row count estimate, critical column integrity,
    and crime type distribution.
    """
    log.info("=" * 60)
    log.info(f"STEP 3 - Validating cleaned dataset (sample: {SAMPLE_ROWS:,} rows)...")
    log.info("=" * 60)

    df = download_blob_df(
        client, "processed-data",
        "chicago_crime_cleaned_v1.0.csv",
        nrows=SAMPLE_ROWS
    )

    log.info(f"  Sample loaded: {df.shape[0]:,} rows x {df.shape[1]} columns")

    # Schema check
    expected_cols = [
        "ID", "Case Number", "Date", "Block", "IUCR", "Primary Type",
        "Description", "Location Description", "Arrest", "Domestic",
        "Beat", "District", "Ward", "Community Area", "FBI Code",
        "X Coordinate", "Y Coordinate", "Year", "Updated On",
        "Latitude", "Longitude", "Location"
    ]
    missing_cols = [c for c in expected_cols if c not in df.columns]
    if missing_cols:
        log.error(f"  [FAIL] Missing columns: {missing_cols}")
    else:
        log.info(f"  [OK] All 22 columns present")

    # Critical nulls
    critical = ["Date", "Primary Type", "Latitude", "Longitude"]
    null_counts = df[critical].isnull().sum()
    if null_counts.sum() == 0:
        log.info(f"  [OK] No nulls in critical columns")
    else:
        log.error(f"  [FAIL] Critical nulls found:\n{null_counts[null_counts > 0]}")

    # Coordinate range
    lat_ok = df["Latitude"].between(41.60, 42.05).all()
    lon_ok = df["Longitude"].between(-88.00, -87.50).all()
    log.info(f"  [{'OK' if lat_ok else 'FAIL'}] Latitude range: "
             f"{df['Latitude'].min():.5f} to {df['Latitude'].max():.5f}")
    log.info(f"  [{'OK' if lon_ok else 'FAIL'}] Longitude range: "
             f"{df['Longitude'].min():.5f} to {df['Longitude'].max():.5f}")

    # Crime type check
    n_types = df["Primary Type"].nunique()
    log.info(f"  [OK] Unique crime types in sample: {n_types}")
    log.info(f"  Top 5 crime types in sample:\n"
             f"{df['Primary Type'].value_counts().head(5).to_string()}")

    return {
        "rows_sampled"  : len(df),
        "columns"       : df.shape[1],
        "critical_nulls": int(null_counts.sum()),
        "lat_ok"        : lat_ok,
        "lon_ok"        : lon_ok,
        "crime_types"   : n_types,
    }


def validate_feature_matrix(client: BlobServiceClient) -> dict:
    """
    Read a sample of the feature matrix from Blob Storage.
    Validates the 9 selected features are present.
    """
    log.info("=" * 60)
    log.info("STEP 4 - Validating feature matrix...")
    log.info("=" * 60)

    df = download_blob_df(
        client, "processed-data",
        "chicago_crime_features_v1.0.csv",
        nrows=10_000
    )

    log.info(f"  Sample loaded: {df.shape[0]:,} rows x {df.shape[1]} columns")

    expected_features = [
        "hour", "is_weekend", "is_night", "hour_sin", "hour_cos",
        "Latitude", "Longitude", "is_crowded", "community_area_enc"
    ]

    present  = [f for f in expected_features if f in df.columns]
    missing  = [f for f in expected_features if f not in df.columns]

    log.info(f"  [OK] Features present ({len(present)}/9): {present}")
    if missing:
        log.error(f"  [FAIL] Features missing: {missing}")
    else:
        log.info(f"  [OK] All 9 selected features confirmed in feature matrix")

    # Check target label exists
    if "crime_type_label" in df.columns:
        log.info(f"  [OK] Target label present: crime_type_label "
                 f"({df['crime_type_label'].nunique()} classes)")
    else:
        log.error(f"  [FAIL] crime_type_label column missing")

    return {
        "rows_sampled"    : len(df),
        "features_present": len(present),
        "features_missing": missing,
    }


def validate_catalog(client: BlobServiceClient) -> dict:
    """
    Download and parse each catalog JSON file from Blob Storage.
    Confirms metadata is intact.
    """
    log.info("=" * 60)
    log.info("STEP 5 - Validating catalog metadata files...")
    log.info("=" * 60)

    results = {}

    # data_catalog.json
    try:
        raw     = download_blob_text(client, "catalog-data", "data_catalog.json")
        catalog = json.loads(raw)
        n_cols  = len(catalog.get("columns", []))
        log.info(f"  [OK] data_catalog.json — {n_cols} columns documented, "
                 f"target: {catalog.get('target_variable')}, "
                 f"classes: {catalog.get('target_classes')}")
        results["data_catalog"] = True
    except Exception as e:
        log.error(f"  [FAIL] data_catalog.json: {e}")
        results["data_catalog"] = False

    # etl_report.json
    try:
        raw    = download_blob_text(client, "catalog-data", "etl_report.json")
        report = json.loads(raw)
        log.info(f"  [OK] etl_report.json — "
                 f"input: {report.get('rows_raw'):,} rows, "
                 f"output: {report.get('rows_cleaned'):,} rows, "
                 f"retention: {report.get('retention_rate_%')}%")
        results["etl_report"] = True
    except Exception as e:
        log.error(f"  [FAIL] etl_report.json: {e}")
        results["etl_report"] = False

    # assumptions.json
    try:
        raw         = download_blob_text(client, "catalog-data", "assumptions.json")
        assumptions = json.loads(raw)
        total = sum(len(v) for v in assumptions.get("categories", {}).values())
        log.info(f"  [OK] assumptions.json — {total} assumptions across "
                 f"{len(assumptions.get('categories', {}))} categories")
        results["assumptions"] = True
    except Exception as e:
        log.error(f"  [FAIL] assumptions.json: {e}")
        results["assumptions"] = False

    # lineage.json
    try:
        raw     = download_blob_text(client, "catalog-data", "lineage.json")
        lineage = json.loads(raw)
        stages  = len(lineage.get("stages", []))
        log.info(f"  [OK] lineage.json — {stages} pipeline stages documented")
        results["lineage"] = True
    except Exception as e:
        log.error(f"  [FAIL] lineage.json: {e}")
        results["lineage"] = False

    return results


def print_summary(container_results, file_results, data_results,
                  feature_results, catalog_results):
    log.info("=" * 60)
    log.info("AZURE DEPLOYMENT VALIDATION SUMMARY")
    log.info("=" * 60)
    log.info(f"  Run date          : {TODAY}")
    log.info(f"  Containers OK     : {container_results}")
    log.info(f"  All files present : {all(file_results.values())}")
    log.info(f"  Cleaned data OK   : {data_results['critical_nulls'] == 0}")
    log.info(f"    Rows sampled    : {data_results['rows_sampled']:,}")
    log.info(f"    Columns         : {data_results['columns']}")
    log.info(f"    Crime types     : {data_results['crime_types']}")
    log.info(f"  Feature matrix OK : {feature_results['features_missing'] == []}")
    log.info(f"    Features present: {feature_results['features_present']}/9")
    log.info(f"  Catalog files OK  : {all(catalog_results.values())}")
    log.info("=" * 60)

    all_passed = (
        container_results and
        all(file_results.values()) and
        data_results["critical_nulls"] == 0 and
        not feature_results["features_missing"] and
        all(catalog_results.values())
    )

    if all_passed:
        log.info("  RESULT: ALL CHECKS PASSED - Azure deployment validated.")
    else:
        log.warning("  RESULT: SOME CHECKS FAILED - review logs above.")

    log.info("=" * 60)


# ---------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------

def run_azure_validation():
    log.info("=" * 60)
    log.info("AZURE DEPLOYMENT VALIDATION STARTED")
    log.info("=" * 60)

    client = get_client()

    container_ok    = validate_containers(client)
    file_results    = validate_files(client)
    data_results    = validate_cleaned_data(client)
    feature_results = validate_feature_matrix(client)
    catalog_results = validate_catalog(client)

    print_summary(
        container_ok, file_results,
        data_results, feature_results, catalog_results
    )


if __name__ == "__main__":
    run_azure_validation()