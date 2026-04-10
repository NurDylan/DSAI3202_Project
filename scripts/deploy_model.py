"""
=============================================================
DSAI3202 - Phase 2  |  Task II.4: Model Deployment
=============================================================

PURPOSE:
    Deploy the champion XGBoost model as a BATCH scoring job on
    Azure ML. Reads the feature matrix from Blob Storage, runs
    predictions, and writes results back to Blob Storage.

    Serving mode: BATCH (aligned with project hypothesis — predictions
    generated at shift start for resource allocation planning, not
    real-time incident response).

    Input interface:
        CSV with columns: hour_sin, hour_cos, is_weekend, is_night,
                          Latitude, Longitude, community_area_enc

    Output interface:
        CSV with all input columns + predicted_crime_type (label string)
                                   + predicted_label (integer)
                                   + confidence_score (max class probability)

USAGE:
    python scripts/deploy_model.py

OUTPUTS:
    - deployment/batch_predictions_v1.0.csv  <- predictions file
    - deployment/deployment_config.json      <- deployment configuration record
    - logs/deploy_model.log
"""

import os
import sys
import io
import json
import logging
import datetime
import pickle
import joblib
import pandas as pd
import numpy as np

# CONFIGURATION
CHAMPION_MODEL = os.path.join("data", "phase_ii", "advanced_xgboost_model.pkl")
LABEL_ENCODING = os.path.join("data", "catalog", "label_encoding.json")

FEATURE_COLUMNS = [
    "hour_sin", "hour_cos", "is_weekend", "is_night",
    "Latitude", "Longitude", "community_area_enc"
]

# Batch size — process in chunks to avoid memory issues
BATCH_SIZE  = 100_000
# How many rows to score in this deployment run (None = all)
SCORE_ROWS  = 500_000

INPUT_CONTAINER  = "processed-data"
INPUT_BLOB       = "chicago_crime_features_v1.0.csv"
OUTPUT_CONTAINER = "model-outputs"
OUTPUT_BLOB      = "batch_predictions_v1.0.csv"

TODAY = datetime.date.today().isoformat()

# SETUP

os.makedirs("deployment", exist_ok=True)
os.makedirs("logs", exist_ok=True)

LOG_FILE = os.path.join("logs", "deploy_model.log")

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


# STEP 1 - LOAD CHAMPION MODEL & LABEL ENCODER

def load_model_and_encoder():
    log.info("=" * 60)
    log.info("STEP 1 - Loading champion model and label encoder...")
    log.info("=" * 60)

    if not os.path.exists(CHAMPION_MODEL):
        raise FileNotFoundError(
            f"Champion model not found: {CHAMPION_MODEL}\n"
            "Run crime_modeling.ipynb first."
        )

    model    = joblib.load(CHAMPION_MODEL)
    size_mb  = os.path.getsize(CHAMPION_MODEL) / (1024 ** 2)
    log.info(f"  [OK] Model loaded: {CHAMPION_MODEL} ({size_mb:.1f} MB)")
    log.info(f"  Model type: {type(model).__name__}")

    # Load label encoding for decoding predictions back to crime type strings
    with open(LABEL_ENCODING, encoding="utf-8") as f:
        label_map = json.load(f)

    # label_encoding.json maps crime_type_string -> integer
    # We need integer -> crime_type_string for decoding
    # Keys in the file are the crime type labels as strings of integers
    # Build reverse: int -> crime type name from crime_type_merges context
    # Since the encoding stores numeric keys, we load it directly
    int_to_label = {int(v): k for k, v in label_map.items()}
    log.info(f"  [OK] Label encoder loaded: {len(int_to_label)} classes")

    return model, int_to_label


# STEP 2 - LOAD INPUT DATA

def load_input_data(model):
    """
    Load feature data from Azure Blob Storage if connection string
    is available, otherwise fall back to local file.
    """
    log.info("=" * 60)
    log.info("STEP 2 - Loading input data...")
    log.info("=" * 60)

    if CONN_STR:
        log.info(f"  Reading from Azure Blob Storage: {INPUT_CONTAINER}/{INPUT_BLOB}")
        try:
            from azure.storage.blob import BlobServiceClient
            client      = BlobServiceClient.from_connection_string(CONN_STR)
            blob_client = client.get_blob_client(
                container=INPUT_CONTAINER, blob=INPUT_BLOB
            )
            data = blob_client.download_blob().readall()
            df   = pd.read_csv(io.BytesIO(data), nrows=SCORE_ROWS)
            log.info(f"  [OK] Loaded from Blob: {len(df):,} rows")
        except Exception as e:
            log.warning(f"  Blob read failed: {e} — falling back to local file")
            df = _load_local_features()
    else:
        log.warning("  AZURE_STORAGE_CONNECTION_STRING not set — using local file")
        df = _load_local_features()

    # Validate feature columns present
    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns in input data: {missing}")

    log.info(f"  Input shape: {df.shape[0]:,} rows x {df.shape[1]} columns")
    log.info(f"  Features used: {FEATURE_COLUMNS}")
    return df


def _load_local_features():
    local = os.path.join("data", "processed", "chicago_crime_features_v1.0.csv")
    if not os.path.exists(local):
        raise FileNotFoundError(
            f"Local feature file not found: {local}\n"
            "Run src/features.py or set AZURE_STORAGE_CONNECTION_STRING."
        )
    df = pd.read_csv(local, nrows=SCORE_ROWS)
    log.info(f"  [OK] Loaded from local: {len(df):,} rows")
    return df


# STEP 3 - RUN BATCH PREDICTIONS

def run_batch_predictions(df, model, int_to_label):
    log.info("=" * 60)
    log.info(f"STEP 3 - Running batch predictions ({len(df):,} rows)...")
    log.info("=" * 60)

    X           = df[FEATURE_COLUMNS].values
    predictions = []
    confidences = []

    # Process in batches to manage memory
    n_batches = max(1, len(X) // BATCH_SIZE)
    log.info(f"  Processing {n_batches} batches of {BATCH_SIZE:,} rows...")

    for i in range(0, len(X), BATCH_SIZE):
        batch     = X[i:i + BATCH_SIZE]
        preds     = model.predict(batch)
        # Get confidence score (max class probability) if available
        try:
            probs = model.predict_proba(batch)
            confs = probs.max(axis=1)
        except AttributeError:
            confs = np.ones(len(preds))  # fallback if predict_proba not available

        predictions.extend(preds.tolist())
        confidences.extend(confs.tolist())

        if (i // BATCH_SIZE + 1) % 5 == 0 or i == 0:
            log.info(f"  Batch {i // BATCH_SIZE + 1}/{n_batches} complete "
                     f"({min(i + BATCH_SIZE, len(X)):,}/{len(X):,} rows)")

    # Decode integer predictions back to crime type strings
    df_out = df.copy()
    df_out["predicted_label"]      = predictions
    df_out["predicted_crime_type"] = [int_to_label.get(p, f"CLASS_{p}") for p in predictions]
    df_out["confidence_score"]     = [round(c, 4) for c in confidences]

    log.info(f"  [OK] Predictions complete: {len(df_out):,} rows")
    log.info(f"  Prediction distribution (top 5):")
    top5 = df_out["predicted_crime_type"].value_counts().head(5)
    for crime, count in top5.items():
        pct = count / len(df_out) * 100
        log.info(f"    {crime:<35} {count:>8,}  ({pct:.1f}%)")

    avg_confidence = df_out["confidence_score"].mean()
    log.info(f"  Average confidence score: {avg_confidence:.4f}")

    return df_out


# STEP 4 - SAVE PREDICTIONS

def save_predictions(df_out):
    log.info("=" * 60)
    log.info("STEP 4 - Saving predictions...")
    log.info("=" * 60)

    # Save locally first
    local_path = os.path.join("deployment", "batch_predictions_v1.0.csv")
    df_out.to_csv(local_path, index=False)
    size_mb = os.path.getsize(local_path) / (1024 ** 2)
    log.info(f"  [OK] Saved locally: {local_path} ({size_mb:.1f} MB)")

    # Upload to Azure Blob Storage if connection available
    if CONN_STR:
        try:
            from azure.storage.blob import BlobServiceClient, ContainerClient
            client = BlobServiceClient.from_connection_string(CONN_STR)

            # Create output container if it doesn't exist
            try:
                client.create_container(OUTPUT_CONTAINER)
                log.info(f"  Created container: {OUTPUT_CONTAINER}")
            except Exception:
                pass  # Container already exists

            blob_client = client.get_blob_client(
                container=OUTPUT_CONTAINER, blob=OUTPUT_BLOB
            )
            with open(local_path, "rb") as f:
                blob_client.upload_blob(f, overwrite=True)
            log.info(f"  [OK] Uploaded to Blob: {OUTPUT_CONTAINER}/{OUTPUT_BLOB}")
        except Exception as e:
            log.warning(f"  Blob upload failed: {e} — predictions saved locally only")

    return local_path


# STEP 5 - SAVE DEPLOYMENT CONFIG

def save_deployment_config(df_out):
    config = {
        "deployment_version"   : "1.0",
        "deployment_date"      : TODAY,
        "serving_mode"         : "batch",
        "serving_justification": (
            "Batch mode aligns with the project hypothesis — predictions are "
            "generated at shift start for resource allocation planning, not "
            "during active incident response. Latency < 500ms per record is "
            "not required; total batch completion time is the relevant metric."
        ),
        "champion_model"       : "crime-classifier-xgboost",
        "model_file"           : CHAMPION_MODEL,
        "input_interface"      : {
            "source"   : f"{INPUT_CONTAINER}/{INPUT_BLOB}",
            "format"   : "CSV",
            "features" : FEATURE_COLUMNS,
        },
        "output_interface"     : {
            "destination"      : f"{OUTPUT_CONTAINER}/{OUTPUT_BLOB}",
            "format"           : "CSV",
            "added_columns"    : [
                "predicted_label",
                "predicted_crime_type",
                "confidence_score",
            ],
        },
        "batch_stats"          : {
            "rows_scored"         : int(len(df_out)),
            "batch_size"          : BATCH_SIZE,
            "avg_confidence"      : round(float(df_out["confidence_score"].mean()), 4),
            "prediction_distribution": df_out["predicted_crime_type"].value_counts().to_dict(),
        },
        "feature_parity"       : (
            "Feature columns at serving time are identical to training: "
            + ", ".join(FEATURE_COLUMNS)
        ),
    }

    out = os.path.join("deployment", "deployment_config.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False, default=str)
    log.info(f"  [OK] Deployment config saved -> {out}")
    return config


# MAIN

def run_deployment():
    log.info("=" * 60)
    log.info("TASK II.4 - MODEL DEPLOYMENT (BATCH) STARTED")
    log.info("=" * 60)

    model, int_to_label = load_model_and_encoder()
    df                  = load_input_data(model)
    df_out              = run_batch_predictions(df, model, int_to_label)
    local_path          = save_predictions(df_out)
    config              = save_deployment_config(df_out)

    log.info("=" * 60)
    log.info("TASK II.4 - DEPLOYMENT COMPLETE")
    log.info(f"  Rows scored        : {len(df_out):,}")
    log.info(f"  Avg confidence     : {config['batch_stats']['avg_confidence']:.4f}")
    log.info(f"  Predictions file   : {local_path}")
    log.info(f"  Deployment config  : deployment/deployment_config.json")
    log.info("=" * 60)

    return df_out


if __name__ == "__main__":
    run_deployment()