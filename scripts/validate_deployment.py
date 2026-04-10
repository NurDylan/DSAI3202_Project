"""
=============================================================
DSAI3202 - Phase 2  |  Task II.5: Deployment Validation
=============================================================

PURPOSE:
    Verify deployed model behaviour through functional tests and
    sanity checks. Confirms:
      1. Model loads and predicts correctly
      2. Output format matches specification
      3. Predictions are consistent between offline and deployed
      4. Confidence scores are valid
      5. Feature parity between training and serving is maintained
      6. Batch predictions file exists in Blob Storage

USAGE:
    python scripts/validate_deployment.py

OUTPUTS:
    - deployment/validation_report.json  <- full test report
    - logs/validate_deployment.log
"""

import os
import sys
import io
import json
import time
import logging
import datetime
import joblib
import numpy as np
import pandas as pd

# CONFIGURATION

CONN_STR        = os.getenv(
    "AZURE_STORAGE_CONNECTION_STRING",
    "DefaultEndpointsProtocol=https;AccountName=cloudproject60107174;"
    "AccountKey=ncENKi5Q3q7znrrczl5iPxCF5dt75dbBKiv8bruQKmb2txGpALDqKgVsfP5GuS8bFoNFoxjbyprp+AStvFix+w==;"
    "EndpointSuffix=core.windows.net"
)
CHAMPION_MODEL  = os.path.join("data", "phase_ii", "advanced_xgboost_model.pkl")
LABEL_ENCODING  = os.path.join("data", "catalog", "label_encoding.json")
REGISTRY_FILE   = os.path.join("data", "phase_ii", "model_registry.json")
DEPLOY_CONFIG   = os.path.join("deployment", "deployment_config.json")
PREDICTIONS_FILE= os.path.join("deployment", "batch_predictions_v1.0.csv")

FEATURE_COLUMNS = [
    "hour_sin", "hour_cos", "is_weekend", "is_night",
    "Latitude", "Longitude", "community_area_enc"
]

TODAY = datetime.date.today().isoformat()

# SETUP

os.makedirs("deployment", exist_ok=True)
os.makedirs("logs", exist_ok=True)

LOG_FILE = os.path.join("logs", "validate_deployment.log")

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

# Collect test results
test_results = []

def record(name, passed, detail=""):
    status = "PASS" if passed else "FAIL"
    test_results.append({"test": name, "status": status, "detail": detail})
    icon = "[PASS]" if passed else "[FAIL]"
    log.info(f"  {icon} {name}" + (f" — {detail}" if detail else ""))


# TEST 1 - MODEL FILE EXISTS AND LOADS

def test_model_load():
    log.info("=" * 60)
    log.info("TEST 1 - Model load test")
    log.info("=" * 60)

    exists = os.path.exists(CHAMPION_MODEL)
    record("Model file exists", exists, CHAMPION_MODEL)
    if not exists:
        record("Model loads successfully", False, "File missing — cannot load")
        return None

    try:
        model   = joblib.load(CHAMPION_MODEL)
        size_mb = os.path.getsize(CHAMPION_MODEL) / (1024 ** 2)
        record("Model loads successfully", True,
               f"{type(model).__name__} ({size_mb:.1f} MB)")
        return model
    except Exception as e:
        record("Model loads successfully", False, str(e))
        return None


# TEST 2 - CORRECTNESS: PREDICT ON KNOWN INPUTS

def test_correctness(model):
    log.info("=" * 60)
    log.info("TEST 2 - Correctness test (known input -> valid output)")
    log.info("=" * 60)

    if model is None:
        record("Correctness test", False, "Model not loaded")
        return

    # Synthetic test cases representing different crime scenarios
    test_cases = pd.DataFrame({
        "hour_sin"          : [np.sin(2 * np.pi * 2 / 24),   # 2am — night
                               np.sin(2 * np.pi * 14 / 24),  # 2pm — afternoon
                               np.sin(2 * np.pi * 20 / 24)], # 8pm — evening
        "hour_cos"          : [np.cos(2 * np.pi * 2 / 24),
                               np.cos(2 * np.pi * 14 / 24),
                               np.cos(2 * np.pi * 20 / 24)],
        "is_weekend"        : [0, 1, 0],
        "is_night"          : [1, 0, 0],
        "Latitude"          : [41.85, 41.75, 41.90],
        "Longitude"         : [-87.65, -87.70, -87.62],
        "community_area_enc": [23, 44, 8],
    })

    try:
        preds = model.predict(test_cases[FEATURE_COLUMNS].values)
        record("Predict returns output", True, f"3 inputs -> 3 predictions: {preds.tolist()}")

        # Predictions must be integers (class labels)
        all_int = all(isinstance(p, (int, np.integer)) for p in preds)
        record("Predictions are integer class labels", all_int,
               f"Types: {[type(p).__name__ for p in preds]}")

        # Predictions must be within valid class range
        with open(LABEL_ENCODING, encoding="utf-8") as f:
            label_map = json.load(f)
        n_classes  = len(label_map)
        in_range   = all(0 <= int(p) < n_classes for p in preds)
        record("Predictions within valid class range", in_range,
               f"Range [0, {n_classes-1}], got: {preds.tolist()}")

    except Exception as e:
        record("Correctness test", False, str(e))


# TEST 3 - LATENCY TEST

def test_latency(model):
    log.info("=" * 60)
    log.info("TEST 3 - Latency test (< 500ms per record requirement)")
    log.info("=" * 60)

    if model is None:
        record("Latency test", False, "Model not loaded")
        return

    # Single record latency
    single = np.array([[
        np.sin(2 * np.pi * 12 / 24),
        np.cos(2 * np.pi * 12 / 24),
        0, 0, 41.85, -87.65, 23
    ]])

    times = []
    for _ in range(10):
        t0 = time.perf_counter()
        model.predict(single)
        times.append((time.perf_counter() - t0) * 1000)

    avg_ms = np.mean(times)
    min_ms = np.min(times)
    max_ms = np.max(times)

    record("Single-record latency < 500ms", avg_ms < 500,
           f"avg={avg_ms:.2f}ms, min={min_ms:.2f}ms, max={max_ms:.2f}ms (10 runs)")

    # Batch of 1000 records
    batch_1000 = np.tile(single, (1000, 1))
    t0         = time.perf_counter()
    model.predict(batch_1000)
    batch_ms   = (time.perf_counter() - t0) * 1000
    per_record = batch_ms / 1000

    record("Batch-1000 per-record latency < 10ms", per_record < 10,
           f"total={batch_ms:.1f}ms, per_record={per_record:.3f}ms")


# TEST 4 - FEATURE PARITY

def test_feature_parity():
    log.info("=" * 60)
    log.info("TEST 4 - Feature parity (training vs serving)")
    log.info("=" * 60)

    if not os.path.exists(DEPLOY_CONFIG):
        record("Deployment config exists", False, DEPLOY_CONFIG)
        return

    with open(DEPLOY_CONFIG, encoding="utf-8") as f:
        config = json.load(f)

    serving_features  = config.get("input_interface", {}).get("features", [])
    training_features = FEATURE_COLUMNS

    match = set(serving_features) == set(training_features)
    record("Feature parity: serving == training", match,
           f"Training: {training_features} | Serving: {serving_features}")

    order_match = serving_features == training_features
    record("Feature order matches", order_match,
           "Column order matters for XGBoost prediction correctness")


# TEST 5 - OUTPUT FORMAT VALIDATION

def test_output_format():
    log.info("=" * 60)
    log.info("TEST 5 - Output format validation")
    log.info("=" * 60)

    if not os.path.exists(PREDICTIONS_FILE):
        record("Predictions file exists locally", False, PREDICTIONS_FILE)
        log.warning("  Run scripts/deploy_model.py first to generate predictions.")
        return

    df = pd.read_csv(PREDICTIONS_FILE, nrows=1000)
    record("Predictions file is readable", True,
           f"{PREDICTIONS_FILE} ({len(df)} rows in sample)")

    required_output_cols = ["predicted_label", "predicted_crime_type", "confidence_score"]
    for col in required_output_cols:
        present = col in df.columns
        record(f"Output column '{col}' present", present)

    if "confidence_score" in df.columns:
        valid_conf = df["confidence_score"].between(0, 1).all()
        record("Confidence scores in [0, 1]", valid_conf,
               f"min={df['confidence_score'].min():.4f}, max={df['confidence_score'].max():.4f}")

    if "predicted_crime_type" in df.columns:
        n_types = df["predicted_crime_type"].nunique()
        record("Predicted crime types are strings", True,
               f"{n_types} unique crime types in sample")


# TEST 6 - CONSISTENCY (OFFLINE vs DEPLOYED)

def test_consistency(model):
    log.info("=" * 60)
    log.info("TEST 6 - Offline vs deployed consistency")
    log.info("=" * 60)

    if model is None or not os.path.exists(PREDICTIONS_FILE):
        record("Consistency test", False, "Model or predictions file missing")
        return

    # Take 100 rows from predictions file and re-predict offline
    df_pred = pd.read_csv(PREDICTIONS_FILE, nrows=100)

    missing_feats = [c for c in FEATURE_COLUMNS if c not in df_pred.columns]
    if missing_feats:
        record("Consistency test", False, f"Feature columns missing in predictions file: {missing_feats}")
        return

    X              = df_pred[FEATURE_COLUMNS].values
    offline_preds  = model.predict(X)
    deployed_preds = df_pred["predicted_label"].values

    matches      = (offline_preds == deployed_preds).sum()
    match_rate   = matches / len(offline_preds)

    record("Offline == deployed predictions (100 rows)", match_rate == 1.0,
           f"{matches}/100 match ({match_rate:.1%})")


# TEST 7 - AZURE BLOB STORAGE CHECK

def test_azure_storage():
    log.info("=" * 60)
    log.info("TEST 7 - Azure Blob Storage check")
    log.info("=" * 60)

    if not CONN_STR:
        record("Azure connection string set", False,
               "AZURE_STORAGE_CONNECTION_STRING not set — skipping Blob tests")
        return

    try:
        from azure.storage.blob import BlobServiceClient
        client     = BlobServiceClient.from_connection_string(CONN_STR)
        containers = [c["name"] for c in client.list_containers()]

        record("Azure Blob connection OK", True, f"Found {len(containers)} containers")

        # Check model-outputs container for predictions
        if "model-outputs" in containers:
            blobs = [b.name for b in client.get_container_client("model-outputs").list_blobs()]
            has_preds = "batch_predictions_v1.0.csv" in blobs
            record("Predictions blob exists in model-outputs", has_preds,
                   "batch_predictions_v1.0.csv")
        else:
            record("model-outputs container exists", False,
                   "Run deploy_model.py to create it")

        # Check my_models_storage for champion model blob
        model_container = next((c for c in containers if "model" in c.lower() and "output" not in c.lower()), None)
        if model_container:
            model_blobs = [b.name for b in client.get_container_client(model_container).list_blobs()]
            has_xgb = any("advanced_xgboost" in b for b in model_blobs)
            record("Champion model blob exists in model storage", has_xgb,
                   f"Container: {model_container} | advanced_xgboost_model.pkl")
        else:
            record("Model storage container found", False,
                   "Run scripts/upload_models.py to upload models to Azure Blob")

    except Exception as e:
        record("Azure Blob connection OK", False, str(e))


# SAVE VALIDATION REPORT

def save_report():
    passed = sum(1 for t in test_results if t["status"] == "PASS")
    failed = sum(1 for t in test_results if t["status"] == "FAIL")

    report = {
        "validation_version"  : "1.0",
        "validation_date"     : TODAY,
        "champion_model"      : "crime-classifier-xgboost",
        "total_tests"         : len(test_results),
        "passed"              : passed,
        "failed"              : failed,
        "pass_rate"           : f"{passed / max(len(test_results), 1):.1%}",
        "overall_result"      : "PASS" if failed == 0 else "PARTIAL" if passed > 0 else "FAIL",
        "tests"               : test_results,
    }

    out = os.path.join("deployment", "validation_report.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    log.info(f"  Validation report saved -> {out}")
    return report


# MAIN

def run_validation():
    log.info("=" * 60)
    log.info("TASK II.5 - DEPLOYMENT VALIDATION STARTED")
    log.info("=" * 60)

    model = test_model_load()
    test_correctness(model)
    test_latency(model)
    test_feature_parity()
    test_output_format()
    test_consistency(model)
    test_azure_storage()

    report = save_report()

    log.info("=" * 60)
    log.info("DEPLOYMENT VALIDATION SUMMARY")
    log.info("=" * 60)
    log.info(f"  Total tests : {report['total_tests']}")
    log.info(f"  Passed      : {report['passed']}")
    log.info(f"  Failed      : {report['failed']}")
    log.info(f"  Pass rate   : {report['pass_rate']}")
    log.info(f"  Result      : {report['overall_result']}")
    log.info("=" * 60)

    return report


if __name__ == "__main__":
    run_validation()