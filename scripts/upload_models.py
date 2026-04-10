"""
=============================================================
DSAI3202 - Phase 2  |  Task II.3: Upload Models to Azure Blob
=============================================================

PURPOSE:
    Upload all 3 trained model .pkl files to the Azure Blob
    Storage datastore 'my_models_storage' under the 'models/'
    prefix. This makes models accessible to the Azure ML
    workspace and provides the Azure-side artifact required
    for Task II.3 (Model Versioning and Registration).

    After running this script, all three models will be
    visible in the Azure ML Studio under:
        Data > my_models_storage > Browse > models/

USAGE:
    python scripts/upload_models.py

    Requires AZURE_STORAGE_CONNECTION_STRING to be set, OR
    the hardcoded connection string in deploy_model.py to
    be used as fallback.

OUTPUTS:
    - models/advanced_xgboost_model.pkl   -> my_models_storage
    - models/decision_tree_model.pkl      -> my_models_storage
    - models/baseline_logreg_model.pkl    -> my_models_storage
    - logs/upload_models.log
"""

import os
import sys
import json
import logging
import datetime

# CONFIGURATION

CONN_STR = os.getenv(
    "AZURE_STORAGE_CONNECTION_STRING",
    "DefaultEndpointsProtocol=https;AccountName=cloudproject60107174;"
    "AccountKey=ncENKi5Q3q7znrrczl5iPxCF5dt75dbBKiv8bruQKmb2txGpALDqKgVsfP5GuS8bFoNFoxjbyprp+AStvFix+w==;"
    "EndpointSuffix=core.windows.net"
)

CONTAINER = "my-models-storage"   # container backing the 'my_models_storage' datastore
BLOB_PREFIX = "models"

MODELS_TO_UPLOAD = [
    {
        "local_path" : os.path.join("data", "phase_ii", "advanced_xgboost_model.pkl"),
        "blob_name"  : "models/advanced_xgboost_model.pkl",
        "label"      : "crime-classifier-xgboost (champion)",
    },
    {
        "local_path" : os.path.join("data", "phase_ii", "decision_tree_model.pkl"),
        "blob_name"  : "models/decision_tree_model.pkl",
        "label"      : "crime-classifier-decision-tree",
    },
    {
        "local_path" : os.path.join("data", "phase_ii", "baseline_logreg_model.pkl"),
        "blob_name"  : "models/baseline_logreg_model.pkl",
        "label"      : "crime-classifier-logreg-baseline",
    },
]

TODAY = datetime.date.today().isoformat()

# LOGGING

os.makedirs("logs", exist_ok=True)
LOG_FILE = os.path.join("logs", "upload_models.log")

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


# MAIN

def upload_models():
    log.info("=" * 60)
    log.info("TASK II.3 - UPLOAD MODELS TO AZURE BLOB STORAGE")
    log.info("=" * 60)

    try:
        from azure.storage.blob import BlobServiceClient
    except ImportError:
        log.error("azure-storage-blob not installed. Run: pip install azure-storage-blob")
        sys.exit(1)

    client = BlobServiceClient.from_connection_string(CONN_STR)

    # Ensure container exists
    try:
        client.create_container(CONTAINER)
        log.info(f"  Created container: {CONTAINER}")
    except Exception:
        log.info(f"  Container already exists: {CONTAINER}")

    results = []
    for m in MODELS_TO_UPLOAD:
        if not os.path.exists(m["local_path"]):
            log.error(f"  [MISSING] {m['local_path']} — run crime_modeling.ipynb first")
            results.append({"model": m["label"], "status": "SKIPPED", "reason": "file not found"})
            continue

        size_mb = os.path.getsize(m["local_path"]) / (1024 ** 2)
        log.info(f"  Uploading {m['label']} ({size_mb:.1f} MB) -> {m['blob_name']} ...")

        blob_client = client.get_blob_client(container=CONTAINER, blob=m["blob_name"])
        with open(m["local_path"], "rb") as f:
            blob_client.upload_blob(f, overwrite=True)

        log.info(f"  [OK] Uploaded: {m['blob_name']}")
        results.append({
            "model"     : m["label"],
            "blob_path" : m["blob_name"],
            "size_mb"   : round(size_mb, 1),
            "status"    : "OK",
            "uploaded"  : TODAY,
        })

    log.info("=" * 60)
    log.info("UPLOAD SUMMARY")
    log.info("=" * 60)
    for r in results:
        log.info(f"  {r['status']:6s}  {r['model']}")
    log.info(f"  Container : {CONTAINER}")
    log.info(f"  Log       : {LOG_FILE}")
    log.info("=" * 60)

    return results


if __name__ == "__main__":
    upload_models()
