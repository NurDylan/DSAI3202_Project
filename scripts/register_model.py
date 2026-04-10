"""
=============================================================
DSAI3202 - Phase 2  |  Task II.3: Model Versioning & Registration
=============================================================

PURPOSE:
    Register trained models in Azure ML Model Registry with full
    metadata: training data version, feature set, metrics, and
    limitations. Ensures full traceability between data, code,
    and deployed models.

PREREQUISITES:
    - Models trained and saved in data/phase_ii/ as .pkl files
    - Azure ML workspace accessible
    - Environment variables set (see below)

OUTPUTS:
    - 3 registered models in Azure ML Model Registry
    - data/phase_ii/model_registry.json  <- local registration record
    - logs/register_model.log
"""

import os
import sys
import json
import logging
import datetime
import pickle

# CONFIGURATION

SUBSCRIPTION_ID = os.getenv("AZURE_SUBSCRIPTION_ID", "a00dcbea-fd05-4973-82dc-120208b60116")
RESOURCE_GROUP  = os.getenv("AZURE_RESOURCE_GROUP",  "rg-60107174")
WORKSPACE_NAME  = os.getenv("AZURE_ML_WORKSPACE",    "cloudproject60107174")

# Model files — located in data/phase_ii/ as produced by the notebook
MODELS = [
    {
        "name"        : "crime-classifier-xgboost",
        "path"        : os.path.join("data", "phase_ii", "advanced_xgboost_model.pkl"),
        "version"     : "1",
        "description" : "XGBoost multi-class classifier for Chicago crime type prediction. Best performing model.",
        "algorithm"   : "XGBoost",
        "is_champion" : True,
        "metrics"     : {
            "accuracy"          : 0.2876,
            "balanced_accuracy" : 0.0918,
            "macro_f1"          : 0.0768,
            "weighted_precision": 0.2376,
        },
        "params"      : {
            "n_estimators" : 300,
            "max_depth"    : 10,
            "learning_rate": 0.1,
            "tree_method"  : "hist",
            "random_state" : 42,
        },
    },
    {
        "name"        : "crime-classifier-decision-tree",
        "path"        : os.path.join("data", "phase_ii", "decision_tree_model.pkl"),
        "version"     : "1",
        "description" : "Decision Tree multi-class classifier. Intermediate baseline.",
        "algorithm"   : "DecisionTree",
        "is_champion" : False,
        "metrics"     : {
            "accuracy"          : 0.2739,
            "balanced_accuracy" : 0.0614,
            "macro_f1"          : 0.0458,
            "weighted_precision": 0.2152,
        },
        "params"      : {
            "max_depth"  : 12,
            "random_state": 42,
        },
    },
    {
        "name"        : "crime-classifier-logreg-baseline",
        "path"        : os.path.join("data", "phase_ii", "baseline_logreg_model.pkl"),
        "version"     : "1",
        "description" : "Logistic Regression baseline with StandardScaler pipeline.",
        "algorithm"   : "LogisticRegression",
        "is_champion" : False,
        "metrics"     : {
            "accuracy"          : 0.2551,
            "balanced_accuracy" : 0.0432,
            "macro_f1"          : 0.0279,
            "weighted_precision": 0.1386,
        },
        "params"      : {
            "max_iter"   : 1000,
            "scaler"     : "StandardScaler",
        },
    },
]

# Shared metadata for all models
SHARED_METADATA = {
    "training_data_version" : "v1.0",
    "training_data_path"    : "processed-data/chicago_crime_features_v1.0.csv",
    "feature_set"           : [
        "hour_sin", "hour_cos", "is_weekend", "is_night",
        "Latitude", "Longitude", "community_area_enc"
    ],
    "target"                : "crime_type_label",
    "n_classes"             : 29,
    "training_rows"         : 500_000,
    "test_rows"             : 100_000,
    "train_test_split"      : 0.8,
    "random_seed"           : 42,
    "limitations"           : [
        "Trained on 500k sample of 7.28M available rows due to compute constraints.",
        "Class imbalance not corrected — rare crime types underperform.",
        "Geographic features may reflect historical policing bias.",
        "Temporal split not enforced in training — future leakage possible in logreg baseline.",
        "Model predicts crime TYPE only, not individuals or locations.",
    ],
}

# SETUP - directories & logging

os.makedirs(os.path.join("data", "phase_ii"), exist_ok=True)
os.makedirs("logs", exist_ok=True)

LOG_FILE = os.path.join("logs", "register_model.log")

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


# STEP 1 - VERIFY MODEL FILES EXIST

def verify_model_files():
    log.info("=" * 60)
    log.info("STEP 1 - Verifying model files...")
    log.info("=" * 60)

    all_ok = True
    for m in MODELS:
        if os.path.exists(m["path"]):
            size_mb = os.path.getsize(m["path"]) / (1024 ** 2)
            log.info(f"  [OK] {m['name']}: {m['path']} ({size_mb:.1f} MB)")
        else:
            log.error(f"  [MISSING] {m['path']} — run crime_modeling.ipynb first.")
            all_ok = False

    if not all_ok:
        raise FileNotFoundError(
            "One or more model files are missing. "
            "Run data/phase_ii/crime_modeling.ipynb to train and save all models."
        )
    return True


# STEP 2 - REGISTER MODELS IN AZURE ML

def register_models_azure():
    """
    Register all models in Azure ML Model Registry using the
    azure-ai-ml SDK (v2). Falls back to azureml-core SDK (v1)
    if v2 is not available.
    """
    log.info("=" * 60)
    log.info("STEP 2 - Registering models in Azure ML...")
    log.info("=" * 60)

    registry_records = []

    try:
        # Try Azure ML SDK v2 first (recommended)
        from azure.ai.ml import MLClient
        from azure.ai.ml.entities import Model
        from azure.ai.ml.constants import AssetTypes
        from azure.identity import DefaultAzureCredential

        log.info("  Using Azure ML SDK v2 (azure-ai-ml)")
        credential = DefaultAzureCredential()
        ml_client  = MLClient(
            credential,
            subscription_id=SUBSCRIPTION_ID,
            resource_group_name=RESOURCE_GROUP,
            workspace_name=WORKSPACE_NAME,
        )
        log.info(f"  Connected to workspace: {WORKSPACE_NAME}")

        for m in MODELS:
            log.info(f"  Registering: {m['name']} ...")
            tags = {
                "algorithm"          : m["algorithm"],
                "accuracy"           : str(round(m["metrics"]["accuracy"], 4)),
                "macro_f1"           : str(round(m["metrics"]["macro_f1"], 4)),
                "balanced_accuracy"  : str(round(m["metrics"]["balanced_accuracy"], 4)),
                "training_data"      : SHARED_METADATA["training_data_version"],
                "features"           : ",".join(SHARED_METADATA["feature_set"]),
                "n_classes"          : str(SHARED_METADATA["n_classes"]),
                "champion"           : str(m["is_champion"]),
            }

            model = Model(
                path        = m["path"],
                name        = m["name"],
                description = m["description"],
                type        = AssetTypes.CUSTOM_MODEL,
                tags        = tags,
            )

            registered = ml_client.models.create_or_update(model)
            log.info(f"  [OK] {m['name']} registered — version: {registered.version}, id: {registered.id}")

            registry_records.append({
                "name"        : m["name"],
                "version"     : str(registered.version),
                "azure_id"    : registered.id,
                "algorithm"   : m["algorithm"],
                "is_champion" : m["is_champion"],
                "metrics"     : m["metrics"],
                "params"      : m["params"],
                "tags"        : tags,
            })

    except ImportError:
        # Fallback: Azure ML SDK v1 (azureml-core)
        log.warning("  azure-ai-ml not found, falling back to azureml-core SDK v1")
        try:
            from azureml.core import Workspace, Model as AzureModel
            from azureml.core.authentication import ServicePrincipalAuthentication

            ws = Workspace(
                subscription_id=SUBSCRIPTION_ID,
                resource_group=RESOURCE_GROUP,
                workspace_name=WORKSPACE_NAME,
            )
            log.info(f"  Connected to workspace: {ws.name}")

            for m in MODELS:
                log.info(f"  Registering: {m['name']} ...")
                tags = {
                    "algorithm"       : m["algorithm"],
                    "accuracy"        : str(round(m["metrics"]["accuracy"], 4)),
                    "macro_f1"        : str(round(m["metrics"]["macro_f1"], 4)),
                    "training_data"   : SHARED_METADATA["training_data_version"],
                    "n_classes"       : str(SHARED_METADATA["n_classes"]),
                    "champion"        : str(m["is_champion"]),
                }

                registered = AzureModel.register(
                    workspace   = ws,
                    model_path  = m["path"],
                    model_name  = m["name"],
                    description = m["description"],
                    tags        = tags,
                )
                log.info(f"  [OK] {m['name']} v{registered.version} registered.")

                registry_records.append({
                    "name"        : m["name"],
                    "version"     : str(registered.version),
                    "algorithm"   : m["algorithm"],
                    "is_champion" : m["is_champion"],
                    "metrics"     : m["metrics"],
                    "params"      : m["params"],
                    "tags"        : tags,
                })

        except Exception as e:
            log.error(f"  Azure ML registration failed: {e}")
            log.warning("  Falling back to LOCAL-ONLY registration record.")
            registry_records = _local_registration_fallback()

    except Exception as e:
        log.error(f"  Azure ML connection failed: {e}")
        log.warning("  Falling back to LOCAL-ONLY registration record.")
        registry_records = _local_registration_fallback()

    return registry_records


def _local_registration_fallback():
    """
    If Azure ML is not reachable, create a local registration record.
    This still satisfies traceability requirements.
    """
    log.info("  Creating local registration record (no Azure ML connection)...")
    records = []
    for m in MODELS:
        size_mb = os.path.getsize(m["path"]) / (1024 ** 2) if os.path.exists(m["path"]) else 0
        records.append({
            "name"        : m["name"],
            "version"     : m["version"],
            "azure_id"    : "local-only",
            "algorithm"   : m["algorithm"],
            "is_champion" : m["is_champion"],
            "metrics"     : m["metrics"],
            "params"      : m["params"],
            "model_size_mb": round(size_mb, 1),
        })
        log.info(f"  [LOCAL] {m['name']} recorded ({size_mb:.1f} MB)")
    return records


# STEP 3 - SAVE LOCAL REGISTRY RECORD

def save_registry_record(registry_records):
    log.info("=" * 60)
    log.info("STEP 3 - Saving local model registry record...")
    log.info("=" * 60)

    record = {
        "registry_version"  : "1.0",
        "created_date"      : datetime.date.today().isoformat(),
        "workspace"         : WORKSPACE_NAME,
        "resource_group"    : RESOURCE_GROUP,
        "shared_metadata"   : SHARED_METADATA,
        "models"            : registry_records,
        "champion_model"    : next(
            (m["name"] for m in registry_records if m.get("is_champion")),
            registry_records[0]["name"]
        ),
    }

    out = os.path.join("data", "phase_ii", "model_registry.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, ensure_ascii=False)
    log.info(f"  Registry record saved -> {out}")
    return record


# STEP 4 - PRINT SUMMARY

def print_summary(registry_records):
    log.info("=" * 60)
    log.info("MODEL REGISTRATION SUMMARY")
    log.info("=" * 60)
    log.info(f"  Models registered  : {len(registry_records)}")
    log.info(f"  Champion model     : {next((m['name'] for m in registry_records if m.get('is_champion')), 'N/A')}")
    log.info("")
    log.info("  Model Comparison:")
    log.info(f"  {'Model':<40} {'Accuracy':>10} {'Macro F1':>10} {'Champion':>10}")
    log.info("  " + "-" * 72)
    for m in registry_records:
        acc  = m["metrics"]["accuracy"]
        f1   = m["metrics"]["macro_f1"]
        chmp = "YES" if m.get("is_champion") else ""
        log.info(f"  {m['name']:<40} {acc:>10.4f} {f1:>10.4f} {chmp:>10}")
    log.info("=" * 60)
    log.info("  Output: data/phase_ii/model_registry.json")
    log.info("  Log:    logs/register_model.log")
    log.info("=" * 60)


# MAIN

def run_registration():
    log.info("=" * 60)
    log.info("TASK II.3 - MODEL REGISTRATION STARTED")
    log.info("=" * 60)

    verify_model_files()
    registry_records = register_models_azure()
    save_registry_record(registry_records)
    print_summary(registry_records)

    log.info("MODEL REGISTRATION COMPLETE")
    return registry_records


if __name__ == "__main__":
    run_registration()