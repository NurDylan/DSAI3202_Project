import os
from azure.storage.blob import BlobServiceClient
from dotenv import load_dotenv

load_dotenv()

CONN_STR = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
client   = BlobServiceClient.from_connection_string(CONN_STR)

def upload_file(local_path, container, blob_name):
    blob = client.get_blob_client(container=container, blob=blob_name)
    with open(local_path, "rb") as f:
        blob.upload_blob(f, overwrite=True)
    size_mb = os.path.getsize(local_path) / (1024**2)
    print(f"  Uploaded: {blob_name} ({size_mb:.1f} MB)")

print("Uploading raw data...")
upload_file(
    "data/raw/chicago_crime_v1.0_2026-03-16.csv",
    "raw-data",
    "chicago_crime_v1.0_2026-03-16.csv"
)

print("Uploading processed data...")
upload_file(
    "data/processed/chicago_crime_cleaned_v1.0.csv",
    "processed-data",
    "chicago_crime_cleaned_v1.0.csv"
)
upload_file(
    "data/processed/chicago_crime_features_v1.0.csv",
    "processed-data",
    "chicago_crime_features_v1.0.csv"
)

print("Uploading catalog files...")
for f in os.listdir("data/catalog"):
    upload_file(f"data/catalog/{f}", "catalog-data", f)

print("Done. All files uploaded to Azure Blob Storage.")