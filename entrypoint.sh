#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${GCS_BUCKET:-}" ]]; then
  echo "GCS_BUCKET is required."
  exit 1
fi

GCS_DATA_OBJECT="${GCS_DATA_OBJECT:-data/processed/unified_v3_2.parquet}"
LOCAL_DATA_PATH="${LOCAL_DATA_PATH:-data/processed/unified_v3_2.parquet}"
GCS_OUTPUT_PREFIX="${GCS_OUTPUT_PREFIX:-artifacts/v3_2}"
TRAIN_MODULE="${TRAIN_MODULE:-src.v3_2.train}"
GCS_ARTIFACTS="${GCS_ARTIFACTS:-models/v3_2/model.pkl,models/v3_2/model.onnx,results/metrics/v3_2.json,configs/v3_2/features.json,results/comparisons/v2_vs_v3_vs_v3_2.json,results/comparisons/v3_2_baseline.json,results/comparisons/v3_2_features.json}"

mkdir -p "$(dirname "${LOCAL_DATA_PATH}")" models/v3_2 models/v4_no_send results/metrics results/comparisons configs/v3_2 configs/v4_no_send

python3 - <<'PY'
import os
from google.cloud import storage

bucket_name = os.environ["GCS_BUCKET"]
object_name = os.environ.get("GCS_DATA_OBJECT", "data/processed/unified_v3_2.parquet")
local_path = os.environ.get("LOCAL_DATA_PATH", "data/processed/unified_v3_2.parquet")

client = storage.Client()
bucket = client.bucket(bucket_name)
blob = bucket.blob(object_name)
blob.download_to_filename(local_path)
print(f"Downloaded gs://{bucket_name}/{object_name} -> {local_path}")
PY

python3 -m "${TRAIN_MODULE}"

python3 - <<'PY'
import os
from pathlib import Path
from google.cloud import storage

bucket_name = os.environ["GCS_BUCKET"]
prefix = os.environ.get("GCS_OUTPUT_PREFIX", "artifacts/v3_2").strip("/")
artifacts = [
    item.strip()
    for item in os.environ.get("GCS_ARTIFACTS", "").split(",")
    if item.strip()
]

client = storage.Client()
bucket = client.bucket(bucket_name)

for rel in artifacts:
    p = Path(rel)
    if not p.exists():
        print(f"Skipping missing artifact: {rel}")
        continue
    remote = f"{prefix}/{p.name}"
    bucket.blob(remote).upload_from_filename(str(p))
    print(f"Uploaded {rel} -> gs://{bucket_name}/{remote}")
PY
