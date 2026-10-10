#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-models/reengagement_clustering/cdp_kmeans.pkl}"
mkdir -p "$(dirname "${MODEL_PATH}")"

if [[ -n "${GCS_BUCKET:-}" ]]; then
  GCS_MODEL_OBJECT="${GCS_MODEL_OBJECT:-artifacts/reengagement_clustering/cdp_kmeans.pkl}"
  python3 - <<PY
import os
from google.cloud import storage

client = storage.Client()
bucket = client.bucket(os.environ["GCS_BUCKET"])
blob = bucket.blob(os.environ.get("GCS_MODEL_OBJECT", "artifacts/reengagement_clustering/cdp_kmeans.pkl"))
local_path = os.environ.get("MODEL_PATH", "models/reengagement_clustering/cdp_kmeans.pkl")
blob.download_to_filename(local_path)
print(f"Downloaded gs://{os.environ['GCS_BUCKET']}/{blob.name} -> {local_path}")
PY
elif [[ ! -f "${MODEL_PATH}" ]]; then
  echo "Model not found at ${MODEL_PATH} and GCS_BUCKET is not set." >&2
  echo "Either mount the pickle at that path, or set GCS_BUCKET (+ optionally GCS_MODEL_OBJECT)." >&2
  exit 1
fi

exec uvicorn serving.clustering_server:app --host 0.0.0.0 --port "${PORT:-8000}"
