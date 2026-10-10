#!/usr/bin/env python3
"""
Re-engagement CDP Clustering API Server.

Serves the current production candidate — CDP mixed k-means, k=10
(`src/reengagement_clustering/approaches.py::fit_cdp_kmeans`) — as a REST
API. Contacts are first routed by the same deterministic rule used in the
notebook/pipeline (COLD_CONTACTS / NOT_TARGETABLE / learned cluster);
human-readable cluster names are a static dictionary, never model output.

This server only depends on the reengagement_clustering package. It does
not import or load the email send-time models in src/v2, v3, v3_2,
v4_no_send.
"""

from __future__ import annotations

import os
import pickle
import time
from datetime import datetime
from typing import Any, Optional

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.reengagement_clustering.approaches import FittedApproach, predict_segments
from src.reengagement_clustering.data import add_rule_segments

MODEL_PATH = os.environ.get("MODEL_PATH", "models/reengagement_clustering/cdp_kmeans.pkl")

# Static, hand-checked names from the notebook (`reengagement_clustering_approaches.ipynb`).
# Not LLM-generated, not re-derived at inference time.
CLUSTER_NAMES = {
    0: "Tichí příjemci",
    1: "Středně aktivní CRM kontakty",
    2: "Oslovení bez reakce",
    3: "Častější hosté bez aktuální reakce",
    4: "Aktivní čtenáři",
    5: "Dřívější hosté s vyhasínající odezvou",
    6: "Kontakty bez pobytové historie",
    7: "Dřívější hosté s velmi nízkou odezvou",
    8: "Aktivní klikající hosté",
    9: "Hodnotnější dřívější hosté bez odezvy",
}

app = FastAPI(
    title="Re-engagement Clustering API",
    description="Routes CRM contacts to COLD_CONTACTS / NOT_TARGETABLE / a learned CDP segment (k=10).",
    version="1.0",
)

artifact: Optional[FittedApproach] = None


@app.on_event("startup")
async def load_model() -> None:
    global artifact
    with open(MODEL_PATH, "rb") as f:
        artifact = pickle.load(f)
    n_clusters = artifact.estimator.named_steps["cluster"].n_clusters
    print(f"Loaded {artifact.name} (k={n_clusters}) from {MODEL_PATH}")


class ContactPayload(BaseModel):
    """Raw contact row. Routing fields are explicit; everything else
    (engagement/booking/profile/categorical features consumed by the
    model) is passed through via `extra = allow` and imputed by the
    pipeline the same way missing export values are today."""

    contact_id: Optional[str] = None
    tenant_id: Optional[str] = None
    hotel_id: Optional[int] = None
    reachable: int = 1
    sents_total: float = 0
    stays_total: float = 0
    engagement_state: Optional[str] = "unknown"

    class Config:
        extra = "allow"


class SegmentResponse(BaseModel):
    segment: str
    cluster_id: Optional[int] = None
    cluster_name: Optional[str] = None
    routing_detail: Optional[str] = None
    model_version: str
    inference_time_ms: float
    timestamp: str


@app.get("/health")
async def health_check() -> dict[str, Any]:
    return {
        "status": "ok" if artifact is not None else "model_not_loaded",
        "model": artifact.name if artifact else None,
        "n_clusters": artifact.estimator.named_steps["cluster"].n_clusters if artifact else None,
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }


@app.post("/segment", response_model=SegmentResponse)
async def segment_contact(payload: ContactPayload) -> SegmentResponse:
    if artifact is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    start_time = time.time()
    row = payload.model_dump()
    df = pd.DataFrame([row])
    routed = add_rule_segments(df)

    result = predict_segments(artifact, routed).iloc[0]
    detail = routed["routing_detail"].iloc[0]

    cluster_id = None
    cluster_name = None
    if result.startswith(f"{artifact.name}:"):
        cluster_id = int(result.split(":")[1])
        cluster_name = CLUSTER_NAMES.get(cluster_id)
        detail = None  # only meaningful for the two rule-based buckets

    inference_time = (time.time() - start_time) * 1000.0
    return SegmentResponse(
        segment=result,
        cluster_id=cluster_id,
        cluster_name=cluster_name,
        routing_detail=detail,
        model_version=f"{artifact.name}_k{artifact.estimator.named_steps['cluster'].n_clusters}",
        inference_time_ms=round(inference_time, 2),
        timestamp=datetime.utcnow().isoformat() + "Z",
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8000")))
