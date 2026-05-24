#!/usr/bin/env python3
"""
Email Send Time Prediction API Server (v2)
FastAPI server for ONNX inference with shared feature pipeline.
"""

from datetime import datetime
import json
import pickle
import time
from typing import Optional

import numpy as np
import onnxruntime as ort
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

from src.v2.feature_pipeline import DEFAULTS, FEATURE_COLS, IDX_TO_HOUR, build_features
from src.v3.feature_pipeline import (
    FEATURE_COLS as FEATURE_COLS_V3,
    IDX_TO_HOUR as IDX_TO_HOUR_V3,
    build_features as build_features_v3,
)
from src.v3_2.feature_pipeline import (
    FEATURE_COLS as FEATURE_COLS_V3_2,
    IDX_TO_HOUR as IDX_TO_HOUR_V3_2,
    TARGET_HOURS_V2 as TARGET_HOURS_V3_2,
    build_features as build_features_v3_2,
    extract_email_domain,
    normalize_iso2,
    normalize_campaign_type,
    _get_provider_category,
    _get_tld,
)

app = FastAPI(
    title="Email Send Time Prediction API",
    description="Predicts optimal email send hour using XGBoost ONNX model",
    version="2.0",
)

# Paths
MODEL_PATH = "models/v2/model.onnx"
SCALER_PATH = "models/v2/scaler.pkl"
METADATA_PATH = "results/metrics/v2.json"
FEATURES_PATH = "configs/v2/features.json"
MODEL_PATH_V3 = "models/v3/model.onnx"
SCALER_PATH_V3 = "models/v3/scaler.pkl"
METADATA_PATH_V3 = "results/metrics/v3.json"
FEATURES_PATH_V3 = "configs/v3/features.json"
MODEL_PATH_V3_2 = "models/v3_2/model.onnx"
METADATA_PATH_V3_2 = "results/metrics/v3_2.json"
FEATURES_PATH_V3_2 = "configs/v3_2/features.json"

# Globals
session = None
scaler = None
metadata = None
features_config = None
session_v3 = None
scaler_v3 = None
metadata_v3 = None
features_config_v3 = None
session_v3_2 = None
metadata_v3_2 = None
features_config_v3_2 = None

DEFAULT_CLASS_MAPPING = {str(k): v for k, v in IDX_TO_HOUR.items()}


@app.on_event("startup")
async def load_model():
    global session, scaler, metadata, features_config
    global session_v3, scaler_v3, metadata_v3, features_config_v3
    global session_v3_2, metadata_v3_2, features_config_v3_2

    print("Loading model...")
    session = ort.InferenceSession(MODEL_PATH)

    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
    print(f"Loaded scaler: {type(scaler).__name__}")

    with open(METADATA_PATH, "r") as f:
        metadata = json.load(f)
    print("Loaded metadata.")

    with open(FEATURES_PATH, "r") as f:
        features_config = json.load(f)
    print("Loaded feature config.")

    # Optional V3 stack. Keep V2 startup robust even if V3 files are missing.
    try:
        session_v3 = ort.InferenceSession(MODEL_PATH_V3)
        with open(SCALER_PATH_V3, "rb") as f:
            scaler_v3 = pickle.load(f)
        with open(METADATA_PATH_V3, "r") as f:
            metadata_v3 = json.load(f)
        with open(FEATURES_PATH_V3, "r") as f:
            features_config_v3 = json.load(f)
        print("Loaded V3 model stack.")
    except FileNotFoundError:
        session_v3 = None
        scaler_v3 = None
        metadata_v3 = None
        features_config_v3 = None
        print("V3 model stack not found. /predict/v3 will be unavailable.")

    # Optional V3.2 stack (no scaler — XGBoost NaN-native).
    try:
        session_v3_2 = ort.InferenceSession(MODEL_PATH_V3_2)
        with open(METADATA_PATH_V3_2, "r") as f:
            metadata_v3_2 = json.load(f)
        with open(FEATURES_PATH_V3_2, "r") as f:
            features_config_v3_2 = json.load(f)
        print("Loaded V3.2 model stack.")
    except FileNotFoundError:
        session_v3_2 = None
        metadata_v3_2 = None
        features_config_v3_2 = None
        print("V3.2 model stack not found. /predict/v3_2 will be unavailable.")

    print("All components ready.")


class PredictionRequest(BaseModel):
    # Existing fields (backward compatibility)
    sent_hour: int
    sent_dow: int
    sent_day: int
    sent_month: int
    is_weekend: int = 0
    hour_squared: Optional[float] = None
    time_to_open: Optional[float] = DEFAULTS["time_to_open"]
    web_id: Optional[int] = 0
    client_id: Optional[int] = 0
    client_hash: Optional[str] = ""

    # New recipient/context fields
    nationality: Optional[str] = DEFAULTS["nationality"]
    template_id: Optional[str] = DEFAULTS["template_id"]
    mailing_type: Optional[str] = DEFAULTS["mailing_type"]
    state: Optional[str] = DEFAULTS["state"]
    dsn: Optional[str] = DEFAULTS["dsn"]
    previous_stays_count: Optional[float] = DEFAULTS["previous_stays_count"]
    sum_previous_stay_days: Optional[float] = DEFAULTS["sum_previous_stay_days"]
    sent_weekofyear: Optional[int] = None
    sent_is_month_start: Optional[int] = None
    sent_is_month_end: Optional[int] = None


class PredictionResponse(BaseModel):
    predicted_hour: int
    confidence: float
    block: str
    inference_time_ms: float
    timestamp: str
    model_version: str


class PredictionRequestV3(BaseModel):
    email_domain: str
    country: Optional[str] = "unknown"
    age: Optional[float] = None
    campaign_type: Optional[str] = "newsletter"


class PredictionResponseV3(BaseModel):
    predicted_hour: int
    top3_hours: list[int]
    confidence: float
    inference_time_ms: float
    timestamp: str
    model_version: str


def get_block(hour: int) -> str:
    if hour in [7, 8]:
        return "7-8h"
    if hour in [9, 10]:
        return "9-10h"
    if hour in [11, 12]:
        return "11-12h"
    if hour in [15, 16]:
        return "15-16h"
    if hour in [19, 20]:
        return "19-20h"
    return f"{hour}h"


def _request_to_feature_row(request: PredictionRequest) -> pd.DataFrame:
    sent_weekofyear = request.sent_weekofyear
    if sent_weekofyear is None:
        # best effort fallback from sent_day only (limited info in current API contract)
        sent_weekofyear = max(1, min(53, int((request.sent_day - 1) // 7 + 1)))

    sent_is_month_start = request.sent_is_month_start
    if sent_is_month_start is None:
        sent_is_month_start = 1 if request.sent_day <= 3 else 0

    sent_is_month_end = request.sent_is_month_end
    if sent_is_month_end is None:
        sent_is_month_end = 1 if request.sent_day >= 28 else 0

    row = {
        "sent_hour": request.sent_hour,
        "sent_dow": request.sent_dow,
        "sent_day": request.sent_day,
        "sent_month": request.sent_month,
        "is_weekend": request.is_weekend,
        "sent_weekofyear": sent_weekofyear,
        "sent_is_month_start": sent_is_month_start,
        "sent_is_month_end": sent_is_month_end,
        "time_to_open": request.time_to_open if request.time_to_open is not None else DEFAULTS["time_to_open"],
        "web_id": request.web_id if request.web_id is not None else DEFAULTS["web_id"],
        "client_id": request.client_id if request.client_id is not None else DEFAULTS["client_id"],
        "previous_stays_count": request.previous_stays_count
        if request.previous_stays_count is not None
        else DEFAULTS["previous_stays_count"],
        "sum_previous_stay_days": request.sum_previous_stay_days
        if request.sum_previous_stay_days is not None
        else DEFAULTS["sum_previous_stay_days"],
        "nationality": request.nationality or DEFAULTS["nationality"],
        "template_id": request.template_id or DEFAULTS["template_id"],
        "mailing_type": request.mailing_type or DEFAULTS["mailing_type"],
        "state": request.state or DEFAULTS["state"],
        "dsn": request.dsn or DEFAULTS["dsn"],
    }
    return pd.DataFrame([row])


@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "model": f"xgboost_{metadata.get('version', '2.0')}",
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }


@app.get("/metrics")
async def get_metrics():
    return {
        "overall_accuracy": metadata.get("accuracy", 0.0),
        "weighted_f1": metadata.get("weighted_f1", 0.0),
        "block_accuracies": metadata.get("block_accuracies", {}),
        "model_version": metadata.get("version", "2.0"),
        "features": features_config.get("feature_names", FEATURE_COLS),
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    start_time = time.time()
    try:
        row_df = _request_to_feature_row(request)
        cat_mappings = metadata.get("category_mappings", {})
        feature_df = build_features(row_df, cat_mappings)
        features_scaled = scaler.transform(feature_df[FEATURE_COLS])
        features = np.asarray(features_scaled, dtype=np.float32)

        input_name = session.get_inputs()[0].name
        output = session.run(None, {input_name: features})

        predicted_class = int(output[0][0])
        class_mapping = metadata.get("class_mapping") or DEFAULT_CLASS_MAPPING
        predicted_hour = class_mapping.get(str(predicted_class))
        if predicted_hour is None:
            predicted_hour = IDX_TO_HOUR.get(predicted_class, 10)

        confidence = 0.65
        if len(output) > 1 and output[1] is not None:
            probs_arr = output[1][0]
            if isinstance(probs_arr, dict):
                confidence = float(max(probs_arr.values()))
            else:
                exp = np.exp(probs_arr - np.max(probs_arr))
                probs = exp / exp.sum()
                confidence = float(probs[predicted_class])

        inference_time = (time.time() - start_time) * 1000
        return PredictionResponse(
            predicted_hour=int(predicted_hour),
            confidence=round(confidence, 4),
            block=get_block(int(predicted_hour)),
            inference_time_ms=round(inference_time, 2),
            timestamp=datetime.utcnow().isoformat() + "Z",
            model_version=metadata.get("version", "2.0"),
        )
    except Exception as e:
        inference_time = (time.time() - start_time) * 1000
        print(f"Prediction error: {e}, using fallback")
        return PredictionResponse(
            predicted_hour=10,
            confidence=0.0,
            block="9-10h",
            inference_time_ms=round(inference_time, 2),
            timestamp=datetime.utcnow().isoformat() + "Z",
            model_version="2.0-fallback",
        )


@app.post("/predict/batch")
async def predict_batch(requests: list[PredictionRequest]):
    results = []
    for req in requests:
        results.append(await predict(req))
    return results


@app.post("/predict/v3", response_model=PredictionResponseV3)
async def predict_v3(request: PredictionRequestV3):
    start_time = time.time()
    if session_v3 is None or scaler_v3 is None or metadata_v3 is None:
        raise RuntimeError("V3 artifacts not loaded. Train/export V3 first.")

    row = pd.DataFrame(
        [
            {
                "email_domain": request.email_domain,
                "country": request.country,
                "age": request.age,
                "campaign_type": request.campaign_type,
            }
        ]
    )

    feature_df = build_features_v3(
        row,
        category_mappings=metadata_v3["category_mappings"],
        age_fill_value=float(metadata_v3.get("age_fill_value", 40.0)),
    )
    features_scaled = scaler_v3.transform(feature_df[FEATURE_COLS_V3])
    features = np.asarray(features_scaled, dtype=np.float32)

    input_name = session_v3.get_inputs()[0].name
    output = session_v3.run(None, {input_name: features})

    pred_class = int(output[0][0])
    probs_raw = output[1][0]
    if isinstance(probs_raw, dict):
        probs = np.array([float(probs_raw.get(i, 0.0)) for i in range(len(IDX_TO_HOUR_V3))], dtype=np.float64)
    else:
        exp = np.exp(probs_raw - np.max(probs_raw))
        probs = exp / exp.sum()

    top3_idx = np.argsort(-probs)[:3].tolist()
    top3_hours = [int(IDX_TO_HOUR_V3[i]) for i in top3_idx]
    predicted_hour = int(IDX_TO_HOUR_V3.get(pred_class, top3_hours[0]))
    confidence = float(probs[pred_class]) if pred_class < len(probs) else float(probs[top3_idx[0]])

    inference_time = (time.time() - start_time) * 1000.0
    return PredictionResponseV3(
        predicted_hour=predicted_hour,
        top3_hours=top3_hours,
        confidence=round(confidence, 4),
        inference_time_ms=round(inference_time, 2),
        timestamp=datetime.utcnow().isoformat() + "Z",
        model_version=metadata_v3.get("version", "3.0"),
    )


class PredictionRequestV3_2(BaseModel):
    email_domain: str
    campaign_type: Optional[str] = "newsletter"
    country: Optional[str] = None
    age: Optional[float] = None
    # Optional sent_* — when provided, model uses full feature set (sent_known=1)
    sent_hour: Optional[int] = None
    sent_dow: Optional[int] = None
    sent_day: Optional[int] = None
    sent_month: Optional[int] = None
    sent_weekofyear: Optional[int] = None
    sent_is_month_start: Optional[int] = None
    sent_is_month_end: Optional[int] = None
    time_to_open: Optional[float] = None
    previous_stays_count: Optional[float] = None
    sum_previous_stay_days: Optional[float] = None


class PredictionResponseV3_2(BaseModel):
    predicted_hour: int
    top3_hours: list[int]
    calibrated_probs: dict[str, float]
    confidence: float
    sent_known: bool
    inference_time_ms: float
    timestamp: str
    model_version: str


def _temperature_scale(probs: np.ndarray, T: float) -> np.ndarray:
    logits = np.log(np.clip(probs, 1e-12, None))
    scaled = logits / T
    exp = np.exp(scaled - scaled.max())
    return exp / exp.sum()


@app.post("/predict/v3_2", response_model=PredictionResponseV3_2)
async def predict_v3_2(request: PredictionRequestV3_2):
    start_time = time.time()
    if session_v3_2 is None or metadata_v3_2 is None:
        raise RuntimeError("V3.2 artifacts not loaded. Run train_v3_2.py first.")

    domain = extract_email_domain(request.email_domain) if "@" in (request.email_domain or "") else (request.email_domain or "unknown")
    country_iso2 = normalize_iso2(request.country) if request.country else "unknown"
    campaign = normalize_campaign_type(request.campaign_type) if request.campaign_type else "newsletter"

    row = {
        "email_domain": domain,
        "campaign_type": campaign,
        "country_iso2": country_iso2,
        "age": request.age,
        "previous_stays_count": request.previous_stays_count,
        "sum_previous_stay_days": request.sum_previous_stay_days,
        "sent_hour": request.sent_hour,
        "sent_dow": request.sent_dow,
        "sent_day": request.sent_day,
        "sent_month": request.sent_month,
        "sent_weekofyear": request.sent_weekofyear,
        "sent_is_month_start": request.sent_is_month_start,
        "sent_is_month_end": request.sent_is_month_end,
        "time_to_open": request.time_to_open,
        "dataset_source": "V3",
        "email_provider_category": _get_provider_category(domain),
        "email_tld": _get_tld(domain),
    }
    df_row = pd.DataFrame([row])

    cat_maps = metadata_v3_2["category_mappings"]
    domain_priors: dict = {}
    campaign_priors: dict = metadata_v3_2.get("campaign_priors", {"_GLOBAL_": [0.1] * 10})
    country_mean_hour_map: dict = metadata_v3_2.get("country_mean_hour_map", {})
    country_mean_hour_global: float = float(metadata_v3_2.get("country_mean_hour_global", 12.0))

    feature_df = build_features_v3_2(
        df_row, cat_maps, domain_priors, campaign_priors,
        country_mean_hour_map=country_mean_hour_map,
        country_mean_hour_global=country_mean_hour_global,
    )
    features = np.asarray(feature_df[FEATURE_COLS_V3_2].to_numpy(), dtype=np.float32)

    input_name = session_v3_2.get_inputs()[0].name
    output = session_v3_2.run(None, {input_name: features})

    pred_class = int(output[0][0])
    probs_raw = np.array(output[1][0], dtype=np.float64) if len(output) > 1 else np.ones(10) / 10

    T = float(metadata_v3_2.get("calibration_temperature", 1.0))
    probs_cal = _temperature_scale(probs_raw, T)

    top3_idx = np.argsort(-probs_cal)[:3].tolist()
    predicted_hour = int(IDX_TO_HOUR_V3_2.get(pred_class, TARGET_HOURS_V3_2[0]))
    top3_hours = [int(IDX_TO_HOUR_V3_2[i]) for i in top3_idx]
    cal_probs_dict = {str(int(IDX_TO_HOUR_V3_2[i])): round(float(probs_cal[i]), 4) for i in range(len(TARGET_HOURS_V3_2))}
    confidence = float(probs_cal[pred_class]) if pred_class < len(probs_cal) else float(probs_cal[top3_idx[0]])

    sent_known_flag = request.sent_hour is not None

    inference_time = (time.time() - start_time) * 1000.0
    return PredictionResponseV3_2(
        predicted_hour=predicted_hour,
        top3_hours=top3_hours,
        calibrated_probs=cal_probs_dict,
        confidence=round(confidence, 4),
        sent_known=sent_known_flag,
        inference_time_ms=round(inference_time, 2),
        timestamp=datetime.utcnow().isoformat() + "Z",
        model_version=metadata_v3_2.get("version", "3.2"),
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
