from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

IDX_TO_HOUR = {
    0: 7,
    1: 8,
    2: 9,
    3: 10,
    4: 11,
    5: 12,
    6: 15,
    7: 16,
    8: 19,
    9: 20,
}
HOUR_TO_IDX = {v: k for k, v in IDX_TO_HOUR.items()}
TARGET_HOURS = sorted(HOUR_TO_IDX.keys())

CATEGORICAL_COLS = ["nationality", "template_id", "mailing_type", "state", "dsn"]
NUMERIC_COLS = ["web_id", "client_id", "previous_stays_count", "sum_previous_stay_days"]

FEATURE_COLS = [
    "sent_hour",
    "sent_dow",
    "sent_day",
    "sent_month",
    "is_weekend",
    "sent_weekofyear",
    "sent_is_month_start",
    "sent_is_month_end",
    "hour_squared",
    "time_to_open",
    "web_id",
    "client_id",
    "previous_stays_count",
    "sum_previous_stay_days",
    "nationality_enc",
    "template_id_enc",
    "mailing_type_enc",
    "state_enc",
    "dsn_enc",
]

DEFAULTS: dict[str, Any] = {
    "time_to_open": 2.0,
    "web_id": 0.0,
    "client_id": 0.0,
    "previous_stays_count": 0.0,
    "sum_previous_stay_days": 0.0,
    "nationality": "unknown",
    "template_id": "unknown",
    "mailing_type": "unknown",
    "state": "unknown",
    "dsn": "unknown",
}


@dataclass
class PreparedTrainingData:
    X: pd.DataFrame
    y: pd.Series
    category_mappings: dict[str, dict[str, int]]
    rows_before: int
    rows_after: int


def _normalize_categorical(series: pd.Series) -> pd.Series:
    normalized = (
        series.fillna("unknown")
        .astype(str)
        .str.strip()
        .replace({"": "unknown", "nan": "unknown", "None": "unknown"})
    )
    return normalized


def fit_category_mappings(df: pd.DataFrame) -> dict[str, dict[str, int]]:
    mappings: dict[str, dict[str, int]] = {}
    for col in CATEGORICAL_COLS:
        values = _normalize_categorical(df.get(col, pd.Series(index=df.index, dtype=object)))
        unique_values = sorted(v for v in values.unique().tolist() if v != "unknown")
        mapping = {"unknown": 0}
        for idx, value in enumerate(unique_values, start=1):
            mapping[value] = idx
        mappings[col] = mapping
    return mappings


def _encode_categorical(values: pd.Series, mapping: dict[str, int]) -> pd.Series:
    normalized = _normalize_categorical(values)
    return normalized.map(mapping).fillna(0).astype(np.float32)


def _as_numeric(series: pd.Series, default: float = 0.0) -> pd.Series:
    parsed = pd.to_numeric(series, errors="coerce")
    if parsed.notna().any():
        fill_value = float(parsed.median())
    else:
        fill_value = default
    return parsed.fillna(fill_value).astype(np.float32)


def build_features(
    df: pd.DataFrame,
    category_mappings: dict[str, dict[str, int]],
) -> pd.DataFrame:
    out = df.copy()

    # Time features (expected to be present for inference; training derives from sent_dt)
    out["sent_hour"] = pd.to_numeric(out.get("sent_hour"), errors="coerce").fillna(-1).astype(np.int16)
    out["sent_dow"] = pd.to_numeric(out.get("sent_dow"), errors="coerce").fillna(-1).astype(np.int16)
    out["sent_day"] = pd.to_numeric(out.get("sent_day"), errors="coerce").fillna(-1).astype(np.int16)
    out["sent_month"] = pd.to_numeric(out.get("sent_month"), errors="coerce").fillna(-1).astype(np.int16)
    out["is_weekend"] = (out["sent_dow"] >= 5).astype(np.int16)
    out["sent_weekofyear"] = pd.to_numeric(out.get("sent_weekofyear"), errors="coerce").fillna(-1).astype(np.int16)
    out["sent_is_month_start"] = pd.to_numeric(out.get("sent_is_month_start"), errors="coerce").fillna(0).astype(np.int16)
    out["sent_is_month_end"] = pd.to_numeric(out.get("sent_is_month_end"), errors="coerce").fillna(0).astype(np.int16)
    out["hour_squared"] = (out["sent_hour"] ** 2).astype(np.float32)

    # Numeric recipient/context features
    out["time_to_open"] = _as_numeric(out.get("time_to_open", DEFAULTS["time_to_open"]), default=DEFAULTS["time_to_open"])
    out["web_id"] = _as_numeric(out.get("web_id", DEFAULTS["web_id"]), default=DEFAULTS["web_id"])
    out["client_id"] = _as_numeric(out.get("client_id", DEFAULTS["client_id"]), default=DEFAULTS["client_id"])
    out["previous_stays_count"] = _as_numeric(
        out.get("previous_stays_count", DEFAULTS["previous_stays_count"]),
        default=DEFAULTS["previous_stays_count"],
    )
    out["sum_previous_stay_days"] = _as_numeric(
        out.get("sum_previous_stay_days", DEFAULTS["sum_previous_stay_days"]),
        default=DEFAULTS["sum_previous_stay_days"],
    )

    for col in CATEGORICAL_COLS:
        values = out.get(col, DEFAULTS[col])
        if not isinstance(values, pd.Series):
            values = pd.Series([values] * len(out), index=out.index)
        out[f"{col}_enc"] = _encode_categorical(values, category_mappings[col])

    return out[FEATURE_COLS].astype(np.float32)


def prepare_training_data(raw_df: pd.DataFrame) -> PreparedTrainingData:
    rows_before = len(raw_df)
    df = raw_df.copy()

    df = df.dropna(subset=["opened"]).copy()
    df["sent_dt"] = pd.to_datetime(df["sent"], errors="coerce", utc=True)
    df["opened_dt"] = pd.to_datetime(df["opened"], errors="coerce", utc=True)
    df = df.dropna(subset=["sent_dt", "opened_dt"]).copy()

    df["open_hour"] = df["opened_dt"].dt.hour
    df = df[df["open_hour"].isin(TARGET_HOURS)].copy()
    df["target"] = df["open_hour"].map(HOUR_TO_IDX).astype(int)

    # Time-derived features from sent timestamp
    df["sent_hour"] = df["sent_dt"].dt.hour.astype(np.int16)
    df["sent_dow"] = df["sent_dt"].dt.dayofweek.astype(np.int16)
    df["sent_day"] = df["sent_dt"].dt.day.astype(np.int16)
    df["sent_month"] = df["sent_dt"].dt.month.astype(np.int16)
    df["sent_weekofyear"] = df["sent_dt"].dt.isocalendar().week.astype(np.int16)
    df["sent_is_month_start"] = df["sent_dt"].dt.is_month_start.astype(np.int16)
    df["sent_is_month_end"] = df["sent_dt"].dt.is_month_end.astype(np.int16)

    # Real signal instead of previous placeholder constant
    delta_hours = (df["opened_dt"] - df["sent_dt"]).dt.total_seconds() / 3600.0
    df["time_to_open"] = delta_hours.clip(lower=0.0, upper=72.0).fillna(DEFAULTS["time_to_open"]).astype(np.float32)

    category_mappings = fit_category_mappings(df)
    X = build_features(df, category_mappings)
    y = df["target"].astype(int)

    return PreparedTrainingData(
        X=X,
        y=y,
        category_mappings=category_mappings,
        rows_before=rows_before,
        rows_after=len(df),
    )
