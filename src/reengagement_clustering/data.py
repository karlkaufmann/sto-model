from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

RAW_ZIP = Path("data/raw/clustering.zip")

ID_COLS = ["tenant_id", "hotel_id", "contact_id"]

NUMERIC_FEATURES = [
    "age",
    "age_known",
    "registration_age_days",
    "consent_marketing_active",
    "consent_profiling_active",
    "consent_doubleoptin_active",
    "newsletter_optout_active",
    "excluded",
    "stays_total",
    "stays_confirmed",
    "stays_past",
    "stays_future",
    "stays_365d",
    "bookings_created_90d",
    "bookings_created_365d",
    "days_since_last_stay",
    "days_to_next_stay",
    "nights_total",
    "currency_count",
    "lead_time_avg_days",
    "nonrefundable_share",
    "insured_share",
    "days_since_last_booking",
    "hotels_count_contact",
    "booking_custom_params_count",
    "sents_total",
    "sents_30d",
    "sents_90d",
    "sents_365d",
    "delivered_90d",
    "opens_total",
    "opens_30d",
    "opens_90d",
    "opens_ts_90d",
    "dsn_90d",
    "dsn_hard_total",
    "clicks_total",
    "clicks_30d",
    "clicks_90d",
    "clicked_links_distinct",
    "clicked_campaigns_distinct",
    "days_since_last_send",
    "days_since_last_open",
    "days_since_last_click",
    "campaigns_distinct",
    "autocampaigns_distinct",
    "open_rate_30d",
    "open_rate_90d",
    "open_rate_lifetime",
    "click_rate_90d",
    "dsn_rate_90d",
    "delivery_rate_90d",
    "rate_delta_30_vs_90",
    "signals_count",
    "recency_stay_score",
    "rank_spend_total",
    "rank_stays_total",
    "rank_nights_total",
    "rank_sents_total",
    "rank_opens_total",
    "rank_clicks_total",
    "rank_open_rate_90d",
    "rank_recency_stay",
]

CATEGORICAL_FEATURES = [
    "language",
    "country",
    "gender",
    "email_domain",
    "booking_channel",
    "booking_source_code",
    "booking_room_type",
    "currency_dominant",
    "mailing_types",
    "engagement_state",
]

CONTEXT_COLS = [
    "as_of_date",
    "hotel_unassigned",
    "has_email",
    "email_valid",
    "reachable",
    "regime_bucket",
    "is_clusterable",
    "spend_total",
]

READ_COLS = ID_COLS + CONTEXT_COLS + NUMERIC_FEATURES + CATEGORICAL_FEATURES

RATE_COLS = [
    "open_rate_30d",
    "open_rate_90d",
    "open_rate_lifetime",
    "click_rate_90d",
    "dsn_rate_90d",
    "delivery_rate_90d",
]

DAY_COLS = [
    "registration_age_days",
    "days_since_last_stay",
    "days_to_next_stay",
    "days_since_last_booking",
    "days_since_last_send",
    "days_since_last_open",
    "days_since_last_click",
]

COUNT_COLS = [
    "stays_total",
    "stays_confirmed",
    "stays_past",
    "stays_future",
    "stays_365d",
    "bookings_created_90d",
    "bookings_created_365d",
    "nights_total",
    "sents_total",
    "sents_30d",
    "sents_90d",
    "sents_365d",
    "delivered_90d",
    "opens_total",
    "opens_30d",
    "opens_90d",
    "opens_ts_90d",
    "dsn_90d",
    "dsn_hard_total",
    "clicks_total",
    "clicks_30d",
    "clicks_90d",
    "clicked_links_distinct",
    "clicked_campaigns_distinct",
    "campaigns_distinct",
    "autocampaigns_distinct",
]


def load_clustering_export(zip_path: str | Path = RAW_ZIP, columns: list[str] | None = None) -> pd.DataFrame:
    selected = columns or READ_COLS
    frames: list[pd.DataFrame] = []
    with zipfile.ZipFile(zip_path) as zf:
        members = [
            name
            for name in zf.namelist()
            if name.endswith(".parquet") and not name.endswith("_schema.json")
        ]
        for name in members:
            with zf.open(name) as fh:
                available = [col for col in selected if col in pq.ParquetFile(fh).schema_arrow.names]
            with zf.open(name) as fh:
                frames.append(pq.read_table(fh, columns=available).to_pandas())
    if not frames:
        raise RuntimeError(f"No parquet files found in {zip_path}")
    return pd.concat(frames, ignore_index=True)


def add_rule_segments(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    reachable = pd.to_numeric(out.get("reachable", 0), errors="coerce").fillna(0).astype(int).eq(1)
    sents_total = pd.to_numeric(out.get("sents_total", 0), errors="coerce").fillna(0)
    engagement_state = out.get("engagement_state", pd.Series("", index=out.index)).fillna("").astype(str)
    stays_total = pd.to_numeric(out.get("stays_total", 0), errors="coerce").fillna(0)
    has_stay_history = stays_total > 0

    out["routing_segment"] = "LEARNED_CLUSTER"
    out.loc[~reachable, "routing_segment"] = "NOT_TARGETABLE"
    out.loc[reachable & ((sents_total <= 0) | engagement_state.eq("never_sent")), "routing_segment"] = "COLD_CONTACTS"
    out["fit_candidate"] = out["routing_segment"].eq("LEARNED_CLUSTER")

    # Heuristic sub-label for the two rule-based buckets, based on booking
    # history alone. Purely descriptive (reporting/labeling) - does not
    # change routing_segment, fit_candidate, or anything the k-means model
    # sees, so it never needs the clustering itself to change.
    is_cold = out["routing_segment"].eq("COLD_CONTACTS")
    is_not_targetable = out["routing_segment"].eq("NOT_TARGETABLE")
    out["routing_detail"] = out["routing_segment"]
    out.loc[is_cold & has_stay_history, "routing_detail"] = "COLD_WITH_HISTORY"
    out.loc[is_cold & ~has_stay_history, "routing_detail"] = "COLD_UNKNOWN"
    out.loc[is_not_targetable & has_stay_history, "routing_detail"] = "UNREACHABLE_WITH_HISTORY"
    out.loc[is_not_targetable & ~has_stay_history, "routing_detail"] = "UNREACHABLE_UNKNOWN"
    return out


def prepare_features(df: pd.DataFrame, categorical_top_k: int = 40) -> pd.DataFrame:
    out = df.copy()
    for col in NUMERIC_FEATURES:
        if col not in out.columns:
            out[col] = np.nan
        out[col] = pd.to_numeric(out[col], errors="coerce")

    for col in RATE_COLS:
        out[col] = out[col].clip(lower=0, upper=1)

    for col in DAY_COLS:
        out[col] = out[col].clip(lower=0, upper=3650)

    for col in COUNT_COLS:
        out[f"log1p_{col}"] = np.log1p(out[col].clip(lower=0))

    for col in CATEGORICAL_FEATURES:
        if col not in out.columns:
            out[col] = "unknown"
        values = out[col].fillna("unknown").astype(str).str.strip().replace({"": "unknown"})
        top = set(values.value_counts(dropna=False).head(categorical_top_k).index)
        out[col] = values.where(values.isin(top), "other")

    return out


def guest_ref(row: pd.Series) -> str:
    raw = f"{row.get('tenant_id')}|{row.get('hotel_id')}|{row.get('contact_id')}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]


def training_mask(df: pd.DataFrame) -> pd.Series:
    if "fit_candidate" not in df.columns:
        df = add_rule_segments(df)
    return df["fit_candidate"].astype(bool)
