from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

TARGET_HOURS = list(range(7, 22))
HOUR_TO_IDX = {h: i for i, h in enumerate(TARGET_HOURS)}
IDX_TO_HOUR = {i: h for h, i in HOUR_TO_IDX.items()}

TOP_K_EMAIL_DOMAIN = 50
TOP_K_COUNTRY = 30

FEATURE_COLS = [
    "campaign_type_enc",
    "email_domain_enc",
    "country_enc",
    "age",
    "age_known",
    "country_known",
]


@dataclass
class PreparedTrainingDataV3:
    X: pd.DataFrame
    y: pd.Series
    category_mappings: dict[str, dict[str, int]]
    age_fill_value: float
    rows_before: int
    rows_after: int
    dropped_rows: dict[str, int]


def _normalize_text(series: pd.Series, unknown: str = "unknown") -> pd.Series:
    return (
        series.fillna(unknown)
        .astype(str)
        .str.strip()
        .replace({"": unknown, "nan": unknown, "None": unknown, "NaN": unknown})
    )


def _parse_age(series: pd.Series) -> tuple[pd.Series, pd.Series]:
    age = pd.to_numeric(series, errors="coerce")
    age = age.where((age >= 0) & (age <= 100))
    age_known = age.notna().astype(np.float32)
    return age.astype(np.float32), age_known


def _fit_topk_mapping(series: pd.Series, top_k: int) -> dict[str, int]:
    counts = series.value_counts(dropna=False)
    top_values = [str(v) for v in counts.head(top_k).index.tolist() if str(v) != "unknown"]
    mapping = {"unknown": 0}
    for idx, value in enumerate(top_values, start=1):
        mapping[value] = idx
    mapping["other"] = len(mapping)
    return mapping


def _encode_with_other(series: pd.Series, mapping: dict[str, int]) -> pd.Series:
    other_code = mapping["other"]
    encoded = series.map(mapping).fillna(other_code).astype(np.float32)
    return encoded


def fit_category_mappings(df: pd.DataFrame) -> dict[str, dict[str, int]]:
    normalized_domain = _normalize_text(df["email_domain"])
    normalized_country = _normalize_text(df["country"])
    normalized_campaign = _normalize_text(df["campaign_type"])

    campaign_values = sorted(v for v in normalized_campaign.unique().tolist() if v != "unknown")
    campaign_map = {"unknown": 0}
    for idx, value in enumerate(campaign_values, start=1):
        campaign_map[value] = idx

    mappings = {
        "email_domain": _fit_topk_mapping(normalized_domain, TOP_K_EMAIL_DOMAIN),
        "country": _fit_topk_mapping(normalized_country, TOP_K_COUNTRY),
        "campaign_type": campaign_map,
    }
    return mappings


def build_features(
    df: pd.DataFrame,
    category_mappings: dict[str, dict[str, int]],
    age_fill_value: float,
) -> pd.DataFrame:
    out = df.copy()

    campaign = _normalize_text(out.get("campaign_type", "unknown"))
    domain = _normalize_text(out.get("email_domain", "unknown"))
    country = _normalize_text(out.get("country", "unknown"))
    age_raw, age_known = _parse_age(out.get("age"))

    out["campaign_type_enc"] = campaign.map(category_mappings["campaign_type"]).fillna(0).astype(np.float32)
    out["email_domain_enc"] = _encode_with_other(domain, category_mappings["email_domain"])
    out["country_enc"] = _encode_with_other(country, category_mappings["country"])
    out["country_known"] = (country != "unknown").astype(np.float32)
    out["age_known"] = age_known
    out["age"] = age_raw.fillna(age_fill_value).astype(np.float32)

    return out[FEATURE_COLS].astype(np.float32)


def prepare_training_data_v3(raw_df: pd.DataFrame) -> PreparedTrainingDataV3:
    rows_before = len(raw_df)
    df = raw_df.copy()

    # Standardize columns used for row filtering.
    df["bounced_norm"] = _normalize_text(df.get("bounced", "unknown")).str.lower()
    unsub = pd.to_numeric(df.get("unsubscribed", 0), errors="coerce").fillna(0).astype(int)
    hour = pd.to_numeric(df.get("open_hour_local"), errors="coerce")

    dropped_bounced = int((df["bounced_norm"] != "none").sum())
    dropped_unsubscribed = int((unsub == 1).sum())
    dropped_outside_hour = int((~hour.isin(TARGET_HOURS)).sum())

    keep_mask = (df["bounced_norm"] == "none") & (unsub != 1) & (hour.isin(TARGET_HOURS))
    df = df[keep_mask].copy()
    df["open_hour_local"] = hour[keep_mask].astype(int)
    df["target"] = df["open_hour_local"].map(HOUR_TO_IDX).astype(int)

    category_mappings = fit_category_mappings(df)
    age_raw, _ = _parse_age(df.get("age"))
    age_fill = float(age_raw.median()) if age_raw.notna().any() else 40.0
    X = build_features(df, category_mappings=category_mappings, age_fill_value=age_fill)
    y = df["target"].astype(int)

    return PreparedTrainingDataV3(
        X=X,
        y=y,
        category_mappings=category_mappings,
        age_fill_value=age_fill,
        rows_before=rows_before,
        rows_after=len(df),
        dropped_rows={
            "bounced_not_none": dropped_bounced,
            "unsubscribed_equals_1": dropped_unsubscribed,
            "open_hour_outside_7_21": dropped_outside_hour,
        },
    )
