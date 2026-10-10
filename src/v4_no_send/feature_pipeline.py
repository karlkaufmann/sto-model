from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

TARGET_HOURS = [7, 8, 9, 10, 11, 12, 15, 16, 19, 20]
HOUR_TO_IDX = {h: i for i, h in enumerate(TARGET_HOURS)}
IDX_TO_HOUR = {i: h for h, i in HOUR_TO_IDX.items()}
N_CLASSES = len(TARGET_HOURS)

BLOCKS = [
    ("7-8h", [7, 8]),
    ("9-10h", [9, 10]),
    ("11-12h", [11, 12]),
    ("15-16h", [15, 16]),
    ("19-20h", [19, 20]),
]

# Pre-send features only. Do not add sent_local/send_hour/send_* or label columns here.
CATEGORICAL_COLS = [
    "tenant_id",
    "mailing_type",
    "recipient_email_domain",
    "recipient_language",
    "recipient_language_alt",
    "recipient_country",
    "recipient_gender",
    "booking_channel_raw",
    "booking_status",
    "booking_currency",
    "custom_booking_channel",
    "custom_booking_device",
    "custom_stay_purpose",
    "custom_room_type",
    "custom_rate_code",
    "custom_recipient_timezone",
    "custom_client_device",
    "custom_client_segment",
    "checkin_stay_purpose",
    "checkin_country",
    "checkin_nationality",
    "checkin_room",
    "campaign_subject_language",
    "campaign_email_format",
    "campaign_template_type",
    "campaign_offer_type",
    "template_type",
]

NUMERIC_COLS = [
    "web_id",
    "client_id",
    "booking_id",
    "template_id",
    "recipient_age_approx",
    "recipient_mail_is_valid",
    "recipient_newsletter_consent_legacy",
    "booking_property_id",
    "booking_created_hour_local",
    "booking_created_dow",
    "booking_lead_time_days",
    "days_to_checkin",
    "stay_nights",
    "booking_price",
    "booking_non_refundable",
    "booking_insured",
    "custom_adults",
    "custom_children",
    "custom_guests",
    "checkin_guest_count",
    "checkin_gdpr_marketing",
    "checkin_gdpr_profile",
    "campaign_target_property_id",
    "campaign_subject_length",
    "campaign_preheader_length",
    "campaign_subject_has_exclamation",
    "campaign_subject_has_question",
    "campaign_ab_test",
    "campaign_spam_score",
    "campaign_recipient_count",
    "template_version",
    "recipient_emails_sent_before",
    "recipient_opens_before",
    "recipient_clicks_before",
    "recipient_delivered_before",
    "recipient_open_rate_before",
    "recipient_click_rate_before",
    "recipient_days_since_last_email",
    "recipient_days_since_last_open",
    "recipient_avg_open_delay_seconds_before",
    "web_visits_count_last_30d",
    "web_pageviews_count_last_30d",
    "hours_since_last_web_visit",
    "web_mobile_visits_last_30d",
    "web_tablet_visits_last_30d",
    "web_desktop_visits_last_30d",
]


@dataclass
class FeatureState:
    category_mappings: dict[str, dict[str, int]]
    numeric_medians: dict[str, float]
    feature_names: list[str]
    top_k: int


def _normalize_text(series: pd.Series) -> pd.Series:
    return (
        series.fillna("unknown")
        .astype(str)
        .str.strip()
        .replace({"": "unknown", "nan": "unknown", "None": "unknown", "NaN": "unknown"})
    )


def fit_feature_state(df: pd.DataFrame, top_k: int = 200) -> FeatureState:
    category_mappings: dict[str, dict[str, int]] = {}
    numeric_medians: dict[str, float] = {}
    feature_names: list[str] = []

    for col in CATEGORICAL_COLS:
        if col not in df.columns:
            continue
        values = _normalize_text(df[col])
        top_values = [v for v in values.value_counts().head(top_k).index.tolist() if v != "unknown"]
        mapping = {"unknown": 0}
        for idx, value in enumerate(sorted(top_values), start=1):
            mapping[value] = idx
        mapping["other"] = len(mapping)
        category_mappings[col] = mapping
        feature_names.append(f"{col}_enc")

    for col in NUMERIC_COLS:
        if col not in df.columns:
            continue
        parsed = pd.to_numeric(df[col], errors="coerce")
        numeric_medians[col] = float(parsed.median()) if parsed.notna().any() else 0.0
        feature_names.extend([col, f"{col}_known"])

    return FeatureState(
        category_mappings=category_mappings,
        numeric_medians=numeric_medians,
        feature_names=feature_names,
        top_k=top_k,
    )


def build_features(df: pd.DataFrame, state: FeatureState) -> pd.DataFrame:
    columns: dict[str, pd.Series] = {}

    for col, mapping in state.category_mappings.items():
        values = _normalize_text(df.get(col, pd.Series("unknown", index=df.index)))
        encoded = values.map(mapping).fillna(mapping["other"]).astype(np.float32)
        columns[f"{col}_enc"] = encoded

    for col, median in state.numeric_medians.items():
        parsed = pd.to_numeric(df.get(col, pd.Series(np.nan, index=df.index)), errors="coerce")
        columns[col] = parsed.fillna(median).astype(np.float32)
        columns[f"{col}_known"] = parsed.notna().astype(np.float32)

    return pd.DataFrame(columns, index=df.index)[state.feature_names].astype(np.float32)
