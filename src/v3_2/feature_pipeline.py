"""
Feature pipeline for V3.2 unified model (V2 + V3 datasets).

Shared constants and build_features() are used by both merge_v2_v3.py (offline)
and train_v3_2.py (training) and inference_server.py (online).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

# ── Target ────────────────────────────────────────────────────────────────────
TARGET_HOURS_V2 = [7, 8, 9, 10, 11, 12, 15, 16, 19, 20]
HOUR_TO_IDX = {h: i for i, h in enumerate(TARGET_HOURS_V2)}
IDX_TO_HOUR = {i: h for h, i in HOUR_TO_IDX.items()}
N_CLASSES = len(TARGET_HOURS_V2)

# V2-style time blocks for evaluation
BLOCKS = [("7-8h", [7, 8]), ("9-10h", [9, 10]), ("11-12h", [11, 12]),
          ("15-16h", [15, 16]), ("19-20h", [19, 20])]

# ── Feature columns ───────────────────────────────────────────────────────────
# NaN-able sent_* columns (present for V2 rows, NaN for V3 rows)
SENT_FEATURES = [
    "sent_hour", "sent_dow", "sent_day", "sent_month",
    "sent_weekofyear", "sent_is_month_start", "sent_is_month_end",
    "time_to_open",
]
# Profile features (both V2 and V3)
PROFILE_FEATURES = [
    "campaign_type_enc",
    "country_iso2_enc",
    "email_domain_enc",
    "email_provider_category_enc",
    "email_tld_enc",
    "age",
    "age_known",
    "age_segment_enc",
    "domain_type_enc",
    "domain_type_x_campaign_enc",
    "country_mean_hour",
    "previous_stays_count",
    "sum_previous_stay_days",
    "sent_known",
    "dataset_source_enc",
]
# Domain priors (10 floats, one per target hour)
DOMAIN_PRIOR_FEATURES = [f"domain_prior_h{h}" for h in TARGET_HOURS_V2]

FEATURE_COLS = SENT_FEATURES + PROFILE_FEATURES + DOMAIN_PRIOR_FEATURES

# ── ISO-2 mapping for V2 nationality ─────────────────────────────────────────
ISO_NORM: dict[str, str] = {
    # ISO-3 → ISO-2
    "AFG": "AF", "ALB": "AL", "ARE": "AE", "ARG": "AR", "ARM": "AM",
    "AUS": "AU", "AUT": "AT", "AZE": "AZ", "BEL": "BE", "BGR": "BG",
    "BHR": "BH", "BLR": "BY", "BRA": "BR", "CAN": "CA", "CHE": "CH",
    "CHL": "CL", "CHN": "CN", "COL": "CO", "CYP": "CY", "CZE": "CZ",
    "DEU": "DE", "DNK": "DK", "EGY": "EG", "ESP": "ES", "EST": "EE",
    "FIN": "FI", "FRA": "FR", "GBR": "GB", "GEO": "GE", "GRC": "GR",
    "HRV": "HR", "HUN": "HU", "IND": "IN", "IRL": "IE", "IRN": "IR",
    "IRQ": "IQ", "ISL": "IS", "ISR": "IL", "ITA": "IT", "JOR": "JO",
    "JPN": "JP", "KAZ": "KZ", "KOR": "KR", "KWT": "KW", "LBN": "LB",
    "LBY": "LY", "LTU": "LT", "LUX": "LU", "LVA": "LV", "MAR": "MA",
    "MDA": "MD", "MEX": "MX", "MNE": "ME", "MYS": "MY", "NLD": "NL",
    "NOR": "NO", "NZL": "NZ", "OMN": "OM", "PAK": "PK", "POL": "PL",
    "PRT": "PT", "QAT": "QA", "ROU": "RO", "RUS": "RU", "SAU": "SA",
    "SER": "RS", "SGP": "SG", "SVK": "SK", "SVN": "SI", "SWE": "SE",
    "TJK": "TJ", "TUN": "TN", "TUR": "TR", "TWN": "TW", "UKR": "UA",
    "USA": "US", "UZB": "UZ", "ZAF": "ZA",
    # Long names → ISO-2
    "Austria": "AT", "Belgium": "BE", "Bulgaria": "BG", "Canada": "CA",
    "Czech Republic": "CZ", "Denmark": "DK", "France": "FR",
    "Germany": "DE", "Great Britain": "GB", "Greece": "GR",
    "Hungary": "HU", "Ireland": "IE", "Italy": "IT",
    "Netherlands": "NL", "Nederlands": "NL", "Norway": "NO",
    "Poland": "PL", "Portugal": "PT", "Romania": "RO",
    "Russia": "RU", "Slovakia": "SK", "Spain": "ES", "Sweden": "SE",
    "Switzerland": "CH", "Turkey": "TR", "Ukraine": "UA",
    "United States": "US", "United Kingdom": "GB",
    "Republic of Latvia": "LV",
    # Misc typos seen in audit
    "ROM": "RO", "GRD": "GD", "H": "HU", "Mo": "MD",
}

# ── Campaign type mapping (V2 mailing_type → shared campaign_type) ────────────
MAILING_TYPE_MAP: dict[str, str] = {
    "campaign": "newsletter",
    "autocampaign": "newsletter",
    "DOI": "transactional",
    "newsletter": "newsletter",
    "transactional": "transactional",
    "other": "other",
}

# ── Email provider category ───────────────────────────────────────────────────
_FREEMAIL_CZ = {"seznam.cz", "centrum.cz", "email.cz", "atlas.cz", "quick.cz",
                "post.cz", "volny.cz", "tiscali.cz"}
_FREEMAIL_SK = {"centrum.sk", "post.sk", "azet.sk", "zoznam.sk",
                "atlas.sk", "pobox.sk"}
_GMAIL = {"gmail.com", "googlemail.com"}
_MICROSOFT = {"outlook.com", "hotmail.com", "live.com", "msn.com",
              "outlook.cz", "hotmail.cz", "hotmail.sk"}
_APPLE = {"icloud.com", "me.com", "mac.com"}
_BOOKING = {"guest.booking.com", "m.expediapartnercentral.com",
            "virtualzoom.com", "noreply.booking.com"}
_YAHOO = {"yahoo.com", "yahoo.co.uk", "yahoo.de", "yahoo.fr",
          "ymail.com", "yahoo.at", "yahoo.pl"}
_PROVIDER_ORDER = [
    ("gmail",    _GMAIL),
    ("microsoft", _MICROSOFT),
    ("apple",    _APPLE),
    ("yahoo",    _YAHOO),
    ("cz_freemail", _FREEMAIL_CZ),
    ("sk_freemail", _FREEMAIL_SK),
    ("booking",  _BOOKING),
]
EMAIL_PROVIDER_CATEGORIES = ["gmail", "microsoft", "apple", "yahoo",
                              "cz_freemail", "sk_freemail", "booking",
                              "corporate", "unknown"]

# All domains reachable via the freemail provider sets above, plus common
# international freemail prefixes used for fast prefix-based lookup.
_ALL_FREEMAIL_DOMAINS: frozenset[str] = frozenset(
    _GMAIL | _MICROSOFT | _APPLE | _YAHOO | _FREEMAIL_CZ | _FREEMAIL_SK
)
_FREEMAIL_TOKENS: frozenset[str] = frozenset({
    "gmail", "yahoo", "seznam", "centrum", "hotmail", "outlook", "icloud",
    "volny", "email", "post", "atlas", "tiscali", "live", "msn", "me",
    "mac", "googlemail", "ymail", "azet", "zoznam", "pobox", "quick",
})

# ── Age segment ───────────────────────────────────────────────────────────────
AGE_SEGMENTS = ["student", "working", "senior", "unknown"]

# ── Domain type ───────────────────────────────────────────────────────────────
DOMAIN_TYPES = ["freemail", "corporate", "unknown"]

# ── domain_type × campaign_type interaction ───────────────────────────────────
DOMAIN_TYPE_X_CAMPAIGN = [
    f"{d}__{c}"
    for d in ["freemail", "corporate", "unknown"]
    for c in ["newsletter", "transactional", "other", "unknown"]
]


def _get_provider_category(domain: str) -> str:
    if not domain or domain == "unknown":
        return "unknown"
    d = domain.lower()
    for name, s in _PROVIDER_ORDER:
        if d in s:
            return name
    return "corporate"


def _get_tld(domain: str) -> str:
    if not domain or domain == "unknown":
        return "unknown"
    parts = domain.lower().rsplit(".", 1)
    if len(parts) < 2:
        return "unknown"
    tld = parts[-1]
    if tld in ("com", "org", "net", "io", "co"):
        return "com_group"
    if tld in ("cz",):
        return "cz"
    if tld in ("sk",):
        return "sk"
    if tld in ("de", "at", "ch"):
        return "dach"
    if tld in ("pl", "hu", "ro", "bg", "hr", "si"):
        return "cee"
    if tld in ("uk", "gb"):
        return "uk"
    if tld in ("fr", "es", "it", "nl", "be"):
        return "west_eu"
    return "other"


EMAIL_TLDS = ["com_group", "cz", "sk", "dach", "cee", "uk", "west_eu", "other", "unknown"]


def _get_age_segment(age: Any) -> str:
    try:
        v = float(age)
    except (TypeError, ValueError):
        return "unknown"
    if v != v:  # NaN check
        return "unknown"
    if v < 25:
        return "student"
    if v < 60:
        return "working"
    return "senior"


def _get_domain_type(domain: str) -> str:
    if not domain or domain == "unknown":
        return "unknown"
    d = domain.lower()
    if d in _ALL_FREEMAIL_DOMAINS:
        return "freemail"
    prefix = d.split(".")[0]
    if prefix in _FREEMAIL_TOKENS:
        return "freemail"
    return "corporate"


def normalize_iso2(val: Any) -> str:
    if pd.isna(val) or str(val).strip() in ("", "nan", "None"):
        return "unknown"
    s = str(val).strip()
    if len(s) == 2 and s.upper() == s:
        return s
    return ISO_NORM.get(s, "unknown")


def normalize_campaign_type(val: Any) -> str:
    if pd.isna(val) or str(val).strip() in ("", "nan", "None"):
        return "unknown"
    return MAILING_TYPE_MAP.get(str(val).strip(), "other")


def extract_email_domain(email: Any) -> str:
    if pd.isna(email) or not isinstance(email, str):
        return "unknown"
    parts = email.strip().lower().split("@")
    if len(parts) != 2 or not parts[1]:
        return "unknown"
    return parts[1]


# ── Category mapping fitting ──────────────────────────────────────────────────
def _fit_topk_map(series: pd.Series, top_k: int) -> dict[str, int]:
    counts = series.value_counts(dropna=False)
    top_vals = [str(v) for v in counts.head(top_k).index if str(v) != "unknown"]
    m = {"unknown": 0}
    for idx, v in enumerate(top_vals, 1):
        m[v] = idx
    m["other"] = len(m)
    return m


def _fit_exact_map(series: pd.Series, known_values: list[str]) -> dict[str, int]:
    """Map known values to indices; everything else → 'other'."""
    m = {"unknown": 0}
    for idx, v in enumerate(known_values, 1):
        m[v] = idx
    m["other"] = len(m)
    return m


def fit_category_mappings(df: pd.DataFrame) -> dict[str, dict[str, int]]:
    return {
        "campaign_type": _fit_exact_map(
            df["campaign_type"],
            ["newsletter", "transactional", "other"],
        ),
        "country_iso2": _fit_topk_map(df["country_iso2"], top_k=50),
        "email_domain": _fit_topk_map(df["email_domain"], top_k=100),
        "email_provider_category": _fit_exact_map(
            df["email_provider_category"],
            EMAIL_PROVIDER_CATEGORIES,
        ),
        "email_tld": _fit_exact_map(df["email_tld"], EMAIL_TLDS),
        "dataset_source": {"V2": 1, "V3": 2},
        "age_segment": _fit_exact_map(pd.Series(dtype=str), AGE_SEGMENTS),
        "domain_type": _fit_exact_map(pd.Series(dtype=str), DOMAIN_TYPES),
        "domain_type_x_campaign": _fit_exact_map(
            pd.Series(dtype=str), DOMAIN_TYPE_X_CAMPAIGN
        ),
    }


def _enc(series: pd.Series, mapping: dict[str, int]) -> pd.Series:
    other = mapping.get("other", 0)
    return series.astype(str).map(mapping).fillna(other).astype(np.float32)


# ── Main feature builder (inference-safe) ────────────────────────────────────
def build_features(
    df: pd.DataFrame,
    category_mappings: dict[str, dict[str, int]],
    domain_priors: dict[str, list[float]],
    campaign_priors: dict[str, list[float]],
    country_mean_hour_map: dict[str, float] | None = None,
    country_mean_hour_global: float = 12.0,
) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)

    # ---- sent_* (NaN-safe) ---------------------------------------------------
    for col in SENT_FEATURES:
        if col in df.columns:
            out[col] = pd.to_numeric(df[col], errors="coerce").astype(np.float32)
        else:
            out[col] = np.nan

    # ---- sent_known indicator ------------------------------------------------
    out["sent_known"] = out["sent_hour"].notna().astype(np.float32)

    # ---- Profile features ----------------------------------------------------
    # campaign_type
    camp = df.get("campaign_type", pd.Series("unknown", index=df.index)).fillna("unknown").astype(str)
    out["campaign_type_enc"] = _enc(camp, category_mappings["campaign_type"])

    # country
    country = df.get("country_iso2", pd.Series("unknown", index=df.index)).fillna("unknown").astype(str)
    out["country_iso2_enc"] = _enc(country, category_mappings["country_iso2"])

    # email domain (derived from email_domain column)
    domain = df.get("email_domain", pd.Series("unknown", index=df.index)).fillna("unknown").astype(str)
    out["email_domain_enc"] = _enc(domain, category_mappings["email_domain"])

    # provider category
    provider = domain.map(_get_provider_category)
    out["email_provider_category_enc"] = _enc(provider, category_mappings["email_provider_category"])

    # TLD
    tld = domain.map(_get_tld)
    out["email_tld_enc"] = _enc(tld, category_mappings["email_tld"])

    # age
    age = pd.to_numeric(df.get("age", np.nan), errors="coerce")
    age = age.where((age >= 0) & (age <= 100))
    out["age"] = age.astype(np.float32)
    out["age_known"] = age.notna().astype(np.float32)

    # age_segment
    age_seg = age.apply(_get_age_segment)
    out["age_segment_enc"] = _enc(age_seg, category_mappings["age_segment"])

    # domain_type
    domain_type = domain.map(_get_domain_type)
    out["domain_type_enc"] = _enc(domain_type, category_mappings["domain_type"])

    # domain_type × campaign_type interaction
    dtxc = domain_type.str.cat(camp, sep="__")
    out["domain_type_x_campaign_enc"] = _enc(dtxc, category_mappings["domain_type_x_campaign"])

    # country_mean_hour (numeric prior — train-derived, no leak)
    cmh_map = country_mean_hour_map or {}
    cmh = country.map(cmh_map)
    out["country_mean_hour"] = cmh.fillna(country_mean_hour_global).astype(np.float32)

    # stays
    out["previous_stays_count"] = pd.to_numeric(
        df.get("previous_stays_count", 0), errors="coerce"
    ).fillna(0).astype(np.float32)
    out["sum_previous_stay_days"] = pd.to_numeric(
        df.get("sum_previous_stay_days", 0), errors="coerce"
    ).fillna(0).astype(np.float32)

    # dataset_source
    src = df.get("dataset_source", pd.Series("unknown", index=df.index)).fillna("unknown").astype(str)
    out["dataset_source_enc"] = _enc(src, category_mappings["dataset_source"])

    # ---- Domain priors -------------------------------------------------------
    fallback_key = "_GLOBAL_"
    global_prior = campaign_priors.get("_GLOBAL_", [1.0 / N_CLASSES] * N_CLASSES)

    def _lookup_prior(row_domain: str, row_campaign: str) -> list[float]:
        key = f"{row_domain}|||{row_campaign}"
        if key in domain_priors:
            return domain_priors[key]
        key2 = f"unknown|||{row_campaign}"
        if key2 in domain_priors:
            return domain_priors[key2]
        return campaign_priors.get(row_campaign, global_prior)

    if len(df) > 0:
        domain_arr = domain.tolist()
        camp_arr = camp.tolist()
        priors_matrix = np.array(
            [_lookup_prior(d, c) for d, c in zip(domain_arr, camp_arr)],
            dtype=np.float32,
        )
    else:
        priors_matrix = np.zeros((0, N_CLASSES), dtype=np.float32)

    for i, h in enumerate(TARGET_HOURS_V2):
        out[f"domain_prior_h{h}"] = priors_matrix[:, i] if len(df) > 0 else np.array([], dtype=np.float32)

    return out[FEATURE_COLS].astype(np.float32)


# ── PreparedTrainingData dataclass ────────────────────────────────────────────
@dataclass
class PreparedTrainingDataV3_2:
    X: pd.DataFrame
    y: pd.Series
    category_mappings: dict[str, dict[str, int]]
    domain_priors: dict[str, list[float]]
    campaign_priors: dict[str, list[float]]
    rows_before: int
    rows_after: int
    rows_v2: int
    rows_v3: int
    dropped: dict[str, int] = field(default_factory=dict)
