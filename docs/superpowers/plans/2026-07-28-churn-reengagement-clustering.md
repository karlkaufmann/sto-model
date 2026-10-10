# Churn Re-Engagement Clustering Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a contact-level unsupervised clustering workflow for hotel CRM re-engagement personas, with rule regimes kept separate from k-means clusters.

**Architecture:** Add a new `src/reengagement_clustering/` package beside existing model versions so the current send-time prediction pipeline is untouched. Dataset builders produce one contact row per tenant identity, regime assignment marks non-clusterable contacts, and a scikit-learn pipeline fits `k=5` only on clusterable rows. Outputs follow existing repo conventions: gitignored data/model artifacts, tracked configs and lightweight JSON summaries.

**Tech Stack:** Python 3.10+, pandas, numpy, pyarrow, scikit-learn, pytest.

## Global Constraints

- Primary output is clustering, not a supervised churn score.
- STO export is prototype-only; Zoomle `nwl3_*` extract is production source of truth.
- Grain is one row per client within tenant or hotel.
- Do not merge cross-tenant email identities.
- STO identity is `tenant_id x recipient_email_sha256`.
- Zoomle identity is `id_klienta` within tenant or hotel.
- Use engagement windows of 30 days and 90 days.
- Keep rule regime buckets separate from k-means clusters.
- Start k-means with `k=5`.
- Do not commit raw data files, archives, parquet datasets, model binaries, database files, or secrets.
- Public repo commit scope currently allows documentation, notebooks, and evaluation scripts only; do not commit `src/`, `tests/`, `configs/`, or `results/` implementation changes unless the user explicitly approves a broader commit scope.

---

## File Structure

- Create `src/reengagement_clustering/__init__.py`: package marker and exported version string.
- Create `src/reengagement_clustering/schema.py`: shared column constants and small dataclasses for dataset/fit outputs.
- Create `src/reengagement_clustering/regimes.py`: deterministic rule regime assignment.
- Create `src/reengagement_clustering/features.py`: feature matrix builder for clustering.
- Create `src/reengagement_clustering/build_sto_contact_dataset.py`: prototype dataset builder from `data/raw/data_6_6_2026.zip`.
- Create `src/reengagement_clustering/build_zl_contact_dataset.py`: production dataset builder from local exported `nwl3_*` table files.
- Create `src/reengagement_clustering/fit.py`: k-means fit, metrics, persistence, and segment summary.
- Create `src/reengagement_clustering/labels.py`: human-readable label candidates from cluster profiles.
- Create `tests/test_reengagement_regimes.py`: regime bucket tests.
- Create `tests/test_reengagement_features.py`: feature builder tests.
- Create `tests/test_reengagement_sto_builder.py`: STO aggregation tests with tiny synthetic frames.
- Create `tests/test_reengagement_fit.py`: clustering fit smoke tests.
- Create `configs/reengagement_clustering/features.json`: tracked v1 feature config.
- Create `docs/reengagement_clustering.md`: operator notes for prototype and production runs.

## Task 1: Package Contract And Feature Config

**Files:**
- Create: `src/reengagement_clustering/__init__.py`
- Create: `src/reengagement_clustering/schema.py`
- Create: `configs/reengagement_clustering/features.json`
- Test: `tests/test_reengagement_features.py`

**Interfaces:**
- Produces: `IDENTITY_COLS: list[str]`
- Produces: `ENGAGEMENT_COLS: list[str]`
- Produces: `SOCIO_DEMO_COLS: list[str]`
- Produces: `BOOKING_COLS: list[str]`
- Produces: `META_COLS: list[str]`
- Produces: `CLUSTER_EXCLUDE_COLS: list[str]`
- Produces: `DatasetBuildResult` dataclass with `frame: pd.DataFrame` and `audit: dict[str, object]`
- Produces: `FitResult` dataclass with `labels: pd.Series`, `metrics: dict[str, object]`, and `segments: list[dict[str, object]]`
- Consumes: no project-local interfaces.

- [ ] **Step 1: Write the config file**

Create `configs/reengagement_clustering/features.json` with:

```json
{
  "version": "v1",
  "identity_columns": ["tenant_id", "contact_id", "identity_source"],
  "engagement_windows_days": [30, 90],
  "numeric_features": [
    "sents_30d",
    "sents_90d",
    "opens_30d",
    "opens_90d",
    "clicks_30d",
    "clicks_90d",
    "dsn_30d",
    "dsn_90d",
    "open_rate_30d",
    "open_rate_90d",
    "click_rate_30d",
    "click_rate_90d",
    "days_since_last_email",
    "days_since_last_open",
    "days_since_last_click",
    "open_rate_delta_30_vs_90",
    "click_rate_delta_30_vs_90",
    "registration_age_days",
    "age",
    "stays_count",
    "days_since_last_stay",
    "total_nights",
    "total_spend",
    "avg_lead_time_days",
    "tenant_open_rate_rank",
    "tenant_click_rate_rank",
    "tenant_booking_value_rank",
    "tenant_booking_recency_rank"
  ],
  "categorical_features": [
    "language",
    "country",
    "gender",
    "consent_state",
    "booking_channel",
    "booking_source"
  ],
  "known_flag_features": [
    "language_known",
    "country_known",
    "gender_known",
    "age_known",
    "booking_known",
    "booking_channel_known",
    "booking_source_known"
  ],
  "excluded_from_clustering": [
    "tenant_id",
    "contact_id",
    "identity_source",
    "regime_bucket",
    "is_clusterable",
    "cluster_id",
    "segment_label",
    "extracted_at"
  ],
  "kmeans": {
    "n_clusters": 5,
    "random_state": 42,
    "n_init": "auto"
  }
}
```

- [ ] **Step 2: Add schema constants**

Create `src/reengagement_clustering/schema.py` with:

```python
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

IDENTITY_COLS = ["tenant_id", "contact_id", "identity_source"]

ENGAGEMENT_COLS = [
    "sents_30d",
    "sents_90d",
    "opens_30d",
    "opens_90d",
    "clicks_30d",
    "clicks_90d",
    "dsn_30d",
    "dsn_90d",
    "open_rate_30d",
    "open_rate_90d",
    "click_rate_30d",
    "click_rate_90d",
    "days_since_last_email",
    "days_since_last_open",
    "days_since_last_click",
    "open_rate_delta_30_vs_90",
    "click_rate_delta_30_vs_90",
]

SOCIO_DEMO_COLS = [
    "language",
    "country",
    "gender",
    "age",
    "consent_state",
    "registration_age_days",
    "language_known",
    "country_known",
    "gender_known",
    "age_known",
]

BOOKING_COLS = [
    "stays_count",
    "days_since_last_stay",
    "total_nights",
    "total_spend",
    "avg_lead_time_days",
    "booking_channel",
    "booking_source",
    "booking_known",
    "booking_channel_known",
    "booking_source_known",
]

META_COLS = [
    "regime_bucket",
    "is_clusterable",
    "tenant_open_rate_rank",
    "tenant_click_rate_rank",
    "tenant_booking_value_rank",
    "tenant_booking_recency_rank",
    "extracted_at",
]

CLUSTER_EXCLUDE_COLS = [
    "tenant_id",
    "contact_id",
    "identity_source",
    "regime_bucket",
    "is_clusterable",
    "cluster_id",
    "segment_label",
    "extracted_at",
]


@dataclass
class DatasetBuildResult:
    frame: pd.DataFrame
    audit: dict[str, object]


@dataclass
class FitResult:
    labels: pd.Series
    metrics: dict[str, object]
    segments: list[dict[str, object]]
```

- [ ] **Step 3: Add package marker**

Create `src/reengagement_clustering/__init__.py` with:

```python
from __future__ import annotations

__version__ = "0.1.0"
```

- [ ] **Step 4: Write schema test**

Create `tests/test_reengagement_features.py` with:

```python
from __future__ import annotations

from src.reengagement_clustering.schema import CLUSTER_EXCLUDE_COLS, ENGAGEMENT_COLS, IDENTITY_COLS


def test_schema_keeps_identity_out_of_clustering_exclusions() -> None:
    assert IDENTITY_COLS == ["tenant_id", "contact_id", "identity_source"]
    assert set(IDENTITY_COLS).issubset(CLUSTER_EXCLUDE_COLS)


def test_schema_contains_dual_window_engagement() -> None:
    assert "sents_30d" in ENGAGEMENT_COLS
    assert "sents_90d" in ENGAGEMENT_COLS
    assert "open_rate_delta_30_vs_90" in ENGAGEMENT_COLS
    assert "click_rate_delta_30_vs_90" in ENGAGEMENT_COLS
```

- [ ] **Step 5: Run test**

Run: `python -m pytest tests/test_reengagement_features.py -v`

Expected: 2 tests pass.

- [ ] **Step 6: Commit safety checkpoint**

Run: `rtk git status --short`.

Expected: only intentional files from this task are changed. Do not commit these implementation files in the public repo unless the user has approved broadening commit scope beyond documentation, notebooks, and evaluation scripts.

## Task 2: Rule Regime Assignment

**Files:**
- Create: `src/reengagement_clustering/regimes.py`
- Modify: `tests/test_reengagement_regimes.py`

**Interfaces:**
- Consumes: contact-level dataframe with engagement, consent, validity, and bounce columns.
- Produces: `assign_regimes(df: pd.DataFrame) -> pd.DataFrame`
- Produces columns: `regime_bucket: str`, `is_clusterable: bool`

- [ ] **Step 1: Write failing regime tests**

Create `tests/test_reengagement_regimes.py` with:

```python
from __future__ import annotations

import pandas as pd

from src.reengagement_clustering.regimes import assign_regimes


BASE = {
    "email_valid": True,
    "has_email_consent": True,
    "hard_bounce_count": 0,
    "sents_90d": 5,
    "opens_90d": 2,
    "clicks_90d": 0,
    "open_rate_30d": 0.3,
    "open_rate_90d": 0.3,
    "click_rate_30d": 0.0,
    "click_rate_90d": 0.0,
    "days_since_last_open": 10,
    "days_since_last_click": 9999,
}


def row(**overrides: object) -> dict[str, object]:
    data = BASE.copy()
    data.update(overrides)
    return data


def test_unreachable_overrides_other_signals() -> None:
    df = pd.DataFrame([row(email_valid=False, opens_90d=4)])

    out = assign_regimes(df)

    assert out.loc[0, "regime_bucket"] == "unreachable"
    assert bool(out.loc[0, "is_clusterable"]) is False


def test_cold_never_engaged_is_not_clusterable() -> None:
    df = pd.DataFrame([row(opens_90d=0, clicks_90d=0, sents_90d=8)])

    out = assign_regimes(df)

    assert out.loc[0, "regime_bucket"] == "cold_never_engaged"
    assert bool(out.loc[0, "is_clusterable"]) is False


def test_low_signal_requires_two_sends() -> None:
    df = pd.DataFrame([row(sents_90d=1, opens_90d=1)])

    out = assign_regimes(df)

    assert out.loc[0, "regime_bucket"] == "inactive_low_signal"
    assert bool(out.loc[0, "is_clusterable"]) is False


def test_fading_contact_remains_clusterable_when_history_is_sufficient() -> None:
    df = pd.DataFrame([row(sents_90d=6, opens_90d=3, open_rate_30d=0.0, open_rate_90d=0.5)])

    out = assign_regimes(df)

    assert out.loc[0, "regime_bucket"] == "fading"
    assert bool(out.loc[0, "is_clusterable"]) is True


def test_healthy_engaged_contact_is_clusterable() -> None:
    df = pd.DataFrame([row(sents_90d=6, opens_90d=3, days_since_last_open=12)])

    out = assign_regimes(df)

    assert out.loc[0, "regime_bucket"] == "healthy_engaged"
    assert bool(out.loc[0, "is_clusterable"]) is True
```

- [ ] **Step 2: Run tests to verify failure**

Run: `python -m pytest tests/test_reengagement_regimes.py -v`

Expected: import failure for `src.reengagement_clustering.regimes`.

- [ ] **Step 3: Implement regime assignment**

Create `src/reengagement_clustering/regimes.py` with:

```python
from __future__ import annotations

import numpy as np
import pandas as pd


def _numeric(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype="float64")
    return pd.to_numeric(df[col], errors="coerce").fillna(default)


def _bool(df: pd.DataFrame, col: str, default: bool) -> pd.Series:
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype="bool")
    return df[col].fillna(default).astype(bool)


def assign_regimes(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    email_valid = _bool(out, "email_valid", True)
    has_consent = _bool(out, "has_email_consent", True)
    hard_bounces = _numeric(out, "hard_bounce_count", 0)
    sents_90d = _numeric(out, "sents_90d", 0)
    opens_90d = _numeric(out, "opens_90d", 0)
    clicks_90d = _numeric(out, "clicks_90d", 0)
    open_rate_30d = _numeric(out, "open_rate_30d", 0)
    open_rate_90d = _numeric(out, "open_rate_90d", 0)
    days_since_last_open = _numeric(out, "days_since_last_open", 9999)
    days_since_last_click = _numeric(out, "days_since_last_click", 9999)

    unreachable = (~email_valid) | (~has_consent) | (hard_bounces > 0)
    low_signal = sents_90d < 2
    never_engaged = (opens_90d <= 0) & (clicks_90d <= 0)
    sufficient_history = (sents_90d >= 2) & ((opens_90d >= 1) | (clicks_90d >= 1))
    recent_engagement = (days_since_last_open <= 30) | (days_since_last_click <= 30)
    fading = sufficient_history & (open_rate_90d >= 0.1) & ((open_rate_90d - open_rate_30d) >= 0.2)
    healthy = sufficient_history & recent_engagement & ~fading

    out["regime_bucket"] = np.select(
        [
            unreachable,
            low_signal & ~unreachable,
            never_engaged & ~unreachable,
            fading & ~unreachable,
            healthy & ~unreachable,
        ],
        [
            "unreachable",
            "inactive_low_signal",
            "cold_never_engaged",
            "fading",
            "healthy_engaged",
        ],
        default="clusterable",
    )
    out["is_clusterable"] = sufficient_history & ~unreachable
    return out
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_reengagement_regimes.py -v`

Expected: 5 tests pass.

- [ ] **Step 5: Commit safety checkpoint**

Run: `rtk git status --short`.

Expected: only intentional files from Tasks 1-2 are changed. Do not commit implementation files without explicit user approval for public repo scope.

## Task 3: Clustering Feature Matrix Builder

**Files:**
- Create: `src/reengagement_clustering/features.py`
- Modify: `tests/test_reengagement_features.py`

**Interfaces:**
- Consumes: contact-level dataframe after `assign_regimes`.
- Produces: `build_feature_matrix(df: pd.DataFrame, config_path: str | Path = "configs/reengagement_clustering/features.json") -> tuple[pd.DataFrame, list[str], list[str]]`
- Returns: transformed input columns only, numeric feature names, categorical feature names.

- [ ] **Step 1: Extend feature tests**

Append to `tests/test_reengagement_features.py`:

```python
import pandas as pd

from src.reengagement_clustering.features import build_feature_matrix


def test_build_feature_matrix_excludes_identity_and_regime_columns() -> None:
    df = pd.DataFrame(
        [
            {
                "tenant_id": "hotel-a",
                "contact_id": "abc",
                "identity_source": "sto_sha256",
                "regime_bucket": "healthy_engaged",
                "is_clusterable": True,
                "sents_30d": 2,
                "sents_90d": 4,
                "opens_30d": 1,
                "opens_90d": 2,
                "clicks_30d": 0,
                "clicks_90d": 1,
                "language": "cs",
                "country": "CZ",
                "gender": "unknown",
                "consent_state": "active",
                "booking_channel": "direct",
                "booking_source": "web",
            }
        ]
    )

    matrix, numeric_features, categorical_features = build_feature_matrix(df)

    assert "tenant_id" not in matrix.columns
    assert "contact_id" not in matrix.columns
    assert "regime_bucket" not in matrix.columns
    assert "sents_30d" in numeric_features
    assert "language" in categorical_features
```

- [ ] **Step 2: Run tests to verify failure**

Run: `python -m pytest tests/test_reengagement_features.py -v`

Expected: import failure for `src.reengagement_clustering.features`.

- [ ] **Step 3: Implement feature matrix builder**

Create `src/reengagement_clustering/features.py` with:

```python
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


DEFAULT_CONFIG_PATH = Path("configs/reengagement_clustering/features.json")


def _load_config(config_path: str | Path) -> dict[str, object]:
    with Path(config_path).open() as f:
        return json.load(f)


def build_feature_matrix(
    df: pd.DataFrame,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    config = _load_config(config_path)
    numeric_features = [col for col in config["numeric_features"] if col in df.columns]
    categorical_features = [col for col in config["categorical_features"] if col in df.columns]
    known_flag_features = [col for col in config["known_flag_features"] if col in df.columns]
    feature_cols = numeric_features + categorical_features + known_flag_features
    matrix = df.loc[:, feature_cols].copy()
    return matrix, numeric_features + known_flag_features, categorical_features
```

- [ ] **Step 4: Run feature tests**

Run: `python -m pytest tests/test_reengagement_features.py -v`

Expected: all feature tests pass.

- [ ] **Step 5: Commit safety checkpoint**

Run: `rtk git status --short`.

Expected: only intentional files from Tasks 1-3 are changed. Do not commit implementation files without explicit user approval for public repo scope.

## Task 4: STO Prototype Contact Dataset Builder

**Files:**
- Create: `src/reengagement_clustering/build_sto_contact_dataset.py`
- Create: `tests/test_reengagement_sto_builder.py`

**Interfaces:**
- Consumes: STO event dataframe or zip of parquet members.
- Consumes: `assign_regimes(df)`.
- Produces: `build_contact_frame(events: pd.DataFrame, as_of: pd.Timestamp | None = None) -> DatasetBuildResult`
- Produces CLI: `python -m src.reengagement_clustering.build_sto_contact_dataset`

- [ ] **Step 1: Write aggregation tests**

Create `tests/test_reengagement_sto_builder.py` with:

```python
from __future__ import annotations

import pandas as pd

from src.reengagement_clustering.build_sto_contact_dataset import build_contact_frame


def test_build_contact_frame_keeps_tenant_email_grain() -> None:
    events = pd.DataFrame(
        [
            {
                "tenant_id": "hotel-a",
                "recipient_email_sha256": "same",
                "sent_local": "2026-07-20 10:00:00",
                "target_opened_anytime": 1,
                "target_clicked_anytime": 0,
                "bounced": "none",
            },
            {
                "tenant_id": "hotel-b",
                "recipient_email_sha256": "same",
                "sent_local": "2026-07-20 10:00:00",
                "target_opened_anytime": 0,
                "target_clicked_anytime": 0,
                "bounced": "none",
            },
        ]
    )

    result = build_contact_frame(events, as_of=pd.Timestamp("2026-07-28"))

    assert len(result.frame) == 2
    assert set(result.frame["tenant_id"]) == {"hotel-a", "hotel-b"}
    assert set(result.frame["contact_id"]) == {"same"}


def test_build_contact_frame_computes_dual_windows() -> None:
    events = pd.DataFrame(
        [
            {
                "tenant_id": "hotel-a",
                "recipient_email_sha256": "one",
                "sent_local": "2026-07-20 10:00:00",
                "target_opened_anytime": 1,
                "target_clicked_anytime": 1,
                "bounced": "none",
            },
            {
                "tenant_id": "hotel-a",
                "recipient_email_sha256": "one",
                "sent_local": "2026-05-20 10:00:00",
                "target_opened_anytime": 1,
                "target_clicked_anytime": 0,
                "bounced": "none",
            },
        ]
    )

    result = build_contact_frame(events, as_of=pd.Timestamp("2026-07-28"))
    row = result.frame.iloc[0]

    assert row["sents_30d"] == 1
    assert row["sents_90d"] == 2
    assert row["opens_30d"] == 1
    assert row["opens_90d"] == 2
    assert row["clicks_30d"] == 1
    assert row["clicks_90d"] == 1
```

- [ ] **Step 2: Run tests to verify failure**

Run: `python -m pytest tests/test_reengagement_sto_builder.py -v`

Expected: import failure for `build_sto_contact_dataset`.

- [ ] **Step 3: Implement in-memory builder**

Create `src/reengagement_clustering/build_sto_contact_dataset.py` with:

```python
from __future__ import annotations

import json
import os
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .regimes import assign_regimes
from .schema import DatasetBuildResult

RAW_ZIP = Path(os.environ.get("REENGAGEMENT_STO_RAW_ZIP", "data/raw/data_6_6_2026.zip"))
OUT_PARQUET = Path(os.environ.get("REENGAGEMENT_STO_OUT_PARQUET", "data/processed/reengagement_sto_contacts.parquet"))
AUDIT_OUT = Path(os.environ.get("REENGAGEMENT_STO_AUDIT_OUT", "results/audits/reengagement_sto_contacts.json"))

READ_COLS = [
    "tenant_id",
    "recipient_email_sha256",
    "sent_local",
    "target_opened_anytime",
    "target_clicked_anytime",
    "bounced",
    "recipient_language",
    "recipient_country",
    "recipient_gender",
    "recipient_age_approx",
    "recipient_mail_is_valid",
    "recipient_newsletter_consent_legacy",
    "booking_created_hour_local",
    "booking_lead_time_days",
    "stay_nights",
    "booking_price",
    "custom_booking_channel",
]


def _safe_rate(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return np.where(denominator > 0, numerator / denominator, 0.0)


def _days_since(last_seen: pd.Series, as_of: pd.Timestamp) -> pd.Series:
    parsed = pd.to_datetime(last_seen, errors="coerce")
    days = (as_of - parsed).dt.days
    return days.fillna(9999).clip(lower=0)


def _read_zip_events(path: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    with zipfile.ZipFile(path) as zf:
        members = [name for name in zf.namelist() if name.endswith(".parquet")]
        for name in members:
            with zf.open(name) as fh:
                schema_names = pq.ParquetFile(fh).schema_arrow.names
            columns = [col for col in READ_COLS if col in schema_names]
            with zf.open(name) as fh:
                frames.append(pq.read_table(fh, columns=columns).to_pandas())
    if not frames:
        raise RuntimeError(f"No parquet members found in {path}")
    return pd.concat(frames, ignore_index=True)


def build_contact_frame(events: pd.DataFrame, as_of: pd.Timestamp | None = None) -> DatasetBuildResult:
    df = events.copy()
    df["tenant_id"] = df["tenant_id"].astype(str)
    df["contact_id"] = df["recipient_email_sha256"].astype(str)
    df["identity_source"] = "sto_sha256"
    df["sent_dt"] = pd.to_datetime(df["sent_local"], errors="coerce")
    df = df.dropna(subset=["tenant_id", "contact_id", "sent_dt"])
    if as_of is None:
        as_of = df["sent_dt"].max().normalize()

    df["opened"] = pd.to_numeric(df.get("target_opened_anytime", 0), errors="coerce").fillna(0).astype(int)
    df["clicked"] = pd.to_numeric(df.get("target_clicked_anytime", 0), errors="coerce").fillna(0).astype(int)
    bounced_norm = df.get("bounced", pd.Series("none", index=df.index)).fillna("none").astype(str).str.lower()
    df["dsn"] = (bounced_norm != "none").astype(int)
    df["email_valid"] = pd.to_numeric(df.get("recipient_mail_is_valid", 1), errors="coerce").fillna(1).astype(int).eq(1)
    df["has_email_consent"] = pd.to_numeric(df.get("recipient_newsletter_consent_legacy", 1), errors="coerce").fillna(1).astype(int).eq(1)

    keys = ["tenant_id", "contact_id", "identity_source"]
    base = df.groupby(keys, dropna=False).agg(
        language=("recipient_language", "last"),
        country=("recipient_country", "last"),
        gender=("recipient_gender", "last"),
        age=("recipient_age_approx", "last"),
        email_valid=("email_valid", "max"),
        has_email_consent=("has_email_consent", "max"),
        hard_bounce_count=("dsn", "sum"),
        last_email_at=("sent_dt", "max"),
        last_open_at=("sent_dt", lambda s: df.loc[s.index[df.loc[s.index, "opened"].eq(1)], "sent_dt"].max()),
        last_click_at=("sent_dt", lambda s: df.loc[s.index[df.loc[s.index, "clicked"].eq(1)], "sent_dt"].max()),
        stays_count=("stay_nights", "count"),
        total_nights=("stay_nights", "sum"),
        total_spend=("booking_price", "sum"),
        avg_lead_time_days=("booking_lead_time_days", "mean"),
        booking_channel=("custom_booking_channel", "last"),
    ).reset_index()

    for days in (30, 90):
        window = df[df["sent_dt"] >= as_of - pd.Timedelta(days=days)]
        agg = window.groupby(keys, dropna=False).agg(
            **{
                f"sents_{days}d": ("sent_dt", "size"),
                f"opens_{days}d": ("opened", "sum"),
                f"clicks_{days}d": ("clicked", "sum"),
                f"dsn_{days}d": ("dsn", "sum"),
            }
        ).reset_index()
        base = base.merge(agg, on=keys, how="left")

    count_cols = ["sents_30d", "sents_90d", "opens_30d", "opens_90d", "clicks_30d", "clicks_90d", "dsn_30d", "dsn_90d"]
    base[count_cols] = base[count_cols].fillna(0)
    base["open_rate_30d"] = _safe_rate(base["opens_30d"], base["sents_30d"])
    base["open_rate_90d"] = _safe_rate(base["opens_90d"], base["sents_90d"])
    base["click_rate_30d"] = _safe_rate(base["clicks_30d"], base["sents_30d"])
    base["click_rate_90d"] = _safe_rate(base["clicks_90d"], base["sents_90d"])
    base["open_rate_delta_30_vs_90"] = base["open_rate_30d"] - base["open_rate_90d"]
    base["click_rate_delta_30_vs_90"] = base["click_rate_30d"] - base["click_rate_90d"]
    base["days_since_last_email"] = _days_since(base["last_email_at"], as_of)
    base["days_since_last_open"] = _days_since(base["last_open_at"], as_of)
    base["days_since_last_click"] = _days_since(base["last_click_at"], as_of)

    base["language_known"] = base["language"].notna()
    base["country_known"] = base["country"].notna()
    base["gender_known"] = base["gender"].notna()
    base["age_known"] = pd.to_numeric(base["age"], errors="coerce").notna()
    base["booking_known"] = base["stays_count"].fillna(0).gt(0)
    base["booking_channel_known"] = base["booking_channel"].notna()
    base["booking_source"] = "unknown"
    base["booking_source_known"] = False
    base["consent_state"] = np.where(base["has_email_consent"], "active", "inactive")
    base["registration_age_days"] = np.nan
    base["days_since_last_stay"] = np.nan

    base["tenant_open_rate_rank"] = base.groupby("tenant_id")["open_rate_90d"].rank(pct=True).fillna(0)
    base["tenant_click_rate_rank"] = base.groupby("tenant_id")["click_rate_90d"].rank(pct=True).fillna(0)
    base["tenant_booking_value_rank"] = base.groupby("tenant_id")["total_spend"].rank(pct=True).fillna(0)
    base["tenant_booking_recency_rank"] = base.groupby("tenant_id")["days_since_last_stay"].rank(pct=True).fillna(0)
    base["extracted_at"] = as_of.isoformat()

    base = assign_regimes(base)
    audit = {
        "rows": int(len(base)),
        "clusterable_rows": int(base["is_clusterable"].sum()),
        "regime_counts": {str(k): int(v) for k, v in base["regime_bucket"].value_counts().to_dict().items()},
        "as_of": as_of.isoformat(),
    }
    return DatasetBuildResult(frame=base, audit=audit)


def main() -> None:
    t0 = time.time()
    OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    AUDIT_OUT.parent.mkdir(parents=True, exist_ok=True)
    events = _read_zip_events(RAW_ZIP)
    result = build_contact_frame(events)
    result.frame.to_parquet(OUT_PARQUET, index=False, compression="snappy")
    result.audit["elapsed_seconds"] = round(time.time() - t0, 1)
    with AUDIT_OUT.open("w") as f:
        json.dump(result.audit, f, indent=2)
    print(f"Written {OUT_PARQUET} rows={len(result.frame):,}")
    print(f"Audit {AUDIT_OUT}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run STO builder tests**

Run: `python -m pytest tests/test_reengagement_sto_builder.py -v`

Expected: 2 tests pass.

- [ ] **Step 5: Run prototype builder on real local STO zip**

Run: `python -m src.reengagement_clustering.build_sto_contact_dataset`

Expected: writes `data/processed/reengagement_sto_contacts.parquet` and `results/audits/reengagement_sto_contacts.json`. The parquet path is gitignored and must not be staged.

- [ ] **Step 6: Commit safety checkpoint**

Run: `rtk git status --short`.

Expected: generated parquet remains untracked or ignored. Do not commit implementation files without explicit user approval for public repo scope.

## Task 5: Zoomle Production Extract Builder

**Files:**
- Create: `src/reengagement_clustering/build_zl_contact_dataset.py`
- Modify: `docs/reengagement_clustering.md`

**Interfaces:**
- Consumes local extracted files under `data/raw/zl_nwl3/`.
- Produces CLI: `python -m src.reengagement_clustering.build_zl_contact_dataset`
- Produces output: `data/processed/reengagement_zl_contacts.parquet`

- [ ] **Step 1: Implement production extract loader**

Create `src/reengagement_clustering/build_zl_contact_dataset.py` with:

```python
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .regimes import assign_regimes
from .schema import DatasetBuildResult

RAW_DIR = Path(os.environ.get("REENGAGEMENT_ZL_RAW_DIR", "data/raw/zl_nwl3"))
OUT_PARQUET = Path(os.environ.get("REENGAGEMENT_ZL_OUT_PARQUET", "data/processed/reengagement_zl_contacts.parquet"))
AUDIT_OUT = Path(os.environ.get("REENGAGEMENT_ZL_AUDIT_OUT", "results/audits/reengagement_zl_contacts.json"))


def _read_table(raw_dir: Path, stem: str) -> pd.DataFrame:
    parquet = raw_dir / f"{stem}.parquet"
    csv = raw_dir / f"{stem}.csv"
    if parquet.exists():
        return pd.read_parquet(parquet)
    if csv.exists():
        return pd.read_csv(csv)
    raise FileNotFoundError(f"Missing {parquet} or {csv}")


def build_contact_frame(raw_dir: Path = RAW_DIR, as_of: pd.Timestamp | None = None) -> DatasetBuildResult:
    clients = _read_table(raw_dir, "nwl3_klienti")
    gdpr = _read_table(raw_dir, "nwl3_gdpr")
    reservations = _read_table(raw_dir, "nwl3_rezervace")
    processed = _read_table(raw_dir, "nwl3_mailer_processed")
    zoomtrack = _read_table(raw_dir, "nwl3_zoomtrack")

    tenant_col = "tenant_id" if "tenant_id" in clients.columns else "web_id"
    client_col = "id_klienta"
    keys = [tenant_col, client_col]

    events = processed.copy()
    events["event_dt"] = pd.to_datetime(events.get("sent_at", events.get("sent", events.get("created_at"))), errors="coerce")
    events = events.dropna(subset=keys + ["event_dt"])
    if as_of is None:
        as_of = events["event_dt"].max().normalize()

    event_type = events.get("event_type", pd.Series("", index=events.index)).fillna("").astype(str).str.lower()
    events["sent_flag"] = 1
    events["open_flag"] = event_type.str.contains("open").astype(int)
    events["dsn_flag"] = event_type.str.contains("dsn|bounce").astype(int)

    clicks = zoomtrack.copy()
    clicks["click_dt"] = pd.to_datetime(clicks.get("clicked_at", clicks.get("created_at")), errors="coerce")
    clicks = clicks.dropna(subset=keys + ["click_dt"])

    base = clients[keys].drop_duplicates().copy()
    base = base.rename(columns={tenant_col: "tenant_id", client_col: "contact_id"})
    base["identity_source"] = "zl_id_klienta"

    profile = clients.copy()
    profile = profile.rename(columns={tenant_col: "tenant_id", client_col: "contact_id"})
    profile_cols = ["tenant_id", "contact_id"]
    for source, target in [("jazyk", "language"), ("zeme", "country"), ("pohlavi", "gender"), ("vek", "age")]:
        if source in profile.columns:
            profile[target] = profile[source]
            profile_cols.append(target)
    base = base.merge(profile[profile_cols].drop_duplicates(["tenant_id", "contact_id"]), on=["tenant_id", "contact_id"], how="left")

    gdpr_work = gdpr.rename(columns={tenant_col: "tenant_id", client_col: "contact_id"})
    gdpr_work["has_email_consent"] = pd.to_numeric(gdpr_work.get("marketing_email", 1), errors="coerce").fillna(1).astype(int).eq(1)
    consent = gdpr_work.groupby(["tenant_id", "contact_id"], dropna=False)["has_email_consent"].max().reset_index()
    base = base.merge(consent, on=["tenant_id", "contact_id"], how="left")
    base["has_email_consent"] = base["has_email_consent"].fillna(False)
    base["email_valid"] = True

    events = events.rename(columns={tenant_col: "tenant_id", client_col: "contact_id"})
    clicks = clicks.rename(columns={tenant_col: "tenant_id", client_col: "contact_id"})
    for days in (30, 90):
        event_window = events[events["event_dt"] >= as_of - pd.Timedelta(days=days)]
        event_agg = event_window.groupby(["tenant_id", "contact_id"], dropna=False).agg(
            **{
                f"sents_{days}d": ("sent_flag", "sum"),
                f"opens_{days}d": ("open_flag", "sum"),
                f"dsn_{days}d": ("dsn_flag", "sum"),
            }
        ).reset_index()
        click_window = clicks[clicks["click_dt"] >= as_of - pd.Timedelta(days=days)]
        click_agg = click_window.groupby(["tenant_id", "contact_id"], dropna=False).size().rename(f"clicks_{days}d").reset_index()
        base = base.merge(event_agg, on=["tenant_id", "contact_id"], how="left")
        base = base.merge(click_agg, on=["tenant_id", "contact_id"], how="left")

    count_cols = ["sents_30d", "sents_90d", "opens_30d", "opens_90d", "clicks_30d", "clicks_90d", "dsn_30d", "dsn_90d"]
    base[count_cols] = base[count_cols].fillna(0)
    base["open_rate_30d"] = np.where(base["sents_30d"] > 0, base["opens_30d"] / base["sents_30d"], 0.0)
    base["open_rate_90d"] = np.where(base["sents_90d"] > 0, base["opens_90d"] / base["sents_90d"], 0.0)
    base["click_rate_30d"] = np.where(base["sents_30d"] > 0, base["clicks_30d"] / base["sents_30d"], 0.0)
    base["click_rate_90d"] = np.where(base["sents_90d"] > 0, base["clicks_90d"] / base["sents_90d"], 0.0)
    base["open_rate_delta_30_vs_90"] = base["open_rate_30d"] - base["open_rate_90d"]
    base["click_rate_delta_30_vs_90"] = base["click_rate_30d"] - base["click_rate_90d"]

    last_email = events.groupby(["tenant_id", "contact_id"], dropna=False)["event_dt"].max().reset_index(name="last_email_at")
    last_open = events[events["open_flag"].eq(1)].groupby(["tenant_id", "contact_id"], dropna=False)["event_dt"].max().reset_index(name="last_open_at")
    last_click = clicks.groupby(["tenant_id", "contact_id"], dropna=False)["click_dt"].max().reset_index(name="last_click_at")
    base = base.merge(last_email, on=["tenant_id", "contact_id"], how="left")
    base = base.merge(last_open, on=["tenant_id", "contact_id"], how="left")
    base = base.merge(last_click, on=["tenant_id", "contact_id"], how="left")
    base["days_since_last_email"] = (as_of - base["last_email_at"]).dt.days.fillna(9999).clip(lower=0)
    base["days_since_last_open"] = (as_of - base["last_open_at"]).dt.days.fillna(9999).clip(lower=0)
    base["days_since_last_click"] = (as_of - base["last_click_at"]).dt.days.fillna(9999).clip(lower=0)

    reservations = reservations.rename(columns={tenant_col: "tenant_id", client_col: "contact_id"})
    reservations["stay_nights"] = pd.to_numeric(reservations.get("nights", reservations.get("pocet_noci")), errors="coerce")
    reservations["booking_price"] = pd.to_numeric(reservations.get("price", reservations.get("cena")), errors="coerce")
    reservations["lead_time_days"] = pd.to_numeric(reservations.get("lead_time_days", reservations.get("predstih_dni")), errors="coerce")
    booking = reservations.groupby(["tenant_id", "contact_id"], dropna=False).agg(
        stays_count=("contact_id", "size"),
        total_nights=("stay_nights", "sum"),
        total_spend=("booking_price", "sum"),
        avg_lead_time_days=("lead_time_days", "mean"),
    ).reset_index()
    base = base.merge(booking, on=["tenant_id", "contact_id"], how="left")

    base["hard_bounce_count"] = base["dsn_90d"]
    base["language_known"] = base.get("language", pd.Series(index=base.index)).notna()
    base["country_known"] = base.get("country", pd.Series(index=base.index)).notna()
    base["gender_known"] = base.get("gender", pd.Series(index=base.index)).notna()
    base["age_known"] = pd.to_numeric(base.get("age", pd.Series(index=base.index)), errors="coerce").notna()
    base["booking_known"] = base["stays_count"].fillna(0).gt(0)
    base["booking_channel"] = "unknown"
    base["booking_source"] = "unknown"
    base["booking_channel_known"] = False
    base["booking_source_known"] = False
    base["consent_state"] = np.where(base["has_email_consent"], "active", "inactive")
    base["registration_age_days"] = np.nan
    base["days_since_last_stay"] = np.nan

    base["tenant_open_rate_rank"] = base.groupby("tenant_id")["open_rate_90d"].rank(pct=True).fillna(0)
    base["tenant_click_rate_rank"] = base.groupby("tenant_id")["click_rate_90d"].rank(pct=True).fillna(0)
    base["tenant_booking_value_rank"] = base.groupby("tenant_id")["total_spend"].rank(pct=True).fillna(0)
    base["tenant_booking_recency_rank"] = base.groupby("tenant_id")["days_since_last_stay"].rank(pct=True).fillna(0)
    base["extracted_at"] = as_of.isoformat()

    base = assign_regimes(base)
    audit = {
        "rows": int(len(base)),
        "clusterable_rows": int(base["is_clusterable"].sum()),
        "regime_counts": {str(k): int(v) for k, v in base["regime_bucket"].value_counts().to_dict().items()},
        "as_of": as_of.isoformat(),
        "raw_dir": str(raw_dir),
    }
    return DatasetBuildResult(frame=base, audit=audit)


def main() -> None:
    t0 = time.time()
    OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    AUDIT_OUT.parent.mkdir(parents=True, exist_ok=True)
    result = build_contact_frame(RAW_DIR)
    result.frame.to_parquet(OUT_PARQUET, index=False, compression="snappy")
    result.audit["elapsed_seconds"] = round(time.time() - t0, 1)
    with AUDIT_OUT.open("w") as f:
        json.dump(result.audit, f, indent=2)
    print(f"Written {OUT_PARQUET} rows={len(result.frame):,}")
    print(f"Audit {AUDIT_OUT}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Add operator note**

Create `docs/reengagement_clustering.md` with:

```markdown
# Re-Engagement Clustering

This workflow builds contact-level unsupervised segments for CRM re-engagement campaigns. It is not a supervised churn score.

## Prototype From STO Export

Place the STO archive at `data/raw/data_6_6_2026.zip`, then run:

```bash
python -m src.reengagement_clustering.build_sto_contact_dataset
```

The command writes `data/processed/reengagement_sto_contacts.parquet` and `results/audits/reengagement_sto_contacts.json`.

## Production From Zoomle Extract

Export these files to `data/raw/zl_nwl3/` as parquet or CSV:

- `nwl3_klienti`
- `nwl3_gdpr`
- `nwl3_rezervace`
- `nwl3_mailer_processed`
- `nwl3_zoomtrack`

Then run:

```bash
python -m src.reengagement_clustering.build_zl_contact_dataset
```

The command writes `data/processed/reengagement_zl_contacts.parquet` and `results/audits/reengagement_zl_contacts.json`.
```

- [ ] **Step 3: Run production builder with a real local extract**

Run: `python -m src.reengagement_clustering.build_zl_contact_dataset`

Expected: writes `data/processed/reengagement_zl_contacts.parquet` and `results/audits/reengagement_zl_contacts.json`. If the local extract is absent, the command fails with a missing file path under `data/raw/zl_nwl3/`.

- [ ] **Step 4: Commit safety checkpoint**

Run: `rtk git status --short`.

Expected: generated parquet remains untracked or ignored. Do not commit implementation files without explicit user approval for public repo scope.

## Task 6: Fit K-Means And Write Segment Summary

**Files:**
- Create: `src/reengagement_clustering/fit.py`
- Create: `src/reengagement_clustering/labels.py`
- Create: `tests/test_reengagement_fit.py`

**Interfaces:**
- Consumes: contact parquet from STO or ZL builder.
- Consumes: `build_feature_matrix`.
- Produces CLI: `python -m src.reengagement_clustering.fit --input data/processed/reengagement_sto_contacts.parquet`
- Produces: `models/reengagement_clustering/model.pkl`
- Produces: `results/metrics/reengagement_clustering.json`
- Produces: `results/segments/reengagement_clustering_segments.json`

- [ ] **Step 1: Write fit smoke test**

Create `tests/test_reengagement_fit.py` with:

```python
from __future__ import annotations

import pandas as pd

from src.reengagement_clustering.fit import fit_clusters


def test_fit_clusters_assigns_only_clusterable_rows() -> None:
    rows = []
    for idx in range(12):
        rows.append(
            {
                "tenant_id": "hotel-a",
                "contact_id": f"c{idx}",
                "identity_source": "sto_sha256",
                "is_clusterable": idx < 10,
                "regime_bucket": "healthy_engaged" if idx < 10 else "cold_never_engaged",
                "sents_30d": 2 + idx,
                "sents_90d": 4 + idx,
                "opens_30d": 1,
                "opens_90d": 2,
                "clicks_30d": idx % 2,
                "clicks_90d": idx % 3,
                "open_rate_30d": 0.2,
                "open_rate_90d": 0.3,
                "click_rate_30d": 0.1,
                "click_rate_90d": 0.1,
                "days_since_last_email": idx,
                "days_since_last_open": idx + 1,
                "days_since_last_click": idx + 2,
                "open_rate_delta_30_vs_90": -0.1,
                "click_rate_delta_30_vs_90": 0.0,
                "language": "cs",
                "country": "CZ",
                "gender": "unknown",
                "consent_state": "active",
                "booking_channel": "direct",
                "booking_source": "web",
            }
        )
    df = pd.DataFrame(rows)

    result = fit_clusters(df, n_clusters=3)

    assert result.labels.notna().sum() == 10
    assert result.labels.isna().sum() == 2
    assert result.metrics["n_clusters"] == 3
    assert result.metrics["clusterable_rows"] == 10
```

- [ ] **Step 2: Run test to verify failure**

Run: `python -m pytest tests/test_reengagement_fit.py -v`

Expected: import failure for `src.reengagement_clustering.fit`.

- [ ] **Step 3: Implement label helper**

Create `src/reengagement_clustering/labels.py` with:

```python
from __future__ import annotations

import pandas as pd


def label_segments(clustered: pd.DataFrame) -> list[dict[str, object]]:
    segments: list[dict[str, object]] = []
    cluster_rows = clustered[clustered["cluster_id"].notna()].copy()
    if cluster_rows.empty:
        return segments

    numeric_cols = [
        "open_rate_90d",
        "click_rate_90d",
        "days_since_last_open",
        "stays_count",
        "total_spend",
        "tenant_booking_value_rank",
    ]
    available = [col for col in numeric_cols if col in cluster_rows.columns]
    global_means = cluster_rows[available].mean(numeric_only=True)

    for cluster_id, group in cluster_rows.groupby("cluster_id"):
        means = group[available].mean(numeric_only=True)
        contrasts = (means - global_means).sort_values(key=lambda s: s.abs(), ascending=False)
        top_contrasts = [
            {"feature": str(name), "mean": float(means[name]), "delta_vs_global": float(delta)}
            for name, delta in contrasts.head(5).items()
        ]
        open_rate = float(means.get("open_rate_90d", 0.0))
        spend_rank = float(means.get("tenant_booking_value_rank", 0.0))
        recency = float(means.get("days_since_last_open", 9999.0))
        if open_rate >= 0.3 and recency <= 30:
            label = "recently engaged"
        elif spend_rank >= 0.7:
            label = "valuable but quiet"
        elif open_rate < 0.1:
            label = "low engagement"
        else:
            label = "moderate engagement"
        segments.append(
            {
                "cluster_id": int(cluster_id),
                "label_candidate": label,
                "size": int(len(group)),
                "top_contrasts": top_contrasts,
                "regime_counts": {str(k): int(v) for k, v in group["regime_bucket"].value_counts().to_dict().items()},
            }
        )
    return sorted(segments, key=lambda row: row["cluster_id"])
```

- [ ] **Step 4: Implement fit module**

Create `src/reengagement_clustering/fit.py` with:

```python
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .features import build_feature_matrix
from .labels import label_segments
from .schema import FitResult

MODEL_OUT = Path("models/reengagement_clustering/model.pkl")
METRICS_OUT = Path("results/metrics/reengagement_clustering.json")
SEGMENTS_OUT = Path("results/segments/reengagement_clustering_segments.json")


def _encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def fit_clusters(df: pd.DataFrame, n_clusters: int = 5, random_state: int = 42) -> FitResult:
    clusterable = df[df["is_clusterable"].astype(bool)].copy()
    if len(clusterable) < n_clusters:
        raise ValueError(f"Need at least {n_clusters} clusterable rows, got {len(clusterable)}")

    matrix, numeric_features, categorical_features = build_feature_matrix(clusterable)
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numeric_features),
            ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", _encoder())]), categorical_features),
        ],
        remainder="drop",
    )
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("kmeans", KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")),
        ]
    )
    cluster_ids = model.fit_predict(matrix)
    labels = pd.Series(pd.NA, index=df.index, dtype="Int64")
    labels.loc[clusterable.index] = cluster_ids

    transformed = model.named_steps["preprocessor"].transform(matrix)
    silhouette = float(silhouette_score(transformed, cluster_ids)) if len(set(cluster_ids)) > 1 else None
    clustered = df.copy()
    clustered["cluster_id"] = labels
    segments = label_segments(clustered)
    metrics = {
        "n_rows": int(len(df)),
        "clusterable_rows": int(len(clusterable)),
        "n_clusters": int(n_clusters),
        "silhouette": silhouette,
        "cluster_sizes": {str(k): int(v) for k, v in pd.Series(cluster_ids).value_counts().sort_index().to_dict().items()},
        "regime_counts": {str(k): int(v) for k, v in df["regime_bucket"].value_counts().to_dict().items()},
    }
    return FitResult(labels=labels, metrics=metrics, segments=segments)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--clusters", type=int, default=5)
    args = parser.parse_args()

    df = pd.read_parquet(args.input)
    result = fit_clusters(df, n_clusters=args.clusters)
    df = df.copy()
    df["cluster_id"] = result.labels

    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    METRICS_OUT.parent.mkdir(parents=True, exist_ok=True)
    SEGMENTS_OUT.parent.mkdir(parents=True, exist_ok=True)
    with MODEL_OUT.open("wb") as f:
        pickle.dump({"input": args.input, "metrics": result.metrics}, f)
    with METRICS_OUT.open("w") as f:
        json.dump(result.metrics, f, indent=2)
    with SEGMENTS_OUT.open("w") as f:
        json.dump(result.segments, f, indent=2)
    print(f"Metrics {METRICS_OUT}")
    print(f"Segments {SEGMENTS_OUT}")
    print(f"Model metadata {MODEL_OUT}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run fit tests**

Run: `python -m pytest tests/test_reengagement_fit.py -v`

Expected: 1 test passes.

- [ ] **Step 6: Fit STO prototype clusters**

Run: `python -m src.reengagement_clustering.fit --input data/processed/reengagement_sto_contacts.parquet`

Expected: writes metrics, segment summary, and gitignored model metadata. Inspect `results/segments/reengagement_clustering_segments.json` before presenting persona labels.

- [ ] **Step 7: Commit safety checkpoint**

Run: `rtk git status --short`.

Expected: generated model artifact remains ignored. Do not commit implementation files without explicit user approval for public repo scope.

## Task 7: Stability Report

**Files:**
- Modify: `src/reengagement_clustering/fit.py`
- Modify: `tests/test_reengagement_fit.py`

**Interfaces:**
- Consumes two clustered contact frames with `tenant_id`, `contact_id`, `regime_bucket`, and `cluster_id`.
- Produces: `compute_stability(previous: pd.DataFrame, current: pd.DataFrame) -> dict[str, object]`

- [ ] **Step 1: Add stability test**

Append to `tests/test_reengagement_fit.py`:

```python
from src.reengagement_clustering.fit import compute_stability


def test_compute_stability_reports_overlap_and_regime_moves() -> None:
    previous = pd.DataFrame(
        [
            {"tenant_id": "h", "contact_id": "a", "regime_bucket": "healthy_engaged", "cluster_id": 0},
            {"tenant_id": "h", "contact_id": "b", "regime_bucket": "fading", "cluster_id": 1},
        ]
    )
    current = pd.DataFrame(
        [
            {"tenant_id": "h", "contact_id": "a", "regime_bucket": "healthy_engaged", "cluster_id": 0},
            {"tenant_id": "h", "contact_id": "b", "regime_bucket": "healthy_engaged", "cluster_id": 1},
            {"tenant_id": "h", "contact_id": "c", "regime_bucket": "cold_never_engaged", "cluster_id": pd.NA},
        ]
    )

    report = compute_stability(previous, current)

    assert report["overlap_rows"] == 2
    assert report["previous_rows"] == 2
    assert report["current_rows"] == 3
    assert report["regime_movement"]["fading -> healthy_engaged"] == 1
```

- [ ] **Step 2: Implement stability helper**

Append to `src/reengagement_clustering/fit.py` above `main()`:

```python
def compute_stability(previous: pd.DataFrame, current: pd.DataFrame) -> dict[str, object]:
    keys = ["tenant_id", "contact_id"]
    prev = previous[keys + ["regime_bucket", "cluster_id"]].rename(
        columns={"regime_bucket": "previous_regime", "cluster_id": "previous_cluster"}
    )
    curr = current[keys + ["regime_bucket", "cluster_id"]].rename(
        columns={"regime_bucket": "current_regime", "cluster_id": "current_cluster"}
    )
    overlap = prev.merge(curr, on=keys, how="inner")
    regime_moves = (
        overlap["previous_regime"].astype(str) + " -> " + overlap["current_regime"].astype(str)
    ).value_counts()
    stable_cluster = overlap["previous_cluster"].astype("string").eq(overlap["current_cluster"].astype("string"))
    return {
        "previous_rows": int(len(previous)),
        "current_rows": int(len(current)),
        "overlap_rows": int(len(overlap)),
        "cluster_retention_rate": float(stable_cluster.mean()) if len(overlap) else None,
        "regime_movement": {str(k): int(v) for k, v in regime_moves.to_dict().items()},
    }
```

- [ ] **Step 3: Run fit tests**

Run: `python -m pytest tests/test_reengagement_fit.py -v`

Expected: 2 tests pass.

- [ ] **Step 4: Commit safety checkpoint**

Run: `rtk git status --short`.

Expected: no raw or model artifacts are staged. Do not commit implementation files without explicit user approval for public repo scope.

## Task 8: Final Verification And Handoff

**Files:**
- Modify: `docs/reengagement_clustering.md`

**Interfaces:**
- Consumes: outputs from Tasks 4-7.
- Produces: a concise operator-facing summary of paths, commands, and interpretation guardrails.

- [ ] **Step 1: Run focused tests**

Run:

```bash
python -m pytest \
  tests/test_reengagement_features.py \
  tests/test_reengagement_regimes.py \
  tests/test_reengagement_sto_builder.py \
  tests/test_reengagement_fit.py \
  -v
```

Expected: all re-engagement clustering tests pass.

- [ ] **Step 2: Run STO prototype end-to-end**

Run:

```bash
python -m src.reengagement_clustering.build_sto_contact_dataset
python -m src.reengagement_clustering.fit --input data/processed/reengagement_sto_contacts.parquet
```

Expected:

- `results/audits/reengagement_sto_contacts.json` includes contact count and regime counts.
- `results/metrics/reengagement_clustering.json` includes clusterable rows, cluster sizes, and silhouette.
- `results/segments/reengagement_clustering_segments.json` includes five segment summaries.

- [ ] **Step 3: Update operator notes**

Append this section to `docs/reengagement_clustering.md`:

```markdown
## Interpretation Guardrails

- Rule regimes and k-means clusters answer different questions. Regimes explain reachability and engagement eligibility; clusters describe personas among contacts with enough signal.
- STO prototype results should be used to validate mechanics and persona readability, not production segment sizes.
- Production decisions should use the Zoomle `nwl3_*` extract because it is the source of truth for clients, consent, reservations, sends, and clicks.
- A segment label is a campaign planning aid, not a churn probability.
```

- [ ] **Step 4: Public repo safety check**

Run:

```bash
rtk git status --short
rtk git diff --stat
```

Expected: raw data, archives, parquet files, model binaries, and secrets are not staged. If committing, stop and ask for user approval before staging implementation files outside the currently allowed public repo scope.

- [ ] **Step 5: Execution handoff**

Report:

- Spec path: `docs/superpowers/specs/2026-07-28-churn-reengagement-clustering-design.md`
- Plan path: `docs/superpowers/plans/2026-07-28-churn-reengagement-clustering.md`
- Prototype command: `python -m src.reengagement_clustering.build_sto_contact_dataset`
- Fit command: `python -m src.reengagement_clustering.fit --input data/processed/reengagement_sto_contacts.parquet`
- Production command: `python -m src.reengagement_clustering.build_zl_contact_dataset`
- Remaining approval needed: whether implementation files may be committed in this public repo.

## Self-Review

Spec coverage:

- Clustering-first output is covered by Tasks 6 and 8.
- STO prototype and ZL production split is covered by Tasks 4 and 5.
- Tenant-preserving identity grain is covered by Tasks 1, 4, and 5.
- 30-day and 90-day engagement windows are covered by Tasks 1, 4, and 5.
- Rule regimes stay outside k-means through Tasks 2, 3, and 6.
- Segment labels and interpretability are covered by Task 6.
- Stability is covered by Task 7.
- Public repo safety is covered in Global Constraints and every checkpoint.

Placeholder scan:

- The plan contains no unresolved implementation markers.
- Every created code file has concrete interfaces and runnable verification commands.

Type consistency:

- `DatasetBuildResult` and `FitResult` are defined in Task 1 and reused by later tasks.
- `assign_regimes`, `build_feature_matrix`, `fit_clusters`, and `compute_stability` names are consistent across tests and implementation steps.
