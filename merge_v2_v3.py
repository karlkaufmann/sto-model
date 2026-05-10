"""
merge_v2_v3.py  —  Build unified V3.2 parquet from V2 zip + V3 CSV.

Outputs:
  unified_v3_2.parquet   (gitignored)

Schema columns written:
  target, dataset_source, sent_known,
  sent_hour, sent_dow, sent_day, sent_month, sent_weekofyear,
  sent_is_month_start, sent_is_month_end, time_to_open,
  email_domain, campaign_type, country_iso2, age,
  previous_stays_count, sum_previous_stay_days,
  email_provider_category, email_tld
"""

import io
import json
import sys
import time
import warnings
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

warnings.filterwarnings("ignore", category=FutureWarning)

from feature_pipeline_v3_2 import (
    TARGET_HOURS_V2,
    HOUR_TO_IDX,
    MAILING_TYPE_MAP,
    extract_email_domain,
    normalize_iso2,
    normalize_campaign_type,
    _get_provider_category,
    _get_tld,
)

V2_ZIP = "nwl3_mailer_processed_view.csv.zip"
V3_CSV = "Data/dataset.csv"
OUT_PARQUET = "unified_v3_2.parquet"
AUDIT_OUT = "merge_audit_v3_2.json"
CHUNK_SIZE = 300_000

V2_USE_COLS = [
    "recipient_email", "mailing_type", "nationality",
    "sent", "opened",
    "previous_stays_count", "sum_previous_stay_days",
    "state",
]

V3_USE_COLS = [
    "open_hour_local", "campaign_type", "email_domain",
    "age", "country", "bounced", "unsubscribed",
]

# PyArrow schema for uniform output
PA_SCHEMA = pa.schema([
    pa.field("target", pa.int8()),
    pa.field("dataset_source", pa.string()),
    pa.field("sent_known", pa.float32()),
    pa.field("sent_hour", pa.float32()),
    pa.field("sent_dow", pa.float32()),
    pa.field("sent_day", pa.float32()),
    pa.field("sent_month", pa.float32()),
    pa.field("sent_weekofyear", pa.float32()),
    pa.field("sent_is_month_start", pa.float32()),
    pa.field("sent_is_month_end", pa.float32()),
    pa.field("time_to_open", pa.float32()),
    pa.field("email_domain", pa.string()),
    pa.field("campaign_type", pa.string()),
    pa.field("country_iso2", pa.string()),
    pa.field("age", pa.float32()),
    pa.field("previous_stays_count", pa.float32()),
    pa.field("sum_previous_stay_days", pa.float32()),
    pa.field("email_provider_category", pa.string()),
    pa.field("email_tld", pa.string()),
])


def _to_f32_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").astype("float32")


def _process_v2_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
    # Filter bounced / deferred
    if "state" in chunk.columns:
        chunk = chunk[~chunk["state"].isin(["bounced", "deferred"])].copy()
    if len(chunk) == 0:
        return pd.DataFrame()

    # Parse timestamps
    chunk["sent_dt"] = pd.to_datetime(chunk["sent"], errors="coerce", utc=True)
    chunk["opened_dt"] = pd.to_datetime(chunk["opened"], errors="coerce", utc=True)
    chunk = chunk.dropna(subset=["sent_dt", "opened_dt"])
    if len(chunk) == 0:
        return pd.DataFrame()

    # Open hour and target
    chunk["open_hour"] = chunk["opened_dt"].dt.hour.astype(int)
    chunk = chunk[chunk["open_hour"].isin(TARGET_HOURS_V2)].copy()
    if len(chunk) == 0:
        return pd.DataFrame()
    chunk["target"] = chunk["open_hour"].map(HOUR_TO_IDX).astype(np.int8)

    # Sent time features
    chunk["sent_hour"] = chunk["sent_dt"].dt.hour.astype(np.float32)
    chunk["sent_dow"] = chunk["sent_dt"].dt.dayofweek.astype(np.float32)
    chunk["sent_day"] = chunk["sent_dt"].dt.day.astype(np.float32)
    chunk["sent_month"] = chunk["sent_dt"].dt.month.astype(np.float32)
    chunk["sent_weekofyear"] = chunk["sent_dt"].dt.isocalendar().week.astype(np.float32)
    chunk["sent_is_month_start"] = chunk["sent_dt"].dt.is_month_start.astype(np.float32)
    chunk["sent_is_month_end"] = chunk["sent_dt"].dt.is_month_end.astype(np.float32)
    delta = (chunk["opened_dt"] - chunk["sent_dt"]).dt.total_seconds() / 3600.0
    chunk["time_to_open"] = delta.clip(0, 72).fillna(2.0).astype(np.float32)

    # Profile
    chunk["email_domain"] = chunk["recipient_email"].apply(extract_email_domain)
    chunk["campaign_type"] = chunk["mailing_type"].apply(normalize_campaign_type)
    chunk["country_iso2"] = chunk["nationality"].apply(normalize_iso2)
    chunk["age"] = np.nan  # V2 has no age column
    chunk["previous_stays_count"] = _to_f32_series(chunk.get("previous_stays_count", 0))
    chunk["sum_previous_stay_days"] = _to_f32_series(chunk.get("sum_previous_stay_days", 0))
    chunk["email_provider_category"] = chunk["email_domain"].map(_get_provider_category)
    chunk["email_tld"] = chunk["email_domain"].map(_get_tld)
    chunk["dataset_source"] = "V2"
    chunk["sent_known"] = np.float32(1.0)

    keep = [
        "target", "dataset_source", "sent_known",
        "sent_hour", "sent_dow", "sent_day", "sent_month",
        "sent_weekofyear", "sent_is_month_start", "sent_is_month_end",
        "time_to_open", "email_domain", "campaign_type", "country_iso2",
        "age", "previous_stays_count", "sum_previous_stay_days",
        "email_provider_category", "email_tld",
    ]
    return chunk[keep].reset_index(drop=True)


def _process_v3_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
    # Filters
    bounced_norm = chunk.get("bounced", "none").fillna("none").astype(str).str.lower()
    unsub = pd.to_numeric(chunk.get("unsubscribed", 0), errors="coerce").fillna(0).astype(int)
    hour = pd.to_numeric(chunk.get("open_hour_local"), errors="coerce")
    keep = (bounced_norm == "none") & (unsub != 1) & (hour.isin(TARGET_HOURS_V2))
    chunk = chunk[keep].copy()
    if len(chunk) == 0:
        return pd.DataFrame()

    chunk["open_hour"] = hour[keep].astype(int)
    chunk["target"] = chunk["open_hour"].map(HOUR_TO_IDX).astype(np.int8)
    chunk["campaign_type"] = chunk.get("campaign_type", "unknown").apply(normalize_campaign_type)
    chunk["email_domain"] = chunk.get("email_domain", "unknown").fillna("unknown").astype(str).str.lower().str.strip()
    chunk["country_iso2"] = chunk.get("country", "unknown").apply(normalize_iso2)
    chunk["age"] = _to_f32_series(chunk.get("age", np.nan))
    chunk["age"] = chunk["age"].where((chunk["age"] >= 0) & (chunk["age"] <= 100))
    chunk["previous_stays_count"] = np.float32(0)
    chunk["sum_previous_stay_days"] = np.float32(0)
    chunk["email_provider_category"] = chunk["email_domain"].map(_get_provider_category)
    chunk["email_tld"] = chunk["email_domain"].map(_get_tld)
    chunk["dataset_source"] = "V3"
    chunk["sent_known"] = np.float32(0.0)

    # NaN for sent_* columns
    for col in ["sent_hour", "sent_dow", "sent_day", "sent_month",
                "sent_weekofyear", "sent_is_month_start", "sent_is_month_end",
                "time_to_open"]:
        chunk[col] = np.float32(np.nan)

    keep_cols = [
        "target", "dataset_source", "sent_known",
        "sent_hour", "sent_dow", "sent_day", "sent_month",
        "sent_weekofyear", "sent_is_month_start", "sent_is_month_end",
        "time_to_open", "email_domain", "campaign_type", "country_iso2",
        "age", "previous_stays_count", "sum_previous_stay_days",
        "email_provider_category", "email_tld",
    ]
    return chunk[keep_cols].reset_index(drop=True)


def main():
    t0 = time.time()
    out_path = Path(OUT_PARQUET)
    if out_path.exists():
        out_path.unlink()

    writer = None
    audit = {
        "v2_chunks": 0, "v2_rows_read": 0, "v2_rows_kept": 0,
        "v3_chunks": 0, "v3_rows_read": 0, "v3_rows_kept": 0,
        "unmapped_mailing_types": {},
    }
    unmapped: dict[str, int] = {}

    # ── V2 ────────────────────────────────────────────────────────────────────
    print("=== Processing V2 ===")
    zf = zipfile.ZipFile(V2_ZIP)
    fname = zf.namelist()[0]
    with zf.open(fname) as raw_f:
        wrapper = io.TextIOWrapper(raw_f, encoding="utf-8-sig")
        reader = pd.read_csv(
            wrapper,
            usecols=V2_USE_COLS,
            chunksize=CHUNK_SIZE,
            low_memory=False,
            on_bad_lines="skip",
        )
        for i, chunk in enumerate(reader):
            audit["v2_rows_read"] += len(chunk)

            # Track unmapped mailing_type values
            for val in chunk["mailing_type"].fillna("NaN").astype(str).unique():
                if val not in MAILING_TYPE_MAP:
                    unmapped[val] = unmapped.get(val, 0) + int((chunk["mailing_type"] == val).sum())

            processed = _process_v2_chunk(chunk)
            if len(processed) > 0:
                audit["v2_rows_kept"] += len(processed)
                table = pa.Table.from_pandas(processed, schema=PA_SCHEMA, preserve_index=False)
                if writer is None:
                    writer = pq.ParquetWriter(out_path, PA_SCHEMA, compression="snappy")
                writer.write_table(table)

            audit["v2_chunks"] += 1
            if (i + 1) % 20 == 0:
                elapsed = time.time() - t0
                print(f"  V2 chunk {i+1}: read {audit['v2_rows_read']:,} | kept {audit['v2_rows_kept']:,} | {elapsed:.0f}s")
                sys.stdout.flush()

    audit["unmapped_mailing_types"] = unmapped
    if unmapped:
        print(f"  Unmapped mailing_type values: {unmapped}")
    print(f"V2 done: {audit['v2_rows_kept']:,} rows kept from {audit['v2_rows_read']:,} read.")

    # ── V3 ────────────────────────────────────────────────────────────────────
    print("\n=== Processing V3 ===")
    for chunk in pd.read_csv(
        V3_CSV, sep=";", chunksize=CHUNK_SIZE, low_memory=False,
        usecols=V3_USE_COLS, on_bad_lines="skip",
    ):
        audit["v3_rows_read"] += len(chunk)
        processed = _process_v3_chunk(chunk)
        if len(processed) > 0:
            audit["v3_rows_kept"] += len(processed)
            table = pa.Table.from_pandas(processed, schema=PA_SCHEMA, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(out_path, PA_SCHEMA, compression="snappy")
            writer.write_table(table)
        audit["v3_chunks"] += 1
        if audit["v3_chunks"] % 10 == 0:
            print(f"  V3 chunk {audit['v3_chunks']}: read {audit['v3_rows_read']:,} | kept {audit['v3_rows_kept']:,}")
            sys.stdout.flush()

    if writer:
        writer.close()

    elapsed = time.time() - t0
    audit["elapsed_seconds"] = round(elapsed, 1)
    audit["total_rows"] = audit["v2_rows_kept"] + audit["v3_rows_kept"]

    with open(AUDIT_OUT, "w") as f:
        json.dump(audit, f, indent=2)

    print(f"\nDone in {elapsed:.0f}s")
    print(f"V2 kept: {audit['v2_rows_kept']:,}")
    print(f"V3 kept: {audit['v3_rows_kept']:,}")
    print(f"Total:   {audit['total_rows']:,}")
    print(f"Written: {OUT_PARQUET}")


if __name__ == "__main__":
    main()
