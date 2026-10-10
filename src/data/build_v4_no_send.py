from __future__ import annotations

import json
import os
import time
import zipfile
from collections import Counter
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from src.v4_no_send.feature_pipeline import CATEGORICAL_COLS, HOUR_TO_IDX, NUMERIC_COLS, TARGET_HOURS

RAW_ZIP = os.environ.get("V4_RAW_ZIP", "data/raw/data_6_6_2026.zip")
OUT_PARQUET = os.environ.get("V4_OUT_PARQUET", "data/processed/v4_no_send.parquet")
AUDIT_OUT = os.environ.get("V4_AUDIT_OUT", "results/audits/v4_no_send_data.json")

LABEL_COLS = ["sent_local", "target_opened_anytime", "label_open_delay_seconds"]
READ_COLS = LABEL_COLS + CATEGORICAL_COLS + NUMERIC_COLS


def _read_member(zip_file: zipfile.ZipFile, name: str, columns: list[str]) -> pd.DataFrame:
    with zip_file.open(name) as fh:
        schema_names = pq.ParquetFile(fh).schema_arrow.names
    available = [col for col in columns if col in schema_names]
    with zip_file.open(name) as fh:
        return pq.read_table(fh, columns=available).to_pandas()


def _build_labeled_frame(df: pd.DataFrame) -> pd.DataFrame:
    opened = pd.to_numeric(df["target_opened_anytime"], errors="coerce").fillna(0).eq(1)
    delay = pd.to_numeric(df["label_open_delay_seconds"], errors="coerce")
    sent = pd.to_datetime(df["sent_local"], errors="coerce")
    mask = opened & delay.notna() & sent.notna() & (delay >= 0)
    if not mask.any():
        return pd.DataFrame()

    open_hour = (sent[mask] + pd.to_timedelta(delay[mask], unit="s")).dt.hour.astype(int)
    target_mask = open_hour.isin(TARGET_HOURS)
    if not target_mask.any():
        return pd.DataFrame()

    feature_cols = [col for col in CATEGORICAL_COLS + NUMERIC_COLS if col in df.columns]
    out = df.loc[mask, feature_cols].loc[target_mask].copy()
    out["target"] = open_hour[target_mask].map(HOUR_TO_IDX).astype("int8").to_numpy()
    return out


def _normalize_output_types(data: pd.DataFrame) -> pd.DataFrame:
    out = data.copy()
    for col in CATEGORICAL_COLS:
        if col in out.columns:
            out[col] = (
                out[col]
                .fillna("unknown")
                .astype(str)
                .str.strip()
                .replace({"": "unknown", "nan": "unknown", "None": "unknown", "NaN": "unknown"})
            )
    for col in NUMERIC_COLS:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    out["target"] = out["target"].astype("int8")
    return out


def main() -> None:
    t0 = time.time()
    raw_path = Path(RAW_ZIP)
    out_path = Path(OUT_PARQUET)
    audit_path = Path(AUDIT_OUT)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.parent.mkdir(parents=True, exist_ok=True)

    frames: list[pd.DataFrame] = []
    target_counts: Counter[int] = Counter()
    file_rows = []
    rows_total = 0
    rows_opened = 0
    rows_labeled = 0
    rows_target = 0

    with zipfile.ZipFile(raw_path) as zf:
        members = [name for name in zf.namelist() if name.endswith(".parquet")]
        for name in members:
            df = _read_member(zf, name, READ_COLS)
            rows_total += len(df)
            rows_opened += int(pd.to_numeric(df["target_opened_anytime"], errors="coerce").fillna(0).eq(1).sum())

            labeled = _build_labeled_frame(df)
            rows_target += len(labeled)
            if len(labeled) > 0:
                target_counts.update(labeled["target"].astype(int).value_counts().to_dict())
                frames.append(labeled)
            file_rows.append({"file": name, "rows": int(len(df)), "target_rows": int(len(labeled))})
            print(f"{name}: rows={len(df):,} target_rows={len(labeled):,}")

    if not frames:
        raise RuntimeError(f"No labeled target rows found in {raw_path}")

    data = _normalize_output_types(pd.concat(frames, ignore_index=True))
    data.to_parquet(out_path, index=False, compression="snappy")
    rows_labeled = int(len(data))

    audit = {
        "raw_zip": str(raw_path),
        "out_parquet": str(out_path),
        "parquet_files": len(file_rows),
        "rows_total": int(rows_total),
        "rows_opened_anytime": int(rows_opened),
        "rows_labeled_10class": rows_labeled,
        "target_hours": TARGET_HOURS,
        "target_counts": {str(TARGET_HOURS[idx]): int(count) for idx, count in sorted(target_counts.items())},
        "categorical_columns": [col for col in CATEGORICAL_COLS if col in data.columns],
        "numeric_columns": [col for col in NUMERIC_COLS if col in data.columns],
        "file_rows": file_rows,
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    with audit_path.open("w") as f:
        json.dump(audit, f, indent=2)

    print(f"Written {out_path} rows={rows_target:,}")
    print(f"Audit {audit_path}")


if __name__ == "__main__":
    main()
