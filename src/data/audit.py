import json
from collections import Counter

import pandas as pd


DATASET_PATH = "data/raw/dataset.csv"
OUTPUT_PATH = "results/audits/v3_data.json"
CHUNK_SIZE = 200_000
TOP_N = 25


def _safe_int(value):
    try:
        return int(value)
    except Exception:
        return None


def _counter_to_top_dict(counter: Counter, top_n: int = TOP_N) -> dict[str, int]:
    return {str(k): int(v) for k, v in counter.most_common(top_n)}


def main():
    global_rows = 0
    columns = None
    non_null_counts: dict[str, int] = {}
    unique_counters: dict[str, Counter] = {}

    open_hour_counter = Counter()
    campaign_counter = Counter()
    bounced_counter = Counter()
    unsubscribed_counter = Counter()
    timezone_counter = Counter()
    device_counter = Counter()
    os_counter = Counter()
    email_client_counter = Counter()

    for chunk in pd.read_csv(
        DATASET_PATH,
        sep=";",
        chunksize=CHUNK_SIZE,
        low_memory=False,
        on_bad_lines="skip",
    ):
        if columns is None:
            columns = list(chunk.columns)
            non_null_counts = {c: 0 for c in columns}
            unique_counters = {c: Counter() for c in columns}

        chunk_rows = len(chunk)
        global_rows += chunk_rows

        for col in columns:
            series = chunk[col]
            non_null_counts[col] += int(series.notna().sum())
            value_counts = series.value_counts(dropna=False)
            unique_counters[col].update({str(k): int(v) for k, v in value_counts.items()})

        open_hour_counter.update(chunk["open_hour_local"].dropna().astype(int).tolist())
        campaign_counter.update(chunk["campaign_type"].fillna("NaN").astype(str).tolist())
        bounced_counter.update(chunk["bounced"].fillna("NaN").astype(str).tolist())
        unsubscribed_counter.update(chunk["unsubscribed"].fillna("NaN").astype(str).tolist())
        timezone_counter.update(chunk["user_timezone"].fillna("NaN").astype(str).tolist())
        device_counter.update(chunk["device_type"].fillna("NaN").astype(str).tolist())
        os_counter.update(chunk["os"].fillna("NaN").astype(str).tolist())
        email_client_counter.update(chunk["email_client"].fillna("NaN").astype(str).tolist())

    if global_rows == 0 or columns is None:
        raise RuntimeError("Dataset is empty or could not be read.")

    missing_rate = {
        col: round(1.0 - (non_null_counts[col] / global_rows), 6)
        for col in columns
    }
    unique_count = {
        col: int(len(unique_counters[col]))
        for col in columns
    }

    open_7_21 = sum(v for k, v in open_hour_counter.items() if 7 <= _safe_int(k) <= 21)
    open_outside = global_rows - open_7_21
    rows_bounced_not_none = sum(
        v for k, v in bounced_counter.items() if str(k).strip().lower() != "none"
    )
    rows_unsubscribed = sum(
        v for k, v in unsubscribed_counter.items() if str(k).strip() in {"1", "1.0"}
    )

    recommended_drop_columns = [
        "click_timestamp",
        "device_type",
        "os",
        "email_client",
        "zip_code",
        "user_timezone",
    ]

    recommended_feature_columns = [
        "email_domain",
        "country",
        "age",
        "campaign_type",
    ]

    row_filters = {
        "keep_open_hour_local_range_inclusive": [7, 21],
        "drop_bounced_not_none": True,
        "drop_unsubscribed_equals_1": True,
    }

    audit = {
        "dataset_path": DATASET_PATH,
        "rows_total": int(global_rows),
        "columns": columns,
        "missing_rate": missing_rate,
        "non_null_count": {k: int(v) for k, v in non_null_counts.items()},
        "unique_count": unique_count,
        "top_values": {
            "open_hour_local": _counter_to_top_dict(open_hour_counter, top_n=24),
            "campaign_type": _counter_to_top_dict(campaign_counter),
            "bounced": _counter_to_top_dict(bounced_counter),
            "unsubscribed": _counter_to_top_dict(unsubscribed_counter),
            "user_timezone": _counter_to_top_dict(timezone_counter),
            "device_type": _counter_to_top_dict(device_counter),
            "os": _counter_to_top_dict(os_counter),
            "email_client": _counter_to_top_dict(email_client_counter),
        },
        "filter_impact": {
            "rows_open_hour_7_21": int(open_7_21),
            "rows_open_hour_outside_7_21": int(open_outside),
            "rows_bounced_not_none": int(rows_bounced_not_none),
            "rows_unsubscribed_equals_1": int(rows_unsubscribed),
        },
        "recommended_drop_columns": recommended_drop_columns,
        "recommended_feature_columns": recommended_feature_columns,
        "row_filters": row_filters,
        "notes": [
            "click_timestamp is leakage-prone and sparse.",
            "device_type/os/email_client/user_timezone are low-information in current dataset profile.",
            "zip_code has high missingness and high cardinality.",
        ],
    }

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2)

    print(f"Audit written to {OUTPUT_PATH}")
    print(f"Rows total: {global_rows:,}")
    print(f"Rows open_hour in [7,21]: {open_7_21:,}")
    print(f"Rows bounced != none: {rows_bounced_not_none:,}")
    print(f"Rows unsubscribed == 1: {rows_unsubscribed:,}")


if __name__ == "__main__":
    main()
