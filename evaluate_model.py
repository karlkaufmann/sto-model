import json
import pickle

import numpy as np
import onnxruntime as ort
import pandas as pd

from feature_pipeline import DEFAULTS, FEATURE_COLS, IDX_TO_HOUR, build_features

TEST_CSV = "test_cases_data.csv"
MODEL_ONNX = "xgboost_model.onnx"
SCALER_PKL = "scaler.pkl"
METADATA_JSON = "model_metadata.json"
OUTPUT_CSV = "evaluated_test_cases.csv"


def main():
    df = pd.read_csv(TEST_CSV)
    print(f"Loaded {len(df)} test cases.")

    with open(METADATA_JSON, "r") as f:
        metadata = json.load(f)
    cat_mappings = metadata.get("category_mappings", {})

    # Backward-compatible defaults for new v2 fields.
    for col, default in DEFAULTS.items():
        if col not in df.columns:
            df[col] = default

    # Additional v2 fields inferred from existing send-time fields.
    if "sent_weekofyear" not in df.columns:
        df["sent_weekofyear"] = ((df["sent_day"].fillna(1).astype(int) - 1) // 7 + 1).clip(1, 53)
    if "sent_is_month_start" not in df.columns:
        df["sent_is_month_start"] = (df["sent_day"].fillna(1).astype(int) <= 3).astype(int)
    if "sent_is_month_end" not in df.columns:
        df["sent_is_month_end"] = (df["sent_day"].fillna(1).astype(int) >= 28).astype(int)

    feature_df = build_features(df, cat_mappings)
    with open(SCALER_PKL, "rb") as f:
        scaler = pickle.load(f)
    X_scaled = np.asarray(scaler.transform(feature_df[FEATURE_COLS]), dtype=np.float32)

    session = ort.InferenceSession(MODEL_ONNX)
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: X_scaled})
    pred_classes = outputs[0].astype(int).ravel()

    pred_hours = [IDX_TO_HOUR.get(c, 10) for c in pred_classes]
    df["new_pred_hour"] = pred_hours

    display_cols = ["sent_hour", "old_time", "new_pred_hour"]
    display_cols = [c for c in display_cols if c in df.columns]
    print("\nFirst 20 predictions:")
    print(df[display_cols].head(20).to_string(index=False))

    print("\nDistribution of predictions (`new_pred_hour`):")
    print(df["new_pred_hour"].value_counts().sort_index())

    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved evaluated dataset to {OUTPUT_CSV}.")


if __name__ == "__main__":
    main()
