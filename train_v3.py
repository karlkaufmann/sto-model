import hashlib
import json
import pickle
import time
from datetime import datetime, timezone

import numpy as np
import onnxmltools
import pandas as pd
from onnxmltools.convert.common.data_types import FloatTensorType
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from feature_pipeline_v3 import (
    FEATURE_COLS,
    IDX_TO_HOUR,
    TARGET_HOURS,
    prepare_training_data_v3,
)

try:
    import xgboost as xgb
except Exception as exc:
    raise RuntimeError("xgboost is required for V3 training") from exc


DATASET_PATH = "Data/dataset.csv"
MODEL_PKL_PATH = "xgboost_model_v3.pkl"
MODEL_ONNX_PATH = "xgboost_model_v3.onnx"
SCALER_PATH = "scaler_v3.pkl"
METADATA_PATH = "model_metadata_v3.json"
FEATURES_CONFIG_PATH = "features_config_v3.json"
COMPARISON_PATH = "comparison_v2_vs_v3.json"
N_ROWS = None
RANDOM_STATE = 42


def topk_accuracy(probs: np.ndarray, y_true: np.ndarray, k: int = 3) -> float:
    topk = np.argsort(-probs, axis=1)[:, :k]
    hits = (topk == y_true.reshape(-1, 1)).any(axis=1)
    return float(hits.mean())


def build_calibration_summary(probs: np.ndarray, y_true: np.ndarray, n_bins: int = 10) -> dict:
    pred_idx = np.argmax(probs, axis=1)
    conf = probs[np.arange(len(probs)), pred_idx]
    correct = (pred_idx == y_true).astype(int)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    rows = []
    for i in range(n_bins):
        left = bins[i]
        right = bins[i + 1]
        if i == n_bins - 1:
            mask = (conf >= left) & (conf <= right)
        else:
            mask = (conf >= left) & (conf < right)

        if mask.sum() == 0:
            rows.append(
                {
                    "bin_left": float(left),
                    "bin_right": float(right),
                    "count": 0,
                    "mean_confidence": None,
                    "empirical_accuracy": None,
                }
            )
            continue

        rows.append(
            {
                "bin_left": float(left),
                "bin_right": float(right),
                "count": int(mask.sum()),
                "mean_confidence": float(conf[mask].mean()),
                "empirical_accuracy": float(correct[mask].mean()),
            }
        )
    return {"n_bins": n_bins, "bins": rows}


def campaign_prior_baseline(y_train: np.ndarray, campaign_train: np.ndarray, y_test: np.ndarray, campaign_test: np.ndarray) -> dict:
    prior_by_campaign: dict[int, np.ndarray] = {}
    for campaign in np.unique(campaign_train):
        mask = campaign_train == campaign
        counts = np.bincount(y_train[mask], minlength=len(TARGET_HOURS)).astype(np.float64)
        dist = counts / counts.sum() if counts.sum() > 0 else np.ones(len(TARGET_HOURS)) / len(TARGET_HOURS)
        prior_by_campaign[int(campaign)] = dist

    global_counts = np.bincount(y_train, minlength=len(TARGET_HOURS)).astype(np.float64)
    global_dist = global_counts / global_counts.sum()

    probs = np.zeros((len(y_test), len(TARGET_HOURS)), dtype=np.float64)
    for i, campaign in enumerate(campaign_test):
        probs[i] = prior_by_campaign.get(int(campaign), global_dist)

    pred = np.argmax(probs, axis=1)
    top1 = float((pred == y_test).mean())
    top3 = topk_accuracy(probs, y_test, k=3)
    return {
        "top1_accuracy": top1,
        "top3_accuracy": top3,
    }


def export_to_onnx(model, n_features: int) -> str:
    initial_type = [("float_input", FloatTensorType([None, n_features]))]
    onnx_model = onnxmltools.convert_xgboost(model, initial_types=initial_type)
    with open(MODEL_ONNX_PATH, "wb") as f:
        f.write(onnx_model.SerializeToString())
    with open(MODEL_ONNX_PATH, "rb") as f:
        md5 = hashlib.md5(f.read()).hexdigest()
    return md5


def write_comparison_json(metrics_v3: dict):
    # Direct apples-to-apples comparison with V2 is not valid due schema + class differences.
    comparison = {
        "v2_vs_v3_comparability": "limited",
        "reason": (
            "V2 and V3 are trained on different schema and target spaces. "
            "V2 uses send-time/context features and 10 classes, V3 uses recipient profile and 15 classes (7-21)."
        ),
        "v3_metrics": metrics_v3,
        "v2_reference_metadata_file": "model_metadata.json",
    }
    with open(COMPARISON_PATH, "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2)


def main():
    print("[1/7] Loading raw data")
    df = pd.read_csv(
        DATASET_PATH,
        sep=";",
        nrows=N_ROWS,
        low_memory=False,
        on_bad_lines="skip",
    )
    print(f"Loaded rows: {len(df):,}")

    print("[2/7] Preparing features")
    prepared = prepare_training_data_v3(df)
    X = prepared.X
    y = prepared.y
    print(f"Rows after filtering: {prepared.rows_after:,} / {prepared.rows_before:,}")

    print("[3/7] Train/val/test split (80/10/10 stratified)")
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.5,
        random_state=RANDOM_STATE,
        stratify=y_temp,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train).astype(np.float32)
    X_val_scaled = scaler.transform(X_val).astype(np.float32)
    X_test_scaled = scaler.transform(X_test).astype(np.float32)

    class_counts = np.bincount(y_train.to_numpy(), minlength=len(TARGET_HOURS))
    class_weights = np.where(class_counts > 0, class_counts.sum() / class_counts, 1.0)
    sample_weight = class_weights[y_train.to_numpy()]

    print("[4/7] Training XGBoost model")
    model = xgb.XGBClassifier(
        objective="multi:softprob",
        num_class=len(TARGET_HOURS),
        max_depth=7,
        n_estimators=240,
        learning_rate=0.08,
        subsample=0.9,
        colsample_bytree=0.9,
        tree_method="hist",
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        eval_metric="mlogloss",
        early_stopping_rounds=30,
        verbosity=1,
    )
    start = time.time()
    model.fit(
        X_train_scaled,
        y_train.to_numpy(),
        sample_weight=sample_weight,
        eval_set=[(X_val_scaled, y_val.to_numpy())],
        verbose=False,
    )
    train_seconds = time.time() - start

    print("[5/7] Evaluating")
    probs_test = model.predict_proba(X_test_scaled)
    y_pred = np.argmax(probs_test, axis=1)

    top1 = float(accuracy_score(y_test, y_pred))
    top3 = topk_accuracy(probs_test, y_test.to_numpy(), k=3)
    per_class_recall = recall_score(y_test, y_pred, average=None, labels=np.arange(len(TARGET_HOURS)))
    conf_mat = confusion_matrix(y_test, y_pred, labels=np.arange(len(TARGET_HOURS)))
    calibration = build_calibration_summary(probs_test, y_test.to_numpy(), n_bins=10)

    campaign_train = X_train["campaign_type_enc"].to_numpy().astype(int)
    campaign_test = X_test["campaign_type_enc"].to_numpy().astype(int)
    baseline = campaign_prior_baseline(
        y_train.to_numpy(),
        campaign_train,
        y_test.to_numpy(),
        campaign_test,
    )

    print(f"Top-1 accuracy: {top1:.4f}")
    print(f"Top-3 accuracy: {top3:.4f}")
    print(f"Baseline top-1: {baseline['top1_accuracy']:.4f}")
    print(f"Baseline top-3: {baseline['top3_accuracy']:.4f}")

    # Keep plan gate: if no gain, stop and force feature review.
    if (top1 < baseline["top1_accuracy"] + 0.01) and (top3 < baseline["top3_accuracy"] + 0.01):
        raise RuntimeError(
            "V3 is not significantly better than baseline. "
            "Stopping export to enforce feature review."
        )

    print("[6/7] Exporting artifacts")
    with open(MODEL_PKL_PATH, "wb") as f:
        pickle.dump(model, f)
    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)
    onnx_md5 = export_to_onnx(model, n_features=len(FEATURE_COLS))

    metrics = {
        "top1_accuracy": top1,
        "top3_accuracy": top3,
        "baseline_top1_accuracy": baseline["top1_accuracy"],
        "baseline_top3_accuracy": baseline["top3_accuracy"],
        "training_seconds": float(train_seconds),
    }

    metadata = {
        "version": "3.0",
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_path": DATASET_PATH,
        "rows_before_filter": int(prepared.rows_before),
        "rows_after_filter": int(prepared.rows_after),
        "dropped_rows": prepared.dropped_rows,
        "target_hours": TARGET_HOURS,
        "class_mapping": {str(i): int(h) for i, h in IDX_TO_HOUR.items()},
        "feature_count": len(FEATURE_COLS),
        "feature_names": FEATURE_COLS,
        "model_family": "xgboost",
        "metrics": metrics,
        "per_class_recall": {str(TARGET_HOURS[i]): float(v) for i, v in enumerate(per_class_recall)},
        "confusion_matrix": conf_mat.tolist(),
        "calibration": calibration,
        "category_mappings": prepared.category_mappings,
        "age_fill_value": prepared.age_fill_value,
        "onnx_md5": onnx_md5,
    }
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    features_cfg = {
        "schema_version": "v3",
        "model_version": "3.0",
        "feature_names": FEATURE_COLS,
        "categorical_source_columns": ["campaign_type", "email_domain", "country"],
        "numeric_source_columns": ["age"],
        "target_hours": TARGET_HOURS,
    }
    with open(FEATURES_CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(features_cfg, f, indent=2)

    write_comparison_json(metrics)

    print("[7/7] Done")
    print(f"Saved: {MODEL_PKL_PATH}, {MODEL_ONNX_PATH}, {SCALER_PATH}, {METADATA_PATH}, {FEATURES_CONFIG_PATH}, {COMPARISON_PATH}")


if __name__ == "__main__":
    main()
