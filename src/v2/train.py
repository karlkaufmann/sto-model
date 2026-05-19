import hashlib
import json
import os
import pickle
import time
from datetime import datetime, timezone

import numpy as np
import onnxmltools
import pandas as pd
from onnxmltools.convert.common.data_types import FloatTensorType
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType as SklearnFloatTensorType
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .feature_pipeline import FEATURE_COLS, IDX_TO_HOUR, prepare_training_data

try:
    import xgboost as xgb
except Exception:
    xgb = None

# ==== CONFIGURATION ==========================================================
CSV_FILENAME = "data/raw/nwl3_mailer_processed_view.csv"  # if missing, .zip variant is used
MODEL_PKL_FILENAME = "models/v2/model.pkl"
MODEL_ONNX_FILENAME = "models/v2/model.onnx"
METADATA_FILENAME = "results/metrics/v2.json"
FEATURES_CONFIG_FILENAME = "configs/v2/features.json"
SCALER_FILENAME = "models/v2/scaler.pkl"
N_ROWS = 1_000_000
MODEL_VERSION = "2.0"
MODEL_FAMILY = "xgboost" if xgb is not None else "sklearn-randomforest"


def load_raw_data(csv_path: str, n_rows: int = N_ROWS) -> pd.DataFrame:
    print(f"\n[1/7] Loading Data from {csv_path}...")
    if not os.path.exists(csv_path) and os.path.exists(csv_path + ".zip"):
        csv_path = csv_path + ".zip"
        print(f"  -> Found zip: {csv_path}")

    use_cols = [
        "sent",
        "opened",
        "web_id",
        "client_id",
        "template_id",
        "mailing_type",
        "state",
        "dsn",
        "nationality",
        "previous_stays_count",
        "sum_previous_stay_days",
    ]
    df = pd.read_csv(csv_path, usecols=use_cols, nrows=n_rows, low_memory=True)
    print(f"  -> Raw rows loaded: {len(df):,}")
    return df


def train_model(X: pd.DataFrame, y: pd.Series):
    print("\n[3/7] Train/Test Split + Scaling")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train: {len(X_train):,} | Test: {len(X_test):,}")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train).astype(np.float32)
    X_test_scaled = scaler.transform(X_test).astype(np.float32)

    print(f"\n[4/7] Model Training (v2, family={MODEL_FAMILY})")
    start = time.time()
    if xgb is not None:
        model = xgb.XGBClassifier(
            objective="multi:softprob",
            num_class=len(IDX_TO_HOUR),
            max_depth=6,
            n_estimators=140,
            learning_rate=0.08,
            subsample=0.9,
            colsample_bytree=0.9,
            tree_method="hist",
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=1,
            eval_metric="mlogloss",
            early_stopping_rounds=20,
            verbosity=0,
        )
        model.fit(
            X_train_scaled,
            y_train,
            eval_set=[(X_test_scaled, y_test)],
            verbose=False,
        )
    else:
        model = RandomForestClassifier(
            n_estimators=240,
            max_depth=20,
            min_samples_leaf=3,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced_subsample",
        )
        model.fit(X_train_scaled, y_train)
    elapsed = time.time() - start

    y_pred = model.predict(X_test_scaled).astype(int)
    probs = model.predict_proba(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    mean_confidence = float(np.mean(np.max(probs, axis=1)))

    print(f"  -> Training time: {elapsed:.1f}s")
    print(f"  -> Accuracy: {acc:.4f}")
    print(f"  -> Weighted F1: {f1:.4f}")
    print(f"  -> Mean confidence: {mean_confidence:.4f}")

    metrics = {
        "accuracy": float(acc),
        "weighted_f1": float(f1),
        "mean_confidence": mean_confidence,
        "training_seconds": float(elapsed),
    }
    return model, scaler, X_test, y_test, X_test_scaled, y_pred, metrics


def evaluate_blocks(y_true_idx: np.ndarray, y_pred_idx: np.ndarray) -> dict[str, float]:
    y_true_hours = np.array([IDX_TO_HOUR[int(i)] for i in y_true_idx])
    y_pred_hours = np.array([IDX_TO_HOUR[int(i)] for i in y_pred_idx])

    blocks = [
        ("7-8h", [7, 8]),
        ("9-10h", [9, 10]),
        ("11-12h", [11, 12]),
        ("15-16h", [15, 16]),
        ("19-20h", [19, 20]),
    ]
    scores: dict[str, float] = {}
    print("\n[5/7] Block-wise accuracy")
    for block_name, hours in blocks:
        mask = np.isin(y_true_hours, hours)
        if np.sum(mask) > 0:
            block_acc = accuracy_score(y_true_hours[mask], y_pred_hours[mask])
            scores[block_name] = float(block_acc)
            print(f"  -> {block_name}: {block_acc:.4f} (n={int(np.sum(mask))})")
    return scores


def export_to_onnx(model, n_features: int):
    print(f"\n[6/7] Export ONNX: {MODEL_ONNX_FILENAME}")
    if xgb is not None:
        initial_type = [("float_input", FloatTensorType([None, n_features]))]
        onnx_model = onnxmltools.convert_xgboost(model, initial_types=initial_type)
    else:
        initial_type = [("float_input", SklearnFloatTensorType([None, n_features]))]
        onnx_model = convert_sklearn(model, initial_types=initial_type)
    with open(MODEL_ONNX_FILENAME, "wb") as f:
        f.write(onnx_model.SerializeToString())

    with open(MODEL_ONNX_FILENAME, "rb") as f:
        md5 = hashlib.md5(f.read()).hexdigest()
    size = os.path.getsize(MODEL_ONNX_FILENAME)
    print(f"  -> ONNX size: {size:,} bytes")
    print(f"  -> ONNX md5:  {md5}")
    return md5


def save_artifacts(
    model,
    scaler,
    prepared_rows: dict[str, int],
    metrics: dict[str, float],
    block_metrics: dict[str, float],
    category_mappings: dict[str, dict[str, int]],
    onnx_md5: str,
):
    with open(MODEL_PKL_FILENAME, "wb") as f:
        pickle.dump(model, f)
    with open(SCALER_FILENAME, "wb") as f:
        pickle.dump(scaler, f)

    metadata = {
        "version": MODEL_VERSION,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "class_mapping": {str(k): v for k, v in IDX_TO_HOUR.items()},
        "dataset_rows_before_filter": prepared_rows["before"],
        "dataset_rows_after_filter": prepared_rows["after"],
        "feature_count": len(FEATURE_COLS),
        "model_family": MODEL_FAMILY,
        "features_schema_version": "v2",
        "accuracy": metrics["accuracy"],
        "weighted_f1": metrics["weighted_f1"],
        "mean_confidence": metrics["mean_confidence"],
        "training_seconds": metrics["training_seconds"],
        "block_accuracies": block_metrics,
        "onnx_md5": onnx_md5,
        "scaler_type": type(scaler).__name__,
        "category_mappings": category_mappings,
    }
    with open(METADATA_FILENAME, "w") as f:
        json.dump(metadata, f, indent=2)

    features_config = {
        "feature_names": FEATURE_COLS,
        "categorical_source_columns": list(category_mappings.keys()),
        "schema_version": "v2",
        "model_version": MODEL_VERSION,
    }
    with open(FEATURES_CONFIG_FILENAME, "w") as f:
        json.dump(features_config, f, indent=2)

    print(f"  -> Saved {MODEL_PKL_FILENAME}")
    print(f"  -> Saved {SCALER_FILENAME} ({type(scaler).__name__})")
    print(f"  -> Saved {METADATA_FILENAME}")
    print(f"  -> Saved {FEATURES_CONFIG_FILENAME}")


if __name__ == "__main__":
    raw_df = load_raw_data(CSV_FILENAME)

    print("\n[2/7] Feature Engineering + Target")
    prepared = prepare_training_data(raw_df)
    print(f"  -> Rows before filtering: {prepared.rows_before:,}")
    print(f"  -> Rows after filtering:  {prepared.rows_after:,}")
    print(f"  -> Final feature matrix:  {prepared.X.shape}")

    model, scaler, X_test, y_test, X_test_scaled, y_pred, metrics = train_model(prepared.X, prepared.y)
    block_metrics = evaluate_blocks(y_test.to_numpy(), y_pred)
    onnx_md5 = export_to_onnx(model, n_features=len(FEATURE_COLS))

    save_artifacts(
        model=model,
        scaler=scaler,
        prepared_rows={"before": prepared.rows_before, "after": prepared.rows_after},
        metrics=metrics,
        block_metrics=block_metrics,
        category_mappings=prepared.category_mappings,
        onnx_md5=onnx_md5,
    )

    print("\n[7/7] Training and export finished.")
