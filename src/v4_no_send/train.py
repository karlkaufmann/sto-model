from __future__ import annotations

import hashlib
import json
import os
import pickle
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import onnxmltools
import pandas as pd
import xgboost as xgb
from onnxmltools.convert.common.data_types import FloatTensorType
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit

from .feature_pipeline import (
    BLOCKS,
    IDX_TO_HOUR,
    N_CLASSES,
    TARGET_HOURS,
    FeatureState,
    build_features,
    fit_feature_state,
)

RANDOM_STATE = 42
PARQUET_PATH = os.environ.get("V4_PARQUET_PATH", "data/processed/v4_no_send.parquet")
MODEL_DIR = Path(os.environ.get("V4_MODEL_DIR", "models/v4_no_send"))
METRICS_PATH = Path(os.environ.get("V4_METRICS_PATH", "results/metrics/v4_no_send.json"))
FEATURES_CFG_PATH = Path(os.environ.get("V4_FEATURES_CFG_PATH", "configs/v4_no_send/features.json"))
COMPARISON_PATH = Path(os.environ.get("V4_COMPARISON_PATH", "results/comparisons/v3_2_vs_v4_no_send.json"))
XGB_DEVICE = os.environ.get("XGB_DEVICE", "cuda")
RANDOM_SEARCH_ITERS = int(os.environ.get("V4_RANDOM_SEARCH_ITERS", "12"))
TOP_K_CATEGORIES = int(os.environ.get("V4_TOP_K_CATEGORIES", "200"))
MAX_ROWS = int(os.environ.get("V4_MAX_ROWS", "0"))

random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)


def topk_accuracy(probs: np.ndarray, y_true: np.ndarray, k: int) -> float:
    topk = np.argsort(-probs, axis=1)[:, :k]
    return float((topk == y_true.reshape(-1, 1)).any(axis=1).mean())


def compute_metrics(y_true: np.ndarray, probs: np.ndarray, tag: str = "") -> dict:
    y_pred = np.argmax(probs, axis=1)
    metrics = {
        "top1_accuracy": float(accuracy_score(y_true, y_pred)),
        "top3_accuracy": topk_accuracy(probs, y_true, 3),
        "precision_at_1_weighted": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "precision_at_1_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "auc_weighted_ovr": float(roc_auc_score(y_true, probs, multi_class="ovr", average="weighted")),
        "n_samples": int(len(y_true)),
    }
    per_class = recall_score(y_true, y_pred, average=None, labels=np.arange(N_CLASSES), zero_division=0)
    metrics["per_class_recall"] = {str(TARGET_HOURS[i]): float(v) for i, v in enumerate(per_class)}
    y_true_h = np.array([TARGET_HOURS[i] for i in y_true])
    y_pred_h = np.array([TARGET_HOURS[i] for i in y_pred])
    block_accuracy = {}
    for name, hours in BLOCKS:
        mask = np.isin(y_true_h, hours)
        if mask.sum() > 0:
            block_accuracy[name] = float(accuracy_score(y_true_h[mask], y_pred_h[mask]))
    metrics["block_accuracy"] = block_accuracy
    if tag:
        print(
            f"  [{tag}] top1={metrics['top1_accuracy']:.4f} "
            f"top3={metrics['top3_accuracy']:.4f} "
            f"p@1_w={metrics['precision_at_1_weighted']:.4f} "
            f"auc={metrics['auc_weighted_ovr']:.4f} n={metrics['n_samples']:,}"
        )
    return metrics


def stratified_split(df: pd.DataFrame):
    y = df["target"].astype(int)
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
    train_idx, temp_idx = next(sss1.split(df, y))
    df_train = df.iloc[train_idx].reset_index(drop=True)
    df_temp = df.iloc[temp_idx].reset_index(drop=True)
    y_temp = df_temp["target"].astype(int)
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=RANDOM_STATE)
    val_idx, test_idx = next(sss2.split(df_temp, y_temp))
    return (
        df_train,
        df_temp.iloc[val_idx].reset_index(drop=True),
        df_temp.iloc[test_idx].reset_index(drop=True),
    )


def random_search_xgb(
    X_train: np.ndarray,
    y_train: np.ndarray,
    sw_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> dict:
    param_grid = {
        "max_depth": [6, 8, 10],
        "learning_rate": [0.05, 0.08],
        "min_child_weight": [1, 5, 10],
        "reg_lambda": [1.0, 2.0, 5.0],
        "reg_alpha": [0.0, 0.1],
        "subsample": [0.85, 0.95],
        "colsample_bytree": [0.85, 0.95],
    }
    best_score = -1.0
    best_params: dict[str, float | int] = {}
    print(f"[5/8] XGBoost random search ({RANDOM_SEARCH_ITERS} trials, device={XGB_DEVICE})")
    for trial in range(RANDOM_SEARCH_ITERS):
        params = {key: random.choice(values) for key, values in param_grid.items()}
        model = xgb.XGBClassifier(
            objective="multi:softprob",
            num_class=N_CLASSES,
            n_estimators=320,
            early_stopping_rounds=25,
            eval_metric="mlogloss",
            tree_method="hist",
            device=XGB_DEVICE,
            random_state=RANDOM_STATE,
            n_jobs=1,
            verbosity=1,
            **params,
        )
        model.fit(X_train, y_train, sample_weight=sw_train, eval_set=[(X_val, y_val)], verbose=False)
        probs = model.predict_proba(X_val)
        preds = np.argmax(probs, axis=1)
        score = precision_score(y_val, preds, average="weighted", zero_division=0)
        print(f"  trial {trial + 1}/{RANDOM_SEARCH_ITERS}: p@1_w={score:.4f} params={params}")
        if score > best_score:
            best_score = score
            best_params = params
    print(f"  best p@1_w={best_score:.4f} params={best_params}")
    return best_params


def _load_previous_no_send_metrics() -> dict:
    path = Path("results/metrics/v3_2.json")
    if not path.exists():
        return {}
    with path.open() as f:
        metrics = json.load(f)
    previous = metrics.get("metrics_sent_known_0", {})
    return {
        "source": str(path),
        "top1_accuracy": previous.get("top1_accuracy"),
        "top3_accuracy": previous.get("top3_accuracy"),
        "precision_at_1_weighted": previous.get("precision_weighted"),
        "auc_weighted_ovr": previous.get("auc_weighted_ovr"),
        "n_samples": previous.get("n_samples"),
    }


def _feature_state_to_json(state: FeatureState) -> dict:
    return {
        "top_k": state.top_k,
        "feature_names": state.feature_names,
        "category_mappings": state.category_mappings,
        "numeric_medians": state.numeric_medians,
    }


def main() -> None:
    t0 = time.time()
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    FEATURES_CFG_PATH.parent.mkdir(parents=True, exist_ok=True)
    COMPARISON_PATH.parent.mkdir(parents=True, exist_ok=True)

    print("[1/8] Loading V4 no-send parquet")
    df = pd.read_parquet(PARQUET_PATH)
    df = df.dropna(subset=["target"]).copy()
    df["target"] = df["target"].astype(int)
    if MAX_ROWS > 0 and len(df) > MAX_ROWS:
        sample_idx, _ = next(
            StratifiedShuffleSplit(n_splits=1, train_size=MAX_ROWS, random_state=RANDOM_STATE).split(
                df, df["target"]
            )
        )
        df = df.iloc[sample_idx].reset_index(drop=True)
        df["target"] = df["target"].astype(int)
        print(f"  sampled_rows={len(df):,} (V4_MAX_ROWS={MAX_ROWS:,})")
    print(f"  rows={len(df):,} columns={len(df.columns):,}")

    print("[2/8] Stratified split 80/10/10")
    df_train, df_val, df_test = stratified_split(df)
    print(f"  train={len(df_train):,} val={len(df_val):,} test={len(df_test):,}")

    print("[3/8] Fitting feature encoders")
    state = fit_feature_state(df_train, top_k=TOP_K_CATEGORIES)
    X_train = build_features(df_train, state).to_numpy(dtype=np.float32)
    X_val = build_features(df_val, state).to_numpy(dtype=np.float32)
    X_test = build_features(df_test, state).to_numpy(dtype=np.float32)
    y_train = df_train["target"].to_numpy(dtype=np.int32)
    y_val = df_val["target"].to_numpy(dtype=np.int32)
    y_test = df_test["target"].to_numpy(dtype=np.int32)
    print(f"  features={len(state.feature_names):,}")

    print("[4/8] Computing class weights")
    class_counts = np.bincount(y_train, minlength=N_CLASSES)
    class_weights = np.where(class_counts > 0, class_counts.sum() / (N_CLASSES * class_counts), 1.0)
    sw_train = class_weights[y_train]

    best_params = random_search_xgb(X_train, y_train, sw_train, X_val, y_val)

    print("[6/8] Final XGBoost training")
    model = xgb.XGBClassifier(
        objective="multi:softprob",
        num_class=N_CLASSES,
        n_estimators=650,
        early_stopping_rounds=45,
        eval_metric="mlogloss",
        tree_method="hist",
        device=XGB_DEVICE,
        random_state=RANDOM_STATE,
        n_jobs=1,
        verbosity=1,
        **best_params,
    )
    t_train = time.time()
    model.fit(X_train, y_train, sample_weight=sw_train, eval_set=[(X_val, y_val)], verbose=False)
    train_seconds = time.time() - t_train
    print(f"  training_seconds={train_seconds:.1f} best_iteration={model.best_iteration}")

    print("[7/8] Evaluation")
    probs_val = model.predict_proba(X_val)
    probs_test = model.predict_proba(X_test)
    metrics_val = compute_metrics(y_val, probs_val, "val")
    metrics_test = compute_metrics(y_test, probs_test, "test")
    y_pred_test = np.argmax(probs_test, axis=1)
    cm = confusion_matrix(y_test, y_pred_test, labels=np.arange(N_CLASSES)).tolist()

    previous = _load_previous_no_send_metrics()
    deltas = {
        key: (
            metrics_test[key] - previous[key]
            if previous.get(key) is not None and key in metrics_test
            else None
        )
        for key in ["top1_accuracy", "top3_accuracy", "precision_at_1_weighted", "auc_weighted_ovr"]
    }
    gate_thresholds = {
        "precision_at_1_weighted": max(0.30, float(previous.get("precision_at_1_weighted") or 0) + 0.05),
        "top3_accuracy": max(0.45, float(previous.get("top3_accuracy") or 0) + 0.05),
    }
    gate_results = {key: metrics_test[key] >= threshold for key, threshold in gate_thresholds.items()}
    gate_passed = all(gate_results.values())

    print("[8/8] Exporting artifacts")
    model_pkl = MODEL_DIR / "model.pkl"
    model_onnx = MODEL_DIR / "model.onnx"
    with model_pkl.open("wb") as f:
        pickle.dump({"model": model, "feature_state": state}, f)

    initial_type = [("float_input", FloatTensorType([None, len(state.feature_names)]))]
    onnx_model = onnxmltools.convert_xgboost(model, initial_types=initial_type)
    with model_onnx.open("wb") as f:
        f.write(onnx_model.SerializeToString())
    onnx_md5 = hashlib.md5(model_onnx.read_bytes()).hexdigest()

    metadata = {
        "version": "v4_no_send",
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_path": PARQUET_PATH,
        "rows_total": int(len(df)),
        "target_hours": TARGET_HOURS,
        "class_mapping": {str(i): int(h) for i, h in IDX_TO_HOUR.items()},
        "feature_count": len(state.feature_names),
        "feature_names": state.feature_names,
        "model_family": "xgboost",
        "xgb_device": XGB_DEVICE,
        "best_hyperparams": best_params,
        "best_iteration": int(model.best_iteration),
        "metrics_val": metrics_val,
        "metrics_test": metrics_test,
        "previous_no_send_reference": previous,
        "metric_deltas_vs_previous_no_send": deltas,
        "gate_thresholds": gate_thresholds,
        "gate_results": gate_results,
        "gate_passed": gate_passed,
        "confusion_matrix": cm,
        "training_seconds": round(train_seconds, 1),
        "elapsed_seconds": round(time.time() - t0, 1),
        "onnx_md5": onnx_md5,
    }
    with METRICS_PATH.open("w") as f:
        json.dump(metadata, f, indent=2)

    features_cfg = {
        "schema_version": "v4_no_send",
        "model_version": "v4_no_send",
        "target": "local open hour among configured send buckets",
        "no_send_time_features": True,
        "excluded_feature_families": ["sent_local", "send_hour_local", "send_*", "label_*", "delivery_*", "target_*"],
        **_feature_state_to_json(state),
    }
    with FEATURES_CFG_PATH.open("w") as f:
        json.dump(features_cfg, f, indent=2)

    comparison = {
        "note": "V4 no-send predicts open hour without using campaign send timestamp features.",
        "previous_v3_2_sent_known_0": previous,
        "v4_no_send_test": metrics_test,
        "deltas": deltas,
        "gate_passed": gate_passed,
    }
    with COMPARISON_PATH.open("w") as f:
        json.dump(comparison, f, indent=2)

    print(f"Saved: {model_pkl}, {model_onnx}, {METRICS_PATH}, {FEATURES_CFG_PATH}, {COMPARISON_PATH}")
    print(f"Gate passed: {gate_passed}")


if __name__ == "__main__":
    main()
