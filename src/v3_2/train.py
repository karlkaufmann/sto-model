"""
train_v3_2.py  —  Train V3.2 unified model (V2 + V3 parquet).

Steps:
  1. Load unified parquet
  2. Fit category mappings
  3. Compute V3-derived domain priors (from V3 train split only, no leak)
  4. Build features (NaN-safe)
  5. Stratified split 80/10/10 by (dataset_source, target)
  6. XGBoost random search (~20 combos) on val set, objective = weighted Precision
  7. Temperature scaling calibration on val set
  8. Full evaluation: total + per-source + per-input-mode
  9. Acceptance gate
  10. Export: pkl, ONNX, metadata, features config, comparison JSON
"""

import hashlib
import json
import pickle
import random
import shutil
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import onnxmltools
from onnxmltools.convert.common.data_types import FloatTensorType
from scipy.optimize import minimize_scalar
from sklearn.metrics import (
    accuracy_score, confusion_matrix, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import StratifiedShuffleSplit

import xgboost as xgb

from .feature_pipeline import (
    FEATURE_COLS, TARGET_HOURS_V2, HOUR_TO_IDX, IDX_TO_HOUR, N_CLASSES,
    BLOCKS, fit_category_mappings, build_features,
    PreparedTrainingDataV3_2,
)

warnings.filterwarnings("ignore")
random.seed(42)
np.random.seed(42)

PARQUET_PATH = "data/processed/unified_v3_2.parquet"
MODEL_PKL = "models/v3_2/model.pkl"
MODEL_ONNX = "models/v3_2/model.onnx"
SCALER_PKL = None   # No scaler for V3.2
METADATA_PATH = "results/metrics/v3_2.json"
FEATURES_CFG_PATH = "configs/v3_2/features.json"
COMPARISON_PATH = "results/comparisons/v2_vs_v3_vs_v3_2.json"
COMPARISON_BASELINE_PATH = "results/comparisons/v3_2_baseline.json"
COMPARISON_FEATURES_PATH = "results/comparisons/v3_2_features.json"
PRIORS_MIN_ROWS = 50
PRIOR_ALPHA = 10.0
RANDOM_STATE = 42
COUNTRY_MEAN_HOUR_MIN_SUPPORT = 200
V3_SEGMENT_BASELINE_PRECISION = 0.2133


# ── helpers ───────────────────────────────────────────────────────────────────
def topk_accuracy(probs: np.ndarray, y_true: np.ndarray, k: int) -> float:
    topk = np.argsort(-probs, axis=1)[:, :k]
    return float((topk == y_true.reshape(-1, 1)).any(axis=1).mean())


def compute_metrics(y_true: np.ndarray, probs: np.ndarray, tag: str = "") -> dict:
    y_pred = np.argmax(probs, axis=1)
    d = {
        "top1_accuracy": float(accuracy_score(y_true, y_pred)),
        "top3_accuracy": topk_accuracy(probs, y_true, 3),
        "precision_weighted": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "auc_weighted_ovr": float(roc_auc_score(y_true, probs, multi_class="ovr", average="weighted")),
        "n_samples": int(len(y_true)),
    }
    # per-class recall
    per_class = recall_score(y_true, y_pred, average=None, labels=np.arange(N_CLASSES), zero_division=0)
    d["per_class_recall"] = {str(TARGET_HOURS_V2[i]): float(v) for i, v in enumerate(per_class)}
    # block accuracy (V2 style)
    y_true_h = np.array([TARGET_HOURS_V2[i] for i in y_true])
    y_pred_h = np.array([TARGET_HOURS_V2[i] for i in y_pred])
    block_acc = {}
    for bname, hours in BLOCKS:
        mask = np.isin(y_true_h, hours)
        if mask.sum() > 0:
            block_acc[bname] = float(accuracy_score(y_true_h[mask], y_pred_h[mask]))
    d["block_accuracy"] = block_acc
    if tag:
        print(f"  [{tag}] top1={d['top1_accuracy']:.4f} top3={d['top3_accuracy']:.4f} "
              f"prec_w={d['precision_weighted']:.4f} AUC={d['auc_weighted_ovr']:.4f} n={d['n_samples']:,}")
    return d


def calibration_bins(probs: np.ndarray, y_true: np.ndarray, n_bins: int = 10) -> list:
    pred_idx = np.argmax(probs, axis=1)
    conf = probs[np.arange(len(probs)), pred_idx]
    correct = (pred_idx == y_true).astype(int)
    bins = np.linspace(0, 1, n_bins + 1)
    rows = []
    for i in range(n_bins):
        mask = (conf >= bins[i]) & (conf < bins[i + 1] if i < n_bins - 1 else conf <= bins[i + 1])
        rows.append({
            "bin_left": float(bins[i]), "bin_right": float(bins[i + 1]),
            "count": int(mask.sum()),
            "mean_confidence": float(conf[mask].mean()) if mask.sum() > 0 else None,
            "empirical_accuracy": float(correct[mask].mean()) if mask.sum() > 0 else None,
        })
    return rows


def temperature_scale(probs: np.ndarray, T: float) -> np.ndarray:
    logits = np.log(np.clip(probs, 1e-12, None))
    scaled = logits / T
    exp = np.exp(scaled - scaled.max(axis=1, keepdims=True))
    return exp / exp.sum(axis=1, keepdims=True)


def fit_temperature(probs_val: np.ndarray, y_val: np.ndarray) -> float:
    def nll(T):
        if T <= 0:
            return 1e9
        scaled = temperature_scale(probs_val, T)
        return -float(np.mean(np.log(np.clip(scaled[np.arange(len(y_val)), y_val], 1e-12, None))))
    result = minimize_scalar(nll, bounds=(0.1, 10.0), method="bounded")
    return float(result.x)


# ── Domain priors (no leak: computed only on V3 train rows) ──────────────────
def compute_domain_priors(
    domain_arr: np.ndarray,
    campaign_arr: np.ndarray,
    target_arr: np.ndarray,
    alpha: float = PRIOR_ALPHA,
    min_rows: int = PRIORS_MIN_ROWS,
) -> tuple[dict[str, list[float]], dict[str, list[float]]]:
    """Return (domain_priors, campaign_priors) dicts with smoothed distributions."""
    # Campaign-level prior
    campaign_counts: dict[str, np.ndarray] = {}
    for camp, tgt in zip(campaign_arr, target_arr):
        if camp not in campaign_counts:
            campaign_counts[camp] = np.zeros(N_CLASSES)
        campaign_counts[camp][tgt] += 1
    campaign_priors: dict[str, list[float]] = {}
    global_counts = np.zeros(N_CLASSES)
    for camp, counts in campaign_counts.items():
        global_counts += counts
        smoothed = (counts + alpha) / (counts.sum() + alpha * N_CLASSES)
        campaign_priors[camp] = smoothed.tolist()
    smoothed_global = (global_counts + alpha) / (global_counts.sum() + alpha * N_CLASSES)
    campaign_priors["_GLOBAL_"] = smoothed_global.tolist()

    # Domain+campaign-level prior
    domain_priors: dict[str, list[float]] = {}
    combo_counts: dict[str, np.ndarray] = {}
    for dom, camp, tgt in zip(domain_arr, campaign_arr, target_arr):
        key = f"{dom}|||{camp}"
        if key not in combo_counts:
            combo_counts[key] = np.zeros(N_CLASSES)
        combo_counts[key][tgt] += 1

    for key, counts in combo_counts.items():
        dom, camp = key.split("|||", 1)
        if counts.sum() >= min_rows:
            smoothed = (counts + alpha) / (counts.sum() + alpha * N_CLASSES)
        else:
            fallback = np.array(campaign_priors.get(camp, campaign_priors["_GLOBAL_"]))
            w = counts.sum() / min_rows
            raw = (counts + alpha) / (counts.sum() + alpha * N_CLASSES)
            smoothed = w * raw + (1 - w) * fallback
        domain_priors[key] = smoothed.tolist()

    return domain_priors, campaign_priors


# ── Stratified split by (dataset_source, target) ─────────────────────────────
def stratified_split(df: pd.DataFrame, val_frac: float = 0.1, test_frac: float = 0.1):
    strat_key = df["dataset_source"].astype(str) + "_" + df["target"].astype(str)
    # First split: 80 / 20
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=val_frac + test_frac, random_state=RANDOM_STATE)
    train_idx, temp_idx = next(sss1.split(df, strat_key))
    df_train = df.iloc[train_idx]
    df_temp = df.iloc[temp_idx]
    strat_temp = strat_key.iloc[temp_idx]
    # Second split: 50 / 50 of the temp → val & test each 10%
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=RANDOM_STATE)
    val_idx, test_idx = next(sss2.split(df_temp, strat_temp))
    df_val = df_temp.iloc[val_idx]
    df_test = df_temp.iloc[test_idx]
    return df_train.reset_index(drop=True), df_val.reset_index(drop=True), df_test.reset_index(drop=True)


# ── XGBoost random search ─────────────────────────────────────────────────────
def random_search_xgb(
    X_train: np.ndarray, y_train: np.ndarray, sw_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    n_iter: int = 20,
) -> dict:
    param_grid = {
        "max_depth": [6, 8, 10],
        "learning_rate": [0.05, 0.08],
        "min_child_weight": [1, 5, 10],
        "reg_lambda": [1.0, 5.0],
        "reg_alpha": [0.0, 0.1],
        "subsample": [0.8, 0.9],
        "colsample_bytree": [0.8, 0.9],
    }
    best_score = -1.0
    best_params = {}
    print(f"  Random search over {n_iter} combos …")
    for trial in range(n_iter):
        params = {k: random.choice(v) for k, v in param_grid.items()}
        model = xgb.XGBClassifier(
            objective="multi:softprob",
            num_class=N_CLASSES,
            n_estimators=300,
            early_stopping_rounds=25,
            eval_metric="mlogloss",
            tree_method="hist",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbosity=0,
            **params,
        )
        model.fit(
            X_train, y_train,
            sample_weight=sw_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
        probs = model.predict_proba(X_val)
        preds = np.argmax(probs, axis=1)
        score = precision_score(y_val, preds, average="weighted", zero_division=0)
        print(f"    trial {trial+1}/{n_iter}: params={params} prec_w={score:.4f}")
        if score > best_score:
            best_score = score
            best_params = params
    print(f"  Best: prec_w={best_score:.4f} params={best_params}")
    return best_params


def main():
    t0 = time.time()

    # ── 0. Snapshot baseline comparison (once, so 0.2133 survives rewrites) ──
    baseline_src = Path(COMPARISON_PATH)
    baseline_dst = Path(COMPARISON_BASELINE_PATH)
    if baseline_src.exists() and not baseline_dst.exists():
        shutil.copy2(baseline_src, baseline_dst)
        print(f"[0] Baseline snapshot saved → {COMPARISON_BASELINE_PATH}")

    # ── 1. Load parquet ───────────────────────────────────────────────────────
    print("[1/10] Loading unified parquet …")
    df = pd.read_parquet(PARQUET_PATH)
    print(f"  Total rows: {len(df):,}  V2: {(df['dataset_source']=='V2').sum():,}  V3: {(df['dataset_source']=='V3').sum():,}")

    # ── 2. Fit category mappings ──────────────────────────────────────────────
    print("[2/10] Fitting category mappings …")
    cat_maps = fit_category_mappings(df)

    # ── 3. Split (before computing priors to avoid leak) ─────────────────────
    print("[3/10] Stratified split 80/10/10 …")
    df_train, df_val, df_test = stratified_split(df)
    print(f"  Train: {len(df_train):,}  Val: {len(df_val):,}  Test: {len(df_test):,}")

    # ── 3b. Country mean hour (from ALL train rows, no leak) ──────────────────
    print("[3b] Computing country_mean_hour from training split …")
    hours_train = df_train["target"].map(IDX_TO_HOUR).astype(float)
    agg = (
        pd.DataFrame({"c": df_train["country_iso2"].astype(str), "h": hours_train})
        .groupby("c")["h"]
        .agg(["mean", "count"])
    )
    country_mean_hour_global = float(hours_train.mean())
    country_mean_hour_map: dict[str, float] = {
        str(c): float(row["mean"])
        for c, row in agg.iterrows()
        if row["count"] >= COUNTRY_MEAN_HOUR_MIN_SUPPORT
        and c not in {"unknown", "nan", ""}
    }
    print(f"  Countries with ≥{COUNTRY_MEAN_HOUR_MIN_SUPPORT} rows: {len(country_mean_hour_map)}"
          f"  global mean: {country_mean_hour_global:.2f}h")

    # ── 4. V3-derived domain priors (from V3 train rows only) ─────────────────
    print("[4/10] Computing V3 domain priors …")
    v3_train = df_train[df_train["dataset_source"] == "V3"]
    domain_priors, campaign_priors = compute_domain_priors(
        v3_train["email_domain"].to_numpy(),
        v3_train["campaign_type"].to_numpy(),
        v3_train["target"].to_numpy(),
    )
    print(f"  Domain+campaign combos with priors: {len(domain_priors):,}")

    # ── 5. Build features ─────────────────────────────────────────────────────
    print("[5/10] Building features …")
    bf_kwargs = dict(
        country_mean_hour_map=country_mean_hour_map,
        country_mean_hour_global=country_mean_hour_global,
    )
    X_train = build_features(df_train, cat_maps, domain_priors, campaign_priors, **bf_kwargs)
    X_val   = build_features(df_val,   cat_maps, domain_priors, campaign_priors, **bf_kwargs)
    X_test  = build_features(df_test,  cat_maps, domain_priors, campaign_priors, **bf_kwargs)
    y_train = df_train["target"].to_numpy().astype(int)
    y_val   = df_val["target"].to_numpy().astype(int)
    y_test  = df_test["target"].to_numpy().astype(int)

    Xtr = X_train.to_numpy(dtype=np.float32)
    Xvl = X_val.to_numpy(dtype=np.float32)
    Xte = X_test.to_numpy(dtype=np.float32)

    # Class weights
    class_counts = np.bincount(y_train, minlength=N_CLASSES)
    cw = np.where(class_counts > 0, class_counts.sum() / (N_CLASSES * class_counts), 1.0)
    sw_train = cw[y_train]

    # ── 6. Random search ──────────────────────────────────────────────────────
    print("[6/10] XGBoost random search …")
    best_params = random_search_xgb(Xtr, y_train, sw_train, Xvl, y_val, n_iter=20)

    # ── 7. Final train with best params ───────────────────────────────────────
    print("[7/10] Final XGBoost training …")
    model = xgb.XGBClassifier(
        objective="multi:softprob",
        num_class=N_CLASSES,
        n_estimators=500,
        early_stopping_rounds=40,
        eval_metric="mlogloss",
        tree_method="hist",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=1,
        **best_params,
    )
    t_train = time.time()
    model.fit(
        Xtr, y_train,
        sample_weight=sw_train,
        eval_set=[(Xvl, y_val)],
        verbose=False,
    )
    train_secs = time.time() - t_train
    print(f"  Training time: {train_secs:.1f}s  Best iter: {model.best_iteration}")

    # ── 8. Calibration (temperature scaling on val) ───────────────────────────
    print("[8/10] Temperature scaling …")
    probs_val = model.predict_proba(Xvl)
    T = fit_temperature(probs_val, y_val)
    print(f"  Calibration temperature T={T:.4f}")
    probs_val_cal = temperature_scale(probs_val, T)

    # ── 9. Evaluation ─────────────────────────────────────────────────────────
    print("[9/10] Evaluation …")
    probs_test = model.predict_proba(Xte)
    probs_test_cal = temperature_scale(probs_test, T)

    print("  === Uncalibrated ===")
    m_total = compute_metrics(y_test, probs_test, "total")
    print("  === Calibrated ===")
    m_total_cal = compute_metrics(y_test, probs_test_cal, "total_cal")

    # Per-source
    src_test = df_test["dataset_source"].to_numpy()
    v2_mask = src_test == "V2"
    v3_mask = src_test == "V3"
    if v2_mask.sum() > 0:
        m_v2 = compute_metrics(y_test[v2_mask], probs_test_cal[v2_mask], "V2_segment")
    else:
        m_v2 = {}
    if v3_mask.sum() > 0:
        m_v3 = compute_metrics(y_test[v3_mask], probs_test_cal[v3_mask], "V3_segment")
    else:
        m_v3 = {}

    # Per-input-mode
    sent_known_test = df_test["sent_known"].to_numpy() if "sent_known" in df_test.columns else v2_mask.astype(float)
    sk1 = sent_known_test == 1.0
    sk0 = sent_known_test == 0.0
    if sk1.sum() > 0:
        m_sk1 = compute_metrics(y_test[sk1], probs_test_cal[sk1], "sent_known=1")
    else:
        m_sk1 = {}
    if sk0.sum() > 0:
        m_sk0 = compute_metrics(y_test[sk0], probs_test_cal[sk0], "sent_known=0")
    else:
        m_sk0 = {}

    # Confusion matrix + calibration bins
    y_pred_test = np.argmax(probs_test_cal, axis=1)
    cm = confusion_matrix(y_test, y_pred_test, labels=np.arange(N_CLASSES)).tolist()
    cal_bins = calibration_bins(probs_test_cal, y_test)

    # ── 10. Acceptance gate ───────────────────────────────────────────────────
    print("[10/10] Acceptance gate …")
    gate = {
        "total_precision_weighted": m_total_cal.get("precision_weighted", 0),
        "total_auc_ovr": m_total_cal.get("auc_weighted_ovr", 0),
        "v2_segment_precision": m_v2.get("precision_weighted", 0),
        "v3_segment_precision": m_v3.get("precision_weighted", 0),
        "v3_segment_top3": m_v3.get("top3_accuracy", 0),
    }
    thresholds = {
        "total_precision_weighted": 0.55,
        "total_auc_ovr": 0.88,
        "v2_segment_precision": 0.75,
        "v3_segment_precision": 0.30,
        "v3_segment_top3": 0.45,
    }
    gate_pass = {k: gate[k] >= thresholds[k] for k in thresholds}
    all_pass = all(gate_pass.values())
    print("  Gate results:")
    for k, v in gate_pass.items():
        print(f"    {'PASS' if v else 'FAIL'} {k}: {gate[k]:.4f} (threshold {thresholds[k]})")

    if not all_pass:
        failing = [k for k, v in gate_pass.items() if not v]
        print(f"\n  WARNING: {len(failing)} gate(s) failed: {failing}")
        print("  Exporting anyway with gate_passed=False flag in metadata.")

    # ── Export ────────────────────────────────────────────────────────────────
    with open(MODEL_PKL, "wb") as f:
        pickle.dump(model, f)

    # ONNX export
    n_features = X_train.shape[1]
    initial_type = [("float_input", FloatTensorType([None, n_features]))]
    onnx_model = onnxmltools.convert_xgboost(model, initial_types=initial_type)
    with open(MODEL_ONNX, "wb") as f:
        f.write(onnx_model.SerializeToString())
    with open(MODEL_ONNX, "rb") as f:
        onnx_md5 = hashlib.md5(f.read()).hexdigest()

    # Metadata
    metadata = {
        "version": "3.2",
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_path": PARQUET_PATH,
        "rows_total": int(len(df)),
        "rows_v2": int((df["dataset_source"] == "V2").sum()),
        "rows_v3": int((df["dataset_source"] == "V3").sum()),
        "target_hours": TARGET_HOURS_V2,
        "class_mapping": {str(i): int(h) for i, h in IDX_TO_HOUR.items()},
        "feature_count": len(FEATURE_COLS),
        "feature_names": FEATURE_COLS,
        "model_family": "xgboost",
        "best_hyperparams": best_params,
        "calibration_temperature": T,
        "metrics_uncalibrated": m_total,
        "metrics_calibrated": m_total_cal,
        "metrics_v2_segment": m_v2,
        "metrics_v3_segment": m_v3,
        "metrics_sent_known_1": m_sk1,
        "metrics_sent_known_0": m_sk0,
        "confusion_matrix": cm,
        "calibration_bins": cal_bins,
        "training_seconds": round(train_secs, 1),
        "gate_passed": all_pass,
        "gate_results": gate,
        "gate_thresholds": thresholds,
        "category_mappings": cat_maps,
        "domain_priors_count": len(domain_priors),
        "campaign_priors": campaign_priors,
        "country_mean_hour_map": country_mean_hour_map,
        "country_mean_hour_global": country_mean_hour_global,
        "country_mean_hour_min_support": COUNTRY_MEAN_HOUR_MIN_SUPPORT,
        "onnx_md5": onnx_md5,
    }
    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)

    features_cfg = {
        "schema_version": "v3_2",
        "model_version": "3.2",
        "feature_names": FEATURE_COLS,
        "sent_features": ["sent_hour", "sent_dow", "sent_day", "sent_month",
                          "sent_weekofyear", "sent_is_month_start", "sent_is_month_end", "time_to_open"],
        "categorical_source_columns": ["campaign_type", "country_iso2", "email_domain",
                                       "email_provider_category", "email_tld", "dataset_source"],
        "numeric_source_columns": ["age", "previous_stays_count", "sum_previous_stay_days"],
        "target_hours": TARGET_HOURS_V2,
        "nan_strategy": "xgboost_native",
        "scaler": None,
    }
    with open(FEATURES_CFG_PATH, "w") as f:
        json.dump(features_cfg, f, indent=2)

    # Comparison JSON
    def _load_ref(path: str, keys: list[str]) -> dict:
        try:
            with open(path) as fh:
                m = json.load(fh)
            return {k: m.get(k) for k in keys}
        except FileNotFoundError:
            return {"error": f"{path} not found"}

    v2_ref = _load_ref("results/metrics/v2.json", ["accuracy", "weighted_f1", "mean_confidence", "block_accuracies"])
    v3_ref = _load_ref("results/metrics/v3.json", ["metrics", "per_class_recall"])
    comparison = {
        "note": (
            "V2 uses 10 classes with sent_* features on CRM data (2025). "
            "V3 uses 15 classes (7-21h) on open-profile data (2026). "
            "V3.2 unifies both with 10 classes, NaN-aware model, and optional sent_* features."
        ),
        "V2_reference": v2_ref,
        "V3_reference": {"metrics_10class_subset": {"auc": 0.5897, "precision_weighted": 0.1909}},
        "V3_2": {
            "metrics_calibrated": m_total_cal,
            "v2_segment": m_v2,
            "v3_segment": m_v3,
            "gate_passed": all_pass,
        },
    }
    with open(COMPARISON_PATH, "w") as f:
        json.dump(comparison, f, indent=2)

    # Features comparison summary
    new_v3_prec = m_v3.get("precision_weighted", 0.0)
    delta = new_v3_prec - V3_SEGMENT_BASELINE_PRECISION
    features_comparison = {
        "baseline_v3_segment_precision": V3_SEGMENT_BASELINE_PRECISION,
        "new_v3_segment_precision": round(new_v3_prec, 6),
        "delta": round(delta, 6),
        "target": 0.30,
        "target_met": bool(new_v3_prec >= 0.30),
        "new_features_added": [
            "age_segment_enc", "domain_type_enc",
            "domain_type_x_campaign_enc", "country_mean_hour",
        ],
    }
    with open(COMPARISON_FEATURES_PATH, "w") as f:
        json.dump(features_comparison, f, indent=2)

    elapsed = time.time() - t0
    print(f"\nTotal elapsed: {elapsed/60:.1f} min")
    print(f"Saved: {MODEL_PKL}, {MODEL_ONNX}, {METADATA_PATH}, {FEATURES_CFG_PATH}, {COMPARISON_PATH}")
    print(f"Gate passed: {all_pass}")


if __name__ == "__main__":
    main()
