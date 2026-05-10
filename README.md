# Send-time prediction (mailing)

Repository for **research and evaluation** of a multiclass model that predicts **which discrete hour** (within allowed send windows) is associated with observed engagement, given features derived from the send timestamp and identifiers.

This repo is intentionally **minimal for public sharing**: notebooks, evaluation code, and documentation only. Raw data, archives, trained weights, and secrets are excluded via `.gitignore`.

---

## Problem statement

- **Task:** Multiclass classification over **10 ordered hour buckets** (mapped from class index 0–9 to wall-clock hours such as 7, 8, …, 20 depending on product rules).
- **Use case:** Support tooling that suggests **when to send** mailings, under the constraint that only certain hours are actionable.
- **Important nuance:** The model is trained on **historical behaviour** (conditional on your label construction). It is **not** a causal “best time” guarantee; offline metrics describe fit to the training distribution, not uplift.

---

## Repository layout

| Path | Role |
|------|------|
| `model3.ipynb` | End-to-end **exploration and training** narrative: data prep, stratified split, XGBoost training, block metrics, optional ONNX export. |
| `evaluate_model.py` | **Batch evaluation** script: loads test rows, builds features (via `feature_pipeline` when present), applies scaler + ONNX, writes prediction distribution. |
| `.cursor/rules/` | Agent rules for safe public commits and commit message style. |
| `.agent/skills/public-repo-packaging/` | Checklist skill for allowlist-only staging before publish. |

Artifacts such as `xgboost_model.onnx`, `scaler.pkl`, `model_metadata.json`, and CSV inputs are **not tracked** here; keep them outside git or in private storage.

---

## Environment

Use Python **3.10+** (3.12 is fine). Typical stack:

- `pandas`, `numpy`
- `xgboost` (training in notebook)
- `onnxruntime` (inference in `evaluate_model.py`)
- `scikit-learn` (splits / metrics in notebook)
- `onnxmltools` (optional export path in notebook)

Pin versions in your own `requirements-lock` for reproducibility; this repo does not ship a frozen lockfile by design.

---

## How to run

### 1. Notebook (`model3.ipynb`)

1. Place your **training CSV** locally (path is configured inside the notebook; do not commit it).
2. Run cells top to bottom: load → features + target → train → evaluate blocks → export if needed.
3. Inspect **per-block accuracy** and **prediction histograms** before treating any export as production-ready.

### 2. Evaluation script (`evaluate_model.py`)

Expects **local-only** files next to the script (not in git):

- `test_cases_data.csv` — rows with at least the columns required by `feature_pipeline.build_features`
- `xgboost_model.onnx`, `scaler.pkl`, `model_metadata.json` — aligned with the training export

Run:

```bash
python evaluate_model.py
```

Output `evaluated_test_cases.csv` is written beside the script; it is gitignored.

---

### 3. V3 retraining on new schema (`Data/dataset.csv`)


V3 is a parallel model family that uses a different schema than V2:

- Target classes are `open_hour_local` filtered to `7..21` (15 classes).
- Features are recipient-profile level: `email_domain`, `country`, `age`, `campaign_type`.
- V3 keeps V2 untouched; artifacts are written with `_v3` suffixes.

Run full pipeline:

```bash
python data_audit_v3.py
python train_v3.py
```

Expected V3 artifacts:

- `xgboost_model_v3.pkl`, `xgboost_model_v3.onnx`, `scaler_v3.pkl`
- `model_metadata_v3.json`, `features_config_v3.json`
- `comparison_v2_vs_v3.json` (documents limited comparability of V2 vs V3)

Known limitations:

- No absolute send timestamp in V3 source, so no time-based split (uses stratified random split).
- No user ID in source, so personalization is segment-level, not user-level.
- V3 predicts likely open-hour profile, not causal uplift from intervention timing.

### 4. V3.2 unified model (`merge_v2_v3.py` + `train_v3_2.py`)

V3.2 merges the V2 CRM dataset and the V3 open-profile dataset into a single NaN-aware model.

**Strategy:**

- Target: same 10 hours as V2 (`[7, 8, 9, 10, 11, 12, 15, 16, 19, 20]`) for direct comparability.
- V2 rows contribute `sent_*` features (`sent_hour`, `sent_dow`, `time_to_open`, …); V3 rows have NaN for these.
- XGBoost uses `tree_method=hist` which handles NaN natively — no imputation needed.
- V3-derived domain priors (`P(open_hour | email_domain, campaign_type)`) are added as 10 float features.
- A `sent_known` indicator flag lets the model differentiate inference modes.
- Temperature scaling calibration is fitted on the val set.

**Schema mapping V2 ↔ V3:**

| V3.2 field | V2 source | V3 source |
|---|---|---|
| `email_domain` | parsed from `recipient_email` | direct |
| `campaign_type` | `mailing_type` → mapped | direct |
| `country_iso2` | `nationality` → ISO-2 norm | `country` → ISO-2 norm |
| `sent_*` features | derived from `sent`/`opened` timestamps | NaN |
| `age` | NaN (not in V2) | direct (74% missing) |

Run full V3.2 pipeline:

```bash
python merge_v2_v3.py        # builds unified_v3_2.parquet  (~30-60 min)
python train_v3_2.py         # trains, evaluates, exports   (~20-40 min)
```

Expected V3.2 artifacts:

- `xgboost_model_v3_2.pkl`, `xgboost_model_v3_2.onnx`
- `model_metadata_v3_2.json`, `features_config_v3_2.json`
- `comparison_v2_vs_v3_vs_v3_2.json`

Inference API — dual-mode endpoint:

```json
POST /predict/v3_2
{
  "email_domain": "gmail.com",
  "campaign_type": "newsletter",
  "country": "CZ",
  "sent_hour": 10,      // optional — include for best accuracy
  "sent_dow": 2         // optional
}
```

Known limitations:

- No time-based split (no send timestamps in V3 source).
- `mailing_type → campaign_type` mapping is opinion-based; logged in `merge_audit_v3_2.json`.
- Personalization is segment-level only (no user_id in either dataset).

---

## Modelling notes (short)

- **Class imbalance:** Prefer stratified splits and consider **class weights** or block-level reporting so a majority hour does not mask weak blocks.
- **Feature–label leakage:** Ensure no column encodes the answer (e.g. future opens) when building training rows.
- **Train vs inference parity:** Feature order, dtypes (`float32` for ONNX), and **scaler type** must match what the exporter assumed; a mismatch often surfaces as collapsed predictions (e.g. one hour dominates).
- **Calibration:** Raw logits from ONNX are not always well-calibrated probabilities; treat “confidence” as a ranking aid unless you validate calibration on a holdout.

---

## Public repo hygiene

Before any commit:

1. Run `git status`.
2. Confirm no staged files match `.gitignore` deny patterns (data, zip, models, secrets).
3. Prefer the **allowlist** workflow described in `.agent/skills/public-repo-packaging/SKILL.md`.

---

## License

Specify your license in this section when you publish (e.g. MIT, Apache-2.0, or proprietary notice).
