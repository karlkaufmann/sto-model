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
