# STO Model v2 Deployment Notes

## Artifacts to deliver

- `xgboost_model.onnx`
- `xgboost_model.pkl`
- `scaler.pkl` (fitted `StandardScaler`)
- `model_metadata.json`
- `features_config.json`

## Model summary

- `model_version`: `2.0`
- `features_schema_version`: `v2`
- Feature count: `19`
- Added recipient/context features:
  - `nationality`, `template_id`, `mailing_type`, `state`, `dsn`
  - `previous_stays_count`, `sum_previous_stay_days`
- Added temporal features:
  - `sent_weekofyear`, `sent_is_month_start`, `sent_is_month_end`

## API compatibility

The API remains backward-compatible: existing payloads continue to work.
New optional fields improve differentiation between recipients:

- `nationality`
- `template_id`
- `mailing_type`
- `state`
- `dsn`
- `previous_stays_count`
- `sum_previous_stay_days`
- `sent_weekofyear`
- `sent_is_month_start`
- `sent_is_month_end`

## Example v2 payload

```json
{
  "sent_hour": 14,
  "sent_dow": 3,
  "sent_day": 24,
  "sent_month": 3,
  "is_weekend": 0,
  "web_id": 659,
  "client_id": 12345,
  "time_to_open": 2.0,
  "nationality": "CZ",
  "template_id": "42",
  "mailing_type": "promo",
  "state": "delivered",
  "dsn": "ok",
  "previous_stays_count": 3,
  "sum_previous_stay_days": 12,
  "sent_weekofyear": 13,
  "sent_is_month_start": 0,
  "sent_is_month_end": 0
}
```

## Runtime expectations

- Inference server must load all artifacts from the same folder.
- `scaler.pkl` must match the exact trained model and feature schema.
- `model_metadata.json` contains class mapping and category mappings used by inference.

## Verification checklist

1. Start API: `./venv/bin/python inference_server.py`
2. Check health: `GET /health`
3. Check metrics and schema: `GET /metrics`
4. Run sample `POST /predict` with the v2 payload above.
