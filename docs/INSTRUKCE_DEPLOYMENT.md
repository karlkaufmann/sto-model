# Instrukce pro kolegu: Oprava predikce času (vždy 10h)

## Soubory k přeposlání

Pošli kolegovi tyto soubory z kořene projektu:

| Soubor | Povinné |
|--------|---------|
| `inference_server.py` | ano |
| `create_scaler_pkl.py` | ano |
| `scaler.pkl` | ano* |
| `model_metadata.json` | doporučeno |
| `features_config.json` | doporučeno |

\* Nebo kolega vygeneruje `scaler.pkl` spuštěním `create_scaler_pkl.py` (viz krok 2).

---

## Kroky pro kolegu

### 1. Připrav soubory
- Nahraď v kontejneru/Docker image `inference_server.py` novou verzí.
- Přidej `create_scaler_pkl.py`, `model_metadata.json`, `features_config.json` (pokud chceš).

### 2. Vygeneruj scaler.pkl
Model očekává neskalované vstupy. Spusť v adresáři s inference serverem:
```bash
python create_scaler_pkl.py
```
Tím vznikne `scaler.pkl` s IdentityScaler (passthrough).

### 3. Zajisti soubory v kontejneru
Tyto soubory musí být na stejném místě jako inference server v kontejneru (typicky `xgboost_model.onnx`, `model_metadata.json`, `features_config.json`):
- `scaler.pkl` ← **nový nebo přegenerovaný**
- `model_metadata.json`
- `features_config.json`

### 4. Rebuild & deploy
```bash
docker build -t predictor .
docker compose up -d  # nebo tvůj deploy příkaz
```

### 5. Ověření
Po startu zkontroluj logy kontejneru – nemělo by se objevit:
```text
WARNING: StandardScaler detected...
```

Otestuj různé časy na https://beta.zoomletter.com/tools/AI_test/testuj_predictor.php – predikce by se měly měnit podle vstupu.
