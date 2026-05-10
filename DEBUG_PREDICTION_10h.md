# Debug: Model v produkci vždy vrací 10. hodinu

## Shrnutí diagnózy

Model vrací `model_version: "1.0"` (ne fallback) → inference nepadá do exception handleru.
Problém tedy není v chybějícím kódu, ale v **logice nebo datech**.

---

## Hypotézy (řazené pravděpodobností)

### 1. **StandardScaler na produkci × model trénovaný na RAW datech**

- `model3.py` trénuje na `X_train` (řádky 190–193), **ne** na `X_train_scaled`
- Scalery `fit_transform`/`transform` se používá, ale **model se nikdy netrénoval na scaled datech**
- Inference server aplikuje `scaler.transform(features)` pokud existuje `scaler.pkl`
- Pokud na produkci existuje `scaler.pkl` s **StandardScaler**, vstupy se zkreslí → model dostává úplně jiné hodnoty → může konzistentně predikovat jednu třídu (např. 3 → 10h)

**Ověření na produkci:** Jaký typ scaleru je v `scaler.pkl`?
```python
import pickle
with open('scaler.pkl','rb') as f: s=pickle.load(f)
print(type(s).__name__)  # StandardScaler vs IdentityScaler
```

**Řešení:** Model se trénuje bez skalování → inference musí použít IdentityScaler (nebo žádný scaler).
- Buď na produkci nasadit `scaler.pkl` s IdentityScaler
- Nebo inference_server nemít scaler (volitelně ho učinit optional)

---

### 2. **Chybný formát `class_mapping` v model_metadata.json**

Kód:
```python
predicted_hour = class_mapping.get(str(predicted_class), class_mapping.get(predicted_class, 10))
```

Pokud `class_mapping` na produkci:
- nemá klíče pro danou třídu (např. jen `"0"`–`"9"` ale s jiným formátem),
- nebo má integer klíče `0`–`9` ale metadata je poškozená,

pak `.get(predicted_class, 10)` vrací výchozích **10**.

**Ověření:** Zkontrolovat `model_metadata.json` na produkci:
```json
{
  "class_mapping": {"0": 7, "1": 8, "2": 9, "3": 10, ...},
  "version": "1.0"
}
```

---

### 3. **Stejné vstupy od PHP klienta**

Pokud PHP vždy posílá stejné feature values (např. výchozí hodnoty), model dostane identický vstup → vždy stejná predikce.
- Ověření: Porovnat request body z testuj_predictor.php napříč různými scénáři
- Ujistit se, že `sent_hour`, `sent_dow`, atd. se mění podle testu

---

### 4. **Drobné chyby v inference_server.py**

- `get_block`: řádek 106 má mrtvý kód (`return` uvnitř `elif`)
- `confidence`: `output[1]` je numpy array (logity), ne dict; `max(probabilities)` funguje, ale není to pravděpodobnost (pro softmax by bylo třeba `softmax(logits)[argmax]`)

Tyto chyby samy o sobě nezpůsobují „vždy 10h“, ale stojí za opravu.

---

## Ověření lokálně

Lokální model (`xgboost_model.onnx`) vrací **různé** hodiny pro různé vstupy:
```
sent_hour= 7 dow=0 -> class=0 -> hour=7
sent_hour=10 dow=2 -> class=3 -> hour=10
sent_hour=19 dow=4 -> class=8 -> hour=19
sent_hour=12 dow=1 -> class=8 -> hour=19
```

→ Na lokále je inference v pořádku. Problém je pravděpodobně v **produkčním prostředí** (scaler, metadata, nebo request data).

---

## Opravy v inference_server.py

1. **Scaler optional**: Když chybí `scaler.pkl`, používá se `IdentityScaler` (passthrough). Model byl trénován na RAW datech.
2. **Metadata optional**: Při chybějícím `model_metadata.json` se použije výchozí `class_mapping`.
3. **Confidence**: Opraven výpočet pro numpy array logitů (softmax).
4. **get_block**: Opraven dead code (`return` byl uvnitř `elif`).
5. **Varování**: Pokud je načten `StandardScaler`, vypíše se WARNING – pravděpodobná příčina „vždy 10h“.

Pro produkci: nasaďte `scaler.pkl` s **IdentityScaler** (viz níže) nebo odstraňte scaler úplně:

```python
# Vytvořit scaler.pkl pro produkci (model trénován bez skalování)
import pickle
class IdentityScaler:
    def fit(self, X, y=None): return self
    def transform(self, X): return X
with open("scaler.pkl", "wb") as f:
    pickle.dump(IdentityScaler(), f)
```

---

## Doporučené kroky

1. Na produkci zkontrolovat:
   - typ scaleru v `scaler.pkl`
   - obsah `model_metadata.json` (zejména `class_mapping`)
2. Logovat jeden request/response (features + predicted_class + predicted_hour) pro diagnózu
3. Zajistit konzistenci: model trénovaný bez skalování → inference bez skalování (IdentityScaler nebo bez scaleru)
4. Doplnit generování `model_metadata.json` a `features_config.json` do `model3.py` při exportu
