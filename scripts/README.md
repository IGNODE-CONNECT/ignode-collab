# scripts/

Local utilities for IGNODE platform testing — NOT customer-facing.

## generate_test_models.py

Builds tiny test artifacts in every model format the Custom Model Upload
wizard accepts, so you can exercise the full pipeline (upload → deploy →
predict) without bringing your own model.

### Run

```bash
python scripts/generate_test_models.py
```

Output lands in `scripts/test-artifacts/<format>-<task>/`. Each subfolder
contains:

- The model file (`model.onnx` / `model.txt` / `model.json` / `model.tflite`)
- Matching sidecars (`class_labels.json`, `feature_columns.json`,
  `preprocess_config.json` as appropriate)
- A ready-to-upload `model-artifacts.zip` bundle

### Coverage

| Format      | Classification | Regression | Image classification |
|-------------|----------------|------------|----------------------|
| `lgbm_text` | iris           | diabetes   | — (not supported)    |
| `xgb_json`  | iris           | diabetes   | — (not supported)    |
| `onnx`      | iris (sklearn) | diabetes (sklearn) | tiny MobileNetV2 |
| `tflite`    | — (not supported) | — (not supported) | tiny CNN |

Models are random-weights tiny — they're for testing format handling, not
accuracy. Predict responses will not be meaningful, only well-formed.

### Selective generation

```bash
python scripts/generate_test_models.py --only lgbm-cls xgb-cls
```

Available keys: `lgbm-cls`, `lgbm-reg`, `xgb-cls`, `xgb-reg`,
`onnx-cls`, `onnx-reg`, `onnx-img`, `tflite-img`.

### Dependencies

The script skips any artifact whose Python deps aren't installed (with a
warning). Install whatever set you need:

- Tabular: `pip install lightgbm xgboost scikit-learn skl2onnx onnx`
- ONNX image: `pip install torch torchvision onnx`
- TFLite image: `pip install tensorflow`
