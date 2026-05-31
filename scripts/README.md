# scripts/

Local utilities for IGNODE platform testing — NOT customer-facing.

## Generate test model artifacts

> [**Open generate_test_models.ipynb in Colab**](https://colab.research.google.com/github/IGNODE-CONNECT/ignode-collab/blob/main/colab/generate_test_models.ipynb)

Builds tiny test artifacts in every model format the Custom Model Upload
wizard accepts so you can exercise the full pipeline (upload → deploy →
predict) without bringing your own model. Each cell downloads one ZIP
to your browser — drag into IGNODE → Custom Model Upload to test.

The notebook is the canonical path. Colab has torch, tensorflow,
scikit-learn etc. pre-installed (or one-line installs); local Python on
Windows wrestles with native deps for hours.

### Coverage

| Format      | Classification     | Regression          | Image classification     |
|-------------|--------------------|---------------------|--------------------------|
| `lgbm_text` | iris               | diabetes            | — (not supported)        |
| `xgb_json`  | iris               | diabetes            | — (not supported)        |
| `onnx`      | iris (sklearn)     | diabetes (sklearn)  | random-weight MobileNetV2 |
| `tflite`    | — (not supported)  | — (not supported)   | tiny CNN                 |

Models are random or minimally-trained — the goal is exercising format
handling, not accuracy. Predictions will be well-formed but not
meaningful.

### generate_test_models.py

Stub that just prints the notebook URL. There is no local-Python
equivalent — open the notebook.
