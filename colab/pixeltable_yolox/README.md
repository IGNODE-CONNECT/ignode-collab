# Pixeltable-yolox helper scripts

Mirror of helpers the `train_image_detector_pixeltable_yolox.ipynb`
notebook fetches at runtime.

| Script | Purpose |
|---|---|
| `export_to_onnx.py` | Custom ONNX exporter — works around the broken `yolox export_onnx` CLI in pixeltable-yolox 0.4.2 (CLI subcommand not registered + module references the deleted `yolox.exp`). Produces opset-18 ONNX matching IGNODE's UINF yolox_raw contract. |

Canonical source for these helpers is
`ignode-trainer/benchmarks/pixeltable-yolox-smoke/` (private). This
folder is the public-GitHub mirror the Colab notebook fetches from
via `raw.githubusercontent.com`. Keep the two in sync.

See `ansible-deploy/documentation/MLOPS/IMAGE_RECOGNITION/IMPLAMENTATION_PLAN/
IR_3_1/PIXEL_TABLE_TEST.md` for the full evaluation + migration
context.
