# Sample datasets

Pre-packaged datasets that the customer Colab notebooks pull via `DATASET_URL` for a known-good first run.

## Catalog

| File | Format | Use with notebook | Size | Classes |
|---|---|---|---|---|
| `bccd.coco.zip` | COCO (Roboflow export) | All three detection notebooks | 7.4 MB | 3 (Platelets, RBC, WBC) |
| `bccd.voc.zip` | Pascal VOC (Roboflow export) | `train_image_detector_yolox.ipynb` · `train_image_detector_rfdetr.ipynb` (auto-converts via supervision) | 7.5 MB | 3 (Platelets, RBC, WBC) |
| `bccd.yolov8.zip` | YOLOv8 (Roboflow export) | `train_image_detector_yolox.ipynb` · `train_image_detector_rfdetr.ipynb` (auto-converts) | 7.4 MB | 3 (Platelets, RBC, WBC) |

## How to use

Paste any of the GitHub raw URLs into the Settings cell of a detection notebook:

```python
# Quick test, COCO input (works on every notebook):
DATASET_URL = 'https://raw.githubusercontent.com/IGNODE-CONNECT/ignode-collab/main/examples/datasets/bccd.coco.zip'

# Quick test, VOC input (exercises the supervision converter):
DATASET_URL = 'https://raw.githubusercontent.com/IGNODE-CONNECT/ignode-collab/main/examples/datasets/bccd.voc.zip'

# Quick test, YOLOv8 input (exercises the supervision converter):
DATASET_URL = 'https://raw.githubusercontent.com/IGNODE-CONNECT/ignode-collab/main/examples/datasets/bccd.yolov8.zip'
```

Then **Runtime → Run all**. The notebook will:

1. `curl` the archive into Colab scratch
2. Auto-extract (.zip or .tar)
3. **Auto-detect format** and (if not already COCO) convert via the `supervision` library — same library IGNODE's platform-side `ignode-format-converter` sidecar uses. Conversion preserves class label order from the loader's `.classes` attribute byte-for-byte (no `sorted(seen)` traps; see IR-3.Y / IR-3.GG.H).
4. Train, export to ONNX, write sidecars
5. Bundle for IGNODE Custom Model Upload

## Why BCCD?

- **Tiny** (~7.5 MB, 364 images) — fast enough for a smoke run on Colab's free T4 GPU
- **Three classes** — exercises the multi-class label-mapping path. Catches the IR-3.Y / IR-3.GG.H class-order traps if they regress, because the alphabetical order (`Platelets, RBC, WBC`) differs from the insertion order in some exports.
- **Three formats from one source** — same images, same annotations, three different on-disk layouts. Letting customer testing cover all three input formats via the same expected output gives a tight signal on format-conversion correctness.
- **Public** — comes from Roboflow Universe; redistribution under the dataset's original CC-BY license.

## Format-conversion contract (what the notebooks do under the hood)

| Notebook | Native input | Auto-conversion via supervision |
|---|---|---|
| `train_image_detector_yolox.ipynb` | YOLOX VOC layout | VOC, COCO, YOLO, and Roboflow variants of each. Delegated to `train_any.py` (downloaded from the production trainer benchmark). |
| `train_image_detector_rfdetr.ipynb` | COCO | VOC + YOLO converted to COCO via `sv.DetectionDataset.from_pascal_voc / from_yolo` + `.as_coco()`. |

All three preserve class label order from supervision's `.classes` attribute verbatim. The sidecar `class_labels.json` is derived from that single source after conversion — never re-sorted, never re-derived in parallel.

## Adding more

Drop a `.zip` or `.tar` under this folder, append a row to the table above, and update the relevant notebook's Settings cell example URL. Keep individual files under 50 MB so plain GitHub serves them; bigger needs `git-lfs`.
