# Sample datasets

Pre-packaged datasets that the customer Colab notebooks can pull via `DATASET_URL` for a known-good first run.

## Catalog

| File | Format | Use with notebook | Size | Classes |
|---|---|---|---|---|
| `bccd.coco.zip` | COCO (Roboflow export) | `train_image_detector_yolox.ipynb` · `train_image_detector_rfdetr.ipynb` | 7.4 MB | 3 (Platelets, RBC, WBC) |

## How to use

Paste the GitHub raw URL into the Settings cell:

```python
DATASET_URL = 'https://raw.githubusercontent.com/IGNODE-CONNECT/ignode-collab/main/examples/datasets/bccd.coco.zip'
```

Then **Runtime → Run all**. The notebook will:

1. `curl` the archive into Colab scratch
2. Auto-extract (.zip or .tar)
3. Detect the COCO layout
4. Train, export to ONNX, write sidecars
5. Bundle for IGNODE Custom Model Upload

## Why BCCD?

- **Tiny** (7.4 MB, 364 images) — fast enough for a smoke run on Colab's free T4 GPU
- **Three classes** — exercises the multi-class label-mapping code path (catches the IR-3.Y / IR-3.GG.H class-order traps if they regress)
- **Public** — comes from Roboflow Universe; redistribution under the dataset's original CC-BY license

## Adding more

Drop a `.zip` or `.tar` under this folder, append a row to the table, and update the relevant notebook's Settings cell example URL. Keep individual files under 50 MB so plain GitHub serves them; bigger needs git-lfs.
