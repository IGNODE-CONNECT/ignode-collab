# Colab Notebooks

The four maintained training notebooks. All four:

- Run on Google Colab's free tier (T4 GPU for image notebooks, CPU for tabular)
- Use pinned dependency versions so the notebook still works in 6 months
- Have a single "settings" cell near the top so you can tune without reading code
- Produce an `model.onnx` file plus the sidecars the IGNODE upload wizard expects
- End with a download cell + clear instructions for the upload step

| Notebook | Task | Target runtime |
|---|---|---|
| `train_image_classifier.ipynb` | Image classification (cat/dog, defect/no-defect, etc.) | 5-10 min on T4 |
| `train_image_detector.ipynb` | Image detection (bounding boxes around objects) | 30-60 min on T4 |
| `train_tabular_classifier.ipynb` | Tabular classification (Yes/No, Normal/Warning/Critical) | < 1 min on CPU |
| `train_tabular_regression.ipynb` | Tabular regression (price, remaining life, temperature) | < 1 min on CPU |

## Dataset format expected by each notebook

**Image Classification** — a `.tar` archive of folders, one per class:

```
flowers.tar
├── roses/
│   ├── img_001.jpg
│   ├── img_002.jpg
│   └── ...
├── tulips/
│   └── ...
└── daisies/
    └── ...
```

**Image Detection** — image folder + `annotations.json` in COCO format. Use [LabelImg](https://github.com/HumanSignal/labelImg), [Roboflow](https://roboflow.com/), or [CVAT](https://www.cvat.ai/) to produce the annotations.

**Tabular** — a `.csv` with a header row, one column per feature, and one label column. Pick the label column inside the notebook's settings cell.

## What the notebooks produce

Each notebook downloads two things at the end:

1. `model.onnx` — the trained model in ONNX format
2. A small set of metadata you'll paste into the IGNODE upload wizard:
   - **Class labels** (Classification + Image Classification + Detection)
   - **Feature column names** (Tabular)
   - **Target / label column name** (Tabular)
   - **Image preprocessing config** (Image notebooks) — input size, channel order, mean/std

## How the notebooks stay reliable

- Every cell that installs a dependency pins the exact version
- A linter runs on every push to catch syntax breakage
- We re-verify against current Colab quarterly and bump pinned versions when needed

Last verification date appears in each notebook's header cell.

## Hacking on a notebook

```bash
git clone git@github.com:IGNODE-CONNECT/ignode-collab.git
cd ignode-collab/colab
# edit a notebook
git commit -am "your change"
# open a pull request
```

Linter check runs automatically on the PR.
