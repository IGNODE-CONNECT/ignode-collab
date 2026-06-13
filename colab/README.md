# Colab Notebooks

The maintained training notebooks. All of them:

- Run on Google Colab's free tier (T4 GPU for image notebooks, CPU for tabular)
- Use pinned dependency versions so the notebook still works in 6 months
- Have a single "settings" cell near the top so you can tune without reading code
- Produce an `model.onnx` file plus the sidecars the IGNODE upload wizard expects
- End with a download cell + clear instructions for the upload step

| Notebook | Task | Target runtime |
|---|---|---|
| `train_image_classifier.ipynb` | Image classification (cat/dog, defect/no-defect, etc.) | 5-10 min on T4 |
| `train_image_detector_yolox.ipynb` | **Image detection — default.** YOLOX, same recipe as IGNODE's production trainer | 30-60 min on T4 |
| `train_image_detector_rfdetr.ipynb` | Image detection — RF-DETR (transformer / DETR family) for dense + tiny objects | 60-120 min on T4 |
| `train_tabular_classifier.ipynb` | Tabular classification (Yes/No, Normal/Warning/Critical) | < 1 min on CPU |
| `train_tabular_regression.ipynb` | Tabular regression (price, remaining life, temperature) | < 1 min on CPU |

## Which detection notebook do I pick?

| If your scene is… | Pick |
|---|---|
| Anything else (default) | **YOLOX** — matches IGNODE's production trainer byte-for-byte |
| Dense (more than 20 objects per image), small overlapping targets (cells under a microscope, tiny defects) AND you have at least 500 images per class | **RF-DETR** |
| You want a quick smoke run on a small dataset | **YOLOX** with `--epochs 30` |
| Customer already trained an Ultralytics YOLOv5/v8 model and wants to migrate to YOLOX | **YOLOX** (the notebook walks the migration end-to-end) |

Both detection notebooks emit the same upload-bundle layout — switching between them later is a fresh-train job, not a sidecar swap.

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

## Class-labels contract — non-negotiable for ALL detection / classification notebooks (IR-3.S.A)

**Never hand-type `CLASS_LABELS = [...]` with stub names** ("class_0", "class_1") and expect the customer to fill them in. Half the time they paste the names in the wrong order and ship a model where every prediction is mislabeled. The format-converter's IR-3.GG.H bug taught us this is the single most expensive class of failure to debug after deploy.

Every notebook that produces a `class_labels.json` sidecar MUST:

1. **Auto-derive `CLASS_LABELS` from the dataset.** Read whatever source-of-truth the training framework used:
   - **COCO datasets** → `_annotations.coco.json` → sort `categories` by `id` → take `name` field
   - **YOLO datasets** → `data.yaml` → `names` field (preserve order)
   - **VOC datasets** → use `supervision.DetectionDataset.from_pascal_voc(...).classes` (insertion order; do NOT `sorted(seen)`)
   - **Train-script artefacts** → look for `classes.txt` the training script wrote out
   - **Framework-internal** → read `model.classes` / `model.dataset.classes` if the framework exposes it
2. **Print the derived list before bundling** — show the index→name map plainly so the customer notices a surprising order before downloading.
3. **Add an ONNX smoke-test cell after export** that:
   - Loads the freshly-exported `model.onnx` with `onnxruntime`
   - Runs ONE validation image (mirror exactly the preprocess in `preprocess_config.json`)
   - Decodes the highest-confidence detection
   - Prints `top class_id → class_labels[that id]` so the customer can eyeball whether the model labels look sane

The two reference implementations are `train_image_detector_rfdetr.ipynb` (COCO source → auto-derive + DETR-shape smoke-test) and `train_image_detector_yolox.ipynb` (classes.txt source → auto-derive + YOLOX-raw smoke-test). Both ship the cells under an `IR-3.S.A` comment marker so future audits can grep for the pattern.

## Dataset-source contract — also non-negotiable (IR-3.S.B)

Every image notebook (classification + detection) MUST expose the dataset source via a top-of-notebook **Settings cell** with two variables:

- `DATASET_URL` — public download URL (Roboflow Universe "Raw URL", any HTTPS .zip / .tar)
- `DATASET_DIR` — folder path (typically inside a mounted Google Drive)

`DATASET_URL` takes precedence when set. The cell immediately after Settings:

1. **Raises a loud `SystemExit` warning if BOTH are empty** — saves the customer a 30-minute training run on an empty folder.
2. Downloads + extracts the URL when provided, then auto-points `DATASET_DIR` at the extracted path.
3. Falls back to the local/Drive folder otherwise, verifying the path exists.

The Drive-mount call is now its own **OPTIONAL** cell between Settings and Load — customers using `DATASET_URL` skip it. Look for `# IR-3.S.B` marker in patched cells.

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
