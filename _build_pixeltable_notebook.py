"""Generate train_image_detector_pixeltable_yolox.ipynb from cell strings.

Run once to author the notebook; delete after committing.
"""
from __future__ import annotations
import json
from pathlib import Path

CELLS: list[tuple[str, str]] = [
    ("markdown", r"""# Train an IGNODE-compatible detector with YOLOX (Maintained Fork)

**Output:** an ONNX model + sidecars ready to upload via IGNODE's
ML Factory → Models → Upload custom model.

This notebook uses the actively-maintained
[pixeltable-yolox](https://github.com/pixeltable/pixeltable-yolox)
Apache fork instead of the upstream Megvii YOLOX 0.3.0 (last released
Aug 2022). Same architecture, same ONNX output contract (IGNODE UINF
yolox_raw decode) — different install path. If this notebook fails
install in a way the regular YOLOX notebook doesn't, fall back to
`train_image_detector_yolox.ipynb`.
"""),

    ("code", r"""# Step 0 — install pixeltable-yolox + dependencies.
#
# Much simpler than Megvii YOLOX cell 1: a single pip install pulls
# pixeltable-yolox + all deps (onnx / onnxruntime / supervision /
# thop / tensorboard / pycocotools / onnxsim — all bundled).
#
# Two unavoidable extras stay:
#   1. opencv-python-headless override — pixeltable's pyproject pins
#      opencv-python (GUI variant) which needs libxcb.so.1, missing
#      on Colab slim runtimes. -headless ships the same cv2 with no
#      X11 deps. Force-reinstall to override.
#   2. apt build-essential / g++ / python3-dev — YOLOX-family
#      architectural choice: yolox/layers/cocoeval/cocoeval.cpp is
#      lazily compiled at FIRST eval-step inside training. CXX=g++
#      env var forces torch to use g++ (the `c++` symlink may not
#      exist even with build-essential).
#
# TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 forces torch.load's weights_only
# back to False — torch 2.6+ defaults it to True which rejects
# checkpoints containing numpy._core.multiarray.scalar. Set via
# os.environ so !python subprocesses (Cells 6/7) inherit it.
import os
os.environ['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD'] = '1'
os.environ['CXX'] = 'g++'

!pip install -q pixeltable-yolox
!pip install -q --force-reinstall opencv-python-headless
!apt-get install -y -q build-essential g++ python3-dev
"""),

    ("code", r"""# IR-3.S.B — Settings: pick your dataset source + tune training knobs.
# Set ONE of DATASET_URL or DATASET_DIR. URL takes precedence.
# Leave both empty and the next cell will warn + stop.
#
# Supported format: COCO (.zip with annotations/{instances_train2017,
# instances_val2017}.json + train2017/ + val2017/ folders).
# Roboflow's "COCO" export ships in this exact layout — drop the link
# from Roboflow Universe directly into DATASET_URL.

# Quick test — BCCD blood-cell-count sample (7.4 MB, 3 classes):
#   DATASET_URL = 'https://raw.githubusercontent.com/IGNODE-CONNECT/ignode-collab/main/examples/datasets/bccd.coco.zip'
#
# Your own data: any public .zip download link works.
DATASET_URL = ''

# Google Drive folder path. Mount Drive separately if you want this.
DATASET_DIR = ''

# Training knobs — edit ONE cell to tune the run.
EPOCHS = 100         # 30 for a quick smoke; 300 for a real run
BATCH_SIZE = 8       # drop to 4 on a smaller GPU (T4 free Colab)

# Local scratch directory inside Colab — keep as-is.
WORKDIR = '/content/yolox-run'
"""),

    ("code", r"""# IR-3.S.D — OPTIONAL Drive mount, gated so Run-all is safe.
# Only mounts Drive when the Settings cell points DATASET_DIR at a
# /content/drive path.
import os

_needs_drive = (not DATASET_URL) and DATASET_DIR.startswith('/content/drive')
_already_mounted = os.path.ismount('/content/drive') or os.path.isdir('/content/drive/MyDrive')

if _needs_drive and not _already_mounted:
    from google.colab import drive
    drive.mount('/content/drive')
elif _needs_drive:
    print('Drive already mounted — skipping.')
else:
    print('Not using Drive (DATASET_URL set, or DATASET_DIR is a local path).')
    print('Skipping Drive mount.')
"""),

    ("code", r"""# IR-3.S.B — Load dataset + normalize to pixeltable-yolox's expected
# ./datasets/COCO/ layout, then renumber category IDs for Roboflow's
# phantom-supercategory quirk.
import os, pathlib, shutil, zipfile, subprocess, json

if not DATASET_URL and not DATASET_DIR:
    raise SystemExit(
        '\n⚠️  Both DATASET_URL and DATASET_DIR are empty in the Settings cell.\n'
        '   Set ONE of them, then re-run this cell.'
    )

pathlib.Path(WORKDIR).mkdir(parents=True, exist_ok=True)
COCO_ROOT = pathlib.Path(WORKDIR) / 'datasets' / 'COCO'
(COCO_ROOT / 'annotations').mkdir(parents=True, exist_ok=True)
(COCO_ROOT / 'train2017').mkdir(parents=True, exist_ok=True)
(COCO_ROOT / 'val2017').mkdir(parents=True, exist_ok=True)

# 1. Get the dataset into a known location.
if DATASET_URL:
    print(f'Downloading from public URL: {DATASET_URL}')
    _archive = pathlib.Path('/content/_dataset_download')
    _archive.mkdir(parents=True, exist_ok=True)
    _dl_path = _archive / 'dataset.zip'
    subprocess.run(['curl', '-fsSL', '-o', str(_dl_path), DATASET_URL], check=True)
    print(f'Downloaded {_dl_path.stat().st_size:,} bytes — extracting…')
    _extracted = _archive / 'unpacked'
    if _extracted.exists():
        shutil.rmtree(_extracted)
    _extracted.mkdir()
    with zipfile.ZipFile(_dl_path) as zf:
        zf.extractall(_extracted)
    SOURCE_DIR = _extracted
else:
    SOURCE_DIR = pathlib.Path(DATASET_DIR)
    if not SOURCE_DIR.is_dir():
        raise SystemExit(f'❌ DATASET_DIR={DATASET_DIR!r} does not exist.')

# 2. Re-layout Roboflow's train/ + valid/ folders into the COCO layout.
_train_src = next((p for p in SOURCE_DIR.rglob('train') if p.is_dir()), None)
_valid_src = next((p for p in SOURCE_DIR.rglob('valid') if p.is_dir()), None) \
          or next((p for p in SOURCE_DIR.rglob('val')   if p.is_dir()), None)
if _train_src is None or _valid_src is None:
    raise SystemExit(
        '❌ Could not find train/ + valid/ folders in dataset. '
        'Roboflow COCO exports include both. Check your DATASET_URL.'
    )

for jpg in _train_src.glob('*.jpg'):
    shutil.copy(jpg, COCO_ROOT / 'train2017' / jpg.name)
for jpg in _valid_src.glob('*.jpg'):
    shutil.copy(jpg, COCO_ROOT / 'val2017' / jpg.name)
shutil.copy(_train_src / '_annotations.coco.json',
            COCO_ROOT / 'annotations' / 'instances_train2017.json')
shutil.copy(_valid_src / '_annotations.coco.json',
            COCO_ROOT / 'annotations' / 'instances_val2017.json')

n_train = len(list((COCO_ROOT / 'train2017').glob('*.jpg')))
n_val   = len(list((COCO_ROOT / 'val2017').glob('*.jpg')))
print(f'train2017: {n_train} images')
print(f'val2017:   {n_val} images')

# 3. Renumber category IDs — drop Roboflow's phantom supercategory at
# id=0 + remap remaining classes to 0..N-1 so they fit num_classes.
for ann_path in (COCO_ROOT / 'annotations').glob('instances_*.json'):
    data = json.loads(ann_path.read_text())
    real_cats = [c for c in data['categories'] if c.get('supercategory') != 'none']
    real_cats.sort(key=lambda c: c['id'])
    id_map = {c['id']: new for new, c in enumerate(real_cats)}
    data['categories']  = [{**c, 'id': id_map[c['id']]} for c in real_cats]
    data['annotations'] = [{**a, 'category_id': id_map[a['category_id']]} for a in data['annotations']]
    ann_path.write_text(json.dumps(data))
    print(f"{ann_path.name}: classes → {[(c['id'], c['name']) for c in data['categories']]}")

# 4. Derive NUM_CLASSES from the renumbered annotations.
_train_ann = json.loads((COCO_ROOT / 'annotations' / 'instances_train2017.json').read_text())
NUM_CLASSES = len(_train_ann['categories'])
print(f'\nNUM_CLASSES = {NUM_CLASSES}')
"""),

    ("code", r"""# Step 2 — download helper script from the public ignode-collab GitHub
# mirror. Only one helper for the pixeltable variant:
#
#   export_to_onnx.py — custom ONNX exporter. Works around the broken
#   `yolox export_onnx` CLI in pixeltable-yolox 0.4.2. Produces opset-18
#   ONNX matching IGNODE's UINF yolox_raw contract.
HELPERS_RAW = 'https://raw.githubusercontent.com/IGNODE-CONNECT/ignode-collab/main/colab/pixeltable_yolox'
!curl -fsSL {HELPERS_RAW}/export_to_onnx.py -o {WORKDIR}/export_to_onnx.py
!ls -la {WORKDIR}/export_to_onnx.py
"""),

    ("code", r"""# Step 3 — train via pixeltable-yolox's `yolox train` CLI.
#
# Notable differences from Megvii notebook:
#   - Single CLI call (no train_any.py wrapper).
#   - `-D key=value` overrides each YoloxConfig attribute. No re.sub of
#     the exp config file; max_epoch lands directly.
#   - `-c yolox_s` resolves to the YoloxS subclass (depth=0.33, width=0.5).
import os
os.chdir(WORKDIR)
!yolox train -c yolox_s -b {BATCH_SIZE} -d 1 --fp16 \
  -D max_epoch={EPOCHS} \
  -D num_classes={NUM_CLASSES} \
  -D data_num_workers=2 \
  -D data_dir={WORKDIR}/datasets/COCO
"""),

    ("code", r"""# Step 4 — export to ONNX matching IGNODE's UINF yolox_raw contract.
#
# Critical contract:
#   - decode_in_inference=False (UINF host-side decodes via anchor-grid)
#   - opset 18, input "images" [batch, 3, 640, 640], output "output"
#
# We call our own export_to_onnx.py because pixeltable-yolox 0.4.2's
# built-in `yolox export_onnx` is broken (CLI subcommand not registered
# + underlying module references the deleted `yolox.exp` namespace).
import pathlib
CKPT = next(pathlib.Path(f'{WORKDIR}/out').rglob('best_ckpt.pth'), None) \
    or next(pathlib.Path(f'{WORKDIR}/out').rglob('latest_ckpt.pth'), None)
assert CKPT is not None, 'No checkpoint under ./out/ — re-run Step 3 + inspect its logs.'
OUT = f'{WORKDIR}/model.onnx'

!python {WORKDIR}/export_to_onnx.py {CKPT} {OUT} --num-classes {NUM_CLASSES} --arch yolox_s --opset 18
"""),

    ("code", r"""# IR-3.S.A — Step 5: write sidecars + auto-derive CLASS_LABELS.
# DO NOT hand-type class names here. The renumber step in Cell 4
# encodes the EXACT index→name mapping YOLOX trained against.
#
# Sidecar shape MUST match production trainer's (IR-3.BB + IR-3.CC):
#   channel_order: 'BGR'    — YOLOX uses cv2.imread (BGR by convention)
#   rescale:       'none'   — YOLOX trains on raw [0, 255]
#   mean / std:    identity — no normalization
#   postprocess.family: 'yolox_raw' — UINF host-side decode
import json, shutil, pathlib

# 1. Read class names from the renumbered COCO annotations. IDs are now
# 0..N-1 in order.
_ann_path = pathlib.Path(f'{WORKDIR}/datasets/COCO/annotations/instances_train2017.json')
_ann = json.loads(_ann_path.read_text())
CLASS_LABELS = [c['name'] for c in sorted(_ann['categories'], key=lambda c: c['id'])]
print(f'Derived CLASS_LABELS from {_ann_path.name}:')
for i, name in enumerate(CLASS_LABELS):
    print(f'  [{i}] {name}')
print()
print('If this order surprises you, STOP and check your source dataset.')

preprocess_config = {
    'input_size':      [640, 640],
    'mean':            [0.0, 0.0, 0.0],
    'std':             [1.0, 1.0, 1.0],
    'channel_order':   'BGR',
    'image_format':    'CHW',
    'rescale':         'none',
    'resize_method':   'letterbox',
    'letterbox_color': [114, 114, 114],
    'postprocess': {
        'family':               'yolox_raw',
        'nms_required':         True,
        'confidence_threshold': 0.25,
        'nms_iou_threshold':    0.65,
    },
    '_backbone': 'yolox_s',
    '_trainer':  'pixeltable-yolox',
}

out = pathlib.Path(f'{WORKDIR}/upload-bundle')
out.mkdir(exist_ok=True)
shutil.copy(f'{WORKDIR}/model.onnx', out / 'model.onnx')
(out / 'preprocess_config.json').write_text(json.dumps(preprocess_config, indent=2))
(out / 'class_labels.json').write_text(json.dumps(CLASS_LABELS, indent=2))

print()
print('Upload bundle ready at:', out)
!ls -la {out}
"""),

    ("code", r"""# IR-3.S.A — Step 5.5: ONNX smoke-test (DO NOT SKIP).
# Runs the freshly-exported YOLOX-raw ONNX on ONE val image and prints
# the top anchor's argmax class. If the prediction obviously contradicts
# the image, CLASS_LABELS got reordered upstream — fix before uploading.
import json, pathlib, numpy as np, onnxruntime as ort
import cv2

_bundle = pathlib.Path(f'{WORKDIR}/upload-bundle')
_labels = json.loads((_bundle / 'class_labels.json').read_text())
_sess   = ort.InferenceSession(str(_bundle / 'model.onnx'), providers=['CPUExecutionProvider'])
_inp    = _sess.get_inputs()[0].name

_val_dir = pathlib.Path(f'{WORKDIR}/datasets/COCO/val2017')
_imgs = sorted(_val_dir.glob('*.jpg'))
assert _imgs, f'No val images at {_val_dir} — re-run Step 1.'
_img_path = _imgs[0]
print(f'Smoke-testing on: {_img_path.name}')

# YOLOX preprocess: BGR, letterbox 640×640, [0..255], CHW, no norm.
_bgr = cv2.imread(str(_img_path))
_h, _w = _bgr.shape[:2]
_scale = min(640 / _h, 640 / _w)
_nh, _nw = int(_h * _scale), int(_w * _scale)
_resized = cv2.resize(_bgr, (_nw, _nh), interpolation=cv2.INTER_LINEAR)
_canvas = np.full((640, 640, 3), 114, dtype=np.uint8)
_canvas[:_nh, :_nw] = _resized
_arr = _canvas.astype(np.float32).transpose(2, 0, 1)[None, ...]

# YOLOX-raw output: (1, N_anchors, 5 + num_classes).
_outs = _sess.run(None, {_inp: _arr})
_raw  = next((o for o in _outs if o.ndim == 3 and o.shape[-1] >= 5 + len(_labels)), None)
assert _raw is not None, 'Could not find a YOLOX-raw-shaped output — confirm Step 4 export.'
_obj   = 1.0 / (1.0 + np.exp(-_raw[0, :, 4]))
_cls_l = 1.0 / (1.0 + np.exp(-_raw[0, :, 5:5 + len(_labels)]))
_best  = int((_obj[:, None] * _cls_l).max(axis=1).argmax())
_best_class = int(_cls_l[_best].argmax())
_best_obj   = float(_obj[_best])
_best_score = float((_obj[_best] * _cls_l[_best]).max())
print()
print(f'Top anchor: idx={_best_class} → class_labels[{_best_class}] = {_labels[_best_class]!r}')
print(f'Objectness: {_best_obj:.3f}  Composite score: {_best_score:.3f}')
print()
print(f'Sanity check: open {_img_path.name}. If the model thinks it contains')
print(f'a {_labels[_best_class]!r} but you can see it does not, STOP and fix')
print('upload-bundle/class_labels.json before uploading.')
"""),

    ("markdown", r"""## Step 6 — upload to IGNODE

In your IGNODE workspace:
1. ML Factory → Models → **Upload custom model**
2. Drag the 3 files from `upload-bundle/` into the modal
3. Deploy to a UINF instance
4. Verify in Playground — boxes should land tight on your test image

If the boxes are mispositioned: usually a channel-order or class-label-order issue. The smoke test in Cell 9 is the cheap check — sanity it first.

### Differences from the regular YOLOX (Legacy) notebook
- Uses **pixeltable-yolox** (Apache fork, actively maintained 2024-2025) instead of upstream Megvii YOLOX 0.3.0.
- Same architecture, same ONNX output contract — drop-in replacement at deploy time.
- Cell 1 is one `pip install` instead of a multi-step tarball-extract + in-place file-patch dance.
"""),
]


def make_cell(kind: str, src: str) -> dict:
    lines = src.splitlines(keepends=True)
    if kind == "code":
        return {
            "cell_type": "code",
            "metadata": {},
            "source": lines,
            "outputs": [],
            "execution_count": None,
        }
    return {"cell_type": "markdown", "metadata": {}, "source": lines}


nb = {
    "cells": [make_cell(k, s) for k, s in CELLS],
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
        "accelerator": "GPU",
        "colab": {"provenance": []},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out = Path("colab/train_image_detector_pixeltable_yolox.ipynb")
out.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
print(f"wrote {out} ({len(CELLS)} cells)")
