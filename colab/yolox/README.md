# YOLOX detection — Colab helper scripts

This folder hosts the 3 helper scripts the `train_image_detector_yolox.ipynb`
notebook fetches at training time:

| Script | Purpose |
|---|---|
| `train_any.py` | YOLOX training driver. Auto-detects dataset format (COCO / YOLO / VOC) and normalizes to YOLOX's expected VOC layout. Same code IGNODE's production trainer (IR-3.O Path B) runs internally. |
| `prepare_dataset.py` | Dataset extraction + class-list audit. Walks the customer's `.zip` / `.tar` upload, validates per-class image counts, and emits a class manifest the trainer reads. |
| `voc_eval_patch.py` | Patches YOLOX's stock `voc_eval` so empty val splits return mAP=0 instead of crashing. Necessary because customer val splits sometimes lack instances of every class. |

## Why mirror them here?

The canonical source for these scripts is
`ignode-trainer/benchmarks/yolox_upstream_baseline/` — but that repo is on
private Bitbucket and the customer's Colab kernel can't authenticate to it.
This folder is the public-GitHub mirror the Colab notebook fetches from
via `raw.githubusercontent.com`.

When the trainer-side scripts change, mirror the update here so customer
Colabs stay aligned with production. (Drift is captured by the IR-3.1
plan-doc; if it bites in practice we'll wire a CI sync.)
