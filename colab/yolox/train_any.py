#!/usr/bin/env python3
"""Train YOLOX on ANY dataset format.

Auto-detects and normalizes via `supervision` to YOLOX's required VOC layout.

Supported input formats (auto-detected):
  - Pascal VOC          (Annotations/*.xml + JPEGImages/*.jpg)
  - Roboflow VOC export (train/ + valid/ + test/ — XMLs and JPGs side-by-side)
  - COCO                (annotations/*.json + images/)
  - Roboflow COCO       (train/_annotations.coco.json + train/*.jpg)
  - YOLOv5/v8           (images/{train,val}/ + labels/{train,val}/ + data.yaml)
  - Roboflow YOLO       (train/images/ + train/labels/ + data.yaml)

Output (all go to --output dir):
  results.json            — final mAP + per-class AP + recipe used
  prepare.log             — dataset normalization output
  train.log               — per-iteration training log
  eval.log                — pycocotools / VOC eval output
  yolox_voc_s/            — trained model checkpoints + tensorboard events
  demo_predictions/*.jpg  — annotated predictions on a few val images

Usage:
    python /app/train_any.py --data /mnt/my_dataset --output /mnt/runs/run_001

Common options:
    --epochs N          (default 300)
    --batch-size N      (default 16)
    --backbone S        (yolox_s / yolox_m / yolox_l / yolox_x; default yolox_s)
    --image-size N      (default 640)
    --fp16              (enable native torch.cuda.amp; ~3-5 mAP higher on small datasets)
    --no-fp16           (force fp32 training)
    --format F          (override auto-detect: voc | coco | yolo | roboflow_voc | roboflow_coco | roboflow_yolo)
"""
import argparse
import glob
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import supervision as sv


# ─── Backbone architecture lookup (v8) ──────────────────────────────────────
# Megvii YOLOX backbones differ only in depth (number of CSP blocks per stage)
# and width (channel multiplier). The same yolox_voc_s.py exp config can train
# any of them — we just rewrite self.depth + self.width.
BACKBONE_CONFIG = {
    "yolox_nano": (0.33, 0.25),
    "yolox_tiny": (0.33, 0.375),
    "yolox_s":    (0.33, 0.50),
    "yolox_m":    (0.67, 0.75),
    "yolox_l":    (1.00, 1.00),
    "yolox_x":    (1.33, 1.25),
}

# Public Megvii release page for the COCO-pretrained checkpoints.
CHECKPOINT_URLS = {
    "yolox_nano": "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_nano.pth",
    "yolox_tiny": "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_tiny.pth",
    "yolox_s":    "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.pth",
    "yolox_m":    "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_m.pth",
    "yolox_l":    "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_l.pth",
    "yolox_x":    "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_x.pth",
}


def ensure_checkpoint(backbone: str) -> Path:
    """If /app/{backbone}.pth doesn't exist, download it once."""
    ckpt = Path(f"/app/{backbone}.pth")
    if ckpt.is_file() and ckpt.stat().st_size > 1024 * 1024:
        return ckpt
    url = CHECKPOINT_URLS.get(backbone)
    if not url:
        raise SystemExit(f"No known COCO checkpoint URL for backbone {backbone!r}.")
    print(f"  downloading {backbone}.pth from Megvii releases (~one-time setup)…")
    urllib.request.urlretrieve(url, ckpt)
    print(f"  ✓ saved to {ckpt} ({ckpt.stat().st_size // (1024*1024)} MB)")
    return ckpt


def auto_detect_batch_size(backbone: str) -> int:
    """Pick a safe per-step batch size based on GPU VRAM + backbone size.

    Hard-tuned from observed memory at fp16 on real GPUs:
      yolox_s @ 640px: ~5 GB at batch 16, scales linearly per batch
      yolox_m @ 640px: ~10 GB at batch 16
      yolox_l @ 640px: ~16 GB at batch 16
      yolox_x @ 640px: ~22 GB at batch 16

    Mosaic peaks during training are ~1.5× steady-state, so we leave 30% headroom.
    """
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True,
        )
        vram_mb = int(out.stdout.strip().splitlines()[0])
    except Exception:
        return 16  # fallback if nvidia-smi isn't available

    # Per-image memory cost in MB at fp16, batch 1, mosaic-peak headroom included.
    # Calibrated from observed peak VRAM on the BCCD benchmark, NVIDIA L4:
    #   yolox_s @ batch 16, fp16, 640px → 5.2 GB peak → ~325 MB/img
    #   yolox_m @ batch 16, fp16, 640px → ~10 GB peak → ~625 MB/img
    cost_per_img = {
        "yolox_nano": 150, "yolox_tiny": 200, "yolox_s": 325,
        "yolox_m":    520, "yolox_l":    850, "yolox_x": 1200,
    }.get(backbone, 325)

    safe_batch = max(1, int((vram_mb * 0.7) / cost_per_img))
    # Round down to nearest sensible power-of-2-ish for BN stat quality
    for size in (64, 48, 32, 24, 16, 8, 4, 2, 1):
        if size <= safe_batch:
            return size
    return 1


# ─── Format auto-detection ─────────────────────────────────────────────────

def detect_format(data_dir: Path) -> str:
    """Return one of voc | coco | yolo | roboflow_voc | roboflow_coco | roboflow_yolo."""
    # Standard Pascal VOC
    if (data_dir / "Annotations").is_dir() and (data_dir / "JPEGImages").is_dir():
        return "voc"
    # COCO (top-level annotations dir)
    if list(data_dir.glob("annotations/*.json")):
        return "coco"
    # YOLO with data.yaml (Ultralytics convention)
    if (data_dir / "data.yaml").is_file() and any(data_dir.rglob("labels/**/*.txt")):
        return "yolo"
    # Roboflow exports — train/ + valid/ with XMLs/JPGs side-by-side (VOC)
    if (data_dir / "train").is_dir():
        train = data_dir / "train"
        if list(train.glob("*.xml")):
            return "roboflow_voc"
        if list(train.glob("_annotations.coco.json")):
            return "roboflow_coco"
        if (train / "images").is_dir() and (train / "labels").is_dir():
            return "roboflow_yolo"
    raise SystemExit(
        f"Could not auto-detect format under {data_dir}. "
        "Pass --format explicitly: voc | coco | yolo | roboflow_voc | roboflow_coco | roboflow_yolo"
    )


# ─── Format loaders (each returns dict: split → DetectionDataset) ──────────

def _load_roboflow_voc(data_dir: Path) -> dict:
    """train/+valid/+test/ — XMLs and JPGs side-by-side."""
    splits = {}
    for split_name, dest_bucket in [("train", "train"), ("valid", "val"),
                                     ("val", "val"), ("test", "val")]:
        d = data_dir / split_name
        if d.is_dir() and list(d.glob("*.xml")):
            ds = sv.DetectionDataset.from_pascal_voc(
                images_directory_path=str(d), annotations_directory_path=str(d))
            splits.setdefault(dest_bucket, []).append(ds)
    return splits


def _load_voc_standard(data_dir: Path) -> dict:
    """Standard Pascal VOC: Annotations/ + JPEGImages/. All as train; we split 80/20."""
    ds = sv.DetectionDataset.from_pascal_voc(
        images_directory_path=str(data_dir / "JPEGImages"),
        annotations_directory_path=str(data_dir / "Annotations"))
    return {"train_full": [ds]}  # we'll split this 80/20 ourselves


def _load_coco(data_dir: Path) -> dict:
    """COCO format: annotations/*.json (look for instances_train2017.json / val2017.json)."""
    splits = {}
    ann_dir = data_dir / "annotations"
    for name, bucket in [("instances_train2017.json", "train"),
                          ("instances_val2017.json", "val"),
                          ("train.json", "train"), ("val.json", "val")]:
        json_path = ann_dir / name
        if json_path.is_file():
            # Try common image roots
            for img_root in ("images", "train2017", "val2017", str(bucket)):
                images_dir = data_dir / img_root
                if images_dir.is_dir():
                    ds = sv.DetectionDataset.from_coco(
                        images_directory_path=str(images_dir),
                        annotations_path=str(json_path))
                    splits.setdefault(bucket, []).append(ds)
                    break
    return splits


def _load_roboflow_coco(data_dir: Path) -> dict:
    """train/{_annotations.coco.json, *.jpg}, valid/, test/."""
    splits = {}
    for split_name, dest_bucket in [("train", "train"), ("valid", "val"),
                                     ("val", "val"), ("test", "val")]:
        d = data_dir / split_name
        ann = d / "_annotations.coco.json"
        if ann.is_file():
            ds = sv.DetectionDataset.from_coco(
                images_directory_path=str(d), annotations_path=str(ann))
            splits.setdefault(dest_bucket, []).append(ds)
    return splits


def _load_yolo(data_dir: Path) -> dict:
    """Ultralytics YOLO: data.yaml + images/{train,val}/ + labels/{train,val}/."""
    yaml_path = data_dir / "data.yaml"
    splits = {}
    for split_name, bucket in [("train", "train"), ("val", "val"), ("valid", "val")]:
        img_dir = data_dir / "images" / split_name
        lbl_dir = data_dir / "labels" / split_name
        if img_dir.is_dir() and lbl_dir.is_dir():
            ds = sv.DetectionDataset.from_yolo(
                images_directory_path=str(img_dir),
                annotations_directory_path=str(lbl_dir),
                data_yaml_path=str(yaml_path))
            splits.setdefault(bucket, []).append(ds)
    return splits


def _load_roboflow_yolo(data_dir: Path) -> dict:
    """Roboflow YOLO: train/{images,labels}/, valid/, test/ + data.yaml."""
    yaml_path = data_dir / "data.yaml"
    splits = {}
    for split_name, bucket in [("train", "train"), ("valid", "val"), ("test", "val")]:
        d = data_dir / split_name
        img_dir = d / "images"
        lbl_dir = d / "labels"
        if img_dir.is_dir() and lbl_dir.is_dir():
            ds = sv.DetectionDataset.from_yolo(
                images_directory_path=str(img_dir),
                annotations_directory_path=str(lbl_dir),
                data_yaml_path=str(yaml_path) if yaml_path.is_file() else None)
            splits.setdefault(bucket, []).append(ds)
    return splits


LOADERS = {
    "voc": _load_voc_standard,
    "roboflow_voc": _load_roboflow_voc,
    "coco": _load_coco,
    "roboflow_coco": _load_roboflow_coco,
    "yolo": _load_yolo,
    "roboflow_yolo": _load_roboflow_yolo,
}


# ─── Normalize to YOLOX VOC layout ─────────────────────────────────────────

def write_voc_layout(splits: dict, voc_dest: Path):
    """Write supervision DetectionDatasets into VOCdevkit/VOC2007 + VOC2012.
    Returns (sorted_classes, n_train, n_val)."""
    voc2007 = voc_dest / "VOC2007"
    if voc_dest.exists():
        shutil.rmtree(voc_dest)
    (voc2007 / "JPEGImages").mkdir(parents=True)
    (voc2007 / "Annotations").mkdir(parents=True)
    (voc2007 / "ImageSets" / "Main").mkdir(parents=True)

    all_classes = set()
    split_stems = {"train": [], "val": []}

    # If only "train_full" present (standard VOC, single split), do 80/20
    if "train_full" in splits and not splits.get("train") and not splits.get("val"):
        full_dses = splits["train_full"]
        merged_classes = set()
        for ds in full_dses:
            merged_classes.update(ds.classes)
            for img_path, _, _ in ds:
                stem = Path(img_path).stem
                split_stems["train"].append(stem)
            ds.as_pascal_voc(
                images_directory_path=str(voc2007 / "JPEGImages"),
                annotations_directory_path=str(voc2007 / "Annotations"))
        all_classes = merged_classes
        # Shuffle + 80/20 split
        import random
        random.Random(42).shuffle(split_stems["train"])
        n_val = max(1, len(split_stems["train"]) // 5)
        split_stems["val"] = split_stems["train"][:n_val]
        split_stems["train"] = split_stems["train"][n_val:]
    else:
        for bucket_name in ("train", "val"):
            for ds in splits.get(bucket_name, []):
                all_classes.update(ds.classes)
                for img_path, _, _ in ds:
                    stem = Path(img_path).stem
                    split_stems[bucket_name].append(stem)
                ds.as_pascal_voc(
                    images_directory_path=str(voc2007 / "JPEGImages"),
                    annotations_directory_path=str(voc2007 / "Annotations"))

    if not split_stems["train"]:
        raise SystemExit("No train images found in dataset.")
    if not split_stems["val"]:
        # No val split → carve 20% from train
        import random
        random.Random(42).shuffle(split_stems["train"])
        n_val = max(1, len(split_stems["train"]) // 5)
        split_stems["val"] = split_stems["train"][:n_val]
        split_stems["train"] = split_stems["train"][n_val:]

    with (voc2007 / "ImageSets" / "Main" / "trainval.txt").open("w") as f:
        f.write("\n".join(split_stems["train"]))
    with (voc2007 / "ImageSets" / "Main" / "test.txt").open("w") as f:
        f.write("\n".join(split_stems["val"]))

    # YOLOX VOC loader requires both VOC2007 + VOC2012
    shutil.copytree(voc2007, voc_dest / "VOC2012")

    classes = tuple(sorted(all_classes))
    return classes, len(split_stems["train"]), len(split_stems["val"])


def patch_yolox_files(classes: tuple, num_classes: int, epochs: int,
                     no_aug_epochs: int, backbone: str = "yolox_s",
                     seed: int = None):
    """Write voc_classes.py + coco_classes.py + patch exp config.
    Targets BOTH pip-installed yolox AND the /app/YOLOX clone.
    Also (v8) writes depth/width matching the requested backbone so non-yolox_s
    backbones actually use their right architecture instead of silently training
    a yolox_s model with a heavier checkpoint.
    Also (v9) sets self.seed in the exp config — Megvii 0.3.0's tools/train.py
    has no --seed flag, so we plumb it through the exp config instead."""
    spec = importlib.util.find_spec("yolox")
    pip_yolox = Path(spec.origin).parent
    clone_yolox = Path("/app/YOLOX/yolox")
    for base in (pip_yolox, clone_yolox):
        for fname, var in [("voc_classes.py", "VOC_CLASSES"),
                            ("coco_classes.py", "COCO_CLASSES")]:
            p = base / "data" / "datasets" / fname
            if p.is_file():
                p.write_text(f"{var} = {classes}\n")
        # Clear bytecode
        cache = base / "data" / "datasets" / "__pycache__"
        if cache.is_dir():
            shutil.rmtree(cache)

    # Patch the exp config.
    #
    # YOLOX 0.3.0's `yolox_voc_s.py` only sets `self.num_classes` directly —
    # `max_epoch` and `no_aug_epochs` are INHERITED from the base `Exp`
    # class (max_epoch=300, no_aug_epochs=15) and never re-stated. A
    # plain `re.sub(r"self\.max_epoch\s*=\s*\d+", ...)` matched nothing
    # and silently no-op'd, so `--epochs 10` got accepted on the CLI
    # but the trainer still ran 300 epochs. Symptom: customer sets
    # EPOCHS=10 in the Colab Settings cell, training shows ETA ~3 hours
    # with `max_epoch │ 300` in the YOLOX trainer banner.
    #
    # Fix: inject max_epoch + no_aug_epochs via the same after-
    # super().__init__() block used for depth/width/seed below.
    # super(Exp, self).__init__() IS in yolox_voc_s.py (guaranteed), so
    # the injection always lands.
    exp_file = Path("/app/YOLOX/exps/example/yolox_voc/yolox_voc_s.py")
    exp = exp_file.read_text()
    exp = re.sub(r"self\.num_classes\s*=\s*\d+", f"self.num_classes = {num_classes}", exp)

    # v8: inject backbone-specific depth + width if not yolox_s.
    depth, width = BACKBONE_CONFIG.get(backbone, BACKBONE_CONFIG["yolox_s"])
    # Remove any prior injected lines (re-run safe).
    exp = re.sub(r"\n\s*self\.depth\s*=\s*[0-9.]+", "", exp)
    exp = re.sub(r"\n\s*self\.width\s*=\s*[0-9.]+", "", exp)
    exp = re.sub(r"\n\s*self\.seed\s*=\s*\S+", "", exp)
    exp = re.sub(r"\n\s*self\.max_epoch\s*=\s*\d+", "", exp)
    exp = re.sub(r"\n\s*self\.no_aug_epochs\s*=\s*\d+", "", exp)
    # v9: seed via exp config (Megvii tools/train.py has no --seed flag).
    seed_line = f"\n        self.seed = {seed}" if seed is not None else ""
    exp = exp.replace(
        "super(Exp, self).__init__()",
        (
            f"super(Exp, self).__init__()"
            f"\n        self.depth = {depth}"
            f"\n        self.width = {width}"
            f"\n        self.max_epoch = {epochs}"
            f"\n        self.no_aug_epochs = {no_aug_epochs}"
            f"{seed_line}"
        ),
        1,
    )
    exp_file.write_text(exp)


def cleanup_intermediate_checkpoints(yolox_out: Path):
    """Delete periodic epoch_NN_ckpt.pth files. Keep best_ckpt.pth + latest_ckpt.pth.
    Saves ~2 GB per run on 300-epoch trainings (44 intermediate checkpoints × 69 MB)."""
    deleted = 0
    for ckpt in yolox_out.glob("epoch_*_ckpt.pth"):
        ckpt.unlink()
        deleted += 1
    return deleted


def stream_subprocess(cmd: list, cwd: str, log_path: Path) -> int:
    """Run a subprocess, stream its stdout/stderr to BOTH the terminal AND log_path.
    Like `tee` but doesn't depend on shell tee. Returns the exit code."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", buffering=1) as f:
        proc = subprocess.Popen(
            cmd, cwd=cwd,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            bufsize=1, text=True,
        )
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            f.write(line)
        return proc.wait()


def ensure_voc_symlink(voc_dest: Path):
    """YOLOX resolves `datasets/VOCdevkit` relative to the yolox package dir
    (not cwd). Symlink the package-relative path to our data location."""
    spec = importlib.util.find_spec("yolox")
    pip_yolox_parent = Path(spec.origin).parent.parent
    link = pip_yolox_parent / "datasets"
    if link.is_symlink() or link.exists():
        if link.is_symlink():
            link.unlink()
        else:
            shutil.rmtree(link)
    link.symlink_to(voc_dest.parent)


# ─── mAP parsing (VOC eval format) ─────────────────────────────────────────

def parse_eval(eval_log_path: Path):
    """Extract map_50, map_5095, per-class AP from VOC eval log."""
    text = eval_log_path.read_text() if eval_log_path.is_file() else ""
    mAP_50 = mAP_50_95 = None
    per_class = {}
    for line in text.splitlines():
        m = re.match(r"\s*map_5095:\s*([0-9.]+)", line)
        if m: mAP_50_95 = float(m.group(1))
        m = re.match(r"\s*map_50:\s*([0-9.]+)", line)
        if m: mAP_50 = float(m.group(1))
        m = re.match(r"AP for (\S+)\s*=\s*([0-9.]+)", line)
        if m: per_class[m.group(1)] = float(m.group(2))
    return mAP_50, mAP_50_95, per_class


# ─── Demo predictions ──────────────────────────────────────────────────────

def run_demo(model_path: Path, voc_dest: Path, demo_dir: Path, n: int = 5):
    """Run tools/demo.py on first N val images. Save annotated outputs to demo_dir."""
    demo_dir.mkdir(parents=True, exist_ok=True)
    val_imgs = sorted((voc_dest / "VOC2007" / "JPEGImages").glob("*.jpg"))[:n]
    for img in val_imgs:
        subprocess.run([
            "python", "tools/demo.py", "image",
            "-f", "exps/example/yolox_voc/yolox_voc_s.py",
            "-c", str(model_path),
            "--path", str(img),
            "--conf", "0.25", "--nms", "0.45", "--tsize", "640",
            "--save_result", "--device", "gpu",
        ], cwd="/app/YOLOX", capture_output=True, text=True)
    # Flatten vis_res outputs into demo_dir
    vis_res = Path("/app/YOLOX/YOLOX_outputs/yolox_voc_s/vis_res")
    if vis_res.is_dir():
        for jpg in vis_res.rglob("*.jpg"):
            shutil.copy2(jpg, demo_dir / jpg.name)


# ─── Main ──────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Path to dataset folder")
    ap.add_argument("--output", required=True, help="Where to put results")
    ap.add_argument("--format", default="auto",
                    choices=["auto", "voc", "coco", "yolo", "roboflow_voc",
                             "roboflow_coco", "roboflow_yolo"])
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--batch-size", type=int, default=0,
                    help="0 = auto-detect from GPU VRAM (v8 default).")
    ap.add_argument("--backbone", default="yolox_s",
                    choices=list(BACKBONE_CONFIG.keys()))
    ap.add_argument("--image-size", type=int, default=640)
    ap.add_argument("--no-aug-epochs", type=int, default=15)
    ap.add_argument("--seed", type=int, default=None,
                    help="RNG seed for reproducibility. Omit for random.")
    ap.add_argument("--keep-intermediates", action="store_true",
                    help="Don't delete epoch_NN_ckpt.pth files after training.")
    fp = ap.add_mutually_exclusive_group()
    fp.add_argument("--fp16", dest="fp16", action="store_true",
                    help="Enable native torch.cuda.amp (default for v6+).")
    fp.add_argument("--no-fp16", dest="fp16", action="store_false")
    ap.set_defaults(fp16=True)
    args = ap.parse_args()

    # v8: auto-pick batch size from GPU VRAM if user didn't specify
    if args.batch_size == 0:
        args.batch_size = auto_detect_batch_size(args.backbone)

    data_dir = Path(args.data).resolve()
    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    prepare_log = output_dir / "prepare.log"
    train_log = output_dir / "train.log"
    eval_log = output_dir / "eval.log"

    print(f"════════════════════════════════════════════════════════════")
    print(f"  YOLOX trainer — generic dataset")
    print(f"════════════════════════════════════════════════════════════")
    print(f"  data:       {data_dir}")
    print(f"  output:     {output_dir}")
    print(f"  format:     {args.format}")
    print(f"  backbone:   {args.backbone}")
    print(f"  epochs:     {args.epochs}")
    print(f"  batch size: {args.batch_size}")
    print(f"  fp16:       {args.fp16}")
    if args.seed is not None:
        print(f"  seed:       {args.seed}")
    depth, width = BACKBONE_CONFIG[args.backbone]
    print(f"  arch:       depth={depth} width={width}")
    print(f"════════════════════════════════════════════════════════════")

    # STEP 1: detect format + normalize to VOC layout
    fmt = detect_format(data_dir) if args.format == "auto" else args.format
    print(f"\n>>> STEP 1: load + normalize (format={fmt})")
    splits = LOADERS[fmt](data_dir)
    voc_dest = Path("/app/YOLOX/datasets/VOCdevkit")
    classes, n_train, n_val = write_voc_layout(splits, voc_dest)
    print(f"  classes ({len(classes)}): {classes}")
    print(f"  splits: trainval={n_train} test={n_val}")
    patch_yolox_files(classes, len(classes), args.epochs, args.no_aug_epochs,
                      backbone=args.backbone, seed=args.seed)
    ensure_voc_symlink(voc_dest)
    print(f"  ✓ dataset prepared")

    # v8: make sure the COCO checkpoint for the requested backbone exists
    ckpt_path = ensure_checkpoint(args.backbone)

    # STEP 2: train (streamed live to terminal AND train.log)
    print(f"\n>>> STEP 2: train ({args.epochs} epochs, batch {args.batch_size}, fp16={args.fp16})")
    train_cmd = ["python", "tools/train.py",
                 "-f", "exps/example/yolox_voc/yolox_voc_s.py",
                 "-d", "1", "-b", str(args.batch_size),
                 "-c", str(ckpt_path)]
    if args.fp16:
        train_cmd.append("--fp16")
    # Megvii 0.3.0's tools/train.py has no --seed CLI arg; seed is set via
    # the exp config (self.seed) inside patch_yolox_files() above.
    t0 = time.time()
    train_rc = stream_subprocess(train_cmd, cwd="/app/YOLOX", log_path=train_log)
    train_seconds = int(time.time() - t0)
    print(f"  training wall-clock: {train_seconds}s ({train_seconds//60}m)  exit={train_rc}")

    # STEP 3: eval
    print(f"\n>>> STEP 3: eval")
    yolox_out = Path("/app/YOLOX/YOLOX_outputs/yolox_voc_s")
    model_path = yolox_out / "best_ckpt.pth"
    if not model_path.is_file():
        model_path = yolox_out / "latest_ckpt.pth"
    if not model_path.is_file():
        print(f"  ❌ no checkpoint found at {yolox_out}")
        sys.exit(1)

    eval_cmd = ["python", "tools/eval.py",
                "-n", f"{args.backbone.replace('_', '-')}",
                "-c", str(model_path),
                "-b", "64", "-d", "1", "--conf", "0.001",
                "-f", "exps/example/yolox_voc/yolox_voc_s.py"]
    stream_subprocess(eval_cmd, cwd="/app/YOLOX", log_path=eval_log)

    mAP_50, mAP_50_95, per_class = parse_eval(eval_log)
    print(f"  mAP@0.5     = {mAP_50}")
    print(f"  mAP@0.5:0.95 = {mAP_50_95}")
    print(f"  per_class   = {per_class}")

    # STEP 4: demo predictions
    print(f"\n>>> STEP 4: demo predictions on first 5 val images")
    run_demo(model_path, voc_dest, output_dir / "demo_predictions")

    # v8: clean up intermediate epoch_NN_ckpt.pth before copying (~2 GB savings)
    if not args.keep_intermediates:
        n_deleted = cleanup_intermediate_checkpoints(yolox_out)
        if n_deleted:
            print(f"  ✓ deleted {n_deleted} intermediate epoch_NN_ckpt.pth files")

    # STEP 5: copy checkpoints + write results.json
    print(f"\n>>> STEP 5: copy outputs")
    shutil.copytree(yolox_out, output_dir / "yolox_voc_s", dirs_exist_ok=True)
    results = {
        "trainer": "yolox_upstream_megvii_0.3.0",
        "dataset": str(data_dir),
        "format": fmt,
        "backbone": args.backbone,
        "backbone_depth_width": list(BACKBONE_CONFIG[args.backbone]),
        "image_size": args.image_size,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "n_train": n_train,
        "n_val": n_val,
        "classes": list(classes),
        "mAP_50": mAP_50,
        "mAP_50_95": mAP_50_95,
        "per_class_AP": per_class,
        "training_seconds": train_seconds,
        "recipe": {
            "optimizer": "SGD",
            "lr_scheduler": "yoloxwarmcos",
            "amp_fp16": args.fp16,
            "ema": True,
            "mosaic_prob": 1.0,
            "mixup_prob": 1.0,
            "no_aug_epochs": args.no_aug_epochs,
        },
    }
    (output_dir / "results.json").write_text(json.dumps(results, indent=2))

    print(f"\n════════════════════════════════════════════════════════════")
    print(f"  DONE. Results in: {output_dir}")
    print(f"════════════════════════════════════════════════════════════")


if __name__ == "__main__":
    main()
