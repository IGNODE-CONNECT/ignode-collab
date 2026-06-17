"""Prepare BCCD dataset for YOLOX training (Megvii mainline 0.3.0).

Reads the BCCD VOC zip's extracted contents at /app/bccd_extracted/{train,valid},
normalizes via supervision (the same library IGNODE's ignode-format-converter
sidecar uses), writes into YOLOX's expected `VOCdevkit/VOC2007 + VOC2012` layout,
lowercases class names (YOLOX voc.py does `.lower()` before lookup), and
patches the exp config for our num_classes + epoch count.

Megvii mainline differences from Roboflow fork:
  - apex is OPTIONAL (made so upstream in 2022) — no patches needed
  - voc_classes.py still needs lowercase classes (same upstream behaviour)
  - The pip-installed yolox package is the active one — we patch THAT location,
    not the /app/YOLOX/ clone (clone is only used for tools/ + exps/).
"""
import glob
import importlib.util
import os
import re
import shutil
import sys

import supervision as sv

EXTRACTED_ROOT = '/app/bccd_extracted'
VOC_DEST       = '/app/YOLOX/datasets/VOCdevkit'
EPOCHS         = int(os.environ.get('EPOCHS', '40'))
NO_AUG_EPOCHS  = int(os.environ.get('NO_AUG_EPOCHS', '15'))


def yolox_package_dir():
    """Return the directory of the active yolox package (pip-installed)."""
    spec = importlib.util.find_spec('yolox')
    return os.path.dirname(spec.origin)


def find_roboflow_splits(root):
    """Roboflow VOC zips contain train/+valid/ folders with XMLs+JPGs side-by-side."""
    candidates = ['train', 'valid', 'val', 'test']
    return [os.path.join(root, s) for s in candidates
            if os.path.isdir(os.path.join(root, s)) and glob.glob(f'{root}/{s}/*.xml')]


def main():
    print(f'═══ Reading BCCD from {EXTRACTED_ROOT} ═══')
    splits = find_roboflow_splits(EXTRACTED_ROOT)
    assert splits, f'No train/valid splits found under {EXTRACTED_ROOT}'
    print(f'  found splits: {splits}')

    voc2007 = f'{VOC_DEST}/VOC2007'
    shutil.rmtree(VOC_DEST, ignore_errors=True)
    os.makedirs(f'{voc2007}/JPEGImages')
    os.makedirs(f'{voc2007}/Annotations')
    os.makedirs(f'{voc2007}/ImageSets/Main')

    split_stems = {'train': [], 'val': []}
    all_classes = set()

    for split_dir in splits:
        ds = sv.DetectionDataset.from_pascal_voc(
            images_directory_path=split_dir,
            annotations_directory_path=split_dir,
        )
        all_classes.update(ds.classes)

        bucket = 'train' if os.path.basename(split_dir) == 'train' else 'val'
        for image_path, _, _ in ds:
            split_stems[bucket].append(os.path.splitext(os.path.basename(image_path))[0])

        ds.as_pascal_voc(
            images_directory_path=f'{voc2007}/JPEGImages',
            annotations_directory_path=f'{voc2007}/Annotations',
        )
        print(f'  {split_dir}: wrote {len(ds)} images')

    with open(f'{voc2007}/ImageSets/Main/trainval.txt', 'w') as f:
        f.write('\n'.join(split_stems['train']))
    with open(f'{voc2007}/ImageSets/Main/test.txt', 'w') as f:
        f.write('\n'.join(split_stems['val']))
    print(f'  splits: trainval={len(split_stems["train"])} test={len(split_stems["val"])}')

    shutil.copytree(voc2007, f'{VOC_DEST}/VOC2012')
    print(f'  duplicated to {VOC_DEST}/VOC2012')

    # YOLOX resolves `datasets/VOCdevkit/` relative to the yolox package install
    # path (not cwd), so we also need that location to find our data. Use a
    # symlink so we don't have to maintain two copies.
    pip_yolox_parent = os.path.dirname(yolox_package_dir())
    pkg_datasets_link = os.path.join(pip_yolox_parent, 'datasets')
    if os.path.islink(pkg_datasets_link) or os.path.exists(pkg_datasets_link):
        if os.path.islink(pkg_datasets_link):
            os.unlink(pkg_datasets_link)
        else:
            shutil.rmtree(pkg_datasets_link)
    os.symlink('/app/YOLOX/datasets', pkg_datasets_link)
    print(f'  symlinked {pkg_datasets_link} → /app/YOLOX/datasets')

    # Preserve original XML case in voc_classes.py. The Roboflow 2021 fork
    # lowercased the XML <name> tag before dict lookup (so VOC_CLASSES had to
    # be lowercase to match); Megvii mainline 0.3.0 does NOT lowercase — it
    # uses the raw XML name. So VOC_CLASSES must EXACTLY match the XML case.
    classes_named = tuple(sorted(all_classes))
    num_classes = len(classes_named)
    print(f'\n═══ Classes (preserved XML case): {classes_named} ═══')

    pip_yolox = yolox_package_dir()
    clone_yolox = '/app/YOLOX/yolox'
    targets = []
    for base in (pip_yolox, clone_yolox):
        targets.append((f'{base}/data/datasets/voc_classes.py', 'VOC_CLASSES'))
        targets.append((f'{base}/data/datasets/coco_classes.py', 'COCO_CLASSES'))
    for p, var in targets:
        if not os.path.isfile(p):
            print(f'  - skip {p} (does not exist)')
            continue
        with open(p, 'w') as f:
            f.write(f'{var} = {classes_named}\n')
        print(f'  ✓ wrote {p}')

    # Patch the exp config: our num_classes + epoch count
    exp_file = '/app/YOLOX/exps/example/yolox_voc/yolox_voc_s.py'
    with open(exp_file) as f:
        exp = f.read()
    exp = re.sub(r'self\.num_classes\s*=\s*\d+', f'self.num_classes = {num_classes}', exp)
    exp = re.sub(r'self\.max_epoch\s*=\s*\d+', f'self.max_epoch = {EPOCHS}', exp)
    exp = re.sub(r'self\.no_aug_epochs\s*=\s*\d+', f'self.no_aug_epochs = {NO_AUG_EPOCHS}', exp)
    with open(exp_file, 'w') as f:
        f.write(exp)
    print(f'\n  patched {exp_file}: num_classes={num_classes} max_epoch={EPOCHS} no_aug_epochs={NO_AUG_EPOCHS}')

    # Clear bytecode caches so the next subprocess picks up our edits
    for base in (pip_yolox, clone_yolox):
        for cache in (f'{base}/data/datasets/__pycache__',
                      f'{base}/__pycache__'):
            shutil.rmtree(cache, ignore_errors=True)

    print(f'\n✓ Dataset prepared at {VOC_DEST}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
