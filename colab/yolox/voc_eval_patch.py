"""Patches yolox/evaluators/voc_eval.py to be None-safe + per-object try/except.

Supervision's `as_pascal_voc()` writes minimal XMLs (just <name> + <bndbox>),
missing <pose>, <truncated>, <difficult>. Megvii YOLOX's parse_rec assumes
full VOC2007 spec → AttributeError on .text of None.

This patch:
  - adds a `_safe_text` helper that handles missing elements
  - defaults pose='Unspecified', truncated=0, difficult=0 when missing
  - wraps each object's parse in try/except so one bad object doesn't crash
  - skips objects with no <name> (no class label to attach)
"""
import importlib.util as u, os, sys

p = os.path.dirname(u.find_spec('yolox').origin) + '/evaluators/voc_eval.py'
s = open(p).read()

old = '''def parse_rec(filename):
    """ Parse a PASCAL VOC xml file """
    tree = ET.parse(filename)
    objects = []
    for obj in tree.findall("object"):
        obj_struct = {}
        obj_struct["name"] = obj.find("name").text
        obj_struct["pose"] = obj.find("pose").text
        obj_struct["truncated"] = int(obj.find("truncated").text)
        obj_struct["difficult"] = int(obj.find("difficult").text)
        bbox = obj.find("bndbox")
        obj_struct["bbox"] = [
            int(bbox.find("xmin").text),
            int(bbox.find("ymin").text),
            int(bbox.find("xmax").text),
            int(bbox.find("ymax").text),
        ]
        objects.append(obj_struct)

    return objects'''

new = '''def _safe_text(elem, default=None):
    """Return elem.text if elem is not None, else default (patch helper)."""
    return elem.text if elem is not None else default


def parse_rec(filename):
    """ Parse a PASCAL VOC xml file (patched: None-safe + per-object try/except) """
    tree = ET.parse(filename)
    objects = []
    for obj in tree.findall("object"):
        try:
            obj_struct = {}
            obj_struct["name"] = _safe_text(obj.find("name"))
            obj_struct["pose"] = _safe_text(obj.find("pose"), "Unspecified")
            obj_struct["truncated"] = int(_safe_text(obj.find("truncated"), "0"))
            obj_struct["difficult"] = int(_safe_text(obj.find("difficult"), "0"))
            bbox = obj.find("bndbox")
            obj_struct["bbox"] = [
                int(float(_safe_text(bbox.find("xmin"), "0"))),
                int(float(_safe_text(bbox.find("ymin"), "0"))),
                int(float(_safe_text(bbox.find("xmax"), "0"))),
                int(float(_safe_text(bbox.find("ymax"), "0"))),
            ]
            if obj_struct["name"] is None:
                continue
            objects.append(obj_struct)
        except (AttributeError, ValueError, TypeError) as e:
            print(f"voc_eval.parse_rec: skipping malformed object in {filename}: {e}")
            continue

    return objects'''

if old not in s:
    print(f"ERROR: parse_rec pattern not found in {p}")
    sys.exit(1)
s = s.replace(old, new)
open(p, 'w').write(s)
print(f'patched {p}')
