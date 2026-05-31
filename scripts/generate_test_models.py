"""Generate test model artifacts in every format the IGNODE Custom Model
Upload wizard accepts. Output is dropped into `test-artifacts/` next to
this script — each subfolder contains model + sidecars + a ready-to-upload
`.zip` bundle.

Coverage matrix:

  Format     | Classification               | Regression                   | Image
  -----------+------------------------------+------------------------------+-------------------
  lgbm_text  | iris (3-class)               | diabetes                     | (not supported)
  xgb_json   | iris (3-class)               | diabetes                     | (not supported)
  onnx       | iris (3-class, via skl2onnx) | diabetes (via skl2onnx)      | tiny pytorch CNN
  tflite     | (not supported)              | (not supported)              | tiny keras CNN

Optional deps (skipped with a warning if not importable):
  - lightgbm, xgboost, scikit-learn, onnx, skl2onnx, onnxmltools, onnxconverter_common
  - torch + torchvision      → ONNX image classifier
  - tensorflow               → TFLite image classifier

Usage:
  python scripts/generate_test_models.py             # generate everything we can
  python scripts/generate_test_models.py --only onnx # filter

Workflow:
  1. Run this script locally (one-time).
  2. `test-artifacts/<kind>/model-artifacts.zip` is the ready-to-upload ZIP.
  3. Drag the ZIP into IGNODE → Custom Model Upload wizard to test the
     full pipeline (upload → deploy to UINF → predict from Playground).

Models are deliberately TINY — they're for testing format handling, not
accuracy. iris + diabetes are sklearn built-ins (no network access needed).
"""
from __future__ import annotations

import argparse
import importlib
import io
import json
import os
import sys
import zipfile
from pathlib import Path
from typing import Any

# ─── Where artifacts land ────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent / "test-artifacts"


# ─── Sidecar helpers (canonical IR-2.15 schemas) ─────────────────────
def write_class_labels(folder: Path, labels: list[str]) -> None:
    (folder / "class_labels.json").write_text(
        json.dumps({"schema_version": 1, "class_labels": labels}, indent=2)
    )


def write_feature_columns(
    folder: Path,
    feature_columns: list[str],
    label_columns: list[str] | None = None,
) -> None:
    (folder / "feature_columns.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "feature_columns": feature_columns,
                "label_columns": label_columns or [],
            },
            indent=2,
        )
    )


def write_preprocess_config(folder: Path) -> None:
    """Standard ImageNet preprocessing — what the IR-2.1 trainer emits."""
    (folder / "preprocess_config.json").write_text(
        json.dumps(
            {
                "input_size": [224, 224],
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
                "channel_order": "RGB",
                "image_format": "CHW",
                "rescale": "imagenet",
                "resize_method": "center_crop",
                "resize_short_side": 256,
            },
            indent=2,
        )
    )


def bundle_zip(folder: Path) -> Path:
    """Zip every file in `folder` (flat, no subdirs) into model-artifacts.zip."""
    zip_path = folder / "model-artifacts.zip"
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for child in sorted(folder.iterdir()):
            if child.is_file() and child.name != "model-artifacts.zip":
                zf.write(child, arcname=child.name)
    return zip_path


def safe_import(name: str) -> Any | None:
    """Import a module; return None (and print a warning) if unavailable."""
    try:
        return importlib.import_module(name)
    except ImportError as ex:
        print(f"  [skip] missing dep: {name} ({ex})", file=sys.stderr)
        return None


def prepare(kind: str) -> Path:
    """Create a clean folder for one test artifact + return its path."""
    folder = ROOT / kind
    folder.mkdir(parents=True, exist_ok=True)
    # Wipe stale files so re-runs don't accumulate cruft.
    for existing in folder.iterdir():
        if existing.is_file():
            existing.unlink()
    return folder


# ─── LightGBM ────────────────────────────────────────────────────────
def gen_lgbm_classifier() -> str | None:
    lgb = safe_import("lightgbm")
    sk = safe_import("sklearn")
    if lgb is None or sk is None:
        return None
    from sklearn.datasets import load_iris
    iris = load_iris()
    X, y = iris.data, iris.target
    feature_names = list(iris.feature_names)
    class_names = list(iris.target_names)

    model = lgb.LGBMClassifier(n_estimators=20, learning_rate=0.1, verbose=-1)
    model.fit(X, y)

    folder = prepare("lightgbm-classifier")
    model.booster_.save_model(str(folder / "model.txt"))
    write_class_labels(folder, class_names)
    write_feature_columns(folder, feature_names, label_columns=["species"])
    bundle_zip(folder)
    return "lightgbm-classifier"


def gen_lgbm_regressor() -> str | None:
    lgb = safe_import("lightgbm")
    sk = safe_import("sklearn")
    if lgb is None or sk is None:
        return None
    from sklearn.datasets import load_diabetes
    diabetes = load_diabetes()
    X, y = diabetes.data, diabetes.target
    feature_names = list(diabetes.feature_names)

    model = lgb.LGBMRegressor(n_estimators=20, learning_rate=0.1, verbose=-1)
    model.fit(X, y)

    folder = prepare("lightgbm-regressor")
    model.booster_.save_model(str(folder / "model.txt"))
    write_feature_columns(folder, feature_names, label_columns=["progression"])
    bundle_zip(folder)
    return "lightgbm-regressor"


# ─── XGBoost ─────────────────────────────────────────────────────────
def gen_xgb_classifier() -> str | None:
    xgb = safe_import("xgboost")
    sk = safe_import("sklearn")
    if xgb is None or sk is None:
        return None
    from sklearn.datasets import load_iris
    iris = load_iris()
    X, y = iris.data, iris.target
    feature_names = list(iris.feature_names)
    class_names = list(iris.target_names)

    model = xgb.XGBClassifier(n_estimators=20, learning_rate=0.1, eval_metric="mlogloss")
    model.fit(X, y)

    folder = prepare("xgboost-classifier")
    # Save the raw booster, not the sklearn wrapper — newer xgboost's
    # XGBClassifier.save_model gates on sklearn metadata that's brittle
    # across versions, and the IGNODE xgb_json loader reads booster JSON
    # anyway (Booster.load_model on the wire-side).
    model.get_booster().save_model(str(folder / "model.json"))
    write_class_labels(folder, class_names)
    write_feature_columns(folder, feature_names, label_columns=["species"])
    bundle_zip(folder)
    return "xgboost-classifier"


def gen_xgb_regressor() -> str | None:
    xgb = safe_import("xgboost")
    sk = safe_import("sklearn")
    if xgb is None or sk is None:
        return None
    from sklearn.datasets import load_diabetes
    diabetes = load_diabetes()
    X, y = diabetes.data, diabetes.target
    feature_names = list(diabetes.feature_names)

    model = xgb.XGBRegressor(n_estimators=20, learning_rate=0.1)
    model.fit(X, y)

    folder = prepare("xgboost-regressor")
    model.get_booster().save_model(str(folder / "model.json"))
    write_feature_columns(folder, feature_names, label_columns=["progression"])
    bundle_zip(folder)
    return "xgboost-regressor"


# ─── Sklearn → ONNX (tabular) ────────────────────────────────────────
def gen_sklearn_onnx_classifier() -> str | None:
    sk = safe_import("sklearn")
    onnx = safe_import("onnx")
    skl2onnx = safe_import("skl2onnx")
    if None in (sk, onnx, skl2onnx):
        return None
    from sklearn.datasets import load_iris
    from sklearn.ensemble import RandomForestClassifier
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType

    iris = load_iris()
    X, y = iris.data, iris.target
    feature_names = list(iris.feature_names)
    class_names = list(iris.target_names)

    model = RandomForestClassifier(n_estimators=10, random_state=0)
    model.fit(X, y)

    initial_types = [("input", FloatTensorType([None, len(feature_names)]))]
    onnx_model = convert_sklearn(model, initial_types=initial_types, target_opset=15)

    folder = prepare("sklearn-onnx-classifier")
    onnx.save_model(onnx_model, str(folder / "model.onnx"))
    write_class_labels(folder, class_names)
    write_feature_columns(folder, feature_names, label_columns=["species"])
    bundle_zip(folder)
    return "sklearn-onnx-classifier"


def gen_sklearn_onnx_regressor() -> str | None:
    sk = safe_import("sklearn")
    onnx = safe_import("onnx")
    skl2onnx = safe_import("skl2onnx")
    if None in (sk, onnx, skl2onnx):
        return None
    from sklearn.datasets import load_diabetes
    from sklearn.ensemble import RandomForestRegressor
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType

    diabetes = load_diabetes()
    X, y = diabetes.data, diabetes.target
    feature_names = list(diabetes.feature_names)

    model = RandomForestRegressor(n_estimators=10, random_state=0)
    model.fit(X, y)

    initial_types = [("input", FloatTensorType([None, len(feature_names)]))]
    onnx_model = convert_sklearn(model, initial_types=initial_types, target_opset=15)

    folder = prepare("sklearn-onnx-regressor")
    onnx.save_model(onnx_model, str(folder / "model.onnx"))
    write_feature_columns(folder, feature_names, label_columns=["progression"])
    bundle_zip(folder)
    return "sklearn-onnx-regressor"


# ─── PyTorch → ONNX (image classification) ───────────────────────────
def gen_onnx_image_classifier() -> str | None:
    torch = safe_import("torch")
    torchvision = safe_import("torchvision")
    if torch is None or torchvision is None:
        return None
    from torchvision import models

    # MobileNetV2 is tiny enough to ship as a fixture (~14 MB ONNX); pretrained
    # so /predict returns something meaningful even without training.
    model = models.mobilenet_v2(weights=None)  # random weights — we don't need accuracy
    model.eval()

    folder = prepare("onnx-image-classifier")
    dummy = torch.randn(1, 3, 224, 224)
    torch.onnx.export(
        model,
        dummy,
        str(folder / "model.onnx"),
        input_names=["input"],
        output_names=["Score"],
        opset_version=18,
        dynamic_axes={"input": {0: "batch_size"}, "Score": {0: "batch_size"}},
        do_constant_folding=True,
    )
    # Synthetic 10-class labels — enough to exercise the wizard's class-list UI.
    write_class_labels(folder, [f"class_{i}" for i in range(1000)])
    write_preprocess_config(folder)
    bundle_zip(folder)
    return "onnx-image-classifier"


# ─── TFLite (image classification) ───────────────────────────────────
def gen_tflite_image_classifier() -> str | None:
    tf = safe_import("tensorflow")
    if tf is None:
        return None
    # Tiny CNN — random weights are fine; we only need a valid .tflite file
    # that loads under the IR-1 tflite-runtime loader.
    inputs = tf.keras.Input(shape=(224, 224, 3), name="input")
    x = tf.keras.layers.Conv2D(8, 3, activation="relu")(inputs)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    outputs = tf.keras.layers.Dense(10, activation="softmax", name="Score")(x)
    model = tf.keras.Model(inputs, outputs)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_bytes = converter.convert()

    folder = prepare("tflite-image-classifier")
    (folder / "model.tflite").write_bytes(tflite_bytes)
    write_class_labels(folder, [f"class_{i}" for i in range(10)])
    write_preprocess_config(folder)
    bundle_zip(folder)
    return "tflite-image-classifier"


# ─── Driver ──────────────────────────────────────────────────────────
GENERATORS = {
    "lgbm-cls":   gen_lgbm_classifier,
    "lgbm-reg":   gen_lgbm_regressor,
    "xgb-cls":    gen_xgb_classifier,
    "xgb-reg":    gen_xgb_regressor,
    "onnx-cls":   gen_sklearn_onnx_classifier,
    "onnx-reg":   gen_sklearn_onnx_regressor,
    "onnx-img":   gen_onnx_image_classifier,
    "tflite-img": gen_tflite_image_classifier,
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--only",
        nargs="+",
        choices=list(GENERATORS),
        help="Generate only specific artifacts (default: all)",
    )
    args = parser.parse_args()

    targets = args.only if args.only else list(GENERATORS)
    ROOT.mkdir(parents=True, exist_ok=True)

    generated: list[str] = []
    skipped: list[str] = []
    for key in targets:
        print(f"\n>> {key}")
        result = GENERATORS[key]()
        if result is None:
            skipped.append(key)
        else:
            zip_path = ROOT / result / "model-artifacts.zip"
            print(f"  [ok] {zip_path.relative_to(ROOT.parent)}  ({zip_path.stat().st_size / 1024:.1f} KB)")
            generated.append(key)

    print()
    print(f"Generated: {len(generated)}/{len(targets)}")
    if skipped:
        print(f"Skipped:   {skipped}  (install missing deps to enable)")
    print(f"\nArtifacts directory: {ROOT}")
    print("Drag any model-artifacts.zip into the Custom Model Upload wizard to test.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
