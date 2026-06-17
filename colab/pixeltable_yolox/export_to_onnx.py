#!/usr/bin/env python3
"""Custom ONNX exporter for pixeltable-yolox trained checkpoints.

Workaround for the broken pixeltable-yolox 0.4.2 `yolox export_onnx`
CLI (subcommand not registered + the underlying module references
`yolox.exp` which doesn't exist in the refactored package). See
ansible-deploy/documentation/MLOPS/IMAGE_RECOGNITION/IMPLAMENTATION_PLAN/
IR_3_1/PIXEL_TABLE_TEST.md for the full evaluation.

Matches IGNODE's UINF `yolox_raw` inference contract:
    - decode_in_inference = False  (UINF host-side decodes via anchor-grid
      + stride projection on the 3 raw output heads at strides 8/16/32)
    - input  name "images"   shape [N, 3, 640, 640]   dynamic batch
    - output name "output"                              dynamic batch
    - opset 18

Usage:
    python export_to_onnx.py <ckpt.pth> <out.onnx> \\
        --num-classes 3 \\
        --arch yolox_s \\
        [--input-size 640] \\
        [--opset 18]
"""
from __future__ import annotations

import argparse
import sys

import torch

# Per-arch YoloxConfig subclasses, looked up by `--arch` flag. Each
# subclass's __init__ pre-sets depth + width + (for tiny/nano) input
# size + augmentation knobs to match the canonical Megvii values.
ARCH_CLASSES = {
    "yolox_s":    "YoloxS",
    "yolox_m":    "YoloxM",
    "yolox_l":    "YoloxL",
    "yolox_x":    "YoloxX",
    "yolox_tiny": "YoloxTiny",
    "yolox_nano": "YoloxNano",
}


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ckpt", help="Path to trained .pth checkpoint")
    ap.add_argument("out", help="Output .onnx path")
    ap.add_argument("--num-classes", type=int, required=True,
        help="Number of classes the checkpoint was trained for")
    ap.add_argument("--arch", default="yolox_s", choices=list(ARCH_CLASSES.keys()),
        help="YOLOX architecture (default yolox_s)")
    ap.add_argument("--input-size", type=int, default=640,
        help="Square input size (default 640)")
    ap.add_argument("--opset", type=int, default=18,
        help="ONNX opset version (default 18; matches IGNODE platform pin)")
    return ap.parse_args()


def main() -> int:
    args = parse_args()

    # Import inside main() so --help doesn't require yolox installed.
    # Use the per-arch YoloxConfig subclass — it pre-sets depth + width
    # correctly (the base YoloxConfig defaults to yolox_l / depth=width=1.0).
    # NOTE: do NOT use `yolox.models.create_yolox_model` or `yolox.models.yolox_s`
    # — those factories import `yolox.exp` which doesn't exist in 0.4.2
    # (half-finished refactor, same trap as the broken `yolox export_onnx`
    # CLI subcommand this script replaces).
    import yolox.config as yc
    config = getattr(yc, ARCH_CLASSES[args.arch])()  # e.g. YoloxS()
    config.num_classes = args.num_classes

    # get_model() returns a fresh YoloxModule (backbone + head).
    # It calls .train() internally — we override to .eval() for export.
    model = config.get_model()
    model.eval()

    # Load the trained checkpoint.
    # weights_only=False because the checkpoint contains numpy globals
    # (torch 2.6+ defaults to weights_only=True which rejects YOLOX).
    # Safe — we trust our own training output.
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)

    # Pixeltable + Megvii both wrap the state dict in a "model" key inside
    # the checkpoint dict. If that key is missing, treat the whole load
    # as the state dict (handles bare-tensor checkpoints).
    state_dict = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print(f"  state_dict missing keys:    {len(missing)}", file=sys.stderr)
        print(f"  state_dict unexpected keys: {len(unexpected)}", file=sys.stderr)
        if missing:
            print(f"  first missing:    {missing[:3]}", file=sys.stderr)
        if unexpected:
            print(f"  first unexpected: {unexpected[:3]}", file=sys.stderr)

    # IGNODE UINF yolox_raw contract: raw outputs at strides 8/16/32.
    # UINF does anchor-grid + stride projection host-side, so the network
    # must NOT do that decoding itself.
    if hasattr(model, "head"):
        model.head.decode_in_inference = False

    # Export.
    #
    # `dynamo=False` pins us to the legacy TorchScript-based exporter.
    # torch 2.6+ flipped the default to the new `dynamo` exporter which
    # requires `onnxscript` as a dep we don't install (and would change
    # the produced ONNX graph shape — we want byte-stable output vs
    # the smoke harness validation that ran on torch 2.5.x).
    dummy = torch.randn(1, 3, args.input_size, args.input_size)
    torch.onnx.export(
        model, dummy, args.out,
        input_names=["images"],
        output_names=["output"],
        opset_version=args.opset,
        dynamic_axes={
            "images": {0: "batch"},
            "output": {0: "batch"},
        },
        dynamo=False,
    )

    # Verify the file landed + report shape.
    import onnx
    m = onnx.load(args.out)
    print(f"exported {args.out}", file=sys.stderr)
    print(f"  opset:        {m.opset_import[0].version}", file=sys.stderr)
    print(f"  inputs:       "
          f"{[(i.name, [d.dim_value or d.dim_param for d in i.type.tensor_type.shape.dim]) for i in m.graph.input]}",
          file=sys.stderr)
    print(f"  outputs:      "
          f"{[(o.name, [d.dim_value or d.dim_param for d in o.type.tensor_type.shape.dim]) for o in m.graph.output]}",
          file=sys.stderr)
    print(f"  nodes:        {len(m.graph.node)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
