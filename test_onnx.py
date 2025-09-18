#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.optim as optim
import onnx
import onnx2torch
from pathlib import Path

# Optional: if you still want to sanity-check the detector cache import
try:
    from open_image_models import LicensePlateDetector
except Exception:
    LicensePlateDetector = None

# ----------------------------
# Utilities
# ----------------------------


def _onnx_path_must_exist(path: Path, label: str):
    if not path.exists():
        raise FileNotFoundError(f"{label} ONNX model not found at: {path}")
    return path


def _ocr_layout_from_onnx(onnx_model: onnx.ModelProto) -> str:
    """
    Heuristic: if the very first op is a Transpose with perm [0,3,1,2] that takes the graph input,
    the ONNX graph is NHWC and onnx2torch will convert to NCHW internally.
    Otherwise assume it's already NCHW.
    """
    if not onnx_model.graph.node:
        return "nchw"

    graph_input = onnx_model.graph.input[0].name if onnx_model.graph.input else None
    first = onnx_model.graph.node[0]

    if first.op_type == "Transpose" and graph_input and first.input and first.input[0] == graph_input:
        # read perms if present
        perms = None
        for attr in first.attribute:
            if attr.name == "perm":
                perms = list(attr.ints)
                break
        if perms == [0, 3, 1, 2]:
            return "nhwc"  # expects NHWC at the graph boundary
    return "nchw"


def _print_model_io_info(onnx_model: onnx.ModelProto, title: str):
    def _fmt_value_info(vi):
        t = vi.type.tensor_type
        shape = []
        for d in t.shape.dim:
            shape.append(d.dim_value if d.HasField("dim_value") else (
                d.dim_param if d.HasField("dim_param") else "?"))
        return f"{vi.name}: {shape}"
    ins = ", ".join(_fmt_value_info(i) for i in onnx_model.graph.input)
    outs = ", ".join(_fmt_value_info(o) for o in onnx_model.graph.output)
    print(f"[{title}] ONNX IO -> inputs: {ins} | outputs: {outs}")

# ----------------------------
# Detector
# ----------------------------


def test_gradient_descent_detector():
    print("Loading detector ONNX from cache...")
    # If you want to instantiate the helper library, keep it optional
    if LicensePlateDetector is not None:
        _ = LicensePlateDetector(detection_model="yolo-v9-t-384-license-plate-end2end")

    # Resolve ONNX from the known cache path
    model_cache_dir = Path.home() / ".cache/open-image-models/yolo-v9-t-384-license-plate-end2end"
    onnx_path = _onnx_path_must_exist(
        model_cache_dir / "yolo-v9-t-384-license-plates-end2end.onnx",
        "Detector"
    )

    print(f"Converting detector ONNX to PyTorch: {onnx_path}")
    onnx_model = onnx.load(str(onnx_path))
    _print_model_io_info(onnx_model, "Detector")

    pytorch_model = onnx2torch.convert(onnx_model)
    pytorch_model.train()

    print("Testing differentiability (detector)...")
    # Most YOLO exports are NCHW. 3x384x384 is a safe default.
    inp = torch.randn(1, 3, 384, 384, requires_grad=True)
    out = pytorch_model(inp)

    target = torch.randn_like(out)
    loss = nn.MSELoss()(out, target)

    opt = optim.SGD(pytorch_model.parameters(), lr=1e-3)
    opt.zero_grad()
    loss.backward()
    opt.step()

    print(f"✅ Detector SUCCESS: Loss = {loss.item():.4f}")
    return pytorch_model

# ----------------------------
# OCR
# ----------------------------


def _onnx_input_layout_and_size(onnx_model):
    """Return ('nhwc'|'nchw', (H,W,C_or_None)) from the first graph input."""
    vi = onnx_model.graph.input[0].type.tensor_type.shape.dim
    dims = [d.dim_value if d.HasField("dim_value") else None for d in vi]  # [N, ?, ?, ?]
    # Heuristics: prefer last-dim=3/1 => NHWC. prefer second-dim=3/1 => NCHW.
    if len(dims) == 4:
        n, d1, d2, d3 = dims
        if d3 in (3, 1):  # ...xC
            return "nhwc", (d1, d2, d3)
        if d1 in (3, 1):  # Cx...
            return "nchw", (d2, d3, d1)
    # Fallback default for OCR exports
    return "nhwc", (64, 128, 3)


def test_gradient_descent_ocr():
    print("Loading OCR ONNX...")
    onnx_path = Path.home() / ".cache/fast-plate-ocr/cct-xs-v1-global-model/cct_xs_v1_global.onnx"
    if not onnx_path.exists():
        raise FileNotFoundError(f"OCR ONNX not found at {onnx_path}")

    print(f"Converting OCR ONNX to PyTorch: {onnx_path}")
    onnx_model = onnx.load(str(onnx_path))

    layout, (H, W, C) = _onnx_input_layout_and_size(onnx_model)
    print(f"[OCR] Boundary layout from ONNX value_info: {layout.upper()} with H={H}, W={W}, C={C}")

    pytorch_model = onnx2torch.convert(onnx_model)
    pytorch_model.train()

    print("Testing differentiability (OCR)...")
    # Build candidate inputs (primary from ONNX; fallback to the opposite layout)
    primary = torch.randn(
        1, H, W, C, requires_grad=True) if layout == "nhwc" else torch.randn(
        1, C, H, W, requires_grad=True)
    fallback = torch.randn(
        1, C, H, W, requires_grad=True) if layout == "nhwc" else torch.randn(
        1, H, W, C, requires_grad=True)

    # Try primary, otherwise fallback automatically
    try:
        out = pytorch_model(primary)
        print(out)
        used = layout
        inp = primary
    except Exception as e_primary:
        print(
            f"[OCR] Primary layout {layout} failed, trying fallback... ({type(e_primary).__name__}: {e_primary})")
        out = pytorch_model(fallback)
        used = "nchw" if layout == "nhwc" else "nhwc"
        inp = fallback

    target = torch.randn_like(out)
    loss = nn.MSELoss()(out, target)

    opt = optim.SGD(pytorch_model.parameters(), lr=1e-3)
    opt.zero_grad()
    loss.backward()
    opt.step()

    print(f"✅ OCR SUCCESS ({used.upper()}): Loss = {loss.item():.4f}")
    return pytorch_model

# ----------------------------
# Main
# ----------------------------


if __name__ == "__main__":
    torch.set_grad_enabled(True)
    det_model = test_gradient_descent_detector()
    ocr_model = test_gradient_descent_ocr()
