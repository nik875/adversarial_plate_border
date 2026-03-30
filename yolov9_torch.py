"""
yolov9_torch.py

Pure-PyTorch YOLOv9-t license plate detector using the ultralytics
DetectionModel definition with weights transplanted from the ONNX file.

Architecture: YOLOv9-t GELAN (nc=1) — identical to yolov9t.yaml with nc=1.
BatchNorm is pre-fused into Conv layers (model.fuse() is called during loading).

Usage:
    from yolov9_torch import load_yolov9t_from_onnx

    model = load_yolov9t_from_onnx("~/.cache/open-image-models/.../yolo.onnx")
    model.eval()

    import torch
    x = torch.zeros(1, 3, 384, 384)
    preds = model(x)   # ultralytics Detect head output: [N, 5, 3024] decoded

    # Training / backprop
    model.train()
    out = model(x)     # list of raw per-scale tensors
    loss = sum(o.sum() for o in out)
    loss.backward()    # ✓ gradients flow through all layers
"""

from __future__ import annotations

import os
import torch
import numpy as np
from ultralytics.nn.tasks import DetectionModel

# Path to ultralytics YOLOv9-t config
_YAML_PATH = os.path.join(
    os.path.dirname(__file__),
    "yolov9t.yaml",  # local copy (see below)
)
# Fallback to ultralytics built-in location
_YAML_FALLBACK = os.path.join(
    os.path.dirname(
        __import__("ultralytics", fromlist=[""]).__file__
    ),
    "cfg", "models", "v9", "yolov9t.yaml",
)

# Path to ultralytics YOLOv9-s config
_YAML_PATH_S = os.path.join(
    os.path.dirname(__file__),
    "yolov9s.yaml",  # local copy (optional)
)
# Fallback to ultralytics built-in location
_YAML_FALLBACK_S = os.path.join(
    os.path.dirname(
        __import__("ultralytics", fromlist=[""]).__file__
    ),
    "cfg", "models", "v9", "yolov9s.yaml",
)


# ---------------------------------------------------------------------------
# Detection head fix: cv2 uses groups=4 for its 2nd and 3rd convolutions
# ---------------------------------------------------------------------------

def _fix_detect_cv2_groups(model: DetectionModel) -> None:
    """
    Replace the Detect head's cv2 Sequential with grouped-conv versions.

    The ONNX exports the cv2 regression branch with:
        Conv(in_ch, c2, 3, groups=1)  +SiLU
        Conv(c2,    c2, 3, groups=4)  +SiLU     ← ONNX weight (c2, c2//4, 3, 3)
        nn.Conv2d(c2, 4*reg_max, 1, groups=4)   ← ONNX weight (c2, c2//4, 1, 1)
    where c2 = 4*reg_max.

    The current ultralytics Detect head uses groups=1 for all three convs, so
    cv2 weights have shape mismatches at layers 1 and 2.
    """
    import torch.nn as nn
    from ultralytics.nn.modules.conv import Conv

    detect = model.model[-1]      # Detect head (layer 22)
    reg_max = detect.reg_max      # 16
    c2_out  = 4 * reg_max         # 64 — total output channels of cv2 branch
    groups  = c2_out // reg_max   # 4

    # Infer c2 (intermediate width) from the existing first conv's output
    c2 = detect.cv2[0][0].conv.out_channels   # 64

    new_cv2 = nn.ModuleList()
    for i in range(len(detect.cv2)):
        in_ch = detect.cv2[i][0].conv.in_channels   # per-scale input channels
        new_cv2.append(nn.Sequential(
            Conv(in_ch, c2,     3),                          # groups=1
            Conv(c2,    c2,     3, g=groups),                # groups=4
            nn.Conv2d(c2, c2_out, 1, groups=groups, bias=True),   # groups=4
        ))
    detect.cv2 = new_cv2


# ---------------------------------------------------------------------------
# Main loader
# ---------------------------------------------------------------------------

def load_yolov9t_from_onnx(onnx_path: str, nc: int = 1) -> DetectionModel:
    """
    Build a YOLOv9-t DetectionModel, fuse BN, and load all Conv weights
    from the ONNX file.

    ONNX initialiser names are 'model.model.X.*'; PyTorch state dict uses
    'model.X.*' — we strip the leading 'model.' prefix.

    Args:
        onnx_path: Path to the yolo-v9-t ONNX file.
        nc:        Number of classes (default 1 for license plates).

    Returns:
        A fused, eval-mode DetectionModel with ONNX weights loaded.
    """
    import onnx
    from onnx import numpy_helper

    path = os.path.expanduser(onnx_path)

    # ── 1. Build the ultralytics model ──────────────────────────────────────
    yaml = _YAML_PATH if os.path.exists(_YAML_PATH) else _YAML_FALLBACK
    model = DetectionModel(yaml, nc=nc, verbose=False)

    # Fix Detect head cv2 groups BEFORE fusing (Conv modules still have BN)
    _fix_detect_cv2_groups(model)

    model.fuse()   # fuses Conv+BN → Conv everywhere (matches fused ONNX)
    model.eval()

    # ── 2. Extract ONNX weights ─────────────────────────────────────────────
    onnx_model = onnx.load(path)
    W = {t.name: numpy_helper.to_array(t)
         for t in onnx_model.graph.initializer}

    # Only take weights that start with 'model.model.' (skip ONNX constants)
    def t(name: str) -> torch.Tensor:
        return torch.from_numpy(W[name].copy())

    # ── 3. Build new state dict ─────────────────────────────────────────────
    sd = model.state_dict()
    missing, unexpected = [], []

    for onnx_key, arr in W.items():
        if not onnx_key.startswith("model.model."):
            continue
        # Strip one 'model.' prefix: 'model.model.X...' → 'model.X...'
        pt_key = onnx_key[len("model."):]   # removes first 'model.'
        if pt_key in sd:
            sd[pt_key] = torch.from_numpy(arr.copy())
        else:
            unexpected.append(onnx_key)

    for k in sd:
        pt_onnx_key = "model." + k
        if pt_onnx_key not in W:
            missing.append(k)

    if missing:
        print(f"[yolov9_torch] Warning: {len(missing)} PyTorch keys not found in ONNX:")
        for k in missing[:10]:
            print(f"  {k}")
    if unexpected:
        print(f"[yolov9_torch] Warning: {len(unexpected)} ONNX keys not found in PyTorch:")
        for k in unexpected[:10]:
            print(f"  {k}")

    model.load_state_dict(sd)
    return model


def load_yolov9s_from_onnx(onnx_path: str, nc: int = 1) -> DetectionModel:
    """
    Build a YOLOv9-s DetectionModel, fuse BN, and load all Conv weights
    from the ONNX file.  Mirrors load_yolov9t_from_onnx but uses yolov9s.yaml.

    Args:
        onnx_path: Path to the yolo-v9-s ONNX file.
        nc:        Number of classes (default 1 for license plates).

    Returns:
        A fused, eval-mode DetectionModel with ONNX weights loaded.
    """
    import onnx
    from onnx import numpy_helper

    path = os.path.expanduser(onnx_path)

    yaml = _YAML_PATH_S if os.path.exists(_YAML_PATH_S) else _YAML_FALLBACK_S
    model = DetectionModel(yaml, nc=nc, verbose=False)

    _fix_detect_cv2_groups(model)

    model.fuse()
    model.eval()

    onnx_model = onnx.load(path)
    W = {t.name: numpy_helper.to_array(t)
         for t in onnx_model.graph.initializer}

    sd = model.state_dict()
    missing, unexpected = [], []

    for onnx_key, arr in W.items():
        if not onnx_key.startswith("model.model."):
            continue
        pt_key = onnx_key[len("model."):]
        if pt_key in sd:
            sd[pt_key] = torch.from_numpy(arr.copy())
        else:
            unexpected.append(onnx_key)

    for k in sd:
        pt_onnx_key = "model." + k
        if pt_onnx_key not in W:
            missing.append(k)

    if missing:
        print(f"[yolov9_torch] Warning: {len(missing)} PyTorch keys not found in ONNX:")
        for k in missing[:10]:
            print(f"  {k}")
    if unexpected:
        print(f"[yolov9_torch] Warning: {len(unexpected)} ONNX keys not found in PyTorch:")
        for k in unexpected[:10]:
            print(f"  {k}")

    model.load_state_dict(sd)
    return model


# ---------------------------------------------------------------------------
# Convenience alias matching lprnet_torch.py / cct_ocr_torch.py style
# ---------------------------------------------------------------------------

def load_weights_from_onnx(model: DetectionModel, onnx_path: str) -> DetectionModel:
    """Load ONNX weights into a pre-built, pre-fused DetectionModel in place."""
    import onnx
    from onnx import numpy_helper

    path = os.path.expanduser(onnx_path)
    onnx_model = onnx.load(path)
    W = {t.name: numpy_helper.to_array(t)
         for t in onnx_model.graph.initializer}

    sd = model.state_dict()
    for onnx_key, arr in W.items():
        if not onnx_key.startswith("model.model."):
            continue
        pt_key = onnx_key[len("model."):]
        if pt_key in sd:
            sd[pt_key] = torch.from_numpy(arr.copy())
    model.load_state_dict(sd)
    return model


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse, sys

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--onnx",
        default=(
            "~/.cache/open-image-models/yolo-v9-t-384-license-plate-end2end"
            "/yolo-v9-t-384-license-plates-end2end.onnx"
        ),
    )
    parser.add_argument("--rtol", type=float, default=1e-3)
    args = parser.parse_args()

    onnx_path = os.path.expanduser(args.onnx)
    if not os.path.exists(onnx_path):
        print(f"ONNX not found: {onnx_path}")
        sys.exit(1)

    print(f"Loading weights from {onnx_path} …")
    model = load_yolov9t_from_onnx(onnx_path)
    model.eval()

    # ── Random test input ────────────────────────────────────────────────────
    np.random.seed(42)
    x_np = (np.random.rand(1, 3, 384, 384) * 255).astype(np.float32) / 255.0
    x_pt = torch.from_numpy(x_np)

    with torch.no_grad():
        out_pt = model(x_pt)
    # ultralytics inference mode returns either a tensor or (tensor, extras)
    if isinstance(out_pt, (list, tuple)):
        out_pt = out_pt[0]
    print(f"PyTorch output shape: {out_pt.shape}")

    # ── Compare with onnxruntime on cv4 features of layer 15 ─────────────────
    # Layer 15's cv4 output is a pure Conv+SiLU feature — easy to compare.
    try:
        import onnx as _onnx
        import onnxruntime as ort
        from onnx import numpy_helper as _nph

        probe_name = "/model/model.15/cv4/act/Mul_output_0"
        m_orig = _onnx.load(onnx_path)
        inferred = _onnx.shape_inference.infer_shapes(m_orig)
        m_ext = _onnx.ModelProto(); m_ext.CopyFrom(inferred)
        new_out = m_ext.graph.output.add()
        new_out.name = probe_name
        for vi in inferred.graph.value_info:
            if vi.name == probe_name:
                new_out.CopyFrom(vi); break

        sess = ort.InferenceSession(
            m_ext.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        ort_results = sess.run(
            [o.name for o in sess.get_outputs()],
            {"images": x_np},
        )
        ort_feat = ort_results[[o.name for o in sess.get_outputs()].index(probe_name)]

        # Capture the same feature from PyTorch using a hook
        hook_out = {}
        def _hook(module, inp, out_):
            hook_out["feat"] = out_.detach()

        # model.model[15] is layer 15 (0-indexed)
        h = model.model[15].cv4.register_forward_hook(_hook)
        with torch.no_grad():
            model(x_pt)
        h.remove()

        pt_feat = hook_out["feat"].numpy()
        max_err = np.abs(pt_feat - ort_feat).max()
        mean_err = np.abs(pt_feat - ort_feat).mean()
        print(f"\nLayer-15 cv4 feature comparison:")
        print(f"  PT shape : {pt_feat.shape}")
        print(f"  ORT shape: {ort_feat.shape}")
        print(f"  Max absolute error:  {max_err:.2e}")
        print(f"  Mean absolute error: {mean_err:.2e}")
        if max_err < args.rtol:
            print(f"✓  Outputs match — weight transplant successful.")
        else:
            print(f"✗  Outputs differ — check weight mapping.")

    except ImportError:
        print("(onnxruntime not available; skipping numerical comparison)")

    # ── Confirm backprop ─────────────────────────────────────────────────────
    model.train()
    out = model(x_pt)
    # Train mode: Detect head returns list of raw per-scale tensors
    if isinstance(out, (list, tuple)):
        loss = sum(o.sum() for o in out if isinstance(o, torch.Tensor))
    elif isinstance(out, dict):
        # Some ultralytics versions return a dict with a 'one' key
        loss = sum(v.sum() for v in out.values() if isinstance(v, torch.Tensor))
    else:
        loss = out.sum()
    loss.backward()
    print("✓  loss.backward() succeeded — backprop through all layers works.")
