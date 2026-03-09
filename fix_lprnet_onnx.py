"""
Patches the LPRNet ONNX model so it can be loaded by onnx2torch:
  1. Replaces SAME_UPPER auto_pad in MaxPool with explicit pads.
  2. Replaces the graph outputs (ArgMax / ReduceMax) with the raw Softmax
     logits so gradients can flow through for CTC training.

Output: weights/lprnet_deployable_onnx_v1.1/us_lprnet_patched.onnx

Run on server:
    python fix_lprnet_onnx.py
"""

import math
import onnx
import onnx2torch
import torch
import numpy as np

SRC  = "weights/lprnet_deployable_onnx_v1.1/us_lprnet_baseline18_deployable.onnx"
DST  = "weights/lprnet_deployable_onnx_v1.1/us_lprnet_patched.onnx"

model = onnx.load(SRC)
graph = model.graph

# ── helpers ────────────────────────────────────────────────────────────────────

def get_attr(node, name):
    for a in node.attribute:
        if a.name == name:
            return a
    return None

def del_attr(node, name):
    for a in node.attribute:
        if a.name == name:
            node.attribute.remove(a)
            return

# ── 1. Fix SAME_UPPER in MaxPool ───────────────────────────────────────────────
# For SAME_UPPER padding:
#   out = ceil(in / stride)
#   pad_total = max(0, (out-1)*stride + kernel - in)
#   SAME_UPPER: extra pad goes at the start → pads = [extra, remainder] per dim

print("=== Fixing MaxPool SAME_UPPER padding ===")
for node in graph.node:
    if node.op_type != "MaxPool":
        continue
    auto_pad = get_attr(node, "auto_pad")
    if auto_pad is None or auto_pad.s not in (b"SAME_UPPER", b"SAME_LOWER"):
        continue

    is_upper = (auto_pad.s == b"SAME_UPPER")
    kernel   = list(get_attr(node, "kernel_shape").ints)
    strides  = list(get_attr(node, "strides").ints)
    ndim     = len(kernel)

    # LPRNet input to MaxPool is [B, C, 48, 96] (spatial)
    # We don't know the exact spatial dims here, so we compute symbolically.
    # For fixed input [48, 96] we can compute directly.
    in_spatial = [48, 96]   # H, W of the model input — adjust if needed

    pads = []
    for i in range(ndim):
        out_dim   = math.ceil(in_spatial[i] / strides[i])
        pad_total = max(0, (out_dim - 1) * strides[i] + kernel[i] - in_spatial[i])
        if is_upper:
            pad_start = (pad_total + 1) // 2   # extra goes to start
            pad_end   = pad_total // 2
        else:
            pad_start = pad_total // 2
            pad_end   = (pad_total + 1) // 2
        pads.append((pad_start, pad_end))

    flat_pads = [p[0] for p in pads] + [p[1] for p in pads]  # [top, left, bottom, right]

    print(f"  kernel={kernel} strides={strides}  auto_pad={auto_pad.s.decode()}"
          f"  → explicit pads={flat_pads}")

    del_attr(node, "auto_pad")
    pads_attr = onnx.helper.make_attribute("pads", flat_pads)
    node.attribute.append(pads_attr)

# ── 2. Find Softmax output tensor name ────────────────────────────────────────
softmax_output = None
for node in graph.node:
    if node.op_type == "Softmax":
        softmax_output = node.output[0]
        print(f"\n=== Softmax output tensor: '{softmax_output}' ===")
        break

if softmax_output is None:
    raise RuntimeError("No Softmax node found in graph.")

# ── 3. Replace graph outputs with Softmax tensor ──────────────────────────────
print("\n=== Original graph outputs ===")
for o in graph.output:
    shape = [d.dim_value for d in o.type.tensor_type.shape.dim]
    print(f"  {o.name}: {shape}")

# Infer the shape of the Softmax output by running shape inference
model_inferred = onnx.shape_inference.infer_shapes(model)
softmax_info = None
for vi in model_inferred.graph.value_info:
    if vi.name == softmax_output:
        softmax_info = vi
        break

# Build a new output ValueInfo for the Softmax tensor
if softmax_info is not None:
    new_out = onnx.helper.make_value_info(
        softmax_output,
        softmax_info.type.tensor_type.elem_type,
        [d.dim_value for d in softmax_info.type.tensor_type.shape.dim],
    )
else:
    # Fall back: float32, unknown shape
    new_out = onnx.helper.make_value_info(softmax_output, onnx.TensorProto.FLOAT, None)

while graph.output:
    graph.output.pop()
graph.output.append(new_out)

print("\n=== New graph outputs ===")
for o in graph.output:
    shape = [d.dim_value for d in o.type.tensor_type.shape.dim]
    print(f"  {o.name}: {shape}")

# ── 4. Save and verify ─────────────────────────────────────────────────────────
onnx.checker.check_model(model)
onnx.save(model, DST)
print(f"\nSaved patched model → {DST}")

# ── 5. Try onnx2torch conversion ───────────────────────────────────────────────
print("\n=== onnx2torch conversion ===")
torch_model = onnx2torch.convert(DST)
torch_model.eval()
print("Conversion succeeded.")

# ── 6. Test forward pass ───────────────────────────────────────────────────────
dummy = torch.rand(1, 3, 48, 96)
with torch.no_grad():
    out = torch_model(dummy)

if isinstance(out, (list, tuple)):
    for i, o in enumerate(out):
        print(f"Output[{i}]: shape={list(o.shape)}, min={o.min():.4f}, max={o.max():.4f}")
else:
    print(f"Output shape: {list(out.shape)}, min={out.min():.4f}, max={out.max():.4f}")
    print("(Softmax output: values should sum to ~1 along class dim)")
    print(f"  Sum along class dim (should be ~1): {out.sum(-1)[0, :3]}")

# ── 7. Print number of output classes ─────────────────────────────────────────
if not isinstance(out, (list, tuple)):
    print(f"\nOutput dimensions: {list(out.shape)}")
    print("  Likely interpretation: [batch, time_steps, num_classes]")
    print(f"  → num_classes = {out.shape[-1]}")
    print(f"  → time_steps  = {out.shape[-2] if out.dim() == 3 else '?'}")
