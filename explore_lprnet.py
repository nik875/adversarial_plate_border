"""
Quick script to explore the LPRNet ONNX model and test loading via onnx2torch.
Run on the Ubuntu server:
    python explore_lprnet.py
"""

import onnx
import onnx2torch
import torch

ONNX_PATH = "weights/lprnet_deployable_onnx_v1.1/us_lprnet_baseline18_deployable.onnx"

# ── 1. ONNX graph inspection ─────────────────────────────────────────────────
model_onnx = onnx.load(ONNX_PATH)

print("=== INPUTS ===")
for inp in model_onnx.graph.input:
    shape = [d.dim_value for d in inp.type.tensor_type.shape.dim]
    print(f"  {inp.name}: {shape}")

print("\n=== OUTPUTS ===")
for out in model_onnx.graph.output:
    shape = [d.dim_value for d in out.type.tensor_type.shape.dim]
    print(f"  {out.name}: {shape}")

print(f"\n=== GRAPH NODES ({len(model_onnx.graph.node)}) ===")
op_counts = {}
for node in model_onnx.graph.node:
    op_counts[node.op_type] = op_counts.get(node.op_type, 0) + 1
for op, count in sorted(op_counts.items()):
    print(f"  {op}: {count}")

# ── 2. Load via onnx2torch ────────────────────────────────────────────────────
print("\n=== ONNX2TORCH CONVERSION ===")
model = onnx2torch.convert(ONNX_PATH)
model.eval()
print("Converted successfully.")

total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,}")

# ── 3. Test forward pass ──────────────────────────────────────────────────────
# Infer input shape from ONNX graph
inp_shape = [d.dim_value for d in model_onnx.graph.input[0].type.tensor_type.shape.dim]
print(f"\n=== FORWARD PASS (input shape {inp_shape}) ===")

# Replace batch dim=0 with 1 if dynamic
test_shape = [1 if (d == 0 or d is None) else d for d in inp_shape]
dummy = torch.zeros(*test_shape)
print(f"Dummy input: {list(dummy.shape)}, range [{dummy.min():.2f}, {dummy.max():.2f}]")

with torch.no_grad():
    out = model(dummy)

if isinstance(out, (list, tuple)):
    for i, o in enumerate(out):
        print(f"Output[{i}]: shape={list(o.shape)}, dtype={o.dtype}")
else:
    print(f"Output shape: {list(out.shape)}, dtype={out.dtype}")

# ── 4. Try a realistic input ──────────────────────────────────────────────────
# LPRNet typically expects [B, C, H, W] in [0, 1] or [0, 255]
# Try [0, 1] normalized first
dummy_real = torch.rand(*test_shape)
with torch.no_grad():
    out_real = model(dummy_real)

if isinstance(out_real, (list, tuple)):
    out_real = out_real[0]
print(f"\nWith rand input: output min={out_real.min():.4f}, max={out_real.max():.4f}, mean={out_real.mean():.4f}")
print("(If output looks like logits/log-probs, values should span a wide range)")
