"""
lprnet_torch.py

Pure-PyTorch reimplementation of the NVIDIA TAO US LPRNet, with a weight
loader that transplants all parameters from the ONNX file.

Architecture (traced from us_lprnet_patched.onnx):
  Input [N, 3, 48, 96]
  → ReduceSum(axis=1, keepdims) → [N, 1, 48, 96]
  → conv1 (64, 3×3, s1, p1) + ReLU + MaxPool(3×3, s1, p1) → [N, 64, 48, 96]
  → res2  (64→64,   s1) × 2 blocks                         → [N,  64, 48, 96]
  → res3  (64→128,  s2) × 2 blocks                         → [N, 128, 24, 48]
  → res4  (128→256, s2) × 2 blocks                         → [N, 256, 12, 24]
  → res5  (256→300, s1) × 2 blocks                         → [N, 300, 12, 24]
  → permute(0,3,2,1) → reshape(N,24,3600) → seq-first     → [24, N, 3600]
  → LSTM(3600→512, forward)                                → [24, N, 512]
  → permute(1,0,2) → Linear(512→36) → Softmax              → [N, 24, 36]

Usage:
    from lprnet_torch import LPRNetTorch, load_weights_from_onnx

    model = LPRNetTorch()
    load_weights_from_onnx(model, "us_lprnet_patched.onnx")
    model.eval()

    # inference
    import torch
    x = torch.zeros(1, 3, 48, 96)
    logits = model(x)   # [1, 24, 36]

    # training / backprop
    model.train()
    loss = some_loss(model(x))
    loss.backward()   # ✓ gradients flow through LSTM and all conv layers
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class LPRNetTorch(nn.Module):
    """
    Faithful PyTorch reconstruction of us_lprnet_patched.onnx.

    All Conv2d + LSTM + Linear modules have the same shapes, strides, and
    padding behaviour as the original ONNX graph.  ReLU6 clips are replaced
    with plain ReLU because all clip_min values in the ONNX file are 0.0 with
    no upper-bound (i.e. standard ReLU).
    """

    def __init__(self):
        super().__init__()

        # ── stem ────────────────────────────────────────────────────────────
        self.conv1  = nn.Conv2d(1, 64, 3, stride=1, padding=1, bias=True)
        self.pool1  = nn.MaxPool2d(3, stride=1, padding=1)

        # ── res2 : 64 → 64, stride 1 ────────────────────────────────────────
        # block a  (shortcut has a 1×1 conv even though channels don't change)
        self.res2a_branch1  = nn.Conv2d(64, 64, 1, stride=1, bias=True)
        self.res2a_branch2a = nn.Conv2d(64, 64, 3, stride=1, padding=1, bias=True)
        self.res2a_branch2b = nn.Conv2d(64, 64, 3, stride=1, padding=1, bias=True)
        # block b  (no shortcut conv — identity skip)
        self.res2b_branch2a = nn.Conv2d(64, 64, 3, stride=1, padding=1, bias=True)
        self.res2b_branch2b = nn.Conv2d(64, 64, 3, stride=1, padding=1, bias=True)

        # ── res3 : 64 → 128, stride 2 ───────────────────────────────────────
        # branch2a uses asymmetric padding (0-top/left, 1-bottom/right); handled in forward
        self.res3a_branch1  = nn.Conv2d( 64, 128, 1, stride=2, bias=True)
        self.res3a_branch2a = nn.Conv2d( 64, 128, 3, stride=2, padding=0, bias=True)
        self.res3a_branch2b = nn.Conv2d(128, 128, 3, stride=1, padding=1, bias=True)
        self.res3b_branch2a = nn.Conv2d(128, 128, 3, stride=1, padding=1, bias=True)
        self.res3b_branch2b = nn.Conv2d(128, 128, 3, stride=1, padding=1, bias=True)

        # ── res4 : 128 → 256, stride 2 ──────────────────────────────────────
        self.res4a_branch1  = nn.Conv2d(128, 256, 1, stride=2, bias=True)
        self.res4a_branch2a = nn.Conv2d(128, 256, 3, stride=2, padding=0, bias=True)
        self.res4a_branch2b = nn.Conv2d(256, 256, 3, stride=1, padding=1, bias=True)
        self.res4b_branch2a = nn.Conv2d(256, 256, 3, stride=1, padding=1, bias=True)
        self.res4b_branch2b = nn.Conv2d(256, 256, 3, stride=1, padding=1, bias=True)

        # ── res5 : 256 → 300, stride 1 ──────────────────────────────────────
        self.res5a_branch1  = nn.Conv2d(256, 300, 1, stride=1, bias=True)
        self.res5a_branch2a = nn.Conv2d(256, 300, 3, stride=1, padding=1, bias=True)
        self.res5a_branch2b = nn.Conv2d(300, 300, 3, stride=1, padding=1, bias=True)
        self.res5b_branch2a = nn.Conv2d(300, 300, 3, stride=1, padding=1, bias=True)
        self.res5b_branch2b = nn.Conv2d(300, 300, 3, stride=1, padding=1, bias=True)

        # ── sequence head ────────────────────────────────────────────────────
        self.lstm  = nn.LSTM(input_size=3600, hidden_size=512,
                             num_layers=1, batch_first=False, bidirectional=False)
        self.dense = nn.Linear(512, 36)

    # -------------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [N, 3, 48, 96]  float32 in [0, 1]
        Returns:
            [N, 24, 36]  softmax probabilities over 36 classes per time-step
        """
        # ReduceSum over channel dim (mirrors the ONNX ReduceSum opset 14 node)
        x = x.sum(dim=1, keepdim=True)          # [N, 1, 48, 96]

        # stem
        x = F.relu(self.conv1(x))               # [N, 64, 48, 96]
        x = self.pool1(x)                        # [N, 64, 48, 96]

        # res2a
        sc = self.res2a_branch1(x)
        y  = F.relu(self.res2a_branch2a(x))
        y  = self.res2a_branch2b(y)
        x  = F.relu(sc + y)                     # [N, 64, 48, 96]

        # res2b  (identity skip — no branch1 conv)
        y  = F.relu(self.res2b_branch2a(x))
        y  = self.res2b_branch2b(y)
        x  = F.relu(x + y)                      # [N, 64, 48, 96]

        # res3a  (stride=2, asymmetric pad on branch2a)
        sc = self.res3a_branch1(x)               # 1×1 stride-2 → [N,128,24,48]
        y  = F.pad(x, (0, 1, 0, 1))             # right+bottom pad
        y  = F.relu(self.res3a_branch2a(y))
        y  = self.res3a_branch2b(y)
        x  = F.relu(sc + y)                     # [N,128,24,48]

        # res3b
        y  = F.relu(self.res3b_branch2a(x))
        y  = self.res3b_branch2b(y)
        x  = F.relu(x + y)                      # [N,128,24,48]

        # res4a  (stride=2, asymmetric pad on branch2a)
        sc = self.res4a_branch1(x)               # → [N,256,12,24]
        y  = F.pad(x, (0, 1, 0, 1))
        y  = F.relu(self.res4a_branch2a(y))
        y  = self.res4a_branch2b(y)
        x  = F.relu(sc + y)                     # [N,256,12,24]

        # res4b
        y  = F.relu(self.res4b_branch2a(x))
        y  = self.res4b_branch2b(y)
        x  = F.relu(x + y)                      # [N,256,12,24]

        # res5a  (stride=1, channel change 256→300)
        sc = self.res5a_branch1(x)               # 1×1 → [N,300,12,24]
        y  = F.relu(self.res5a_branch2a(x))
        y  = self.res5a_branch2b(y)
        x  = F.relu(sc + y)                     # [N,300,12,24]

        # res5b
        y  = F.relu(self.res5b_branch2a(x))
        y  = self.res5b_branch2b(y)
        x  = F.relu(x + y)                      # [N,300,12,24]

        # ── feature → sequence ──────────────────────────────────────────────
        # Matches ONNX: Transpose(0,3,2,1) → Reshape(N,24,3600) → Transpose(1,0,2)
        N = x.shape[0]
        x = x.permute(0, 3, 2, 1)              # [N, 24, 12, 300]
        x = x.reshape(N, 24, 3600)             # [N, 24, 3600]
        x = x.permute(1, 0, 2)                 # [24, N, 3600]  (seq-first)

        x, _ = self.lstm(x)                    # [24, N, 512]

        x = x.permute(1, 0, 2)                 # [N, 24, 512]
        x = self.dense(x)                      # [N, 24, 36]
        x = F.softmax(x, dim=-1)               # [N, 24, 36]
        return x


# ---------------------------------------------------------------------------
# Weight loader
# ---------------------------------------------------------------------------

# ONNX LSTM gate order:  i=0, o=1, f=2, c=3
# PyTorch LSTM gate order: i=0, f=1, g=2, o=3  (g = cell / tanh gate)
# Reorder: pick ONNX gates [i, f, c, o] = indices [0, 2, 3, 1]
_ONNX_TO_PT_GATE_ORDER = [0, 2, 3, 1]


def _reorder_lstm_weights(arr: np.ndarray, hidden: int) -> np.ndarray:
    """Reorder ONNX LSTM weight block from [i,o,f,c] to PyTorch [i,f,g,o]."""
    # arr shape: [4*hidden, *]
    arr = arr.reshape(4, hidden, *arr.shape[1:])
    arr = arr[_ONNX_TO_PT_GATE_ORDER]
    return arr.reshape(4 * hidden, *arr.shape[2:])


def load_weights_from_onnx(model: LPRNetTorch, onnx_path: str) -> LPRNetTorch:
    """
    Load all weights from the ONNX file into a LPRNetTorch instance.

    Handles the ONNX→PyTorch LSTM gate-order permutation automatically.
    Returns the model (mutates in-place, also returned for convenience).
    """
    import onnx
    from onnx import numpy_helper

    onnx_model = onnx.load(onnx_path)
    W = {t.name: numpy_helper.to_array(t) for t in onnx_model.graph.initializer}

    def t(name: str) -> torch.Tensor:
        return torch.from_numpy(W[name].copy())

    sd = model.state_dict()

    # ── stem ────────────────────────────────────────────────────────────────
    sd["conv1.weight"] = t("conv1_W_new")
    sd["conv1.bias"]   = t("conv1_B_new")

    # ── ResNet conv layers (all follow the same naming convention) ──────────
    conv_names = [
        "res2a_branch1",  "res2a_branch2a", "res2a_branch2b",
        "res2b_branch2a", "res2b_branch2b",
        "res3a_branch1",  "res3a_branch2a", "res3a_branch2b",
        "res3b_branch2a", "res3b_branch2b",
        "res4a_branch1",  "res4a_branch2a", "res4a_branch2b",
        "res4b_branch2a", "res4b_branch2b",
        "res5a_branch1",  "res5a_branch2a", "res5a_branch2b",
        "res5b_branch2a", "res5b_branch2b",
    ]
    for name in conv_names:
        sd[f"{name}.weight"] = t(f"{name}_W_new")
        sd[f"{name}.bias"]   = t(f"{name}_B_new")

    # ── LSTM (with gate reordering) ─────────────────────────────────────────
    H = 512

    lstm_W = W["lstm_W"][0]                     # [2048, 3600]
    sd["lstm.weight_ih_l0"] = torch.from_numpy(
        _reorder_lstm_weights(lstm_W, H).copy())

    lstm_R = W["lstm_R"][0]                     # [2048, 512]
    sd["lstm.weight_hh_l0"] = torch.from_numpy(
        _reorder_lstm_weights(lstm_R, H).copy())

    lstm_B = W["lstm_B"][0]                     # [4096] = [8*H]
    Wb = _reorder_lstm_weights(lstm_B[:4*H].reshape(4*H, 1), H).reshape(4*H)
    Rb = _reorder_lstm_weights(lstm_B[4*H:].reshape(4*H, 1), H).reshape(4*H)
    sd["lstm.bias_ih_l0"] = torch.from_numpy(Wb.copy())
    sd["lstm.bias_hh_l0"] = torch.from_numpy(Rb.copy())

    # ── dense head ───────────────────────────────────────────────────────────
    # ONNX uses MatMul with kernel [512, 36]; PyTorch Linear.weight is [36, 512]
    sd["dense.weight"] = t("td_dense/kernel:0").T.contiguous()
    sd["dense.bias"]   = t("td_dense/bias:0")

    model.load_state_dict(sd)
    return model


# ---------------------------------------------------------------------------
# Quick smoke-test / numerical comparison against onnxruntime
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys, argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", default="us_lprnet_patched.onnx")
    parser.add_argument("--rtol", type=float, default=1e-4)
    args = parser.parse_args()

    print(f"Loading weights from {args.onnx} …")
    model = LPRNetTorch()
    load_weights_from_onnx(model, args.onnx)
    model.eval()

    x_np = np.random.rand(2, 3, 48, 96).astype(np.float32)
    x_pt = torch.from_numpy(x_np)

    with torch.no_grad():
        out_pt = model(x_pt).numpy()

    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(args.onnx,
                                    providers=["CPUExecutionProvider"])
        in_name  = sess.get_inputs()[0].name
        out_name = sess.get_outputs()[0].name
        out_ort  = sess.run([out_name], {in_name: x_np})[0]

        # ONNX output is [seq, batch, classes]; PyTorch is [batch, seq, classes]
        if out_ort.shape != out_pt.shape:
            out_ort = out_ort.transpose(1, 0, 2)

        max_err = np.abs(out_pt - out_ort).max()
        print(f"Max absolute error vs onnxruntime: {max_err:.2e}")
        if max_err < args.rtol:
            print("✓  Outputs match — weight transplant successful.")
        else:
            print("✗  Outputs differ — check gate ordering or reshape logic.")
    except ImportError:
        print("(onnxruntime not available; skipping numerical comparison)")
        print(f"PyTorch output shape: {out_pt.shape}, sum={out_pt.sum():.4f}")

    # confirm backprop works
    model.train()
    y = model(x_pt)
    loss = y.sum()
    loss.backward()
    print("✓  loss.backward() succeeded — backprop through LSTM works.")
