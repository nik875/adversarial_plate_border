"""
cct_ocr_torch.py

Pure-PyTorch reimplementation of the fast-plate-ocr CCT (Compact Convolutional
Transformer) OCR models, with weight loaders that transplant all parameters
from the ONNX files.

Supports cct-xs-v1-global (XS) and cct-s-v1-global (S) automatically; the
`load_weights_from_onnx` function detects the variant from the ONNX file.

XS architecture (input [N, 64, 128, 3] uint8 NHWC):
  ConvStem: 5×Conv(no-pad)+ReLU, 2×MaxBlurPool(stride-2) → [N, 96, 10, 26]
  embed_dim=64, 4 transformer blocks, 1-head self-attention
  TokenReducer: 9 learned queries, 4-head cross-attention

S architecture (same input):
  ConvStem: 4×Conv(no-pad)+GELU, 1×MaxBlurPool(stride-2) → [N, 128, 32, 64]
  embed_dim=128, 6 transformer blocks, 2-head self-attention
  TokenReducer: 9 learned queries, 4-head cross-attention

Both output: [N, 9, 37] softmax character probabilities.

Usage:
    from cct_ocr_torch import CCTOCRTorch, load_weights_from_onnx

    model = CCTOCRTorch.from_onnx("~/.cache/fast-plate-ocr/cct-xs-v1-global-model/cct_xs_v1_global.onnx")
    model.eval()

    import torch, numpy as np
    x = torch.randint(0, 256, (2, 64, 128, 3), dtype=torch.uint8)
    logits = model(x)   # [2, 9, 37]

    model.train()
    logits.sum().backward()   # ✓ gradients flow through all layers
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _gelu(x: torch.Tensor) -> torch.Tensor:
    """Standard GELU: x * 0.5 * (1 + erf(x / sqrt(2)))."""
    return x * 0.5 * (1.0 + torch.erf(x * (1.0 / math.sqrt(2.0))))


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class MaxBlurPool2d(nn.Module):
    """
    MaxPool(2×2, stride=1) + fixed depthwise Gaussian blur (stride=2).
    The blur kernel is loaded from the ONNX initialiser.
    """

    def __init__(self, channels: int,
                 pool_pads: tuple = (0, 0, 1, 1),
                 blur_pads: tuple = (0, 0, 1, 1)):
        super().__init__()
        self.pool_pad = pool_pads
        self.blur_pad = blur_pads
        self.register_buffer("blur_weight", torch.zeros(channels, 1, 3, 3))

    def _pad(self, x: torch.Tensor, pads: tuple, value: float = 0.0) -> torch.Tensor:
        top, left, bottom, right = pads
        return F.pad(x, (left, right, top, bottom), value=value)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pad with -inf so MaxPool matches ONNX semantics (ONNX MaxPool pads = -inf)
        x = self._pad(x, self.pool_pad, value=float('-inf'))
        x = F.max_pool2d(x, kernel_size=2, stride=1)
        x = self._pad(x, self.blur_pad)
        x = F.conv2d(x, self.blur_weight, stride=2, groups=x.shape[1])
        return x


class DyTNorm(nn.Module):
    """Dynamic Tanh Normalization: norm(x) = tanh(α·x) ⊙ w + b"""

    def __init__(self, dim: int):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(()))
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.alpha * x) * self.weight + self.bias


class TransformerBlock(nn.Module):
    """
    CCT transformer block: DyTNorm → multi-head self-attention → residual
                         + DyTNorm → 2-layer MLP (each GELU) → residual

    Weights are stored in ONNX MatMul convention (x @ W, W=[in, out]).
    """

    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        d = embed_dim
        self.num_heads = num_heads
        self.head_dim = d                  # head_dim of each head is always embed_dim
        self.inner = num_heads * d         # total projected dim
        self.scale = (d ** -0.5)

        self.norm1 = DyTNorm(d)
        # Q,K,V weights: [embed_dim, inner] (in, out convention from ONNX MatMul)
        self.W_q = nn.Parameter(torch.empty(d, self.inner))
        self.b_q = nn.Parameter(torch.zeros(num_heads, d))     # [heads, head_dim]
        self.W_k = nn.Parameter(torch.empty(d, self.inner))
        self.b_k = nn.Parameter(torch.zeros(num_heads, d))
        self.W_v = nn.Parameter(torch.empty(d, self.inner))
        self.b_v = nn.Parameter(torch.zeros(num_heads, d))
        # Output projection: [inner, embed_dim] (in, out)
        self.W_o = nn.Parameter(torch.empty(self.inner, d))
        self.b_o = nn.Parameter(torch.zeros(d))

        self.norm2 = DyTNorm(d)
        self.W_m1 = nn.Parameter(torch.empty(d, d))
        self.b_m1 = nn.Parameter(torch.zeros(d))
        self.W_m2 = nn.Parameter(torch.empty(d, d))
        self.b_m2 = nn.Parameter(torch.zeros(d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N, L, d = x.shape
        h = self.num_heads

        # ── self-attention ───────────────────────────────────────────────────
        normed = self.norm1(x)
        Q = normed @ self.W_q  # [N, L, inner]
        K = normed @ self.W_k
        V = normed @ self.W_v

        # Split into heads and add per-head biases
        Q = Q.reshape(N, L, h, -1).permute(0, 2, 1, 3)  # [N, h, L, d_h]
        K = K.reshape(N, L, h, -1).permute(0, 2, 1, 3)
        V = V.reshape(N, L, h, -1).permute(0, 2, 1, 3)
        Q = Q + self.b_q.unsqueeze(1)   # b_q: [h, d_h]
        K = K + self.b_k.unsqueeze(1)
        V = V + self.b_v.unsqueeze(1)

        # Scaled dot-product attention
        score = (Q * self.scale) @ K.transpose(-1, -2)  # [N, h, L, L]
        attn = score.softmax(dim=-1)
        out = attn @ V                                   # [N, h, L, d_h]

        # Merge heads
        out = out.permute(0, 2, 1, 3).reshape(N, L, self.inner)  # [N, L, inner]
        out = out @ self.W_o + self.b_o
        x = x + out

        # ── MLP ──────────────────────────────────────────────────────────────
        h_n = self.norm2(x)
        h_n = _gelu(h_n @ self.W_m1 + self.b_m1)
        h_n = _gelu(h_n @ self.W_m2 + self.b_m2)
        x = x + h_n
        return x


class TokenReducer(nn.Module):
    """
    Cross-attention with learned queries (4 heads) to reduce seq → num_queries tokens.
    """

    def __init__(self, embed_dim: int, num_queries: int = 9, num_heads: int = 4):
        super().__init__()
        self.h = num_heads
        self.head_dim = embed_dim
        self.inner = num_heads * embed_dim
        self.scale = (embed_dim ** -0.5)

        self.queries = nn.Parameter(torch.empty(1, num_queries, embed_dim))
        # Q: Gemm weight (inner, embed_dim) — PyTorch Linear(embed_dim, inner) weight
        self.W_q = nn.Parameter(torch.empty(self.inner, embed_dim))
        self.b_q = nn.Parameter(torch.zeros(num_heads, embed_dim))
        # K, V: stored as (inner, embed_dim) for x @ W_k.T convention
        self.W_k = nn.Parameter(torch.empty(self.inner, embed_dim))
        self.b_k = nn.Parameter(torch.zeros(num_heads, embed_dim))
        self.W_v = nn.Parameter(torch.empty(self.inner, embed_dim))
        self.b_v = nn.Parameter(torch.zeros(num_heads, embed_dim))
        # Output: (embed_dim, inner) for Gemm(out, W_o, transB=1)
        self.W_o = nn.Parameter(torch.empty(embed_dim, self.inner))
        self.b_o = nn.Parameter(torch.zeros(embed_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N, L, _ = x.shape
        nq = self.queries.shape[1]
        h = self.h

        q = self.queries.expand(N, -1, -1)  # [N, nq, embed_dim]

        # Q via Gemm (transB=1): q @ W_q.T
        Q = (q.reshape(N * nq, -1) @ self.W_q.T).reshape(N, nq, self.inner)
        K = x @ self.W_k.T   # [N, L, inner]
        V = x @ self.W_v.T

        Q = Q.reshape(N, nq, h, -1).permute(0, 2, 1, 3)  # [N, h, nq, d_h]
        K = K.reshape(N, L,  h, -1).permute(0, 2, 1, 3)
        V = V.reshape(N, L,  h, -1).permute(0, 2, 1, 3)
        Q = Q + self.b_q.unsqueeze(1)
        K = K + self.b_k.unsqueeze(1)
        V = V + self.b_v.unsqueeze(1)

        scores = (Q * self.scale) @ K.transpose(-1, -2)  # [N, h, nq, L]
        attn = scores.softmax(dim=-1)
        out = (attn @ V).permute(0, 2, 1, 3).reshape(N, nq, self.inner)

        # Output projection via Gemm (transB=1): out @ W_o.T
        out = (out.reshape(N * nq, self.inner) @ self.W_o.T).reshape(N, nq, -1)
        return out + self.b_o


# ---------------------------------------------------------------------------
# Top-level model
# ---------------------------------------------------------------------------

@dataclass
class CCTConfig:
    """Architecture hyper-parameters for the CCT OCR model."""
    # ConvStem
    stem_channels: List[int] = field(default_factory=lambda: [32, 48, 64, 80, 96])
    stem_act: str = "relu"       # "relu" or "gelu"
    stem_bias: bool = True       # XS: True (conv has bias), S: False
    stem_padding: int = 0        # XS: 0 (VALID), S: 1 (SAME)
    blur_pool_after: List[int] = field(default_factory=lambda: [0, 2])  # 0-indexed conv
    blur_pool_pads: List[tuple] = field(
        default_factory=lambda: [(0,0,1,1), (1,1,1,1)])
    # Patch extractor
    patch_in_ch: int = 96
    patch_out_ch: int = 384
    # Sequence
    seq_in: int = 384            # = patch_out_ch
    embed_dim: int = 64
    posemb_size: int = 65
    # Transformer
    num_blocks: int = 4
    num_heads: int = 1
    # TokenReducer
    tr_num_queries: int = 9
    tr_num_heads: int = 4
    # Head
    num_classes: int = 37


class CCTOCRTorch(nn.Module):
    """
    Configurable pure-PyTorch CCT OCR model.
    Input:  [N, 64, 128, 3] uint8 NHWC
    Output: [N, 9, 37] softmax character probabilities
    """

    def __init__(self, cfg: CCTConfig):
        super().__init__()
        self.cfg = cfg
        act = cfg.stem_act

        # ── ConvStem ────────────────────────────────────────────────────────
        # First conv: 3 → stem_channels[0]
        in_ch = 3
        self.stem_convs = nn.ModuleList()
        self.stem_pools = nn.ModuleDict()
        self.stem_act_fn = F.relu if act == "relu" else _gelu
        self.blur_pool_after = set(cfg.blur_pool_after)

        for i, out_ch in enumerate(cfg.stem_channels):
            self.stem_convs.append(nn.Conv2d(in_ch, out_ch, 3,
                                             padding=cfg.stem_padding,
                                             bias=cfg.stem_bias))
            in_ch = out_ch
            if i in self.blur_pool_after:
                j = cfg.blur_pool_after.index(i)
                bpads = cfg.blur_pool_pads[j]
                self.stem_pools[str(i)] = MaxBlurPool2d(out_ch, blur_pads=bpads)

        # ── Patch extractor ──────────────────────────────────────────────────
        self.patch_conv = nn.Conv2d(cfg.patch_in_ch, cfg.patch_out_ch, 2, stride=2, bias=False)

        # ── Sequence projection ──────────────────────────────────────────────
        d = cfg.embed_dim
        self.seq_W = nn.Parameter(torch.empty(cfg.seq_in, d))
        self.seq_b = nn.Parameter(torch.zeros(d))

        # ── Positional embedding ─────────────────────────────────────────────
        self.register_buffer("pos_emb", torch.zeros(cfg.posemb_size, d))

        # ── Transformer blocks ───────────────────────────────────────────────
        self.blocks = nn.ModuleList([
            TransformerBlock(d, cfg.num_heads)
            for _ in range(cfg.num_blocks)
        ])

        # ── Token reducer ────────────────────────────────────────────────────
        self.token_reducer = TokenReducer(d, cfg.tr_num_queries, cfg.tr_num_heads)

        # ── Vocabulary projection ─────────────────────────────────────────────
        self.vocab_W = nn.Parameter(torch.empty(d, cfg.num_classes))
        self.vocab_b = nn.Parameter(torch.zeros(cfg.num_classes))

    # -------------------------------------------------------------------------
    @classmethod
    def from_onnx(cls, onnx_path: str) -> "CCTOCRTorch":
        """Auto-detect variant from ONNX and return a configured model."""
        import onnx as _onnx
        from onnx import numpy_helper
        import os

        path = os.path.expanduser(onnx_path)
        m = _onnx.load(path)
        W = {t.name: numpy_helper.to_array(t) for t in m.graph.initializer}

        # Detect embed_dim from seq_proj weight
        seq_w = W["CCT_OCR_1/mlp_1/dense_1/Cast/ReadVariableOp:0"]
        seq_in, embed_dim = seq_w.shape  # [seq_in, embed_dim]

        # Detect patch channels
        patch_w = W["CCT_OCR_1/patch_extractor_1/Reshape__6"]
        patch_out_ch, patch_in_ch = patch_w.shape[:2]

        # Detect positional embedding size
        posemb = W["CCT_OCR_1/pos_emb_1/Slice:0"]
        posemb_size = posemb.shape[0]

        # Count transformer blocks
        num_blocks = sum(
            1 for k in W if k.startswith("CCT_OCR_1/transformer_block_") and "dy_t_" in k
            and k.endswith("/Mul/ReadVariableOp:0")
        ) // 2  # two dy_t per block

        # Detect num_heads from transformer block 1 Q bias shape
        qb = W["CCT_OCR_1/transformer_block_1_1/multi_head_attention_1/query_1/add/ReadVariableOp:0"]
        num_heads = qb.shape[0]  # [num_heads, head_dim]

        # Detect stem (count conv weights)
        stem_keys = sorted([k for k in W if "conv_stem" in k and "convolution/ReadVariableOp" in k])
        stem_channels = [W[k].shape[0] for k in stem_keys]  # [out_ch, in_ch, H, W]

        # Detect activation (ReLU or GELU in stem)
        has_gelu_in_stem = any("Gelu" in k for k in W if "conv_stem" in k)
        stem_act = "gelu" if has_gelu_in_stem else "relu"

        # Detect blur pool positions from ONNX node names
        blur_after = []
        blur_pads_list = []
        blur_pool_idx = 0
        for node in m.graph.node:
            if node.op_type == "Conv" and "depthwise" in str(node.output):
                # Find which conv this follows
                attrs = {a.name: list(a.ints) for a in node.attribute if a.ints}
                pads_onnx = attrs.get("pads", [0, 0, 1, 1])
                # The blur pool is after stem_conv[blur_pool_idx]
                blur_after.append(blur_pool_idx)
                blur_pads_list.append(tuple(pads_onnx))
                blur_pool_idx += len(stem_channels) // len([n for n in m.graph.node if n.op_type == "Conv" and "depthwise" in str(n.output)])

        # Better: just count by position
        # Recount: blur comes after conv whose output feeds into MaxPool
        blur_after = []
        blur_pads_list = []
        conv_count = 0
        for node in m.graph.node:
            if node.op_type == "Conv" and "depthwise" not in str(node.output) and "patch_extractor" not in str(node.output):
                if any("conv2d" in str(inp) for inp in node.input):
                    conv_count += 1
            if node.op_type == "Conv" and "depthwise" in str(node.output):
                attrs = {a.name: list(a.ints) for a in node.attribute if a.ints}
                pads_onnx = attrs.get("pads", [0, 0, 1, 1])
                blur_after.append(conv_count - 1)
                blur_pads_list.append(tuple(pads_onnx))

        # For simplicity, use known configs for XS and S
        if embed_dim == 64:  # XS
            cfg = CCTConfig(
                stem_channels=[32, 48, 64, 80, 96],
                stem_act="relu",
                blur_pool_after=[0, 2],
                blur_pool_pads=[(0,0,1,1), (1,1,1,1)],
                patch_in_ch=96, patch_out_ch=384,
                seq_in=384, embed_dim=64, posemb_size=65,
                num_blocks=4, num_heads=1,
                tr_num_queries=9, tr_num_heads=4, num_classes=37,
            )
        else:  # S (embed_dim=128)
            cfg = CCTConfig(
                stem_channels=[48, 80, 96, 128],
                stem_act="gelu",
                stem_bias=False,
                stem_padding=1,
                blur_pool_after=[0],
                blur_pool_pads=[(0,0,1,1)],
                patch_in_ch=128, patch_out_ch=512,
                seq_in=512, embed_dim=128, posemb_size=512,
                num_blocks=6, num_heads=2,
                tr_num_queries=9, tr_num_heads=4, num_classes=37,
            )

        model = cls(cfg)
        load_weights_from_onnx(model, onnx_path)
        return model

    # -------------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [N, H, W, 3] uint8 NHWC
        Returns:
            [N, 9, 37] softmax character probabilities
        """
        x = x.float() * (1.0 / 255.0)
        x = x.permute(0, 3, 1, 2)          # NHWC → NCHW [N, 3, H, W]

        for i, conv in enumerate(self.stem_convs):
            x = self.stem_act_fn(conv(x))
            if i in self.blur_pool_after:
                x = self.stem_pools[str(i)](x)

        # Patch extraction
        x = self.patch_conv(x)              # [N, patch_out_ch, H', W']
        N = x.shape[0]
        x = x.permute(0, 2, 3, 1)          # [N, H', W', patch_out_ch]
        x = x.reshape(N, -1, x.shape[-1])  # [N, H'*W', patch_out_ch]

        # Sequence projection + GELU
        x = _gelu(x @ self.seq_W + self.seq_b)

        # Positional embedding
        x = x + self.pos_emb[:x.shape[1]].unsqueeze(0)

        for blk in self.blocks:
            x = blk(x)

        x = self.token_reducer(x)           # [N, 9, embed_dim]

        x = x @ self.vocab_W + self.vocab_b
        return x.softmax(dim=-1)


# ---------------------------------------------------------------------------
# Weight loader
# ---------------------------------------------------------------------------

def load_weights_from_onnx(model: CCTOCRTorch, onnx_path: str) -> CCTOCRTorch:
    """
    Load all parameters from a CCT-XS or CCT-S ONNX file into a CCTOCRTorch
    instance.  The model's config must already match the ONNX file.
    """
    import onnx
    from onnx import numpy_helper
    import os

    path = os.path.expanduser(onnx_path)
    m = onnx.load(path)
    W = {t.name: numpy_helper.to_array(t) for t in m.graph.initializer}

    def t(name: str) -> torch.Tensor:
        return torch.from_numpy(W[name].copy())

    cfg = model.cfg

    # ── ConvStem ──────────────────────────────────────────────────────────────
    conv_name_map = ["conv2d_1", "conv2d_1_2", "conv2d_2_1", "conv2d_3_1", "conv2d_4_1"]
    blur_pool_name_map = ["max_blur_pooling2d_1", "max_blur_pooling2d_1_2"]
    blur_keys = sorted([k for k in W if "const_fold_opt" in k and W[k].ndim == 4
                        and W[k].shape[2] == 3 and W[k].shape[3] == 3])

    for i, conv in enumerate(model.stem_convs):
        cname = conv_name_map[i]
        conv.weight.data = t(f"CCT_OCR_1/conv_stem_1/{cname}/convolution/ReadVariableOp:0")
        if cfg.stem_bias:
            conv.bias.data = t(f"CCT_OCR_1/conv_stem_1/{cname}/Squeeze:0")

    for j, (pool_idx, pool) in enumerate(
            sorted((int(k), v) for k, v in model.stem_pools.items())):
        blur_key = blur_keys[j]
        pool.blur_weight.data = t(blur_key)

    # ── Patch extractor ───────────────────────────────────────────────────────
    model.patch_conv.weight.data = t("CCT_OCR_1/patch_extractor_1/Reshape__6")

    # ── Sequence projection ───────────────────────────────────────────────────
    model.seq_W.data = t("CCT_OCR_1/mlp_1/dense_1/Cast/ReadVariableOp:0")
    model.seq_b.data = t("CCT_OCR_1/mlp_1/dense_1/BiasAdd/ReadVariableOp:0")

    # ── Positional embedding ──────────────────────────────────────────────────
    model.pos_emb.data = t("CCT_OCR_1/pos_emb_1/Slice:0")

    # ── Transformer blocks ─────────────────────────────────────────────────────
    # Dynamically build block/layer name tables
    block_names = sorted(set(
        k.split("/")[1] for k in W if k.startswith("CCT_OCR_1/transformer_block_")
    ))  # e.g. ['transformer_block_1_1', 'transformer_block_2_1', ...]

    # Collect DyT names and MHA names within each block
    def _get_mha_names(blk_name):
        keys = [k for k in W if f"CCT_OCR_1/{blk_name}/" in k and "multi_head_attention" in k]
        mha_set = sorted(set(k.split("/")[2] for k in keys))
        return mha_set

    def _get_dyt_names(blk_name):
        keys = [k for k in W if f"CCT_OCR_1/{blk_name}/" in k and "dy_t_" in k]
        dyt_set = sorted(set(k.split("/")[2] for k in keys))
        return dyt_set

    def _get_mlp_names(blk_name):
        keys = [k for k in W if f"CCT_OCR_1/{blk_name}/" in k and "/mlp_" in k]
        mlp_set = sorted(set(k.split("/")[2] for k in keys))
        return mlp_set

    for i, blk_name in enumerate(block_names):
        b = model.blocks[i]
        pfx = f"CCT_OCR_1/{blk_name}"

        dyts = _get_dyt_names(blk_name)   # should be 2 per block
        mhas = _get_mha_names(blk_name)   # should be 1 per block
        mlps = _get_mlp_names(blk_name)   # should be 1 per block

        dyt1, dyt2 = dyts[0], dyts[1]
        mha = mhas[0]

        # DyT norm 1
        b.norm1.alpha.data  = t(f"{pfx}/{dyt1}/Mul/ReadVariableOp:0")
        b.norm1.weight.data = t(f"{pfx}/{dyt1}/mul_1/ReadVariableOp:0")
        b.norm1.bias.data   = t(f"{pfx}/{dyt1}/add/ReadVariableOp:0")

        # Attention: Q/K/V weights in ONNX [in, out] format; for single-head square
        # and multi-head rectangular — stored directly (x @ W_q gives correct shape)
        b.W_q.data = t(f"{pfx}/{mha}/query_1/Reshape:0")
        b.b_q.data = t(f"{pfx}/{mha}/query_1/add/ReadVariableOp:0")
        b.W_k.data = t(f"{pfx}/{mha}/key_1/Reshape:0")
        b.b_k.data = t(f"{pfx}/{mha}/key_1/add/ReadVariableOp:0")
        b.W_v.data = t(f"{pfx}/{mha}/value_1/Reshape:0")
        b.b_v.data = t(f"{pfx}/{mha}/value_1/add/ReadVariableOp:0")
        b.W_o.data = t(f"{pfx}/{mha}/attention_output_1/Reshape_1:0")
        b.b_o.data = t(f"{pfx}/{mha}/attention_output_1/add/ReadVariableOp:0")

        # DyT norm 2
        b.norm2.alpha.data  = t(f"{pfx}/{dyt2}/Mul/ReadVariableOp:0")
        b.norm2.weight.data = t(f"{pfx}/{dyt2}/mul_1/ReadVariableOp:0")
        b.norm2.bias.data   = t(f"{pfx}/{dyt2}/add/ReadVariableOp:0")

        # MLP — sort numerically (avoid '10' < '9' string ordering)
        mlp = mlps[0]
        mlp_pfx = f"{pfx}/{mlp}"
        _raw = [k.split("/")[3] for k in W if f"{pfx}/{mlp}/" in k
                and "Cast/ReadVariableOp" in k]
        import re as _re
        dense_keys = sorted(_raw, key=lambda s: int(_re.search(r"\d+", s).group()))
        d1, d2 = dense_keys[0], dense_keys[1]
        b.W_m1.data = t(f"{mlp_pfx}/{d1}/Cast/ReadVariableOp:0")
        b.b_m1.data = t(f"{mlp_pfx}/{d1}/BiasAdd/ReadVariableOp:0")
        b.W_m2.data = t(f"{mlp_pfx}/{d2}/Cast/ReadVariableOp:0")
        b.b_m2.data = t(f"{mlp_pfx}/{d2}/BiasAdd/ReadVariableOp:0")

    # ── Token reducer ──────────────────────────────────────────────────────────
    tr = model.token_reducer
    # Find the MHA name used in token reducer
    tr_keys = [k for k in W if "token_reducer_1/multi_head_attention" in k]
    tr_mha = sorted(set(k.split("/")[2] for k in tr_keys))[0]
    pfx_tr = f"CCT_OCR_1/token_reducer_1/{tr_mha}"

    tr.queries.data = t("CCT_OCR_1/token_reducer_1/Reshape_1:0")

    # Q weight via Gemm (transB=1): find the Gemm const that is (inner, embed_dim)
    embed_dim = cfg.embed_dim
    inner_dim = cfg.tr_num_heads * embed_dim
    q_w_key = next(k for k in W if "const_fold_opt" in k
                   and W[k].shape == (inner_dim, embed_dim))
    tr.W_q.data = t(q_w_key)
    tr.b_q.data = t(f"{pfx_tr}/query_1/add/ReadVariableOp:0")

    # K, V: ONNX MatMul weight (embed_dim, inner), transpose for PyTorch [inner, embed]
    tr.W_k.data = t(f"{pfx_tr}/key_1/Reshape:0").T.contiguous()
    tr.b_k.data = t(f"{pfx_tr}/key_1/add/ReadVariableOp:0")
    tr.W_v.data = t(f"{pfx_tr}/value_1/Reshape:0").T.contiguous()
    tr.b_v.data = t(f"{pfx_tr}/value_1/add/ReadVariableOp:0")

    # Output proj: find Gemm const (embed_dim, inner)
    o_w_key = next(k for k in W if k.startswith("Reshape__")
                   and W[k].shape == (embed_dim, inner_dim))
    tr.W_o.data = t(o_w_key)
    tr.b_o.data = t(f"{pfx_tr}/attention_output_1/add/ReadVariableOp:0")

    # ── Vocabulary projection ──────────────────────────────────────────────────
    vocab_keys = sorted([k for k in W if "vocab_projection_1" in k])
    for k in vocab_keys:
        if "Cast/ReadVariableOp" in k:
            model.vocab_W.data = t(k)
        elif "BiasAdd/ReadVariableOp" in k:
            model.vocab_b.data = t(k)

    return model


# ---------------------------------------------------------------------------
# Smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse, os

    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx-xs",
        default="~/.cache/fast-plate-ocr/cct-xs-v1-global-model/cct_xs_v1_global.onnx")
    parser.add_argument("--onnx-s",
        default="~/.cache/fast-plate-ocr/cct-s-v1-global-model/cct_s_v1_global.onnx")
    parser.add_argument("--rtol", type=float, default=1e-4)
    args = parser.parse_args()

    try:
        import onnxruntime as ort
        has_ort = True
    except ImportError:
        has_ort = False

    x_np = np.random.randint(0, 256, (2, 64, 128, 3), dtype=np.uint8)
    x_pt = torch.from_numpy(x_np)

    for variant, onnx_path in [("XS", args.onnx_xs), ("S", args.onnx_s)]:
        path = os.path.expanduser(onnx_path)
        if not os.path.exists(path):
            print(f"Skipping {variant}: {path} not found")
            continue

        print(f"\n=== CCT-{variant} ===")
        model = CCTOCRTorch.from_onnx(onnx_path)
        model.eval()

        with torch.no_grad():
            out_pt = model(x_pt).numpy()
        print(f"PyTorch output shape: {out_pt.shape}")

        if has_ort:
            sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
            in_name = sess.get_inputs()[0].name
            out_name = sess.get_outputs()[0].name
            out_ort = sess.run([out_name], {in_name: x_np})[0]

            max_err = np.abs(out_pt - out_ort).max()
            mean_err = np.abs(out_pt - out_ort).mean()
            print(f"Max absolute error vs onnxruntime: {max_err:.2e}")
            print(f"Mean absolute error:                {mean_err:.2e}")
            if max_err < args.rtol:
                print(f"✓  CCT-{variant} outputs match — weight transplant successful.")
            else:
                print(f"✗  CCT-{variant} outputs differ.")

        model.train()
        loss = model(x_pt).sum()
        loss.backward()
        print(f"✓  CCT-{variant} loss.backward() succeeded.")
