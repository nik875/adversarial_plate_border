"""
swin.py — Swin-T (Swin Transformer Tiny) wrapper for ImageNet classification.

  - ImageNet1K_V1 weights
  - Standard ImageNet normalization (same as ResNet/ConvNeXt)
  - 224×224 inputs → [B, 1000] logits
  - Frozen (eval mode, no gradients)
"""
from __future__ import annotations

from typing import Callable, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as M


class SwinTWrapper(nn.Module):
    """Frozen Swin-T (hierarchical ViT), ImageNet1K_V1 weights."""

    def __init__(self):
        super().__init__()
        self.model = M.swin_t(weights=M.Swin_T_Weights.IMAGENET1K_V1)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)  # [B, 1000]

    @staticmethod
    def get_preprocess_fn() -> Callable:
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

        def preprocess(x: torch.Tensor) -> torch.Tensor:
            x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
            return (x - mean.to(x.device)) / std.to(x.device)

        return preprocess

    @staticmethod
    def input_size() -> Tuple[int, int]:
        return (224, 224)
