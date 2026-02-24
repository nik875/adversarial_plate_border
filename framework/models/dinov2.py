"""
dinov2.py — DINOv2 ViT-B/14 wrapper (self-supervised dense features).

  - Loaded via torch.hub from 'facebookresearch/dinov2'
  - Standard ImageNet normalization
  - 224×224 inputs → [B, 768] CLS token embedding
  - Frozen (eval mode, no gradients)

NeuronSampler hooks into the ViT's MHA and MLP leaf modules, which fire
regardless of which output (forward vs forward_features) is returned.
"""
from __future__ import annotations

from typing import Callable, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class DINOv2Wrapper(nn.Module):
    """Frozen DINOv2 ViT-B/14 (self-supervised, facebookresearch/dinov2)."""

    def __init__(self):
        super().__init__()
        self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # default forward() returns [B, 768] CLS token
        return self.model(x)

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
