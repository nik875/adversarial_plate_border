"""
clip_encoder.py — CLIP ViT-L/14 vision encoder wrapper.

  - openai/clip-vit-large-patch14 via HuggingFace transformers
  - CLIP-specific normalization (different from ImageNet)
  - 224×224 inputs → [B, 1024] pooled embedding  (ViT-L hidden dim = 1024)
  - Frozen (eval mode, no gradients)

Note: ViT-L/14 hidden dim is 1024, NOT 768 (that is ViT-B).
out.last_hidden_state is [B, 257, 1024]  (1 cls + 256 patches).
"""
from __future__ import annotations

from typing import Callable, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPVisionModel


# CLIP-specific normalization constants (NOT the same as ImageNet)
_CLIP_MEAN = [0.48145466, 0.4578275,  0.40821073]
_CLIP_STD  = [0.26862954, 0.26130258, 0.27577711]


class CLIPVisionWrapper(nn.Module):
    """Frozen CLIP ViT-L/14 vision encoder (openai/clip-vit-large-patch14)."""

    def __init__(self):
        super().__init__()
        self.model = CLIPVisionModel.from_pretrained('openai/clip-vit-large-patch14')
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.model(pixel_values=x)
        return out.pooler_output   # [B, 1024]

    @staticmethod
    def get_preprocess_fn() -> Callable:
        mean = torch.tensor(_CLIP_MEAN).view(1, 3, 1, 1)
        std  = torch.tensor(_CLIP_STD).view(1, 3, 1, 1)

        def preprocess(x: torch.Tensor) -> torch.Tensor:
            x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
            return (x - mean.to(x.device)) / std.to(x.device)

        return preprocess

    @staticmethod
    def input_size() -> Tuple[int, int]:
        return (224, 224)
