"""
trocr.py — TrOCR-base-printed encoder wrapper (OCR / text recognition).

  - microsoft/trocr-base-printed via HuggingFace transformers
  - Only the ViT encoder is kept; decoder weights are freed immediately
  - Mean=0.5, std=0.5 normalization (NOT ImageNet — TrOCR ViT convention)
  - 384×384 inputs → [B, 577, 768] patch embeddings
    (577 = 1 cls + 576 patches, since 384/16 × 384/16 = 576)
  - Frozen (eval mode, no gradients)

We bypass TrOCRProcessor entirely: it rescales [0,255]→[0,1] then normalizes.
Since our pipeline delivers [0,1], we apply only the normalization step.
"""
from __future__ import annotations

from typing import Callable, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import VisionEncoderDecoderModel


class TrOCREncoderWrapper(nn.Module):
    """Frozen TrOCR-base-printed ViT encoder (microsoft/trocr-base-printed)."""

    def __init__(self):
        super().__init__()
        full_model = VisionEncoderDecoderModel.from_pretrained(
            'microsoft/trocr-base-printed'
        )
        self.encoder = full_model.encoder   # ViTModel (nn.Module)
        del full_model                       # free decoder weights
        self.encoder.eval()
        for p in self.encoder.parameters():
            p.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.encoder(pixel_values=x)
        return out.last_hidden_state   # [B, 577, 768]

    @staticmethod
    def get_preprocess_fn() -> Callable:
        mean = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1)
        std  = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1)

        def preprocess(x: torch.Tensor) -> torch.Tensor:
            x = F.interpolate(x, size=(384, 384), mode='bilinear', align_corners=False)
            return (x - mean.to(x.device)) / std.to(x.device)

        return preprocess

    @staticmethod
    def input_size() -> Tuple[int, int]:
        return (384, 384)
