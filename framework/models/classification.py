"""
classification.py — ResNet-50, ConvNeXt-Small, and MobileNetV3-Small wrappers for ImageNet classification.

All models:
  - Use standard ImageNet normalization (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
  - Accept 224×224 inputs
  - Return [B, 1000] logits
  - Are frozen (eval mode, no gradients)
"""
from __future__ import annotations

from typing import Callable, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as M


class ResNet50Wrapper(nn.Module):
    """Frozen ResNet-50 (ImageNet1K_V2 weights)."""

    def __init__(self):
        super().__init__()
        self.model = M.resnet50(weights=M.ResNet50_Weights.IMAGENET1K_V2)
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


class ConvNeXtSmallWrapper(nn.Module):
    """Frozen ConvNeXt-Small (ImageNet1K_V1 weights)."""

    def __init__(self):
        super().__init__()
        self.model = M.convnext_small(weights=M.ConvNeXt_Small_Weights.IMAGENET1K_V1)
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


class MobileNetV3SmallWrapper(nn.Module):
    """Frozen MobileNetV3-Small (ImageNet1K_V1 weights)."""

    def __init__(self):
        super().__init__()
        self.model = M.mobilenet_v3_small(weights=M.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
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
