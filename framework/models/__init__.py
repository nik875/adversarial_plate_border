"""
framework/models/__init__.py — Model registry for the ensemble adversarial training pipeline.

Usage::

    from framework.models import build_model, REGISTRY

    model = build_model('resnet50')      # returns a frozen nn.Module
    model = build_model('yolov8s')
    model = build_model('smolvlm_500m')

    # Check available architectures
    print(list(REGISTRY))
"""
from __future__ import annotations

import torch.nn as nn

from .classification import ResNet50Wrapper, ConvNeXtSmallWrapper
from .swin import SwinTWrapper
from .dinov2 import DINOv2Wrapper
from .clip_encoder import CLIPVisionWrapper
from .yolo_wrapper import YOLOv8Wrapper
from .trocr import TrOCREncoderWrapper
from .vlm import SmolVLMWrapper

REGISTRY: dict[str, type] = {
    'resnet50':        ResNet50Wrapper,
    'convnext_small':  ConvNeXtSmallWrapper,
    'swin_t':          SwinTWrapper,
    'dinov2_vitb14':   DINOv2Wrapper,
    'clip_vit_l14':    CLIPVisionWrapper,
    'yolov8s':         YOLOv8Wrapper,
    'trocr_base':      TrOCREncoderWrapper,
    'smolvlm_500m':    SmolVLMWrapper,
}


def build_model(arch: str) -> nn.Module:
    """Instantiate and return a frozen wrapper for the given architecture name.

    Args:
        arch: One of the keys in REGISTRY (case-sensitive).

    Returns:
        A frozen ``nn.Module`` in eval mode with all parameters' requires_grad=False.

    Raises:
        ValueError: If *arch* is not in REGISTRY.
    """
    if arch not in REGISTRY:
        raise ValueError(
            f"Unknown architecture '{arch}'. "
            f"Available architectures: {sorted(REGISTRY)}"
        )
    return REGISTRY[arch]()


__all__ = [
    'REGISTRY',
    'build_model',
    'ResNet50Wrapper',
    'ConvNeXtSmallWrapper',
    'SwinTWrapper',
    'DINOv2Wrapper',
    'CLIPVisionWrapper',
    'YOLOv8Wrapper',
    'TrOCREncoderWrapper',
    'SmolVLMWrapper',
]
