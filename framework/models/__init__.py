"""
framework/models/__init__.py — Model registry for the ensemble adversarial training pipeline.

All imports are lazy: a wrapper module is only imported when that specific
architecture is first requested.  This means importing ``framework.models``
does NOT trigger heavy downloads or fail due to optional dependencies
(e.g. ultralytics, a newer transformers version for SmolVLM) that may not
be installed in the current environment.

Usage::

    from framework.models import build_model, REGISTRY

    model = build_model('resnet50')      # returns a frozen nn.Module
    model = build_model('yolov8s')
    model = build_model('smolvlm_500m')

    # Check available architecture names
    print(list(REGISTRY))

    # Get the wrapper class (lazy import happens here)
    cls = REGISTRY['clip_vit_l14']
    preprocess = cls.get_preprocess_fn()
"""
from __future__ import annotations

import importlib
import torch.nn as nn


# ---------------------------------------------------------------------------
# Internal map: arch_name -> (module_path, class_name)
# Only the (module, class) strings are stored at import time; the actual
# Python module is loaded on first access via _LazyRegistry.__getitem__.
# ---------------------------------------------------------------------------
_MAP: dict[str, tuple[str, str]] = {
    'resnet50':       ('framework.models.classification', 'ResNet50Wrapper'),
    'convnext_small': ('framework.models.classification', 'ConvNeXtSmallWrapper'),
    'swin_t':         ('framework.models.swin',           'SwinTWrapper'),
    'dinov2_vitb14':  ('framework.models.dinov2',         'DINOv2Wrapper'),
    'clip_vit_l14':   ('framework.models.clip_encoder',   'CLIPVisionWrapper'),
    'yolov8s':        ('framework.models.yolo_wrapper',   'YOLOv8Wrapper'),
    'trocr_base':     ('framework.models.trocr',          'TrOCREncoderWrapper'),
    'smolvlm_500m':   ('framework.models.vlm',            'SmolVLMWrapper'),
}


class _LazyRegistry:
    """Dict-like object that imports wrapper modules on first access.

    Supports: ``arch in REGISTRY``, ``REGISTRY[arch]``,
    ``list(REGISTRY)``, ``set(REGISTRY.keys())``, ``REGISTRY.keys()``.
    """

    def __contains__(self, key: str) -> bool:
        return key in _MAP

    def __getitem__(self, key: str) -> type:
        if key not in _MAP:
            raise KeyError(key)
        mod_path, cls_name = _MAP[key]
        mod = importlib.import_module(mod_path)
        return getattr(mod, cls_name)

    def __iter__(self):
        return iter(_MAP)

    def __len__(self) -> int:
        return len(_MAP)

    def keys(self):
        return _MAP.keys()

    def items(self):
        for k in _MAP:
            yield k, self[k]

    def values(self):
        for k in _MAP:
            yield self[k]

    def __repr__(self) -> str:
        return f"LazyRegistry({list(_MAP)})"


REGISTRY = _LazyRegistry()


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
            f"Available architectures: {sorted(_MAP)}"
        )
    cls = REGISTRY[arch]
    return cls()


__all__ = [
    'REGISTRY',
    'build_model',
]
