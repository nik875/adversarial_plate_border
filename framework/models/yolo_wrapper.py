"""
yolo_wrapper.py — YOLOv8s object detection wrapper.

  - ultralytics YOLOv8s (downloads yolov8s.pt automatically on first use)
  - Input: [B, 3, 640, 640] float32 in [0, 1]  (no further normalization)
  - Output: [B, 84, 8400] raw detection tensor
    (84 = 4 bbox coords + 80 COCO class scores; 8400 anchors)
  - NeuronSampler hooks into C2f / SPPF / Conv leaf modules — output ignored
  - Frozen (eval mode, no gradients)

IMPORTANT: yolo.model(x) (the nn.Module DetectionModel) expects [0, 1] float32.
It does NOT run ultralytics preprocessing — that only happens via .predict().
"""
from __future__ import annotations

from typing import Callable, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class YOLOv8Wrapper(nn.Module):
    """Frozen YOLOv8s DetectionModel (ultralytics)."""

    def __init__(self):
        super().__init__()
        from ultralytics import YOLO
        yolo = YOLO('yolov8s.pt')
        self.model = yolo.model   # DetectionModel (nn.Module)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Direct DetectionModel forward — no postprocessing
        return self.model(x)   # [B, 84, 8400] in eval mode

    @staticmethod
    def get_preprocess_fn() -> Callable:
        def preprocess(x: torch.Tensor) -> torch.Tensor:
            x = F.interpolate(x, size=(640, 640), mode='bilinear', align_corners=False)
            return x.clamp(0.0, 1.0)
        return preprocess

    @staticmethod
    def input_size() -> Tuple[int, int]:
        return (640, 640)
