"""
ObjectDetectionDomain — stub DomainAdapter for object detection (YOLO).

Provides the interface skeleton.  Fill in the concrete implementation
when adding YOLO/Faster-RCNN support.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import torch
from torch import Tensor
from torch.utils.data import Dataset

from framework.base.domain import DomainAdapter, LayerConfig


class ObjectDetectionDomain(DomainAdapter):
    """
    DomainAdapter stub for object detection.

    Supports YOLO-style models.  Override the abstract methods to provide
    real dataset loading, preprocessing, and layer schedule.
    """

    def __init__(
        self,
        model_name: str = 'yolov5s',
        dataset_root: str = '/data/coco',
        device: Optional[torch.device] = None,
    ):
        self._model_name = model_name
        self._dataset_root = dataset_root

        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._device = device

        # Placeholder: load model here
        # Example (requires ultralytics):
        #   from ultralytics import YOLO
        #   self._model = YOLO(model_name).model.to(device)
        #   self._model.eval()
        #   for p in self._model.parameters(): p.requires_grad_(False)
        self._model = None
        print(f"ObjectDetectionDomain stub: {model_name} (not loaded — override __init__)")

    @property
    def input_shape(self) -> Tuple[int, int]:
        return (640, 640)   # YOLO default

    @property
    def model(self) -> torch.nn.Module:
        if self._model is None:
            raise NotImplementedError(
                "ObjectDetectionDomain: model not loaded. "
                "Override __init__ to load your YOLO model."
            )
        return self._model

    @property
    def device(self) -> torch.device:
        return self._device

    def preprocess_for_model(self, image: Tensor) -> Tensor:
        """Resize and normalise for YOLO (stub)."""
        import torch.nn.functional as F
        img = F.interpolate(image, size=(640, 640), mode='bilinear', align_corners=False)
        return img.to(self._device)

    def get_layer_progression(self) -> List[LayerConfig]:
        """YOLOv5 backbone layer progression (stub)."""
        return [
            LayerConfig(name="model.0", description="YOLO Backbone Conv0", max_epochs=30),
            LayerConfig(name="model.4", description="YOLO Backbone C3_1", max_epochs=30),
            LayerConfig(name="model.6", description="YOLO Backbone C3_2", max_epochs=30),
            LayerConfig(name="model.10", description="YOLO Backbone C3_3", max_epochs=40),
            LayerConfig(name="model.17", description="YOLO FPN head", max_epochs=50),
            LayerConfig(name="model.24", description="YOLO Detection head", max_epochs=100),
        ]

    def build_dataset(self, split: str = 'train') -> Dataset:
        """COCO dataset loader (stub)."""
        raise NotImplementedError(
            "ObjectDetectionDomain.build_dataset: implement COCO loading here. "
            "Each item should yield {'image': Tensor[3,H,W], 'index': int}."
        )
