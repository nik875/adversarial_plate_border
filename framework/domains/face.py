"""
FaceRecognitionDomain — stub DomainAdapter for face recognition (ArcFace).

Provides the interface skeleton.  Fill in the concrete implementation
when adding ArcFace / FaceNet support.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import torch
from torch import Tensor
from torch.utils.data import Dataset

from framework.base.domain import DomainAdapter, LayerConfig


class FaceRecognitionDomain(DomainAdapter):
    """
    DomainAdapter stub for face recognition.

    Supports ArcFace-style models.  Override the abstract methods to provide
    real dataset loading, preprocessing, and layer schedule.
    """

    def __init__(
        self,
        model_name: str = 'arcface',
        dataset_root: str = '/data/faces',
        device: Optional[torch.device] = None,
    ):
        self._model_name = model_name
        self._dataset_root = dataset_root

        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._device = device

        # Placeholder: load model here
        # Example (requires insightface):
        #   from insightface.app import FaceAnalysis
        #   ...
        self._model = None
        print(f"FaceRecognitionDomain stub: {model_name} (not loaded — override __init__)")

    @property
    def input_shape(self) -> Tuple[int, int]:
        return (112, 112)   # ArcFace default

    @property
    def model(self) -> torch.nn.Module:
        if self._model is None:
            raise NotImplementedError(
                "FaceRecognitionDomain: model not loaded. "
                "Override __init__ to load your ArcFace model."
            )
        return self._model

    @property
    def device(self) -> torch.device:
        return self._device

    def preprocess_for_model(self, image: Tensor) -> Tensor:
        """Resize to 112×112 and normalise for ArcFace (stub)."""
        import torch.nn.functional as F
        img = F.interpolate(image, size=(112, 112), mode='bilinear', align_corners=False)
        # ArcFace expects values in [-1, 1]
        return (img * 2.0 - 1.0).to(self._device)

    def get_layer_progression(self) -> List[LayerConfig]:
        """ArcFace ResNet-50 backbone layer progression (stub)."""
        return [
            LayerConfig(name="body.0", description="ArcFace Block0", max_epochs=30),
            LayerConfig(name="body.4", description="ArcFace Block4", max_epochs=30),
            LayerConfig(name="body.8", description="ArcFace Block8", max_epochs=40),
            LayerConfig(name="body.12", description="ArcFace Block12", max_epochs=50),
            LayerConfig(name="output_layer", description="ArcFace Embedding", max_epochs=100),
        ]

    def build_dataset(self, split: str = 'train') -> Dataset:
        """Face dataset loader (stub)."""
        raise NotImplementedError(
            "FaceRecognitionDomain.build_dataset: implement face dataset loading here. "
            "Each item should yield {'image': Tensor[3,H,W], 'index': int}."
        )
