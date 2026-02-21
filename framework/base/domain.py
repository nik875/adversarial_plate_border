"""
DomainAdapter ABC — the central interface for the generalized attack framework.

Each domain (classification, detection, face recognition, OCR) provides one
concrete subclass that bundles dataset, target model, preprocessing, and
layer-progression schedule.  GenericPatchTrainer talks ONLY to this interface.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
from torch import Tensor
from torch.utils.data import Dataset


@dataclass
class LayerConfig:
    """Configuration for a target layer in the progressive attack schedule."""
    name: str               # module name as returned by model.named_modules()
    description: str        # human-readable label
    max_epochs: int = 50    # max epochs to train on this layer
    convergence_threshold: float = 1.0  # diversity score for early stopping (≤0 disables)

    def __repr__(self) -> str:
        return f"{self.description} ({self.name})"


class DomainAdapter(ABC):
    """
    Abstract base class that encapsulates everything domain-specific.

    Concrete subclasses must implement:
      - input_shape  → (H, W) the target model expects
      - model        → frozen, eval-mode target model
      - device       → torch.device
      - preprocess_for_model(image) → model input tensor
      - get_layer_progression()     → ordered list of LayerConfig
      - build_dataset(split)        → Dataset yielding dicts with 'image' key

    Optionally override get_baseline_image() for domain-specific baseline logic.
    """

    # ------------------------------------------------------------------
    # Abstract properties
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def input_shape(self) -> Tuple[int, int]:
        """(H, W) that the target model expects as spatial dimensions."""

    @property
    @abstractmethod
    def model(self) -> torch.nn.Module:
        """Frozen, eval-mode target model."""

    @property
    @abstractmethod
    def device(self) -> torch.device:
        """Device on which the model lives."""

    # ------------------------------------------------------------------
    # Abstract methods
    # ------------------------------------------------------------------

    @abstractmethod
    def preprocess_for_model(self, image: Tensor) -> Tensor:
        """
        Preprocess a [B, 3, H, W] image tensor for the target model.

        Handles resizing, normalization, channel reordering, etc.
        Returns the tensor in whatever format the model expects.
        """

    @abstractmethod
    def get_layer_progression(self) -> List[LayerConfig]:
        """Return ordered list of LayerConfig objects defining the attack schedule."""

    @abstractmethod
    def build_dataset(self, split: str = 'train') -> Dataset:
        """
        Build the dataset for the given split.

        Each item should be a dict containing at least:
          - 'image': float Tensor [3, H, W] in [0, 1]
          - 'index': int (optional, for caching)
        """

    # ------------------------------------------------------------------
    # Default implementations (override if needed)
    # ------------------------------------------------------------------

    def get_baseline_image(self, image: Tensor) -> Tensor:
        """
        Return the baseline image used for neutral-comparison activations.

        Default: identity (return image unchanged).
        Override in domains where the patch compositing changes spatial layout
        (e.g., border-style attack wraps the image in a neutral frame).

        Args:
            image: [B, 3, H, W] in [0, 1]
        Returns:
            baseline: [B, 3, H, W] in [0, 1]
        """
        return image
