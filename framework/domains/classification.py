"""
ImageClassificationDomain — concrete DomainAdapter for image classification.

Reference implementation validating the framework design end-to-end.
Uses a torchvision ResNet-50 (or ViT) on ImageNet-style data.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as T
from torch import Tensor
from torch.utils.data import Dataset

from framework.base.domain import DomainAdapter, LayerConfig


# ImageNet normalisation constants
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


class _ImageFolderSubset(Dataset):
    """
    Thin wrapper around torchvision.datasets.ImageFolder with optional sample cap.
    Yields dicts with 'image' and 'index'.
    """

    def __init__(self, root: str, transform, max_samples: Optional[int] = None):
        try:
            from torchvision.datasets import ImageFolder
        except ImportError:
            raise ImportError("torchvision is required for ImageClassificationDomain")

        self._ds = ImageFolder(root=root, transform=None)
        self._to_tensor = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor()])
        self._max = max_samples
        if max_samples is not None:
            self._ds.samples = self._ds.samples[:max_samples]
            self._ds.targets = self._ds.targets[:max_samples]

    def __len__(self) -> int:
        return len(self._ds)

    def __getitem__(self, idx: int) -> dict:
        img_pil, label = self._ds[idx]
        img_tensor = self._to_tensor(img_pil)  # [3, 224, 224] in [0, 1]
        return {'image': img_tensor, 'label': label, 'index': idx}


class ImageClassificationDomain(DomainAdapter):
    """
    DomainAdapter for image classification using a torchvision backbone.

    Supports resnet50, resnet18, resnet101, vit_b_16, etc.
    Defaults to ResNet-50 pretrained on ImageNet.
    """

    def __init__(
        self,
        model_name: str = 'resnet50',
        pretrained: bool = True,
        dataset_root: str = '/data/imagenet',
        max_samples: Optional[int] = None,
        patch_height: int = 224,
        patch_width: int = 224,
        device: Optional[torch.device] = None,
    ):
        if device is None:
            if torch.cuda.is_available():
                device = torch.device('cuda')
            elif torch.backends.mps.is_available():
                device = torch.device('mps')
            else:
                device = torch.device('cpu')
        self._device = device
        self._patch_height = patch_height
        self._patch_width = patch_width
        self._dataset_root = dataset_root
        self._max_samples = max_samples

        # Load model
        weights = 'IMAGENET1K_V1' if pretrained else None
        print(f"Loading {model_name} (pretrained={pretrained})...")
        try:
            self._model = getattr(models, model_name)(weights=weights).to(device)
        except AttributeError:
            raise ValueError(
                f"Unknown model: {model_name}. "
                f"Use any torchvision model name (resnet50, vit_b_16, etc.)"
            )
        self._model.eval()
        for p in self._model.parameters():
            p.requires_grad_(False)
        print(f"✓ {model_name} loaded on {device}")

        # ImageNet normalisation transform (applied in preprocess_for_model)
        self._normalize = T.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD)

    # ------------------------------------------------------------------
    # DomainAdapter interface
    # ------------------------------------------------------------------

    @property
    def input_shape(self) -> Tuple[int, int]:
        return (self._patch_height, self._patch_width)

    @property
    def model(self) -> torch.nn.Module:
        return self._model

    @property
    def device(self) -> torch.device:
        return self._device

    def preprocess_for_model(self, image: Tensor) -> Tensor:
        """
        Resize to 224×224, apply ImageNet normalisation.

        Args:
            image: [B, 3, H, W] float in [0, 1]
        Returns:
            [B, 3, 224, 224] normalised tensor
        """
        if image.shape[2] != 224 or image.shape[3] != 224:
            image = F.interpolate(image, size=(224, 224),
                                  mode='bilinear', align_corners=False)
        # Normalise each image in the batch
        norm = torch.stack([self._normalize(img) for img in image])
        return norm.to(self._device)

    def get_layer_progression(self) -> List[LayerConfig]:
        """
        6-layer progressive schedule for ResNet-50.

        Starts at early conv features and ends at the fully-connected head.
        Module names match torchvision ResNet-50 named_modules().
        """
        return [
            LayerConfig(
                name="layer1.0.relu",
                description="ResNet Layer1 Block0 ReLU (64ch)",
                max_epochs=30,
                convergence_threshold=1.0,
            ),
            LayerConfig(
                name="layer2.0.relu",
                description="ResNet Layer2 Block0 ReLU (128ch)",
                max_epochs=30,
                convergence_threshold=1.0,
            ),
            LayerConfig(
                name="layer3.0.relu",
                description="ResNet Layer3 Block0 ReLU (256ch)",
                max_epochs=30,
                convergence_threshold=1.0,
            ),
            LayerConfig(
                name="layer4.0.relu",
                description="ResNet Layer4 Block0 ReLU (512ch)",
                max_epochs=40,
                convergence_threshold=1.0,
            ),
            LayerConfig(
                name="avgpool",
                description="ResNet Global AvgPool (2048-d)",
                max_epochs=50,
                convergence_threshold=0.5,
            ),
            LayerConfig(
                name="fc",
                description="ResNet FC Head (1000-class logits)",
                max_epochs=100,
                convergence_threshold=0.0,   # no early stopping on final layer
            ),
        ]

    def build_dataset(self, split: str = 'train') -> Dataset:
        """
        Build ImageFolder dataset.

        Expects ImageNet directory layout: root/{split}/{class}/{image.jpg}

        Args:
            split: 'train' or 'val'
        """
        root = Path(self._dataset_root) / split
        if not root.exists():
            # Fallback: treat root itself as the dataset directory
            root = Path(self._dataset_root)
        return _ImageFolderSubset(str(root), transform=None,
                                  max_samples=self._max_samples)

    def get_baseline_image(self, image: Tensor) -> Tensor:
        """
        For classification with border strategy, return image unchanged.

        The BaseStrategy's apply_neutral() handles the neutral composite;
        this default is correct for classification (no border needed for baseline).
        """
        return image
