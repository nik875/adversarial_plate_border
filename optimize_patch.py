#!/usr/bin/env python3
"""
Adversarial patch optimization for license plate detection/OCR systems.

This module provides:
- ALPRModels: Shared detector and OCR models that can be fine-tuned externally
- AdversarialPatchTrainer: Patch optimization using white-box gradients

The models are designed to be shared between this module and external scripts
(e.g., model extraction) that may fine-tune them between patch optimization epochs.
"""

import os
from typing import Tuple, Optional, Dict, Any, List, Union
from dataclasses import dataclass
import warnings
import argparse
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
import numpy as np
from tqdm import tqdm
import kornia
import kornia.geometry as K
import torchvision.transforms as T
import matplotlib.pyplot as plt
from matplotlib import patches
import onnx
import onnx2torch
from open_image_models import LicensePlateDetector

from dataset import create_dataloaders

warnings.filterwarnings("ignore")


# =============================================================================
# Configuration
# =============================================================================

PATCH_WIDTH = 512
PATCH_HEIGHT = 256
OCR_INPUT_SHAPE = (64, 128, 3)
OCR_ALPHABET = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ_'
OCR_MAX_SLOTS = 9


# =============================================================================
# Patch Adapter Layer
# =============================================================================

class PatchAdapter(nn.Module):
    """
    U-Net style encoder-decoder that transforms adversarial patches.

    This module operates on the fixed-size patch tensor (512×256×3) and learns
    to transform it into an equivalent representation that makes the surrogate
    perceive it the same way the black-box does.

    The bottleneck captures the low-dimensional adversarial manifold, and the
    decoder reconstructs pixels that express the same adversarial intent in the
    surrogate's perceptual space.
    """

    def __init__(self, patch_height: int = 256, patch_width: int = 512):
        """
        Args:
            patch_height: Fixed patch height (256)
            patch_width: Fixed patch width (512)
        """
        super().__init__()

        # Encoder (downsampling path)
        self.enc1 = self._conv_block(3, 32)      # 512×256
        self.enc2 = self._conv_block(32, 64)     # 256×128
        self.enc3 = self._conv_block(64, 128)    # 128×64
        self.enc4 = self._conv_block(128, 256)   # 64×32

        # Bottleneck
        self.bottleneck = self._conv_block(256, 256)  # 32×16

        # Decoder (upsampling path)
        self.dec4 = self._upconv_block(256, 128)  # 64×32 (takes bottleneck)
        self.dec3 = self._upconv_block(384, 64)   # 128×64 (takes d4+e4: 128+256=384)
        self.dec2 = self._upconv_block(192, 32)   # 256×128 (takes d3+e3: 64+128=192)
        self.dec1 = self._upconv_block(96, 16)    # 512×256 (takes d2+e2: 32+64=96)

        # Final output
        self.out_conv = nn.Conv2d(48, 3, 1)  # 1×1 conv (takes d1+e1: 16+32=48)
        self.out_activation = nn.Sigmoid()   # Keep in [0, 1]

        self.pool = nn.MaxPool2d(2, 2)

    def _conv_block(self, in_ch: int, out_ch: int) -> nn.Module:
        """Double convolution block."""
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def _upconv_block(self, in_ch: int, out_ch: int) -> nn.Module:
        """Upsampling + convolution block."""
        return nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Transform adversarial patch.

        Args:
            x: Patch tensor [B, 3, 256, 512]

        Returns:
            Transformed patch [B, 3, 256, 512]
        """
        # Encoder
        e1 = self.enc1(x)          # 32 × 256 × 512
        e2 = self.enc2(self.pool(e1))  # 64 × 128 × 256
        e3 = self.enc3(self.pool(e2))  # 128 × 64 × 128
        e4 = self.enc4(self.pool(e3))  # 256 × 32 × 64

        # Bottleneck
        b = self.bottleneck(self.pool(e4))  # 256 × 16 × 32

        # Decoder with skip connections
        d4 = self.dec4(b)                    # 128 × 32 × 64
        d4 = torch.cat([d4, e4], dim=1)      # 256 × 32 × 64

        d3 = self.dec3(d4)                   # 64 × 64 × 128
        d3 = torch.cat([d3, e3], dim=1)      # 128 × 64 × 128

        d2 = self.dec2(d3)                   # 32 × 128 × 256
        d2 = torch.cat([d2, e2], dim=1)      # 64 × 128 × 256

        d1 = self.dec1(d2)                   # 16 × 256 × 512
        d1 = torch.cat([d1, e1], dim=1)      # 32 × 256 × 512

        # Output
        out = self.out_conv(d1)              # 3 × 256 × 512
        out = self.out_activation(out)

        return out


# =============================================================================
# Shared ALPR Models
# =============================================================================

class ALPRModels:
    """
    Shared container for ALPR detection and OCR models.

    This class manages the YOLO detector and OCR models, providing:
    - Lazy loading of models
    - Weight freezing/unfreezing for gradient control
    - Easy access for external fine-tuning scripts

    Usage:
        # Create and load models
        models = ALPRModels(device='cuda')
        models.load()

        # Use in training (weights frozen by default)
        output = models.detector(image)

        # External script can unfreeze and fine-tune
        models.unfreeze_detector()
        # ... fine-tune ...
        models.freeze_all()  # Re-freeze before next patch epoch
    """

    def __init__(self, device: str = None):
        """
        Initialize ALPRModels container.

        Args:
            device: Target device ('cuda', 'mps', 'cpu', or None for auto-detect)
        """
        if device is None:
            if torch.cuda.is_available():
                self.device = 'cuda'
            elif torch.backends.mps.is_available():
                self.device = 'mps'
            else:
                self.device = 'cpu'
        else:
            self.device = device

        self._detector: Optional[nn.Module] = None
        self._ocr: Optional[nn.Module] = None
        self._adapter: Optional[PatchAdapter] = None
        self._loaded = False

    @property
    def detector(self) -> nn.Module:
        """Get the YOLO detector model."""
        if not self._loaded:
            raise RuntimeError("Models not loaded. Call load() first.")
        return self._detector

    @property
    def ocr(self) -> nn.Module:
        """Get the OCR model."""
        if not self._loaded:
            raise RuntimeError("Models not loaded. Call load() first.")
        return self._ocr

    @property
    def adapter(self) -> PatchAdapter:
        """Get the patch adapter module."""
        if not self._loaded:
            raise RuntimeError("Models not loaded. Call load() first.")
        return self._adapter

    @property
    def is_loaded(self) -> bool:
        """Check if models are loaded."""
        return self._loaded

    def load(self) -> 'ALPRModels':
        """
        Load detector and OCR models from cached ONNX files.

        Returns:
            self for method chaining
        """
        if self._loaded:
            return self

        print("Loading ALPR models...")

        # Ensure models are downloaded
        LicensePlateDetector(detection_model="yolo-v9-t-384-license-plate-end2end")

        # Paths to ONNX models
        detector_path = (
            Path.home() / ".cache/open-image-models/yolo-v9-t-384-license-plate-end2end"
            / "yolo-v9-t-384-license-plates-end2end.onnx"
        )
        ocr_path = (
            Path.home() / ".cache/fast-plate-ocr/cct-xs-v1-global-model"
            / "cct_xs_v1_global.onnx"
        )

        if not detector_path.exists():
            raise FileNotFoundError(f"Detector ONNX not found: {detector_path}")
        if not ocr_path.exists():
            raise FileNotFoundError(f"OCR ONNX not found: {ocr_path}")

        # Load and convert detector
        print(f"  Loading detector from {detector_path}")
        detector_onnx = onnx.load(str(detector_path))
        self._detector = onnx2torch.convert(detector_onnx)
        self._detector.to(self.device)
        self._detector.eval()

        # Load and convert OCR
        print(f"  Loading OCR from {ocr_path}")
        ocr_onnx = onnx.load(str(ocr_path))
        self._ocr = onnx2torch.convert(ocr_onnx)
        self._ocr.to(self.device)
        self._ocr.eval()

        # Create patch adapter
        print(f"  Initializing patch adapter (U-Net)")
        self._adapter = PatchAdapter(patch_height=PATCH_HEIGHT, patch_width=PATCH_WIDTH)
        self._adapter.to(self.device)
        self._adapter.train()  # Adapter starts trainable

        # Freeze by default (but not adapter)
        self.freeze_all()

        self._loaded = True
        print(f"  Models loaded on {self.device}")

        return self

    def freeze_all(self) -> None:
        """Freeze all model parameters (no gradients). Does NOT freeze adapter."""
        self.freeze_detector()
        self.freeze_ocr()

    def unfreeze_all(self) -> None:
        """Unfreeze all model parameters (enable gradients). Does NOT affect adapter."""
        self.unfreeze_detector()
        self.unfreeze_ocr()

    def freeze_detector(self) -> None:
        """Freeze detector parameters."""
        if self._detector is not None:
            for param in self._detector.parameters():
                param.requires_grad = False

    def unfreeze_detector(self) -> None:
        """Unfreeze detector parameters for fine-tuning."""
        if self._detector is not None:
            for param in self._detector.parameters():
                param.requires_grad = True

    def freeze_ocr(self) -> None:
        """Freeze OCR parameters."""
        if self._ocr is not None:
            for param in self._ocr.parameters():
                param.requires_grad = False

    def unfreeze_ocr(self) -> None:
        """Unfreeze OCR parameters for fine-tuning."""
        if self._ocr is not None:
            for param in self._ocr.parameters():
                param.requires_grad = True

    def freeze_adapter(self) -> None:
        """Freeze adapter parameters."""
        if self._adapter is not None:
            for param in self._adapter.parameters():
                param.requires_grad = False

    def unfreeze_adapter(self) -> None:
        """Unfreeze adapter parameters for fine-tuning."""
        if self._adapter is not None:
            for param in self._adapter.parameters():
                param.requires_grad = True

    def get_detector_parameters(self):
        """Get detector parameters for optimizer."""
        if self._detector is None:
            return []
        return list(self._detector.parameters())

    def get_ocr_parameters(self):
        """Get OCR parameters for optimizer."""
        if self._ocr is None:
            return []
        return list(self._ocr.parameters())

    def get_adapter_parameters(self):
        """Get adapter parameters for optimizer."""
        if self._adapter is None:
            return []
        return list(self._adapter.parameters())

    def save_state(self, path: str) -> None:
        """Save model states to file."""
        torch.save({
            'detector': self._detector.state_dict() if self._detector else None,
            'ocr': self._ocr.state_dict() if self._ocr else None,
            'adapter': self._adapter.state_dict() if self._adapter else None,
        }, path)

    def load_state(self, path: str) -> None:
        """Load model states from file."""
        if not self._loaded:
            raise RuntimeError("Models must be loaded before loading state.")
        state = torch.load(path, map_location=self.device)
        if state['detector'] is not None:
            self._detector.load_state_dict(state['detector'])
        if state['ocr'] is not None:
            self._ocr.load_state_dict(state['ocr'])
        if 'adapter' in state and state['adapter'] is not None:
            self._adapter.load_state_dict(state['adapter'])


# =============================================================================
# Loss Functions
# =============================================================================

def create_focal_cce_loss(
    vocabulary_size: int,
    alpha: float = 0.25,
    gamma: float = 2.0,
    label_smoothing: float = 0.01,
):
    """
    Create a focal categorical cross-entropy loss function.

    Args:
        vocabulary_size: Size of the character vocabulary
        alpha: Focal loss alpha parameter
        gamma: Focal loss gamma parameter
        label_smoothing: Label smoothing factor

    Returns:
        Loss function callable
    """
    def focal_cce(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        # Flatten inputs
        y_true = y_true.reshape(-1, vocabulary_size)
        y_pred = y_pred.reshape(-1, vocabulary_size)

        # Ensure probabilities
        if y_pred.max() > 1.0 or y_pred.min() < 0.0:
            y_pred = F.softmax(y_pred, dim=-1)

        # Apply label smoothing
        if label_smoothing > 0.0:
            y_true = y_true * (1.0 - label_smoothing) + label_smoothing / vocabulary_size

        # Focal loss computation
        p_t = torch.sum(y_true * y_pred, dim=-1)
        focal_weight = (1.0 - p_t) ** gamma
        ce_loss = -torch.log(p_t + 1e-8)
        focal_loss = alpha * focal_weight * ce_loss

        return torch.mean(focal_loss)

    return focal_cce


def text_to_target_tensor(
    plate_text: str,
    max_slots: int,
    alphabet: str,
    device: str = 'cpu'
) -> torch.Tensor:
    """
    Convert plate text to one-hot encoded target tensor.

    Args:
        plate_text: License plate text (e.g., 'ABC123')
        max_slots: Maximum sequence length
        alphabet: Character vocabulary string
        device: Target device

    Returns:
        One-hot tensor of shape [1, max_slots, vocab_size]
    """
    padded = (plate_text + '_' * max_slots)[:max_slots]
    indices = [alphabet.index(char) for char in padded]

    target = torch.zeros(1, max_slots, len(alphabet))
    for i, idx in enumerate(indices):
        target[0, i, idx] = 1.0

    return target.to(device)


# =============================================================================
# Geometry Utilities
# =============================================================================

def invert_bbox(corners: torch.Tensor, transform: Tuple) -> torch.Tensor:
    """Invert preprocessing transformation to get original coordinates."""
    r, dw, dh = transform
    corners = corners.clone()
    corners[::2] = corners[::2] - dw
    corners[1::2] = corners[1::2] - dh
    corners = corners / r
    return corners


def bbox_to_corners(bbox: torch.Tensor, device: str = 'cpu') -> torch.Tensor:
    """Convert [x1, y1, x2, y2] bbox to corner points."""
    x1, y1, x2, y2 = bbox
    return torch.tensor([[
        [x1, y1], [x2, y1], [x2, y2], [x1, y2]
    ]], device=device)


def corners_to_bbox(corners: torch.Tensor) -> torch.Tensor:
    """Convert corner points to [x1, y1, x2, y2] bbox."""
    min_x = torch.min(corners[:, 0])
    max_x = torch.max(corners[:, 0])
    min_y = torch.min(corners[:, 1])
    max_y = torch.max(corners[:, 1])
    return torch.stack([min_x, min_y, max_x, max_y])


def compute_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """Compute IoU between two sets of boxes."""
    area1 = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    area2 = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

    boxes1 = boxes1.unsqueeze(1)
    boxes2 = boxes2.unsqueeze(0)

    inter_x1 = torch.max(boxes1[..., 0], boxes2[..., 0])
    inter_y1 = torch.max(boxes1[..., 1], boxes2[..., 1])
    inter_x2 = torch.min(boxes1[..., 2], boxes2[..., 2])
    inter_y2 = torch.min(boxes1[..., 3], boxes2[..., 3])

    inter_width = torch.clamp(inter_x2 - inter_x1, min=0)
    inter_height = torch.clamp(inter_y2 - inter_y1, min=0)
    inter_area = inter_width * inter_height

    union_area = area1 + area2 - inter_area
    return inter_area / (union_area + 1e-8)


def apply_plate_blur(
    image: torch.Tensor,
    corners: torch.Tensor,
    sigma: float
) -> torch.Tensor:
    """
    Apply Gaussian blur to the license plate region.

    Args:
        image: Image tensor [C, H, W]
        corners: Plate corner coordinates [4, 2]
        sigma: Blur sigma (0 = no blur)

    Returns:
        Blurred image tensor
    """
    if sigma <= 0:
        return image

    # Create a copy
    result = image.clone()
    C, H, W = image.shape

    # Get bounding box of plate
    min_x = int(max(0, corners[:, 0].min().item()))
    max_x = int(min(W, corners[:, 0].max().item()))
    min_y = int(max(0, corners[:, 1].min().item()))
    max_y = int(min(H, corners[:, 1].max().item()))

    if max_x <= min_x or max_y <= min_y:
        return result

    # Extract plate region
    plate_region = image[:, min_y:max_y, min_x:max_x].unsqueeze(0)

    # Apply Gaussian blur
    kernel_size = int(sigma * 6) | 1  # Ensure odd
    kernel_size = max(3, kernel_size)

    blurred_region = kornia.filters.gaussian_blur2d(
        plate_region,
        kernel_size=(kernel_size, kernel_size),
        sigma=(sigma, sigma)
    ).squeeze(0)

    # Replace plate region
    result[:, min_y:max_y, min_x:max_x] = blurred_region

    return result


# =============================================================================
# Adversarial Patch Trainer
# =============================================================================

@dataclass
class TrainerConfig:
    """Configuration for AdversarialPatchTrainer."""
    patch_width: int = PATCH_WIDTH
    patch_height: int = PATCH_HEIGHT
    grad_accumulate: Optional[int] = None
    match_detection: bool = False
    impersonation_target: Optional[str] = None
    print_blur: float = 0.0
    blur_sigma: float = 0.0
    use_tv_loss: bool = True
    use_homography: bool = True
    border_scale: float = 1.4


class AdversarialPatchTrainer:
    """
    Trainer for adversarial patch optimization against ALPR systems.

    This class optimizes a patch to either:
    - Disrupt detection/OCR (default)
    - Impersonate a target plate (with impersonation_target)

    The trainer can use externally-provided ALPRModels, allowing other scripts
    to fine-tune the models between epochs.

    Usage:
        # Standalone usage (creates own models)
        trainer = AdversarialPatchTrainer(csv_path='labels.csv')
        history = trainer.train(num_epochs=100)

        # With shared models (for model extraction)
        models = ALPRModels(device='cuda').load()
        trainer = AdversarialPatchTrainer(csv_path='labels.csv', models=models)

        for epoch in range(100):
            # External script can fine-tune models here
            models.unfreeze_all()
            fine_tune_on_blackbox_outputs(models)

            # Train patch for one epoch (auto-freezes models)
            loss = trainer.train_single_epoch(epoch)
    """

    def __init__(
        self,
        csv_path: str,
        models: Optional[ALPRModels] = None,
        device: str = None,
        config: Optional[TrainerConfig] = None,
        **kwargs
    ):
        """
        Initialize the trainer.

        Args:
            csv_path: Path to dataset CSV file
            models: Optional shared ALPRModels instance
            device: Target device (uses models.device if models provided)
            config: Trainer configuration
            **kwargs: Override config fields (grad_accumulate, match_detection, etc.)
        """
        # Handle configuration
        self.config = config or TrainerConfig()
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)

        # Device setup
        if models is not None:
            self.device = models.device
        elif device is not None:
            self.device = device
        elif torch.cuda.is_available():
            self.device = 'cuda'
        elif torch.backends.mps.is_available():
            self.device = 'mps'
        else:
            self.device = 'cpu'

        # Models setup
        if models is not None:
            self.models = models
            if not models.is_loaded:
                models.load()
        else:
            self.models = ALPRModels(device=self.device).load()

        # Data loading - use all data for training
        self.transform = T.Compose([T.ToTensor()])
        self.train_loader, self.val_loader = create_dataloaders(
            csv_path,
            transform=self.transform,
            preload=True,
            batch_size=1,
            n_jobs=0,
            use_all_for_train=True
        )

        # Initialize adversarial patch
        self.patch = nn.Parameter(
            torch.randn(
                3, self.config.patch_height, self.config.patch_width,
                device=self.device
            ) * 0.1
        )

        # OCR setup
        self.ocr_loss_fn = create_focal_cce_loss(len(OCR_ALPHABET))

        if self.config.impersonation_target:
            self.ocr_target = text_to_target_tensor(
                self.config.impersonation_target,
                OCR_MAX_SLOTS,
                OCR_ALPHABET,
                self.device
            )
        else:
            self.ocr_target = text_to_target_tensor(
                'VRJ7774',
                OCR_MAX_SLOTS,
                OCR_ALPHABET,
                self.device
            )

        # Calculate baseline losses
        self.detection_baseline, self.ocr_baseline = self._calculate_baseline_loss()

        # Training state
        self._optimizer: Optional[optim.Optimizer] = None
        self._scheduler = None
        self._current_epoch = 0
        self._training_mode = False

    # -------------------------------------------------------------------------
    # Properties for external access
    # -------------------------------------------------------------------------

    @property
    def detector(self) -> nn.Module:
        """Access the detector model."""
        return self.models.detector

    @property
    def ocr(self) -> nn.Module:
        """Access the OCR model."""
        return self.models.ocr

    def set_blur_sigma(self, blur_sigma: float) -> None:
        """
        Update the blur sigma for patch optimization.

        This allows external scripts (e.g., model_extraction.py) to adjust
        the blur level dynamically during training. Recalculates baseline losses
        based on the new blur level.

        Args:
            blur_sigma: New blur sigma value
        """
        self.config.blur_sigma = blur_sigma
        # Recalculate baseline losses with new blur level
        self.detection_baseline, self.ocr_baseline = self._calculate_baseline_loss()

    # -------------------------------------------------------------------------
    # Patch Application
    # -------------------------------------------------------------------------

    def get_patch_bounding_box(self, corners: torch.Tensor) -> torch.Tensor:
        """Calculate bounding box of the patch area (scaled border)."""
        plate_corners = corners

        center_x = plate_corners[:, 0].mean()
        center_y = plate_corners[:, 1].mean()
        center = torch.tensor([center_x, center_y], device=self.device)

        border_corners = center.unsqueeze(0) + (
            plate_corners - center.unsqueeze(0)
        ) * self.config.border_scale

        min_x = torch.min(border_corners[:, 0])
        max_x = torch.max(border_corners[:, 0])
        min_y = torch.min(border_corners[:, 1])
        max_y = torch.max(border_corners[:, 1])

        return torch.stack([min_x, min_y, max_x, max_y])

    def _apply_patch_simple(
        self,
        image: torch.Tensor,
        corners: torch.Tensor,
        patch_normalized: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Apply patch as simple rectangular overlay without homography."""
        batch_size = image.shape[0]
        image_height, image_width = image.shape[2], image.shape[3]

        plate_corners = corners[0]

        center_x = plate_corners[:, 0].mean()
        center_y = plate_corners[:, 1].mean()
        center = torch.tensor([center_x, center_y], device=self.device)

        border_corners = center.unsqueeze(0) + (
            plate_corners - center.unsqueeze(0)
        ) * self.config.border_scale

        border_min_x = torch.clamp(torch.min(border_corners[:, 0]), 0, image_width).int()
        border_max_x = torch.clamp(torch.max(border_corners[:, 0]), 0, image_width).int()
        border_min_y = torch.clamp(torch.min(border_corners[:, 1]), 0, image_height).int()
        border_max_y = torch.clamp(torch.max(border_corners[:, 1]), 0, image_height).int()

        plate_min_x = torch.clamp(torch.min(plate_corners[:, 0]), 0, image_width).int()
        plate_max_x = torch.clamp(torch.max(plate_corners[:, 0]), 0, image_width).int()
        plate_min_y = torch.clamp(torch.min(plate_corners[:, 1]), 0, image_height).int()
        plate_max_y = torch.clamp(torch.max(plate_corners[:, 1]), 0, image_height).int()

        result_image = image.clone()
        final_mask = torch.zeros(
            batch_size, 3, image_height, image_width,
            device=self.device, dtype=torch.float32
        )

        border_h = border_max_y - border_min_y
        border_w = border_max_x - border_min_x

        if border_h > 0 and border_w > 0:
            patch_resized = F.interpolate(
                patch_normalized.unsqueeze(0),
                size=(border_h, border_w),
                mode='bilinear',
                align_corners=True
            )

            for b in range(batch_size):
                result_image[b, :, border_min_y:border_max_y,
                             border_min_x:border_max_x] = patch_resized[0]
                final_mask[b, :, border_min_y:border_max_y, border_min_x:border_max_x] = 1.0

                if plate_max_y > plate_min_y and plate_max_x > plate_min_x:
                    result_image[b, :, plate_min_y:plate_max_y, plate_min_x:plate_max_x] = \
                        image[b, :, plate_min_y:plate_max_y, plate_min_x:plate_max_x]
                    final_mask[b, :, plate_min_y:plate_max_y, plate_min_x:plate_max_x] = 0.0

        return torch.clamp(result_image, 0, 1), final_mask

    def apply_patch_to_image(
        self,
        image: torch.Tensor,
        corners: torch.Tensor,
        patch_override: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Apply adversarial patch as border around license plate.

        Args:
            image: Input image tensor [B, C, H, W]
            corners: Plate corner coordinates [B, 4, 2]
            patch_override: Optional pre-transformed patch [C, H, W] to use instead of self.patch

        Returns:
            Tuple of (patched_image, mask)
        """
        batch_size = image.shape[0]
        image_height, image_width = image.shape[2], image.shape[3]
        dsize = (image_height, image_width)

        # Use provided patch or default to self.patch
        if patch_override is not None:
            # Patch already transformed, just normalize to [0, 1] if needed
            if patch_override.max() <= 1.0 and patch_override.min() >= 0.0:
                patch_normalized = patch_override
            else:
                patch_normalized = torch.tanh(patch_override) * 0.5 + 0.5
        else:
            # Normalize patch to [0, 1]
            patch_normalized = torch.tanh(self.patch) * 0.5 + 0.5

        # Optional blur
        if self.config.print_blur > 0:
            patch_normalized = kornia.filters.gaussian_blur2d(
                patch_normalized.unsqueeze(0),
                kernel_size=(3, 3),
                sigma=(self.config.print_blur, self.config.print_blur)
            ).squeeze(0)

        # Training augmentation
        if self._training_mode:
            darkening_factor = torch.rand(1, device=self.device) * 0.2
            patch_normalized = patch_normalized * (1.0 - darkening_factor)

        # Use simple overlay if homography disabled
        if not self.config.use_homography:
            return self._apply_patch_simple(image, corners, patch_normalized)

        # Homography-based application
        plate_corners = corners[0]

        center_x = plate_corners[:, 0].mean()
        center_y = plate_corners[:, 1].mean()
        center = torch.tensor([center_x, center_y], device=self.device)

        border_corners = center.unsqueeze(0) + (
            plate_corners - center.unsqueeze(0)
        ) * self.config.border_scale
        border_corners = border_corners.unsqueeze(0)

        patch_h, patch_w = self.config.patch_height, self.config.patch_width
        src_corners = torch.tensor([
            [0, 0], [patch_w, 0], [patch_w, patch_h], [0, patch_h]
        ], dtype=torch.float32, device=self.device).unsqueeze(0)

        M_border = K.get_perspective_transform(src_corners, border_corners)
        M_plate = K.get_perspective_transform(src_corners, corners)

        patch_batch = patch_normalized.unsqueeze(0).repeat(batch_size, 1, 1, 1)

        warped_patch = K.warp_perspective(
            patch_batch, M_border, dsize=dsize,
            mode='bilinear', padding_mode='zeros', align_corners=True
        )

        patch_mask = torch.ones(
            batch_size, 1, self.config.patch_height, self.config.patch_width,
            dtype=torch.float32, device=self.device
        )

        warped_border_mask = K.warp_perspective(
            patch_mask, M_border, dsize=dsize,
            mode='bilinear', padding_mode='zeros', align_corners=True
        )

        warped_plate_mask = K.warp_perspective(
            patch_mask, M_plate, dsize=dsize,
            mode='bilinear', padding_mode='zeros', align_corners=True
        )

        final_mask = torch.clamp(warped_border_mask - warped_plate_mask, 0, 1)
        final_mask = final_mask.expand(-1, 3, -1, -1)

        result_image = image * (1 - final_mask) + warped_patch * final_mask
        return torch.clamp(result_image, 0, 1), final_mask

    # -------------------------------------------------------------------------
    # Loss Computation
    # -------------------------------------------------------------------------

    def _compute_partial_loss(
        self,
        batch: Dict[str, torch.Tensor],
        use_ocr_baseline: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute detection and OCR losses for a batch."""
        prep_image = batch['prep_image'].to(self.device)
        corners = batch['new_corners'].to(self.device)

        if self.config.match_detection:
            target_box = self.get_patch_bounding_box(corners)
        else:
            target_box = corners_to_bbox(corners)

        model_output = self.models.detector(prep_image.unsqueeze(0))

        best_detection = None
        det_loss = torch.tensor(0.0, device=self.device)

        for detection in model_output:
            pred_box = detection[1:5]
            conf = detection[6]
            iou = compute_iou(pred_box.unsqueeze(0), target_box.unsqueeze(0))

            if self.config.match_detection:
                this_det_loss = -iou * conf
            else:
                this_det_loss = iou * conf

            if self.config.match_detection:
                if -this_det_loss > -det_loss:
                    det_loss = this_det_loss
                    best_detection = detection
            else:
                if this_det_loss > det_loss:
                    det_loss = this_det_loss
                    best_detection = detection

        ocr_loss = torch.tensor(0.0, device=self.device)
        if best_detection is not None:
            pred_box = best_detection[1:5]
            orig_projection = invert_bbox(pred_box.to('cpu'), batch['transform'])
            corners_box = bbox_to_corners(orig_projection, device='cpu')

            cropped_plate = kornia.geometry.crop_and_resize(
                batch['orig_image'].unsqueeze(0),
                corners_box,
                OCR_INPUT_SHAPE[:2],
                mode='bilinear',
                align_corners=True
            ).to(self.device)

            ocr_input = cropped_plate.permute(0, 2, 3, 1) * 255
            ocr_output = self.models.ocr(ocr_input)
            ocr_loss = torch.sqrt(self.ocr_loss_fn(self.ocr_target, ocr_output))

            if use_ocr_baseline:
                if self.config.impersonation_target:
                    ocr_loss = ocr_loss / self.ocr_baseline
                else:
                    ocr_loss = self.ocr_baseline / ocr_loss

        return det_loss, ocr_loss

    def _compute_tv_loss(self) -> torch.Tensor:
        """Compute total variation regularization loss."""
        patch = self.patch
        C, H, W = patch.shape

        tv_h = torch.pow(patch[:, :, 1:] - patch[:, :, :-1], 2).sum()
        tv_v = torch.pow(patch[:, 1:, :] - patch[:, :-1, :], 2).sum()

        num_comparisons = C * (H * (W - 1) + (H - 1) * W)
        loss = (tv_h + tv_v) / num_comparisons

        return loss * 2.5

    def compute_loss(
        self,
        batch: Dict[str, torch.Tensor],
        use_ocr_baseline: bool = True,
        return_components: bool = False
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute total loss for a batch.

        Args:
            batch: Batch dict from dataloader
            use_ocr_baseline: Whether to use OCR baseline normalization
            return_components: If True, return dict with individual loss components

        Returns:
            Total loss tensor, or dict with 'total', 'det', 'ocr', 'tv' if return_components=True
        """
        batch = {k: v[0] for k, v in batch.items()}

        # Normalize patch to [0,1] before adapter
        patch_normalized = torch.sigmoid(self.patch)

        # Transform patch through adapter (for surrogate perception)
        with torch.no_grad() if not self.patch.requires_grad else torch.enable_grad():
            adapted_patch = self.models.adapter(patch_normalized.unsqueeze(0)).squeeze(0)

        # Apply adapted patch to preprocessed image
        patched_image, _ = self.apply_patch_to_image(
            batch['prep_image'].to(self.device).unsqueeze(0),
            batch['new_corners'].to(self.device).unsqueeze(0),
            patch_override=adapted_patch
        )
        batch['prep_image'] = patched_image.squeeze()

        # Apply adapted patch to original image
        patched_image, _ = self.apply_patch_to_image(
            batch['orig_image'].to(self.device).unsqueeze(0),
            batch['orig_corners'].to(self.device).unsqueeze(0),
            patch_override=adapted_patch
        )
        batch['orig_image'] = patched_image.squeeze()

        # Apply blur after patching (simulates real-world degradation)
        if self.config.blur_sigma > 0:
            # Blur original image
            batch['orig_image'] = apply_plate_blur(
                batch['orig_image'],
                batch['orig_corners'],
                self.config.blur_sigma
            )

            # Blur preprocessed image with scaled sigma
            orig_image_size = max(batch['orig_image'].shape[1], batch['orig_image'].shape[2])
            prep_blur_sigma = self.config.blur_sigma * (384 / orig_image_size)
            batch['prep_image'] = apply_plate_blur(
                batch['prep_image'],
                batch['new_corners'],
                prep_blur_sigma
            )

        det_loss, ocr_loss = self._compute_partial_loss(batch, use_ocr_baseline)

        loss = (det_loss + ocr_loss) / 2

        tv_loss = torch.tensor(0.0, device=self.device)
        if self.config.use_tv_loss:
            tv_loss = self._compute_tv_loss()
            loss = loss + tv_loss

        if return_components:
            return {
                'total': loss,
                'det': det_loss,
                'ocr': ocr_loss,
                'tv': tv_loss
            }
        return loss

    def _calculate_baseline_loss(self) -> Tuple[float, float]:
        """Calculate baseline losses across the dataset."""
        total_ocr_loss = 0.0
        total_det_loss = 0.0
        total_plates = 0

        with tqdm(self.train_loader, desc="Calculating baseline", leave=False) as pbar:
            with torch.no_grad():
                for batch in pbar:
                    batch = {k: v[0] for k, v in batch.items()}
                    det_loss, ocr_loss = self._compute_partial_loss(batch, use_ocr_baseline=False)
                    total_det_loss += det_loss
                    total_ocr_loss += ocr_loss
                    total_plates += 1

                    pbar.set_postfix({
                        'det': f'{(total_det_loss / total_plates).item():.4f}',
                        'ocr': f'{(total_ocr_loss / total_plates).item():.4f}'
                    })

        return total_det_loss / total_plates, total_ocr_loss / total_plates

    # -------------------------------------------------------------------------
    # Training
    # -------------------------------------------------------------------------

    def train_single_epoch(
        self,
        epoch: int,
        optimizer: Optional[optim.Optimizer] = None
    ) -> Dict[str, float]:
        """
        Train for a single epoch.

        This method is designed for external control of the training loop.
        It ensures models are frozen before training.

        Args:
            epoch: Current epoch number (for logging)
            optimizer: Optional optimizer (uses internal if not provided)

        Returns:
            Dict with 'total', 'det', 'ocr', 'tv' average losses for the epoch
        """
        # IMPORTANT: Freeze models at the start of each epoch
        # This ensures weights are frozen even if an external script unfroze them
        self.models.freeze_all()

        self._training_mode = True

        if optimizer is None:
            if self._optimizer is None:
                self._optimizer = optim.AdamW([self.patch], lr=0.01, weight_decay=1e-4)
            optimizer = self._optimizer

        # Track individual loss components
        total_losses = {'total': 0.0, 'det': 0.0, 'ocr': 0.0, 'tv': 0.0}
        accum_losses = {'total': 0.0, 'det': 0.0, 'ocr': 0.0, 'tv': 0.0}
        step_count = 0
        num_updates = 0

        update_every = (
            len(self.train_loader)
            if self.config.grad_accumulate is None
            else self.config.grad_accumulate
        )
        effective_batch_size = update_every

        desc = f"Epoch {epoch + 1} (AccumSteps={update_every})"
        with tqdm(enumerate(self.train_loader), desc=desc, leave=False,
                  total=len(self.train_loader)) as pbar:

            for idx, batch in pbar:
                loss_dict = self.compute_loss(batch, return_components=True)
                loss = loss_dict['total']
                scaled_loss = loss / effective_batch_size
                scaled_loss.backward()

                # Accumulate component losses
                for key in accum_losses:
                    accum_losses[key] += loss_dict[key].item()
                step_count += 1

                if step_count % update_every == 0:
                    torch.nn.utils.clip_grad_norm_([self.patch], max_norm=1.0)
                    optimizer.step()
                    optimizer.zero_grad()

                    for key in total_losses:
                        total_losses[key] += accum_losses[key]
                    num_updates += 1

                    del loss, scaled_loss, loss_dict
                    if self.device == 'cuda':
                        torch.cuda.empty_cache()
                    elif self.device == 'mps':
                        torch.mps.empty_cache()

                    avg_loss = total_losses['total'] / (num_updates * update_every)
                    pbar.set_postfix({'Loss': f"{avg_loss:.4f}", 'Updates': num_updates})

                    accum_losses = {'total': 0.0, 'det': 0.0, 'ocr': 0.0, 'tv': 0.0}
                else:
                    del loss, scaled_loss, loss_dict
                    current_avg = accum_losses['total'] / (step_count % update_every)
                    pbar.set_postfix({
                        'AccumLoss': f"{current_avg:.4f}",
                        'Progress': f"{step_count % update_every}/{update_every}"
                    })

            # Handle remaining gradients
            if step_count % update_every != 0 and self.config.grad_accumulate is not None:
                torch.nn.utils.clip_grad_norm_([self.patch], max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
                for key in total_losses:
                    total_losses[key] += accum_losses[key]
                num_updates += 1

        self._training_mode = False
        self._current_epoch = epoch + 1

        total_batches = num_updates * \
            update_every if self.config.grad_accumulate else len(self.train_loader)

        return {key: val / total_batches for key, val in total_losses.items()}

    def validate(self) -> float:
        """Run validation and return average loss."""
        losses = []

        with torch.no_grad():
            for batch in self.val_loader:
                loss = self.compute_loss(batch)
                losses.append(loss.detach().cpu().item())

        return np.mean(losses)

    def save_patch(self, epoch: int, save_dir: str = "patches") -> None:
        """Save current patch state."""
        Path(save_dir).mkdir(exist_ok=True)

        with torch.no_grad():
            patch_img = torch.tanh(self.patch) * 0.5 + 0.5
            patch_img = patch_img.detach().cpu()
            patch_pil = T.ToPILImage()(patch_img)

            patch_pil.save(f"{save_dir}/patch_epoch_{epoch:04d}.png")

            torch.save({
                'patch': self.patch.detach().cpu(),
                'epoch': epoch,
                'patch_size': (self.config.patch_height, self.config.patch_width)
            }, f"{save_dir}/patch_epoch_{epoch:04d}.pt")

    def load_patch(self, patch_path: str) -> None:
        """Load patch from a .pt file."""
        state = torch.load(patch_path, map_location=self.device)
        self.patch.data = state['patch'].to(self.device)

    def get_patched_images(self) -> List[Dict[str, Any]]:
        """
        Generate patched versions of all training images.

        Returns:
            List of dicts with keys:
                - 'index': Dataset index
                - 'patched_prep': Patched preprocessed image [C, H, W]
                - 'patched_orig': Patched original image [C, H, W]
                - 'prep_image': Original preprocessed image [C, H, W]
                - 'orig_image': Original full image [C, H, W]
                - 'corners': Plate corners in prep space [4, 2]
                - 'orig_corners': Plate corners in original space [4, 2]
                - 'transform': Transform tuple (ratio, dw, dh)
        """
        results = []

        with torch.no_grad():
            for idx, batch in enumerate(self.train_loader):
                batch = {k: v[0] for k, v in batch.items()}

                orig_image = batch['orig_image']
                prep_image = batch['prep_image']
                corners = batch['new_corners']
                orig_corners = batch['orig_corners']
                transform = batch['transform']

                # Apply patch
                patched_prep, _ = self.apply_patch_to_image(
                    prep_image.to(self.device).unsqueeze(0),
                    corners.to(self.device).unsqueeze(0)
                )
                patched_orig, _ = self.apply_patch_to_image(
                    orig_image.to(self.device).unsqueeze(0),
                    orig_corners.to(self.device).unsqueeze(0)
                )

                results.append({
                    'index': idx,
                    'patched_prep': patched_prep.squeeze(0).cpu(),
                    'patched_orig': patched_orig.squeeze(0).cpu(),
                    'prep_image': prep_image,
                    'orig_image': orig_image,
                    'corners': corners,
                    'orig_corners': orig_corners,
                    'transform': transform,
                })

        return results

    def iterate_dataset(self) -> List[Dict[str, Any]]:
        """
        Iterate over the training dataset without applying patches.

        Returns:
            List of dicts with keys:
                - 'index': Dataset index
                - 'prep_image': Preprocessed image [C, H, W]
                - 'orig_image': Original full image [C, H, W]
                - 'corners': Plate corners in prep space [4, 2]
                - 'orig_corners': Plate corners in original space [4, 2]
                - 'transform': Transform tuple (ratio, dw, dh)
        """
        results = []

        for idx, batch in enumerate(self.train_loader):
            batch = {k: v[0] for k, v in batch.items()}

            results.append({
                'index': idx,
                'prep_image': batch['prep_image'],
                'orig_image': batch['orig_image'],
                'corners': batch['new_corners'],
                'orig_corners': batch['orig_corners'],
                'transform': batch['transform'],
            })

        return results

    def train(
        self,
        num_epochs: int = 100,
        learning_rate: float = 0.01,
        save_interval: int = 10,
        early_stop_patience: int = 15
    ) -> Dict[str, list]:
        """
        Full training loop with learning rate scheduling and early stopping.

        For external control of the training loop (e.g., model extraction),
        use train_single_epoch() instead.

        Args:
            num_epochs: Maximum number of epochs
            learning_rate: Initial learning rate
            save_interval: Save checkpoint every N epochs
            early_stop_patience: Stop if no improvement for N epochs

        Returns:
            Training history dict with 'loss', 'val_score', 'learning_rate'
        """
        optimizer = optim.AdamW([self.patch], lr=learning_rate, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=5, factor=0.5
        )

        history = {'loss': [], 'val_score': [], 'learning_rate': []}
        best_loss = float('inf')
        patience_counter = 0

        print("\nStarting adversarial patch training")
        print(f"   Dataset: {len(self.train_loader) + len(self.val_loader)} images")
        print(f"   Patch size: {self.config.patch_height}×{self.config.patch_width}")
        print(f"   Device: {self.device}")
        print(f"   Epochs: {num_epochs}")
        print(f"   Initial LR: {learning_rate}")
        print(f"   Match detection: {self.config.match_detection}")
        print(f"   Impersonation target: {self.config.impersonation_target or 'None'}")
        print(f"   TV loss: {'Enabled' if self.config.use_tv_loss else 'Disabled'}")
        print(f"   Homography: {'Enabled' if self.config.use_homography else 'Disabled'}")
        print("-" * 60)

        for epoch in range(num_epochs):
            loss_dict = self.train_single_epoch(epoch, optimizer)
            train_loss = loss_dict['total']
            val_loss = self.validate()

            scheduler.step(train_loss)
            current_lr = optimizer.param_groups[0]['lr']

            history['loss'].append(train_loss)
            history['val_score'].append(val_loss)
            history['learning_rate'].append(current_lr)

            initial_loss = history['val_score'][0] if history['val_score'] else 1.0
            loss_change = (val_loss / initial_loss - 1) * 100

            print(f"Epoch {epoch + 1:3d}/{num_epochs} | "
                  f"Loss: {train_loss:.4f} | "
                  f"Val: {val_loss:.3f} | "
                  f"Change: {loss_change:+.1f}% | "
                  f"LR: {current_lr:.2e}")

            if val_loss < best_loss:
                best_loss = val_loss
                patience_counter = 0
                self.save_patch(epoch, "best_patches")
                print(f"   New best: {best_loss:.4f}")
            else:
                patience_counter += 1

            if (epoch + 1) % save_interval == 0:
                self.save_patch(epoch, "checkpoint_patches")

            if patience_counter >= early_stop_patience:
                print(f"   Early stopping after {early_stop_patience} epochs without improvement")
                break

            if len(history['loss']) >= 20:
                recent = history['loss'][-20:]
                if (max(recent) - min(recent)) < 0.0001:
                    print("   Converged")
                    break

        print("\nTraining completed!")
        print(f"   Best loss: {best_loss:.4f}")

        return history


# =============================================================================
# Utility Functions
# =============================================================================

def load_patch_from_file(
    patch_file: str,
    target_height: int,
    target_width: int,
    device: str
) -> torch.Tensor:
    """Load and prepare a patch from an image file."""
    from PIL import Image

    if not os.path.exists(patch_file):
        raise FileNotFoundError(f"Patch file not found: {patch_file}")

    patch_img = Image.open(patch_file).convert('RGB')
    print(f"Loaded patch image: {patch_img.size}")

    transform = T.Compose([
        T.Resize((target_height, target_width)),
        T.ToTensor()
    ])

    patch_tensor = transform(patch_img).to(device)
    patch_tensor = torch.clamp(patch_tensor, 0, 1)

    print(f"Patch tensor shape: {patch_tensor.shape}")
    print(f"Patch value range: [{patch_tensor.min():.3f}, {patch_tensor.max():.3f}]")

    return patch_tensor


def logits_to_text(logits: torch.Tensor, alphabet: str = OCR_ALPHABET) -> str:
    """Convert OCR logits to text string."""
    probs = torch.softmax(logits, dim=-1)
    pred_chars = torch.argmax(probs, dim=-1).squeeze(0)
    text = ""
    for char_idx in pred_chars:
        idx = char_idx.item()
        if idx < len(alphabet) and alphabet[idx] != '_':
            text += alphabet[idx]
    return text.strip()


# =============================================================================
# Debug/Visualization Functions
# =============================================================================

def test_detection_visualization(
    csv_path: str,
    output_path: str = "test_detection.png",
    **kwargs
) -> None:
    """Visualize YOLO detections on a sample image."""
    print("Running test mode - visualizing detections...")

    trainer = AdversarialPatchTrainer(csv_path, **kwargs)

    batch = next(iter(trainer.train_loader))
    batch = {k: v[0] for k, v in batch.items()}

    prep_image = batch['prep_image'].to(trainer.device)
    corners = batch['new_corners'].to(trainer.device)
    ground_truth = corners_to_bbox(corners)

    with torch.no_grad():
        model_output = trainer.models.detector(prep_image.unsqueeze(0))

    print(f"Ground truth box: {ground_truth}")

    all_detections = []
    for detection in model_output:
        pred_box = detection[1:5]
        conf = detection[6]
        class_id = detection[5]
        iou = compute_iou(pred_box.unsqueeze(0), ground_truth.unsqueeze(0)).squeeze()

        print(f"  Detection: Box={pred_box}, Conf={conf:.4f}, IoU={iou:.4f}")

        all_detections.append({
            'box': pred_box.cpu().numpy(),
            'confidence': conf.item(),
            'class_id': int(class_id.item()),
            'iou': iou.item()
        })

    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    prep_img = prep_image.permute(1, 2, 0).detach().cpu().numpy()
    ax1.imshow(prep_img)
    ax1.set_title('Preprocessed Image with Detections')

    gt = ground_truth.detach().cpu().numpy()
    gt_rect = patches.Rectangle(
        (gt[0], gt[1]), gt[2] - gt[0], gt[3] - gt[1],
        linewidth=3, edgecolor='green', facecolor='none', label='Ground Truth'
    )
    ax1.add_patch(gt_rect)

    colors = ['red', 'blue', 'orange', 'purple', 'brown']
    for i, det in enumerate(all_detections):
        box = det['box']
        color = colors[i % len(colors)]
        rect = patches.Rectangle(
            (box[0], box[1]), box[2] - box[0], box[3] - box[1],
            linewidth=2, edgecolor=color, facecolor='none',
            label=f'Det {i}: {det["confidence"]:.3f}'
        )
        ax1.add_patch(rect)
    ax1.legend()

    ax2.axis('off')
    ax2.set_title('Detection Analysis')

    analysis_text = f"Detections: {len(all_detections)}\n\n"
    for i, det in enumerate(all_detections):
        analysis_text += f"{i}: Conf={det['confidence']:.4f}, IoU={det['iou']:.4f}\n"

    if all_detections:
        best = max(all_detections, key=lambda x: x['iou'])
        analysis_text += f"\nBest IoU: {best['iou']:.4f}"

    ax2.text(0.05, 0.95, analysis_text, transform=ax2.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Visualization saved to: {output_path}")


def debug_patch_application(
    csv_path: str,
    patch_file: str = None,
    output_path: str = "debug_patch.png",
    **kwargs
) -> Dict[str, Any]:
    """Debug mode: apply patch and visualize impact on detection."""
    print("Running debug patch mode...")

    trainer = AdversarialPatchTrainer(csv_path, **kwargs)

    batch = next(iter(trainer.train_loader))
    batch = {k: v[0] for k, v in batch.items()}

    prep_image = batch['prep_image'].to(trainer.device)
    corners = batch['new_corners'].to(trainer.device)

    if trainer.config.match_detection:
        target_box = trainer.get_patch_bounding_box(corners)
        target_name = "Patch Bounding Box"
    else:
        target_box = corners_to_bbox(corners)
        target_name = "Ground Truth Box"

    # Original detections
    with torch.no_grad():
        original_output = trainer.models.detector(prep_image.unsqueeze(0))

    original_detections = []
    best_original_iou = 0.0
    best_original_conf = 0.0

    for detection in original_output:
        pred_box = detection[1:5]
        conf = detection[6]
        iou = compute_iou(pred_box.unsqueeze(0), target_box.unsqueeze(0)).squeeze()

        original_detections.append({
            'box': pred_box.cpu().numpy(),
            'confidence': conf.item(),
            'iou': iou.item()
        })

        if iou.item() > best_original_iou:
            best_original_iou = iou.item()
        if conf.item() > best_original_conf:
            best_original_conf = conf.item()

    # Load or create patch
    if patch_file:
        patch_tensor = load_patch_from_file(
            patch_file, PATCH_HEIGHT, PATCH_WIDTH, trainer.device
        )
        patch = torch.arctanh(torch.clamp(patch_tensor * 2 - 1, -0.99, 0.99))
    else:
        patch = torch.full(
            (3, PATCH_HEIGHT, PATCH_WIDTH), 2.6, device=trainer.device
        )

    original_patch = trainer.patch.data.clone()
    trainer.patch.data = patch

    patched_image, patch_mask = trainer.apply_patch_to_image(
        prep_image.unsqueeze(0), corners.unsqueeze(0)
    )
    patched_image = patched_image.squeeze(0)

    # Patched detections
    with torch.no_grad():
        patched_output = trainer.models.detector(patched_image.unsqueeze(0))

    patched_detections = []
    best_patched_iou = 0.0
    best_patched_conf = 0.0

    for detection in patched_output:
        pred_box = detection[1:5]
        conf = detection[6]
        iou = compute_iou(pred_box.unsqueeze(0), target_box.unsqueeze(0)).squeeze()

        patched_detections.append({
            'box': pred_box.cpu().numpy(),
            'confidence': conf.item(),
            'iou': iou.item()
        })

        if iou.item() > best_patched_iou:
            best_patched_iou = iou.item()
        if conf.item() > best_patched_conf:
            best_patched_conf = conf.item()

    trainer.patch.data = original_patch

    # Metrics
    iou_reduction = (
        (best_original_iou - best_patched_iou) / best_original_iou * 100
        if best_original_iou > 0 else 0
    )
    conf_reduction = (
        (best_original_conf - best_patched_conf) / best_original_conf * 100
        if best_original_conf > 0 else 0
    )

    print(f"\nPatch Impact:")
    print(f"  Original IoU: {best_original_iou:.4f}, Patched: {best_patched_iou:.4f}")
    print(f"  IoU reduction: {iou_reduction:.1f}%")
    print(f"  Conf reduction: {conf_reduction:.1f}%")

    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    orig_img = prep_image.permute(1, 2, 0).detach().cpu().numpy()
    axes[0, 0].imshow(orig_img)
    axes[0, 0].set_title(f'Original ({len(original_detections)} detections)')

    patched_img = patched_image.permute(1, 2, 0).detach().cpu().numpy()
    axes[0, 1].imshow(patched_img)
    axes[0, 1].set_title(f'Patched ({len(patched_detections)} detections)')

    patch_display = torch.tanh(patch) * 0.5 + 0.5
    patch_img = patch_display.detach().cpu().permute(1, 2, 0).numpy()
    axes[1, 0].imshow(patch_img)
    axes[1, 0].set_title('Applied Patch')
    axes[1, 0].axis('off')

    axes[1, 1].axis('off')
    axes[1, 1].set_title('Impact Analysis')

    analysis = f"Target: {target_name}\n\n"
    analysis += f"Original:\n  Best IoU: {best_original_iou:.4f}\n  Best Conf: {best_original_conf:.4f}\n\n"
    analysis += f"Patched:\n  Best IoU: {best_patched_iou:.4f}\n  Best Conf: {best_patched_conf:.4f}\n\n"
    analysis += f"Reduction:\n  IoU: {iou_reduction:.1f}%\n  Conf: {conf_reduction:.1f}%"

    axes[1, 1].text(0.1, 0.9, analysis, transform=axes[1, 1].transAxes,
                    fontsize=11, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Visualization saved to: {output_path}")

    return {
        'original_detections': len(original_detections),
        'patched_detections': len(patched_detections),
        'best_original_iou': best_original_iou,
        'best_patched_iou': best_patched_iou,
        'iou_reduction_percent': iou_reduction,
        'conf_reduction_percent': conf_reduction
    }


def debug_ocr_accuracy(
    csv_path: str,
    patch_file: str = None,
    output_path: str = "debug_ocr.png",
    **kwargs
) -> Dict[str, Any]:
    """Test OCR accuracy with and without patch."""
    print("Testing OCR accuracy...")

    trainer = AdversarialPatchTrainer(csv_path, **kwargs)

    if patch_file:
        patch_tensor = load_patch_from_file(
            patch_file, PATCH_HEIGHT, PATCH_WIDTH, trainer.device
        )
        patch = torch.arctanh(torch.clamp(patch_tensor * 2 - 1, -0.99, 0.99))
        trainer.patch.data = patch

    batch = next(iter(trainer.train_loader))
    batch = {k: v[0] for k, v in batch.items()}

    def run_pipeline(use_patch: bool) -> Dict:
        batch_copy = {k: v.clone() if torch.is_tensor(v) else v for k, v in batch.items()}

        if use_patch:
            patched, _ = trainer.apply_patch_to_image(
                batch_copy['prep_image'].to(trainer.device).unsqueeze(0),
                batch_copy['new_corners'].to(trainer.device).unsqueeze(0)
            )
            if patched is not None:
                batch_copy['prep_image'] = patched.squeeze(0)

            patched, _ = trainer.apply_patch_to_image(
                batch_copy['orig_image'].to(trainer.device).unsqueeze(0),
                batch_copy['orig_corners'].to(trainer.device).unsqueeze(0)
            )
            if patched is not None:
                batch_copy['orig_image'] = patched.squeeze(0)

        prep_image = batch_copy['prep_image'].to(trainer.device)
        model_output = trainer.models.detector(prep_image.unsqueeze(0))

        corners = batch_copy['new_corners'].to(trainer.device)
        target_box = corners_to_bbox(corners)

        best_detection = None
        best_iou = 0.0

        for detection in model_output:
            pred_box = detection[1:5]
            iou = compute_iou(pred_box.unsqueeze(0), target_box.unsqueeze(0)).squeeze()
            if iou > best_iou:
                best_iou = iou.item()
                best_detection = detection

        if best_detection is None:
            return {'text': None, 'conf': 0.0, 'iou': 0.0, 'ocr_loss': float('inf')}

        pred_box = best_detection[1:5]
        conf = best_detection[6].item()
        orig_projection = invert_bbox(pred_box.to('cpu'), batch_copy['transform'])
        corners_box = bbox_to_corners(orig_projection, device='cpu')

        cropped_plate = kornia.geometry.crop_and_resize(
            batch_copy['orig_image'].unsqueeze(0),
            corners_box,
            OCR_INPUT_SHAPE[:2],
            mode='bilinear',
            align_corners=True
        ).to(trainer.device)

        ocr_input = cropped_plate.permute(0, 2, 3, 1) * 255
        with torch.no_grad():
            ocr_logits = trainer.models.ocr(ocr_input)

        ocr_loss = trainer.ocr_loss_fn(trainer.ocr_target, ocr_logits).item()
        ocr_text = logits_to_text(ocr_logits)

        return {
            'text': ocr_text,
            'conf': conf,
            'iou': best_iou,
            'ocr_loss': ocr_loss
        }

    results_no_patch = run_pipeline(use_patch=False)
    results_with_patch = run_pipeline(use_patch=True)

    print(f"\nOCR Results:")
    print(
        f"  Without patch: '{results_no_patch['text']}' (loss: {results_no_patch['ocr_loss']:.4f})")
    print(
        f"  With patch:    '{results_with_patch['text']}' (loss: {results_with_patch['ocr_loss']:.4f})")

    # Simple visualization
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].bar(['No Patch', 'With Patch'],
                [results_no_patch['ocr_loss'], results_with_patch['ocr_loss']],
                color=['blue', 'red'], alpha=0.7)
    axes[0].set_ylabel('OCR Loss')
    axes[0].set_title('OCR Loss Comparison')

    axes[1].axis('off')
    summary = f"Without Patch:\n  Text: '{results_no_patch['text']}'\n  Loss: {results_no_patch['ocr_loss']:.4f}\n\n"
    summary += f"With Patch:\n  Text: '{results_with_patch['text']}'\n  Loss: {results_with_patch['ocr_loss']:.4f}"
    axes[1].text(0.1, 0.9, summary, transform=axes[1].transAxes,
                 fontsize=12, verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Visualization saved to: {output_path}")

    return {
        'without_patch': results_no_patch,
        'with_patch': results_with_patch,
        'loss_change': results_with_patch['ocr_loss'] - results_no_patch['ocr_loss']
    }


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Adversarial Patch Optimization')
    parser.add_argument('--test', action='store_true',
                        help='Test mode: visualize detections')
    parser.add_argument('--debug-patch', nargs='?', const=True, default=False,
                        help='Debug patch application')
    parser.add_argument('--debug-ocr', nargs='?', const=True, default=False,
                        help='Debug OCR accuracy')
    parser.add_argument('--output', default='test_detection.png',
                        help='Output path for visualizations')
    parser.add_argument('--match-detection', action='store_true',
                        help='Maximize IoU with patch bounding box')
    parser.add_argument('--impersonation-target', type=str, default=None,
                        help='Target plate text for impersonation')
    parser.add_argument('--disable-tv-loss', action='store_true',
                        help='Disable TV regularization')
    parser.add_argument('--disable-homography', action='store_true',
                        help='Use simple rectangle overlay')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (cuda, mps, cpu)')
    parser.add_argument('--grad-accumulate', type=int, default=64,
                        help='Gradient accumulation steps')
    parser.add_argument('--blur-sigma', type=float, default=0.0,
                        help='Blur sigma to apply to plate region (0 = no blur)')
    args = parser.parse_args()

    CSV_PATH = "preproc_labels.csv"

    trainer_kwargs = {
        'device': args.device,
        'match_detection': args.match_detection,
        'impersonation_target': args.impersonation_target,
        'use_tv_loss': not args.disable_tv_loss,
        'use_homography': not args.disable_homography,
        'grad_accumulate': args.grad_accumulate,
        'blur_sigma': args.blur_sigma,
    }

    if args.test:
        test_detection_visualization(CSV_PATH, args.output, **trainer_kwargs)
        return

    if args.debug_patch:
        patch_file = args.debug_patch if isinstance(args.debug_patch, str) else None
        debug_patch_application(CSV_PATH, patch_file, args.output, **trainer_kwargs)
        return

    if args.debug_ocr:
        patch_file = args.debug_ocr if isinstance(args.debug_ocr, str) else None
        debug_ocr_accuracy(CSV_PATH, patch_file, args.output, **trainer_kwargs)
        return

    # Normal training
    trainer = AdversarialPatchTrainer(CSV_PATH, **trainer_kwargs)

    history = trainer.train(
        num_epochs=100,
        learning_rate=0.1,
        save_interval=1,
        early_stop_patience=20
    )

    # Save history
    import pandas as pd
    history_df = pd.DataFrame(history)
    history_df.insert(0, 'epoch', range(1, len(history_df) + 1))
    history_df.to_csv('training_history.csv', index=False)

    # Plot results
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

    ax1.plot(history['loss'], 'b-', label='Training')
    ax1.plot(history['val_score'], 'r-', label='Validation')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss Over Time')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.semilogy(history['learning_rate'])
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Learning Rate')
    ax2.set_title('Learning Rate Schedule')
    ax2.grid(True, alpha=0.3)

    final_patch = torch.tanh(trainer.patch) * 0.5 + 0.5
    final_patch_np = final_patch.detach().cpu().permute(1, 2, 0).numpy()
    ax3.imshow(final_patch_np)
    ax3.set_title('Final Patch')
    ax3.axis('off')

    plt.tight_layout()
    plt.savefig('adversarial_training_results.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("\nResults saved to 'adversarial_training_results.png'")


if __name__ == "__main__":
    main()
