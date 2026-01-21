#!/usr/bin/env python3
"""
Generator Refinement: Load a frozen generator and add refinement layers
to improve attack effectiveness while maintaining SSIM with the original patch.

The generator provides attack direction, the refiner improves effectiveness
while staying close to the learned manifold.
"""
import os
from typing import Tuple, List, Optional
import logging
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
from kornia.metrics import ssim
import torchvision.transforms as T
import matplotlib.pyplot as plt
from matplotlib import patches
import onnx
import onnx2torch
from open_image_models import LicensePlateDetector
from dataset import create_dataloaders
from progressive_patch import SimplePatchGenerator, FoundationPatchGenerator
warnings.filterwarnings("ignore")


PATCH_WIDTH = 512
PATCH_HEIGHT = 256


class RefinementNetwork(nn.Module):
    """
    Refinement network that takes a base patch from the generator
    and applies targeted adjustments to improve attack effectiveness.

    Uses CNN + Dense architecture with residual connection to preserve
    generator's learned patterns while making strategic improvements.
    """
    def __init__(self, patch_height: int = 256, patch_width: int = 512,
                 use_latent_context: bool = False, latent_dim: int = 16):
        super().__init__()

        self.patch_height = patch_height
        self.patch_width = patch_width
        self.use_latent_context = use_latent_context
        self.latent_dim = latent_dim

        # Input channels: 3 (base patch) + latent_dim (if using context)
        input_channels = 3 + (latent_dim if use_latent_context else 0)

        if use_latent_context:
            # Project latent z to spatial map for concatenation
            self.latent_projection = nn.Sequential(
                nn.Linear(latent_dim, 256),
                nn.ReLU(inplace=True),
                nn.Linear(256, latent_dim * patch_height * patch_width)
            )

        # CNN feature extractor
        self.cnn_features = nn.Sequential(
            # Block 1
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # Block 3
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        # Global pooling and dense layers
        self.global_pool = nn.AdaptiveAvgPool2d((8, 16))
        dense_input_dim = 32 * 8 * 16  # 4096

        self.dense_layers = nn.Sequential(
            nn.Linear(dense_input_dim, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 3 * patch_height * patch_width),
            nn.Tanh()  # Output residual in [-1, 1]
        )

        # Learnable scaling factor for residual (initialized small)
        self.alpha = nn.Parameter(torch.tensor(0.1))

        # Initialize weights
        self._initialize_weights()

        print(f"RefinementNetwork initialized:")
        print(f"  Input: {input_channels} channels (patch + {'latent context' if use_latent_context else 'no context'})")
        print(f"  CNN: {input_channels}→32→32→64→64→32→32")
        print(f"  Dense: {dense_input_dim}→1024→512→{3 * patch_height * patch_width}")
        print(f"  Residual scaling: α = {self.alpha.item():.3f}")

    def _initialize_weights(self):
        """Initialize network weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, base_patch: torch.Tensor, z: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            base_patch: [batch_size, 3, H, W] - output from frozen generator
            z: [batch_size, latent_dim] - latent code (optional, for context)

        Returns:
            refined_patch: [batch_size, 3, H, W] - refined patch
        """
        batch_size = base_patch.shape[0]

        # Prepare input
        if self.use_latent_context and z is not None:
            # Project latent to spatial map
            latent_spatial = self.latent_projection(z)
            latent_spatial = latent_spatial.view(
                batch_size, self.latent_dim, self.patch_height, self.patch_width
            )
            # Concatenate with base patch
            cnn_input = torch.cat([base_patch, latent_spatial], dim=1)
        else:
            cnn_input = base_patch

        # Extract CNN features
        features = self.cnn_features(cnn_input)  # [B, 32, H, W]

        # Global pooling and dense processing
        pooled = self.global_pool(features)  # [B, 32, 8, 16]
        flattened = pooled.view(batch_size, -1)  # [B, 4096]
        residual_flat = self.dense_layers(flattened)  # [B, 3*H*W]
        residual = residual_flat.view(batch_size, 3, self.patch_height, self.patch_width)

        # Apply scaled residual connection
        refined_patch = base_patch + self.alpha * residual

        # Clamp to valid range [0, 1]
        refined_patch = torch.clamp(refined_patch, 0.0, 1.0)

        return refined_patch


class RefineGeneratorTrainer:
    """
    Trainer for refining generator outputs with SSIM constraint.

    Loads a frozen generator and trains a refinement network to improve
    attack effectiveness against both detection and OCR models while
    maintaining high SSIM with the original generator output.
    """
    def __init__(self,
                 csv_path: str,
                 generator_checkpoint: str,
                 device: str = None,
                 batch_size: int = 1,
                 grad_accumulate: int = None,
                 match_detection: bool = False,
                 impersonation_target: str = None,
                 print_blur: float = 0,
                 training: bool = False,
                 use_tv_loss: bool = True,
                 use_homography: bool = True,
                 ssim_weight: float = 1.0,
                 use_latent_context: bool = False,
                 generator_type: str = 'simple',
                 use_all_for_train: bool = True):

        self.training = training
        self.print_blur = print_blur
        self.use_tv_loss = use_tv_loss
        self.use_homography = use_homography
        self.ssim_weight = ssim_weight
        self.use_latent_context = use_latent_context
        self.use_all_for_train = use_all_for_train

        # Image preprocessing
        self.transform = T.Compose([T.ToTensor()])

        self.grad_accumulate = grad_accumulate
        self.match_detection = match_detection
        self.impersonation_target = impersonation_target

        if device is None:
            if torch.cuda.is_available():
                self.device = 'cuda'
            elif torch.backends.mps.is_available():
                self.device = 'mps'
            else:
                self.device = 'cpu'
        else:
            self.device = device

        self.patch_width = PATCH_WIDTH
        self.patch_height = PATCH_HEIGHT

        # Load dataset
        # NOTE: batch_size is forced to 1 because the dataset contains variable-sized images
        # Batching with different image sizes requires either image resizing or custom collate functions.
        # gradient accumulation is used instead to simulate larger batch sizes.
        actual_batch_size = 1
        self.train_loader, self.val_loader = create_dataloaders(
            csv_path, transform=self.transform, preload=True, batch_size=actual_batch_size, n_jobs=0,
            use_all_for_train=use_all_for_train
        )

        # Load frozen generator
        print(f"\nLoading frozen generator from: {generator_checkpoint}")
        self.generator, self.basis_dim = self._load_frozen_generator(
            generator_checkpoint, generator_type
        )
        self.generator.eval()
        for param in self.generator.parameters():
            param.requires_grad = False
        print(f"Generator frozen (latent dim: {self.basis_dim})")

        # Initialize refinement network
        print("\nInitializing refinement network...")
        self.refinement_net = RefinementNetwork(
            patch_height=self.patch_height,
            patch_width=self.patch_width,
            use_latent_context=use_latent_context,
            latent_dim=self.basis_dim
        ).to(self.device)

        # Load models (detection + OCR)
        self.load_yolo_model()

        # Track statistics
        self.epoch_stats = []

        print(f"\nRefineGeneratorTrainer initialized:")
        print(f"  Device: {self.device}")
        print(f"  Batch size: 1 (dataset has variable-sized images)")
        print(f"  Gradient accumulation: {grad_accumulate or 'disabled'}")
        print(f"  Generator type: {generator_type}")
        print(f"  SSIM weight: {ssim_weight}")
        print(f"  Match detection: {match_detection}")
        print(f"  Impersonation target: {impersonation_target or 'None (disruption mode)'}")
        print(f"  TV loss: {'Enabled' if use_tv_loss else 'Disabled'}")
        print(f"  Homography: {'Enabled' if use_homography else 'Disabled'}")

    def _load_frozen_generator(self, checkpoint_path: str, generator_type: str):
        """Load frozen generator from checkpoint"""
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Generator checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Extract basis_dim from checkpoint
        if 'basis_dim' in checkpoint:
            basis_dim = checkpoint['basis_dim']
        else:
            # Try to infer from state dict
            if generator_type == 'simple':
                # First layer weight shape is [hidden_dim, basis_dim]
                first_layer_key = 'network.0.weight'
                if first_layer_key in checkpoint['generator_state_dict']:
                    basis_dim = checkpoint['generator_state_dict'][first_layer_key].shape[1]
                else:
                    raise ValueError("Could not infer basis_dim from checkpoint")
            else:  # foundation
                # adapter first layer: [512, basis_dim]
                first_layer_key = 'adapter.0.weight'
                if first_layer_key in checkpoint['generator_state_dict']:
                    basis_dim = checkpoint['generator_state_dict'][first_layer_key].shape[1]
                else:
                    raise ValueError("Could not infer basis_dim from checkpoint")

        # Initialize generator
        if generator_type == 'simple':
            generator = SimplePatchGenerator(
                latent_dim=basis_dim,
                patch_height=self.patch_height,
                patch_width=self.patch_width
            )
        elif generator_type == 'foundation':
            generator = FoundationPatchGenerator(
                latent_dim=basis_dim,
                patch_height=self.patch_height,
                patch_width=self.patch_width
            )
        else:
            raise ValueError(f"Unknown generator type: {generator_type}")

        # Load state dict
        generator.load_state_dict(checkpoint['generator_state_dict'])
        generator.to(self.device)

        print(f"Loaded {generator_type} generator with basis_dim={basis_dim}")
        if 'epoch' in checkpoint:
            print(f"  Checkpoint from epoch: {checkpoint['epoch']}")

        return generator, basis_dim

    def load_yolo_model(self):
        """Load and convert YOLO detection + OCR models"""
        print("\nLoading YOLO detection model...")
        LicensePlateDetector(detection_model="yolo-v9-t-384-license-plate-end2end")

        # Get ONNX model paths
        model_cache_dir = Path.home() / ".cache/open-image-models/yolo-v9-t-384-license-plate-end2end"
        onnx_path = model_cache_dir / "yolo-v9-t-384-license-plates-end2end.onnx"
        ocr_path = Path.home() / ".cache/fast-plate-ocr/cct-xs-v1-global-model/cct_xs_v1_global.onnx"

        if not onnx_path.exists():
            raise FileNotFoundError(f"ONNX model not found at: {onnx_path}")

        # Load detection model
        onnx_model = onnx.load(str(onnx_path))
        self.model = onnx2torch.convert(onnx_model)
        self.model.to(self.device)
        self.model.eval()

        # Load OCR model
        print("Loading OCR model...")
        ocr_model = onnx.load(str(ocr_path))
        self.ocr_input_shape = (64, 128, 3)
        alphabet = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ_'

        # Set OCR target based on mode
        if self.impersonation_target:
            self.ocr_target = self.text_to_target_tensor(self.impersonation_target, 9, alphabet)
        else:
            self.ocr_target = self.text_to_target_tensor('VRJ7774', 9, alphabet)

        self.ocr = onnx2torch.convert(ocr_model).to(self.device)
        self.ocr_loss = self.focal_cce_loss(len(alphabet))
        self.detection_baseline, self.ocr_baseline = self.calculate_baseline_loss()
        self.ocr.eval()

        # Disable gradients for model parameters
        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.ocr.parameters():
            param.requires_grad = False

    def text_to_target_tensor(self, plate_text: str, max_slots: int, alphabet: str):
        """Convert 'ABC123' -> one-hot tensor [batch, seq_len, vocab_size]"""
        padded = (plate_text + '_' * max_slots)[:max_slots]
        indices = [alphabet.index(char) for char in padded]
        target = torch.zeros(1, max_slots, len(alphabet))
        for i, idx in enumerate(indices):
            target[0, i, idx] = 1.0
        return target.to(self.device)

    def focal_cce_loss(self, vocabulary_size: int, alpha: float = 0.25,
                       gamma: float = 2.0, label_smoothing: float = 0.01):
        """Categorical focal cross-entropy loss"""
        def cce(y_true, y_pred):
            y_true = y_true.reshape(-1, vocabulary_size)
            y_pred = y_pred.reshape(-1, vocabulary_size)

            if y_pred.max() > 1.0 or y_pred.min() < 0.0:
                y_pred = F.softmax(y_pred, dim=-1)

            if label_smoothing > 0.0:
                y_true = y_true * (1.0 - label_smoothing) + label_smoothing / vocabulary_size

            p_t = torch.sum(y_true * y_pred, dim=-1)
            focal_weight = (1.0 - p_t) ** gamma
            ce_loss = -torch.log(p_t + 1e-8)
            focal_loss = alpha * focal_weight * ce_loss

            return torch.mean(focal_loss)

        return cce

    def _apply_patch_simple(self, image: torch.Tensor, corners: torch.Tensor,
                            patch_normalized: torch.Tensor, border_scale: float = 1.4) \
            -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Apply patch as simple rectangular overlay without homography transformation"""
        batch_size = image.shape[0]
        image_height, image_width = image.shape[2], image.shape[3]

        plate_corners = corners[0]
        center_x = plate_corners[:, 0].mean()
        center_y = plate_corners[:, 1].mean()
        center = torch.tensor([center_x, center_y], device=self.device)

        border_corners = center.unsqueeze(0) + (plate_corners - center.unsqueeze(0)) * border_scale

        border_min_x = torch.clamp(torch.min(border_corners[:, 0]), 0, image_width).int()
        border_max_x = torch.clamp(torch.max(border_corners[:, 0]), 0, image_width).int()
        border_min_y = torch.clamp(torch.min(border_corners[:, 1]), 0, image_height).int()
        border_max_y = torch.clamp(torch.max(border_corners[:, 1]), 0, image_height).int()

        plate_min_x = torch.clamp(torch.min(plate_corners[:, 0]), 0, image_width).int()
        plate_max_x = torch.clamp(torch.max(plate_corners[:, 0]), 0, image_width).int()
        plate_min_y = torch.clamp(torch.min(plate_corners[:, 1]), 0, image_height).int()
        plate_max_y = torch.clamp(torch.max(plate_corners[:, 1]), 0, image_height).int()

        result_image = image.clone()
        final_mask = torch.zeros(batch_size, 3, image_height, image_width,
                                device=self.device, dtype=torch.float32)

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
                result_image[b, :, border_min_y:border_max_y, border_min_x:border_max_x] = patch_resized[0]
                final_mask[b, :, border_min_y:border_max_y, border_min_x:border_max_x] = 1.0

                if plate_max_y > plate_min_y and plate_max_x > plate_min_x:
                    result_image[b, :, plate_min_y:plate_max_y, plate_min_x:plate_max_x] = \
                        image[b, :, plate_min_y:plate_max_y, plate_min_x:plate_max_x]
                    final_mask[b, :, plate_min_y:plate_max_y, plate_min_x:plate_max_x] = 0.0

        result_image = torch.clamp(result_image, 0, 1)
        return result_image, final_mask

    def get_patch_bounding_box(self, corners: torch.Tensor, border_scale: float = 1.4) -> torch.Tensor:
        """Calculate bounding box of the patch area"""
        plate_corners = corners
        center_x = plate_corners[:, 0].mean()
        center_y = plate_corners[:, 1].mean()
        center = torch.tensor([center_x, center_y], device=self.device)

        border_corners = center.unsqueeze(0) + (plate_corners - center.unsqueeze(0)) * border_scale

        min_x = torch.min(border_corners[:, 0])
        max_x = torch.max(border_corners[:, 0])
        min_y = torch.min(border_corners[:, 1])
        max_y = torch.max(border_corners[:, 1])

        return torch.stack([min_x, min_y, max_x, max_y])

    def apply_patch_to_image(self, image: torch.Tensor, corners: torch.Tensor,
                             patch: torch.Tensor, border_scale: float = 1.4) \
            -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Apply adversarial patch as border around license plate"""
        batch_size = image.shape[0]
        image_height, image_width = image.shape[2], image.shape[3]
        dsize = (image_height, image_width)

        # Patch is already in [0, 1] range from refinement network
        patch_normalized = patch

        # Apply blur if specified
        if self.print_blur > 0:
            patch_normalized = kornia.filters.gaussian_blur2d(
                patch_normalized.unsqueeze(0),
                kernel_size=(3, 3),
                sigma=(self.print_blur, self.print_blur)
            ).squeeze(0)

        # Random darkening during training
        if self.training:
            darkening_factor = torch.rand(1, device=self.device) * 0.2
            patch_normalized = patch_normalized * (1.0 - darkening_factor)

        # Use simple overlay if homography disabled
        if not self.use_homography:
            return self._apply_patch_simple(image, corners, patch_normalized, border_scale)

        # Homography-based application
        plate_corners = corners[0]
        center_x = plate_corners[:, 0].mean()
        center_y = plate_corners[:, 1].mean()
        center = torch.tensor([center_x, center_y], device=self.device)

        border_corners = center.unsqueeze(0) + (plate_corners - center.unsqueeze(0)) * border_scale
        border_corners = border_corners.unsqueeze(0)

        patch_h, patch_w = self.patch_height, self.patch_width
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

        patch_mask = torch.ones(batch_size, 1, self.patch_height, self.patch_width,
                                dtype=torch.float32, device=self.device)

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
        result_image = torch.clamp(result_image, 0, 1)

        return result_image, final_mask

    def invert_bbox(self, corners, transform):
        """Invert transformation to bring corners back to original image"""
        r, dw, dh = transform
        corners = corners.clone()
        corners[::2] = corners[::2] - dw
        corners[1::2] = corners[1::2] - dh
        corners = corners / r
        return corners

    def bbox_to_corners(self, bbox, device=None):
        """Convert bbox to corner format"""
        x1, y1, x2, y2 = bbox
        corners = torch.tensor([[
            [x1, y1], [x2, y1], [x2, y2], [x1, y2]
        ]], device=device or self.device)
        return corners

    def corners_to_bbox(self, corners):
        """Convert corners to bbox format"""
        min_x = torch.min(corners[:, 0])
        max_x = torch.max(corners[:, 0])
        min_y = torch.min(corners[:, 1])
        max_y = torch.max(corners[:, 1])
        return torch.stack([min_x, min_y, max_x, max_y])

    def boxes_IoU(self, boxes1, boxes2):
        """Calculate IoU between boxes"""
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

    def partial_loss(self, batch, refined_patch, use_ocr_baseline=True):
        """Compute detection and OCR losses for refined patch"""
        prep_image = batch['prep_image'].to(self.device)
        corners = batch['new_corners'].to(self.device)

        if self.match_detection:
            target_box = self.get_patch_bounding_box(corners)
        else:
            target_box = self.corners_to_bbox(corners)

        # Apply refined patch to image
        patched_prep, _ = self.apply_patch_to_image(
            prep_image.unsqueeze(0), corners.unsqueeze(0), refined_patch
        )

        # Run YOLO detection
        model_output = self.model(patched_prep)

        best_detection = None
        det_loss = 0.0

        for detection in model_output:
            pred_box = detection[1:5]
            conf = detection[6]
            IoU = self.boxes_IoU(pred_box.unsqueeze(0), target_box.unsqueeze(0))

            if self.match_detection:
                this_det_loss = -IoU * conf
            else:
                this_det_loss = IoU * conf

            if self.match_detection:
                if -this_det_loss > -det_loss:
                    det_loss = this_det_loss
                    best_detection = detection
            else:
                if this_det_loss > det_loss:
                    det_loss = this_det_loss
                    best_detection = detection

        ocr_loss = 0.0
        if best_detection is not None:
            pred_box = best_detection[1:5]
            orig_projection = self.invert_bbox(pred_box.to('cpu'), batch['transform'])
            corners_box = self.bbox_to_corners(orig_projection, device='cpu')

            # Apply patch to original image for OCR
            patched_orig, _ = self.apply_patch_to_image(
                batch['orig_image'].to(self.device).unsqueeze(0),
                batch['orig_corners'].to(self.device).unsqueeze(0),
                refined_patch
            )

            cropped_plate = kornia.geometry.crop_and_resize(
                patched_orig,
                corners_box,
                self.ocr_input_shape[:2],
                mode='bilinear',
                align_corners=True
            ).to(self.device)

            ocr_input = cropped_plate.permute(0, 2, 3, 1) * 255
            ocr_output = self.ocr(ocr_input)
            ocr_loss = self.ocr_loss(self.ocr_target, ocr_output)

            if use_ocr_baseline:
                if not hasattr(self, 'ocr_baseline'):
                    raise ValueError('Must call calculate_baseline_loss before using!')

                if self.impersonation_target:
                    ocr_loss = ocr_loss / self.ocr_baseline
                else:
                    ocr_loss = self.ocr_baseline / ocr_loss

        return det_loss, ocr_loss

    def patch_reg_loss(self, patch: torch.Tensor):
        """Compute total variation regularization loss for patch"""
        C, H, W = patch.shape

        tv_h = torch.pow(patch[:, :, 1:] - patch[:, :, :-1], 2).sum()
        tv_v = torch.pow(patch[:, 1:, :] - patch[:, :-1, :], 2).sum()

        num_comparisons = C * (H * (W - 1) + (H - 1) * W)
        loss = (tv_h + tv_v) / num_comparisons
        loss = loss * 2.5

        return loss

    def calculate_baseline_loss(self) -> Tuple[float, float]:
        """Calculate baseline OCR loss across dataset"""
        total_ocr_loss = 0.0
        total_det_loss = 0.0
        total_plates = 0

        desc = "Calculating baseline loss"
        with tqdm(self.train_loader, desc=desc, leave=False) as pbar:
            with torch.no_grad():
                for batch in pbar:
                    batch = {k: v[0] for k, v in batch.items()}

                    # Use a dummy white patch for baseline
                    dummy_patch = torch.ones(3, self.patch_height, self.patch_width,
                                            device=self.device) * 0.5

                    det_loss, ocr_loss = self.partial_loss(batch, dummy_patch,
                                                          use_ocr_baseline=False)
                    total_det_loss += det_loss
                    total_ocr_loss += ocr_loss
                    total_plates += 1

                    avg_ocr = total_ocr_loss / total_plates
                    avg_det = total_det_loss / total_plates
                    pbar.set_postfix({
                        'Avg_Detection_Loss': f'{avg_det.item():.4f}',
                        'Avg_OCR_Loss': f'{avg_ocr.item():.4f}'
                    })

        return total_det_loss / total_plates, total_ocr_loss / total_plates

    def create_border_mask(self, height: int, width: int, border_scale: float = 1.4) -> torch.Tensor:
        """
        Create a mask for the visible border region of the patch.

        When applied, the patch fills a border_scale region around the plate,
        but the center plate region gets cut out. This mask identifies which
        parts of the patch are actually visible (border) vs obscured (center).

        Args:
            height: Patch height
            width: Patch width
            border_scale: Scale factor for border (default 1.4)

        Returns:
            mask: [1, 1, H, W] tensor, 1 = visible border, 0 = obscured center
        """
        plate_scale_ratio = 1.0 / border_scale

        center_h = int(height * plate_scale_ratio)
        center_w = int(width * plate_scale_ratio)

        mask = torch.ones(1, 1, height, width)

        h_start = (height - center_h) // 2
        h_end = h_start + center_h
        w_start = (width - center_w) // 2
        w_end = w_start + center_w

        mask[:, :, h_start:h_end, w_start:w_end] = 0.0

        return mask

    def compute_ssim_loss(self, base_patch: torch.Tensor, refined_patch: torch.Tensor) -> torch.Tensor:
        """
        Compute SSIM loss between base and refined patches.

        Only compares the visible border region (excludes center plate region
        that gets obscured when patch is applied).

        Args:
            base_patch: [3, H, W] or [1, 3, H, W] base patch from generator
            refined_patch: [3, H, W] or [1, 3, H, W] refined patch

        Returns:
            torch.Tensor: Scalar SSIM loss (1 - SSIM, lower is better)
        """
        # Ensure 4D input [B, C, H, W]
        base = base_patch.unsqueeze(0) if base_patch.dim() == 3 else base_patch
        refined = refined_patch.unsqueeze(0) if refined_patch.dim() == 3 else refined_patch

        # Create border mask (only score visible border, ignore obscured center)
        _, _, H, W = base.shape
        border_mask = self.create_border_mask(H, W, border_scale=1.4).to(base.device)

        # Compute SSIM map (returns [B, C, H, W] with per-pixel SSIM values)
        ssim_map = ssim(refined, base, window_size=11)  # [1, 3, H, W]

        # Average across channels
        ssim_map_avg = ssim_map.mean(dim=1, keepdim=True)  # [1, 1, H, W]

        # Apply border mask: only consider visible regions
        masked_ssim = ssim_map_avg * border_mask  # [1, 1, H, W]

        # Compute mean only over visible border region
        num_visible_pixels = border_mask.sum()
        ssim_value = masked_ssim.sum() / num_visible_pixels if num_visible_pixels > 0 else torch.tensor(0.0, device=base.device)

        # We want to maximize SSIM (patches should be similar), so minimize (1 - SSIM)
        ssim_loss = 1.0 - ssim_value

        return ssim_loss

    def compute_loss_full(self, batch: dict, use_ocr_baseline=True) -> Tuple[torch.Tensor, dict]:
        """Compute full loss including SSIM, detection, OCR, and TV"""
        batch = {k: v[0] for k, v in batch.items()}

        # Sample random latent
        z = torch.randn(1, self.basis_dim, device=self.device)

        # Generate base patch from frozen generator (no grad)
        with torch.no_grad():
            base_patch = self.generator(z).squeeze(0)  # [3, H, W]

        # Refine patch (with grad)
        refined_patch = self.refinement_net(
            base_patch.unsqueeze(0),
            z if self.use_latent_context else None
        ).squeeze(0)  # [3, H, W]

        # Compute SSIM loss
        ssim_loss = self.compute_ssim_loss(base_patch, refined_patch)

        # Compute detection and OCR losses
        det_loss, ocr_loss = self.partial_loss(batch, refined_patch, use_ocr_baseline)

        # Compute TV regularization
        if self.use_tv_loss:
            reg_loss = self.patch_reg_loss(refined_patch)
        else:
            reg_loss = 0.0

        # Combine losses
        attack_loss = (det_loss + ocr_loss) / 2
        total_loss = self.ssim_weight * ssim_loss + attack_loss + reg_loss

        # Return loss breakdown for logging
        loss_breakdown = {
            'total': total_loss.item(),
            'ssim': ssim_loss.item(),
            'detection': det_loss.item(),
            'ocr': ocr_loss.item(),
            'tv': reg_loss.item() if isinstance(reg_loss, torch.Tensor) else reg_loss,
            'attack': attack_loss.item()
        }

        return total_loss, loss_breakdown

    def train_epoch(self, optimizer: torch.optim.Optimizer, epoch: int) -> dict:
        """Train for one epoch with gradient accumulation"""
        self.refinement_net.train()

        total_losses = {
            'total': 0.0, 'ssim': 0.0, 'detection': 0.0,
            'ocr': 0.0, 'tv': 0.0, 'attack': 0.0
        }
        step_count = 0
        num_updates = 0

        update_every = len(self.train_loader) if self.grad_accumulate is None else self.grad_accumulate
        effective_batch_size = update_every

        desc = f"Epoch {epoch+1} - Training (AccumSteps={update_every})"
        with tqdm(enumerate(self.train_loader), desc=desc, leave=False,
                  total=len(self.train_loader)) as pbar:

            for idx, batch in pbar:
                loss, loss_breakdown = self.compute_loss_full(batch)
                scaled_loss = loss / effective_batch_size

                scaled_loss.backward()

                for key in total_losses:
                    total_losses[key] += loss_breakdown[key]
                step_count += 1

                if step_count % update_every == 0:
                    # Clip gradients
                    torch.nn.utils.clip_grad_norm_(self.refinement_net.parameters(), max_norm=1.0)
                    optimizer.step()
                    optimizer.zero_grad()

                    num_updates += 1

                    # Memory cleanup
                    del loss, scaled_loss
                    if self.device == 'cuda':
                        torch.cuda.empty_cache()
                    elif self.device == 'mps':
                        torch.mps.empty_cache()

                    # Update progress
                    avg_total = total_losses['total'] / (num_updates * update_every)
                    avg_ssim = total_losses['ssim'] / (num_updates * update_every)
                    avg_attack = total_losses['attack'] / (num_updates * update_every)
                    pbar.set_postfix({
                        'Loss': f"{avg_total:.4f}",
                        'SSIM': f"{avg_ssim:.4f}",
                        'Attack': f"{avg_attack:.4f}",
                        'Updates': num_updates
                    })
                else:
                    del loss, scaled_loss

            # Handle remaining accumulated gradients
            if step_count % update_every != 0 and self.grad_accumulate is not None:
                torch.nn.utils.clip_grad_norm_(self.refinement_net.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
                num_updates += 1

                if self.device == 'cuda':
                    torch.cuda.empty_cache()
                elif self.device == 'mps':
                    torch.mps.empty_cache()

        # Return average losses
        total_batches = num_updates * update_every if self.grad_accumulate else len(self.train_loader)
        return {key: val / total_batches for key, val in total_losses.items()}

    def validate(self) -> dict:
        """Validation pass on held-out data"""
        # Skip validation if using all data for training
        if self.use_all_for_train or len(self.val_loader) == 0:
            return {key: 0.0 for key in ['total', 'ssim', 'detection', 'ocr', 'tv', 'attack']}

        self.refinement_net.eval()

        total_losses = {
            'total': 0.0, 'ssim': 0.0, 'detection': 0.0,
            'ocr': 0.0, 'tv': 0.0, 'attack': 0.0
        }

        with torch.no_grad():
            for batch in self.val_loader:
                loss, loss_breakdown = self.compute_loss_full(batch)
                for key in total_losses:
                    total_losses[key] += loss_breakdown[key]

        num_batches = len(self.val_loader)
        return {key: val / num_batches for key, val in total_losses.items()}

    def save_checkpoint(self, epoch: int, save_dir: str = "refined_patches"):
        """Save refinement network checkpoint"""
        Path(save_dir).mkdir(exist_ok=True)

        checkpoint = {
            'epoch': epoch,
            'refinement_state_dict': self.refinement_net.state_dict(),
            'basis_dim': self.basis_dim,
            'use_latent_context': self.use_latent_context,
            'ssim_weight': self.ssim_weight,
            'patch_size': (self.patch_height, self.patch_width)
        }

        torch.save(checkpoint, f"{save_dir}/refinement_epoch_{epoch:04d}.pt")

    def save_sample_patches(self, epoch: int, num_samples: int = 4,
                           save_dir: str = "refined_patches"):
        """Generate and save sample patches"""
        Path(save_dir).mkdir(exist_ok=True)

        self.refinement_net.eval()

        fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5 * num_samples))
        if num_samples == 1:
            axes = axes.reshape(1, -1)

        with torch.no_grad():
            for i in range(num_samples):
                z = torch.randn(1, self.basis_dim, device=self.device)

                # Generate base patch
                base_patch = self.generator(z).squeeze(0)

                # Refine patch
                refined_patch = self.refinement_net(
                    base_patch.unsqueeze(0),
                    z if self.use_latent_context else None
                ).squeeze(0)

                # Compute SSIM (using same method as training)
                ssim_loss = self.compute_ssim_loss(base_patch, refined_patch)
                # For display, show as similarity (1 - loss), higher is more similar
                ssim_value = (1.0 - ssim_loss).item()

                # Convert to numpy for visualization
                base_np = base_patch.cpu().permute(1, 2, 0).numpy()
                refined_np = refined_patch.cpu().permute(1, 2, 0).numpy()
                diff_np = np.abs(refined_np - base_np)

                # Plot
                axes[i, 0].imshow(base_np)
                axes[i, 0].set_title(f'Sample {i+1}: Base (Generator)')
                axes[i, 0].axis('off')

                axes[i, 1].imshow(refined_np)
                axes[i, 1].set_title(f'Refined (SSIM: {ssim_value:.4f})')
                axes[i, 1].axis('off')

                axes[i, 2].imshow(diff_np)
                axes[i, 2].set_title('Absolute Difference')
                axes[i, 2].axis('off')

        plt.tight_layout()
        plt.savefig(f"{save_dir}/samples_epoch_{epoch:04d}.png", dpi=150, bbox_inches='tight')
        plt.close()

    def train(self, num_epochs: int = 100, learning_rate: float = 0.001,
              save_interval: int = 10, early_stop_patience: int = 15,
              warmup_epochs: int = 5, lr_min: float = 1e-5):
        """Main training loop"""

        optimizer = optim.AdamW(self.refinement_net.parameters(), lr=learning_rate, weight_decay=1e-4)

        # Create learning rate scheduler with warmup + cosine annealing
        if warmup_epochs > 0:
            # Warmup from lr_min to learning_rate, then cosine anneal to lr_min
            warmup_scheduler = optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=lr_min / learning_rate,
                end_factor=1.0,
                total_iters=warmup_epochs
            )
            cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=num_epochs - warmup_epochs,
                eta_min=lr_min
            )
            scheduler = optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, cosine_scheduler],
                milestones=[warmup_epochs]
            )
        else:
            # No warmup: start directly at learning_rate, cosine down to lr_min
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=num_epochs,
                eta_min=lr_min
            )

        history = {
            'train_total': [], 'train_ssim': [], 'train_attack': [],
            'val_total': [], 'val_ssim': [], 'val_attack': [],
            'learning_rate': []
        }

        best_loss = float('inf')
        patience_counter = 0

        print("\nStarting refinement network training")
        if self.use_all_for_train:
            print(f"   Dataset: {len(self.train_loader)} images (all for training, no validation)")
        else:
            print(f"   Dataset: {len(self.train_loader) + len(self.val_loader)} images "
                  f"({len(self.train_loader)} train, {len(self.val_loader)} val)")
        print(f"   Device: {self.device}")
        print(f"   Epochs: {num_epochs}")
        print(f"   LR: {learning_rate} (warmup {warmup_epochs} epochs, min {lr_min})")
        print(f"   SSIM weight: {self.ssim_weight}")
        print("-" * 60)

        for epoch in range(num_epochs):
            # Training
            train_losses = self.train_epoch(optimizer, epoch)
            val_losses = self.validate() if not self.use_all_for_train else train_losses

            # Learning rate scheduling (step after each epoch)
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']

            # Record history
            history['train_total'].append(train_losses['total'])
            history['train_ssim'].append(train_losses['ssim'])
            history['train_attack'].append(train_losses['attack'])
            history['val_total'].append(val_losses['total'])
            history['val_ssim'].append(val_losses['ssim'])
            history['val_attack'].append(val_losses['attack'])
            history['learning_rate'].append(current_lr)

            # Print epoch summary
            if self.use_all_for_train:
                print(f"Epoch {epoch+1:3d}/{num_epochs} | "
                      f"Loss: {train_losses['total']:.4f} "
                      f"(SSIM: {train_losses['ssim']:.4f}, Attack: {train_losses['attack']:.4f}) | "
                      f"LR: {current_lr:.2e}")
            else:
                print(f"Epoch {epoch+1:3d}/{num_epochs} | "
                      f"Train: {train_losses['total']:.4f} "
                      f"(SSIM: {train_losses['ssim']:.4f}, Attack: {train_losses['attack']:.4f}) | "
                      f"Val: {val_losses['total']:.4f} "
                      f"(SSIM: {val_losses['ssim']:.4f}, Attack: {val_losses['attack']:.4f}) | "
                      f"LR: {current_lr:.2e}")

            # Save best model (use training loss if no validation)
            loss_for_best = train_losses['total'] if self.use_all_for_train else val_losses['total']
            if loss_for_best < best_loss:
                best_loss = loss_for_best
                patience_counter = 0
                self.save_checkpoint(epoch, "best_refined_patches")
                self.save_sample_patches(epoch, num_samples=4, save_dir="best_refined_patches")
                print(f"   New best loss: {best_loss:.4f}")
            else:
                patience_counter += 1

            # Periodic saves
            if (epoch + 1) % save_interval == 0:
                self.save_checkpoint(epoch, "checkpoint_refined_patches")
                self.save_sample_patches(epoch, num_samples=4, save_dir="checkpoint_refined_patches")

            # Early stopping
            if patience_counter >= early_stop_patience:
                print(f"   Early stopping: No improvement for {early_stop_patience} epochs")
                break

        # Save final checkpoint and samples (matching progressive_patch.py pattern)
        final_epoch = epoch  # Use last epoch number
        final_save_dir = "training_complete_final_refinement"
        self.save_checkpoint(final_epoch, final_save_dir)
        self.save_sample_patches(final_epoch, num_samples=8, save_dir=final_save_dir)

        print("\nTraining completed!")
        print(f"   Best loss: {best_loss:.4f}")
        print(f"\nCheckpoints and samples saved to:")
        print(f"   - best_refined_patches/: Best model across all training")
        print(f"   - checkpoint_refined_patches/: Periodic checkpoints (every {save_interval} epochs)")
        print(f"   - {final_save_dir}/: Final model after training completion")
        print(f"\nTraining history saved to: refinement_training_history.csv")

        return history


def main():
    parser = argparse.ArgumentParser(description='Generator Refinement Training')
    parser.add_argument('--generator-checkpoint', required=True,
                        help='Path to frozen generator checkpoint (.pt file from progressive_patch training)')
    parser.add_argument('--generator-type', choices=['simple', 'foundation'], default='simple',
                        help='Type of generator (simple or foundation)')
    parser.add_argument('--csv-path', default='preproc_labels.csv',
                        help='Path to dataset CSV')
    parser.add_argument('--device', default='cuda', choices=['cuda', 'mps', 'cpu'],
                        help='Device to use for training')
    parser.add_argument('--batch-size', type=int, default=1,
                        help='(Ignored - dataset has variable-sized images) '
                        'Batch size is always 1. Use --grad-accumulate instead to simulate larger batches '
                        'and control memory/performance tradeoffs.')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Peak learning rate after warmup (default: 0.001)')
    parser.add_argument('--lr-min', type=float, default=1e-5,
                        help='Minimum learning rate (initial and final, default: 1e-5). '
                        'Used as start of warmup and end of cosine annealing.')
    parser.add_argument('--warmup-epochs', type=int, default=5,
                        help='Number of epochs for linear warmup (default: 5)')
    parser.add_argument('--grad-accumulate', type=int, default=64,
                        help='Gradient accumulation steps')
    parser.add_argument('--ssim-weight', type=float, default=1.0,
                        help='Weight for SSIM loss (higher = stay closer to generator)')
    parser.add_argument('--use-latent-context', action='store_true',
                        help='Pass latent z to refinement network for context')
    parser.add_argument('--match-detection', action='store_true',
                        help='Maximize IoU with patch area instead of minimizing with ground truth')
    parser.add_argument('--impersonation-target', type=str, default=None,
                        help='Target plate text for impersonation (e.g., "ABC123")')
    parser.add_argument('--disable-tv-loss', action='store_true',
                        help='Disable total variation regularization')
    parser.add_argument('--disable-homography', action='store_true',
                        help='Disable homography-based patch application')
    parser.add_argument('--save-interval', type=int, default=10,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--early-stop-patience', type=int, default=15,
                        help='Early stopping patience (epochs)')
    parser.add_argument('--no-use-all-for-train', action='store_true',
                        help='Disable using all data for training (use 80%% train / 20%% validation split). '
                        'Default: uses 100%% of data for training, no validation.')

    args = parser.parse_args()

    try:
        trainer = RefineGeneratorTrainer(
            csv_path=args.csv_path,
            generator_checkpoint=args.generator_checkpoint,
            device=args.device,
            batch_size=args.batch_size,
            grad_accumulate=args.grad_accumulate,
            match_detection=args.match_detection,
            impersonation_target=args.impersonation_target,
            training=True,
            use_tv_loss=not args.disable_tv_loss,
            use_homography=not args.disable_homography,
            ssim_weight=args.ssim_weight,
            use_latent_context=args.use_latent_context,
            generator_type=args.generator_type,
            use_all_for_train=not args.no_use_all_for_train
        )

        history = trainer.train(
            num_epochs=args.epochs,
            learning_rate=args.lr,
            save_interval=args.save_interval,
            early_stop_patience=args.early_stop_patience,
            warmup_epochs=args.warmup_epochs,
            lr_min=args.lr_min
        )

        # Save training history
        import pandas as pd
        history_df = pd.DataFrame(history)
        history_df.insert(0, 'epoch', range(1, len(history_df) + 1))
        history_df.to_csv('refinement_training_history.csv', index=False)
        print(f"\nTraining history saved to: refinement_training_history.csv")

        # Plot results
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Total loss
        axes[0, 0].plot(history['train_total'], 'b-', label='Train')
        axes[0, 0].plot(history['val_total'], 'r-', label='Val')
        axes[0, 0].set_title('Total Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # SSIM loss
        axes[0, 1].plot(history['train_ssim'], 'b-', label='Train')
        axes[0, 1].plot(history['val_ssim'], 'r-', label='Val')
        axes[0, 1].set_title('SSIM Loss (1 - SSIM)')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Attack loss
        axes[1, 0].plot(history['train_attack'], 'b-', label='Train')
        axes[1, 0].plot(history['val_attack'], 'r-', label='Val')
        axes[1, 0].set_title('Attack Loss (Det + OCR)')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # Learning rate
        axes[1, 1].semilogy(history['learning_rate'], 'purple')
        axes[1, 1].set_title('Learning Rate Schedule')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Learning Rate (log scale)')
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('refinement_training_results.png', dpi=300, bbox_inches='tight')

        # Print summary of all outputs
        print("\n" + "="*70)
        print("TRAINING COMPLETE - OUTPUT SUMMARY")
        print("="*70)
        print("\nCheckpoints and Visualizations:")
        print("  - best_refined_patches/: Best model across all training")
        print("    └─ refinement_epoch_*.pt: Best model checkpoint")
        print("    └─ samples_epoch_*.png: Comparison of base vs. refined patches")
        print("  - checkpoint_refined_patches/: Periodic checkpoints")
        print("    └─ refinement_epoch_*.pt: Checkpoints every N epochs")
        print("    └─ samples_epoch_*.png: Sample visualizations at checkpoints")
        print("  - training_complete_final_refinement/: Final model after training")
        print("    └─ refinement_epoch_*.pt: Final model checkpoint")
        print("    └─ samples_epoch_*.png: 8 final sample comparisons")
        print("\nTraining Curves:")
        print("  - refinement_training_results.png: 4-panel plot (loss, SSIM, attack, LR)")
        print("  - refinement_training_history.csv: Epoch-by-epoch loss metrics")
        print("="*70 + "\n")

    except Exception as e:
        print(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    main()
