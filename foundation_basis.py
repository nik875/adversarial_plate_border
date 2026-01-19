#!/usr/bin/env python3
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
import torchvision.transforms as T
import matplotlib.pyplot as plt
from matplotlib import patches
import onnx
import onnx2torch
from open_image_models import LicensePlateDetector
from dataset import create_dataloaders
from diffusers import AutoencoderKL
warnings.filterwarnings("ignore")


PATCH_WIDTH = 512
PATCH_HEIGHT = 256


class FoundationPatchGenerator(nn.Module):
    """Patch generator using Stable Diffusion VAE decoder with trainable adapter and CNN refinement"""
    def __init__(self, latent_dim: int, patch_height: int = 256, patch_width: int = 512):
        super().__init__()

        self.latent_dim = latent_dim
        self.patch_height = patch_height
        self.patch_width = patch_width

        # SD VAE expects latents of shape [B, 4, H/8, W/8]
        # For 256×512 output, latent is [B, 4, 32, 64]
        self.vae_latent_h = patch_height // 8
        self.vae_latent_w = patch_width // 8
        self.vae_latent_channels = 4
        self.vae_latent_dim = self.vae_latent_channels * self.vae_latent_h * self.vae_latent_w

        # Load SD VAE decoder (trainable)
        print("Loading Stable Diffusion VAE decoder...")
        self.vae = AutoencoderKL.from_pretrained(
            "madebyollin/sdxl-vae-fp16-fix",  # Smaller, optimized VAE
            torch_dtype=torch.float32
        )

        # VAE parameters are trainable (fine-tune pretrained weights)
        self.vae.train()

        print(f"VAE loaded (trainable). Latent space: [{self.vae_latent_channels}, {self.vae_latent_h}, {self.vae_latent_w}]")

        # Trainable adapter: z → VAE latent space
        # Using deeper network for better expressiveness
        self.adapter = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 2048),
            nn.LayerNorm(2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, self.vae_latent_dim),
        )

        # Initialize adapter weights
        for m in self.adapter.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        # Skip connection projection: z → spatial feature map
        # Project z to a feature map that can be concatenated with VAE output
        self.skip_projection = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, patch_height * patch_width),
            nn.ReLU(inplace=True)
        )

        # CNN refinement module - deeper conv architecture
        # Input: VAE output (3 channels) + skip connection (1 channel) = 4 channels
        # Output: Feature maps (64 channels)
        self.cnn_refiner = nn.Sequential(
            # Block 1
            nn.Conv2d(4, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            # Block 3
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # Initialize CNN weights
        for m in self.cnn_refiner.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # DNN block: combines CNN output (64 channels) + skip features (1 channel) = 65 channels
        # Uses global average pooling to reduce spatial dimensions before dense layers
        self.global_pool = nn.AdaptiveAvgPool2d((8, 16))  # Downsample to 8x16 spatial
        self.dnn_input_dim = 65 * 8 * 16  # 8320

        self.dnn_block = nn.Sequential(
            nn.Linear(self.dnn_input_dim, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 3 * patch_height * patch_width),
            nn.Sigmoid()  # Ensure output is in [0, 1]
        )

        # Initialize DNN weights
        for m in self.dnn_block.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        print(f"CNN refiner initialized: 4 → 64 → 64 → 128 → 128 → 64 → 64 channels")
        print(f"DNN block (with global pooling): {self.dnn_input_dim} → 2048 → 2048 → 1024 → {3 * patch_height * patch_width}")

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: [batch_size, latent_dim]
        Returns:
            patches: [batch_size, 3, patch_height, patch_width]
        """
        batch_size = z.shape[0]

        # Main path: z → adapter → VAE decoder (frozen)
        # Adapter: z → VAE latent
        vae_latent_flat = self.adapter(z)  # [B, vae_latent_dim]

        # Reshape to VAE's expected latent format
        vae_latent = vae_latent_flat.view(
            batch_size,
            self.vae_latent_channels,
            self.vae_latent_h,
            self.vae_latent_w
        )  # [B, 4, 32, 64]

        # Decode using trainable VAE (gradients flow through for fine-tuning)
        vae_output = self.vae.decode(vae_latent).sample  # [B, 3, 256, 512]

        # Clamp to [0, 1] and ensure correct size
        vae_output = torch.clamp(vae_output, 0.0, 1.0)

        # Ensure exact output size (in case VAE produces slightly different dimensions)
        if vae_output.shape[2] != self.patch_height or vae_output.shape[3] != self.patch_width:
            vae_output = F.interpolate(
                vae_output,
                size=(self.patch_height, self.patch_width),
                mode='bilinear',
                align_corners=True
            )

        # Skip connection: z → spatial feature map
        skip_features = self.skip_projection(z)  # [B, H*W]
        skip_features = skip_features.view(
            batch_size, 1, self.patch_height, self.patch_width
        )  # [B, 1, H, W]

        # Concatenate VAE output with skip connection for CNN input
        cnn_input = torch.cat([vae_output, skip_features], dim=1)  # [B, 4, H, W]

        # Process through CNN refiner
        cnn_output = self.cnn_refiner(cnn_input)  # [B, 64, H, W]

        # Concatenate CNN output with skip features for DNN input
        dnn_input = torch.cat([cnn_output, skip_features], dim=1)  # [B, 65, H, W]

        # Apply global pooling to reduce spatial dimensions
        dnn_input_pooled = self.global_pool(dnn_input)  # [B, 65, 8, 16]
        dnn_input_flat = dnn_input_pooled.view(batch_size, -1)  # [B, 65*8*16]

        # Process through DNN block
        patch_flat = self.dnn_block(dnn_input_flat)  # [B, 3*H*W]
        patches = patch_flat.view(batch_size, 3, self.patch_height, self.patch_width)  # [B, 3, H, W]

        return patches


class FoundationBasisPatchTrainer:
    def __init__(self,
                 csv_path: str,
                 device: str = None,
                 grad_accumulate: int = None,
                 match_detection: bool = False,
                 impersonation_target: str = None,
                 print_blur=0,
                 training=False,
                 use_tv_loss: bool = True,
                 use_homography: bool = True,
                 basis_dim: int = 16,
                 diversity_weight: float = 1.0):
        self.training = training
        self.print_blur = print_blur
        self.use_tv_loss = use_tv_loss
        self.use_homography = use_homography
        self.basis_dim = basis_dim
        self.diversity_weight = diversity_weight

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

        self.train_loader, self.val_loader = create_dataloaders(csv_path, transform=self.transform,
                                                                preload=True, batch_size=1,
                                                                n_jobs=0)

        # Initialize foundation model generator with trainable VAE
        # Create generator with trainable adapter + trainable SD VAE decoder
        self.generator = FoundationPatchGenerator(
            latent_dim=basis_dim,
            patch_height=self.patch_height,
            patch_width=self.patch_width
        ).to(self.device)

        # Load YOLO model
        self.load_yolo_model()

        # Track statistics
        self.epoch_stats = []

    def load_yolo_model(self):
        """Load and convert YOLO model to PyTorch"""
        print("Loading YOLO model...")
        LicensePlateDetector(detection_model="yolo-v9-t-384-license-plate-end2end")

        # Get ONNX model path
        model_cache_dir = \
            Path.home() / ".cache/open-image-models/yolo-v9-t-384-license-plate-end2end"
        onnx_path = model_cache_dir / "yolo-v9-t-384-license-plates-end2end.onnx"
        ocr_path = \
            Path.home() / ".cache/fast-plate-ocr/cct-xs-v1-global-model/cct_xs_v1_global.onnx"

        if not onnx_path.exists():
            raise FileNotFoundError(f"ONNX model not found at: {onnx_path}")

        onnx_model = onnx.load(str(onnx_path))
        self.model = onnx2torch.convert(onnx_model)
        self.model.to(self.device)
        self.model.eval()

        ocr_model = onnx.load(str(ocr_path))
        self.ocr_input_shape = (64, 128, 3)
        alphabet = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ_'

        # Set OCR target based on mode
        if self.impersonation_target:
            # For impersonation: target is the text we want OCR to read
            self.ocr_target = self.text_to_target_tensor(self.impersonation_target, 9, alphabet)
        else:
            # Original behavior: target is the text we want to prevent OCR from reading
            self.ocr_target = self.text_to_target_tensor('VRJ7774', 9, alphabet)

        self.ocr = onnx2torch.convert(ocr_model).to(self.device)
        self.ocr_loss = self.focal_cce_loss(len(alphabet))
        self.detection_baseline, self.ocr_baseline = self.calculate_baseline_loss()
        self.ocr.eval()

        # Disable gradients for model parameters to save memory
        for param in self.model.parameters():
            param.requires_grad = False

    def sample_coefficients(self, batch_size: int) -> torch.Tensor:
        """Sample z ~ N(0, I) for generating patches"""
        return torch.randn(batch_size, self.basis_dim, device=self.device)

    def generate_patches(self, z: torch.Tensor) -> torch.Tensor:
        """
        Generate patches from latent codes using foundation model:
        z → adapter → frozen VAE decoder → CNN refiner → DNN block (with pooling) → patch

        Args:
            z: Latent codes [batch_size, basis_dim]

        Returns:
            patches: [batch_size, 3, H, W] in [0, 1]
        """
        # Pass through full generator pipeline
        patches = self.generator(z)  # [batch_size, 3, H, W], already in [0, 1]

        return patches

    def compute_diversity_loss(self, patches: torch.Tensor) -> torch.Tensor:
        """
        Compute diversity loss via log determinant of Gram matrix
        
        Applies jitter → blur → downsample to prevent gaming the metric with
        pixel-level variations while preserving high-level structure.

        Args:
            patches: [batch_size, 3, H, W]

        Returns:
            log_det: scalar diversity score
        """
        batch_size = patches.shape[0]

        # Apply random jitter (±2 pixels) to each patch independently
        jittered_patches = []
        for i in range(batch_size):
            jitter_x = torch.randint(-2, 3, (1,), device=self.device).float()
            jitter_y = torch.randint(-2, 3, (1,), device=self.device).float()
            jittered = kornia.geometry.transform.translate(
                patches[i:i+1], 
                torch.tensor([[jitter_x.item(), jitter_y.item()]], device=self.device, dtype=torch.float32)
            )
            jittered_patches.append(jittered)
        
        patches_jittered = torch.cat(jittered_patches, dim=0)  # [batch_size, 3, H, W]
        
        # Apply Gaussian blur (sigma=4px) to remove high-frequency noise
        kernel_size = 17  # ~4 sigma
        patches_blurred = kornia.filters.gaussian_blur2d(
            patches_jittered, 
            (kernel_size, kernel_size), 
            (4.0, 4.0)
        )  # [batch_size, 3, H, W]

        # Downsample from 512x256 to 32x64
        downsampled = F.interpolate(
            patches_blurred,
            size=(32, 64),  # (H, W)
            mode='bilinear',
            align_corners=True
        )  # [batch_size, 3, 32, 64]

        # Mask out center region (covered by license plate)
        # The patch at full resolution represents the border region at border_scale=1.4
        # So the plate region is 1/border_scale of the patch dimensions
        border_scale = 1.4
        plate_h = int(downsampled.shape[2] / border_scale)  # Height of plate in downsampled space
        plate_w = int(downsampled.shape[3] / border_scale)  # Width of plate in downsampled space
        
        # Center the plate region
        h_total = downsampled.shape[2]
        w_total = downsampled.shape[3]
        h_start = (h_total - plate_h) // 2
        h_end = h_start + plate_h
        w_start = (w_total - plate_w) // 2
        w_end = w_start + plate_w
        
        # Create mask (1 for regions to keep, 0 for plate center)
        mask = torch.ones(1, 1, h_total, w_total, device=self.device)
        mask[:, :, h_start:h_end, w_start:w_end] = 0.0
        
        # Apply mask to downsampled patches
        downsampled_masked = downsampled * mask  # [batch_size, 3, H, W]
        
        # Flatten and L2 normalize the masked patches
        flat = downsampled_masked.reshape(batch_size, -1)  # [batch_size, 3*32*64]
        normalized = F.normalize(flat, p=2, dim=1)  # [batch_size, d']

        # Compute Gram matrix
        gram = normalized @ normalized.t()  # [batch_size, batch_size]

        # Add larger epsilon for numerical stability
        # Use scale-dependent epsilon to handle different batch sizes
        epsilon = max(1e-6, 1e-2 / batch_size)
        gram = gram + epsilon * torch.eye(batch_size, device=self.device)

        # Use slogdet for numerical stability (returns sign and log|det|)
        sign, log_det = torch.slogdet(gram)

        # Handle numerical issues without clamping to zero
        # When patches are diverse: det is large → log_det is positive → diversity loss is negative (rewards)
        # When patches are similar: det ≈ 0 → log_det → -∞ → diversity loss is positive (penalizes)
        # We must preserve the sign of log_det for correct gradient direction!
        if torch.isnan(log_det):
            # NaN case: treat as singular matrix (similar patches)
            log_det = torch.tensor(-20.0, device=self.device, dtype=log_det.dtype)
        elif sign <= 0:
            # Singular or negative determinant: use large negative value
            # This will make diversity_loss positive, penalizing similarity
            log_det = torch.tensor(-20.0, device=self.device, dtype=log_det.dtype)
        # else: use log_det as-is (no abs!)

        return log_det

    def text_to_target_tensor(self, plate_text: str, max_slots: int, alphabet: str):
        """Convert 'ABC123' -> one-hot tensor [batch, seq_len, vocab_size]"""
        # Pad with '_' to max_slots
        padded = (plate_text + '_' * max_slots)[:max_slots]

        # Convert to indices
        indices = [alphabet.index(char) for char in padded]

        # One-hot encode
        target = torch.zeros(1, max_slots, len(alphabet))
        for i, idx in enumerate(indices):
            target[0, i, idx] = 1.0

        return target.to(self.device)

    def _apply_patch_simple(self, image: torch.Tensor, corners: torch.Tensor,
                            patch: torch.Tensor, border_scale: float = 1.4) \
            -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Apply patch as simple rectangular overlay without homography transformation"""
        batch_size = image.shape[0]
        image_height, image_width = image.shape[2], image.shape[3]

        # Get the 4 corners of the license plate
        plate_corners = corners[0]  # [4, 2]

        # Calculate center and create larger border box
        center_x = plate_corners[:, 0].mean()
        center_y = plate_corners[:, 1].mean()
        center = torch.tensor([center_x, center_y], device=self.device)

        border_corners = center.unsqueeze(0) + (plate_corners - center.unsqueeze(0)) * border_scale

        # Calculate bounding boxes for border and plate
        border_min_x = torch.clamp(torch.min(border_corners[:, 0]), 0, image_width).int()
        border_max_x = torch.clamp(torch.max(border_corners[:, 0]), 0, image_width).int()
        border_min_y = torch.clamp(torch.min(border_corners[:, 1]), 0, image_height).int()
        border_max_y = torch.clamp(torch.max(border_corners[:, 1]), 0, image_height).int()

        plate_min_x = torch.clamp(torch.min(plate_corners[:, 0]), 0, image_width).int()
        plate_max_x = torch.clamp(torch.max(plate_corners[:, 0]), 0, image_width).int()
        plate_min_y = torch.clamp(torch.min(plate_corners[:, 1]), 0, image_height).int()
        plate_max_y = torch.clamp(torch.max(plate_corners[:, 1]), 0, image_height).int()

        # Create result image and mask
        result_image = image.clone()
        final_mask = torch.zeros(batch_size, 3, image_height, image_width,
                                device=self.device, dtype=torch.float32)

        # Resize patch to border area
        border_h = border_max_y - border_min_y
        border_w = border_max_x - border_min_x

        if border_h > 0 and border_w > 0:
            # Resize patch to fit border area (patch is already [3, H, W])
            patch_resized = F.interpolate(
                patch.unsqueeze(0),
                size=(border_h, border_w),
                mode='bilinear',
                align_corners=True
            )

            # Apply to all batches
            for b in range(batch_size):
                # Fill border area with patch
                result_image[b, :, border_min_y:border_max_y, border_min_x:border_max_x] = patch_resized[0]
                final_mask[b, :, border_min_y:border_max_y, border_min_x:border_max_x] = 1.0

                # Cut out plate area (restore original image)
                if plate_max_y > plate_min_y and plate_max_x > plate_min_x:
                    result_image[b, :, plate_min_y:plate_max_y, plate_min_x:plate_max_x] = \
                        image[b, :, plate_min_y:plate_max_y, plate_min_x:plate_max_x]
                    final_mask[b, :, plate_min_y:plate_max_y, plate_min_x:plate_max_x] = 0.0

        result_image = torch.clamp(result_image, 0, 1)

        return result_image, final_mask

    def get_patch_bounding_box(self, corners: torch.Tensor,
                               border_scale: float = 1.4) -> torch.Tensor:
        """Calculate bounding box of the patch area (border around license plate)"""
        plate_corners = corners  # [4, 2]

        # Calculate center and create larger border quad
        center_x = plate_corners[:, 0].mean()
        center_y = plate_corners[:, 1].mean()
        center = torch.tensor([center_x, center_y], device=self.device)

        border_corners = center.unsqueeze(0) + (plate_corners - center.unsqueeze(0)) * border_scale

        # Calculate bounding box of the border corners
        min_x = torch.min(border_corners[:, 0])
        max_x = torch.max(border_corners[:, 0])
        min_y = torch.min(border_corners[:, 1])
        max_y = torch.max(border_corners[:, 1])

        return torch.stack([min_x, min_y, max_x, max_y])

    def apply_patch_to_image(self, image: torch.Tensor,
                             corners: torch.Tensor,
                             patch: torch.Tensor,
                             border_scale: float = 1.4) \
            -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Apply adversarial patch as border around license plate using homography or simple overlay"""
        batch_size = image.shape[0]

        # Extract image dimensions dynamically
        image_height, image_width = image.shape[2], image.shape[3]
        dsize = (image_height, image_width)

        # Patch is already in [0, 1] range (no tanh normalization)

        # Very light Gaussian blur
        if self.print_blur > 0:
            patch = kornia.filters.gaussian_blur2d(
                patch.unsqueeze(0),
                kernel_size=(3, 3),
                sigma=(self.print_blur, self.print_blur)
            ).squeeze(0)

        if self.training:
            darkening_factor = torch.rand(1, device=self.device) * 0.2
            patch = patch * (1.0 - darkening_factor)

        # If homography is disabled, use simple rectangle overlay
        if not self.use_homography:
            return self._apply_patch_simple(image, corners, patch, border_scale)

        # Get the 4 corners of the license plate
        plate_corners = corners[0]  # [4, 2]

        # Calculate center and create larger border quad
        center_x = plate_corners[:, 0].mean()
        center_y = plate_corners[:, 1].mean()
        center = torch.tensor([center_x, center_y], device=self.device)

        border_corners = center.unsqueeze(
            0) + (plate_corners - center.unsqueeze(0)) * border_scale
        border_corners = border_corners.unsqueeze(0)  # [1, 4, 2]

        # Create patch corner coordinates in patch space
        patch_h, patch_w = self.patch_height, self.patch_width
        src_corners = torch.tensor([
            [0, 0], [patch_w, 0], [patch_w, patch_h], [0, patch_h]
        ], dtype=torch.float32, device=self.device).unsqueeze(0)

        # Compute perspective transformation matrices
        M_border = K.get_perspective_transform(src_corners, border_corners)
        M_plate = K.get_perspective_transform(src_corners, corners)

        # Create and warp patch using dynamic image size
        patch_batch = patch.unsqueeze(0).repeat(batch_size, 1, 1, 1)

        warped_patch = K.warp_perspective(
            patch_batch, M_border, dsize=dsize,
            mode='bilinear', padding_mode='zeros', align_corners=True
        )

        # Create masks using dynamic image size
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

        # Final mask: border area minus license plate area
        final_mask = torch.clamp(warped_border_mask - warped_plate_mask, 0, 1)
        final_mask = final_mask.expand(-1, 3, -1, -1)

        # Validate all tensors have matching spatial dimensions
        if image.shape[2:] != final_mask.shape[2:]:
            raise RuntimeError(
                f"Image spatial dims {image.shape[2:]} != mask spatial dims {final_mask.shape[2:]}")
        if image.shape[2:] != warped_patch.shape[2:]:
            raise RuntimeError(
                f"Image spatial dims {image.shape[2:]} != warped patch spatial dims {warped_patch.shape[2:]}")

        # Apply patch with cutout
        result_image = image * (1 - final_mask) + warped_patch * final_mask
        result_image = torch.clamp(result_image, 0, 1)

        return result_image, final_mask

    def focal_cce_loss(
        self,
        vocabulary_size: int,
        alpha: float = 0.25,
        gamma: float = 2.0,
        label_smoothing: float = 0.01,
    ):
        """
        Categorical focal cross-entropy loss - exact PyTorch replica of Keras version.
        """
        def cce(y_true, y_pred):
            # Exact replica: flatten both inputs to (-1, vocabulary_size)
            y_true = y_true.reshape(-1, vocabulary_size)
            y_pred = y_pred.reshape(-1, vocabulary_size)

            # Ensure y_pred are probabilities (Keras uses from_logits=False)
            if y_pred.max() > 1.0 or y_pred.min() < 0.0:
                y_pred = F.softmax(y_pred, dim=-1)

            # Apply label smoothing to y_true (if specified)
            if label_smoothing > 0.0:
                y_true = y_true * (1.0 - label_smoothing) + label_smoothing / vocabulary_size

            # Compute focal loss
            p_t = torch.sum(y_true * y_pred, dim=-1)  # [batch*seq]
            focal_weight = (1.0 - p_t) ** gamma
            ce_loss = -torch.log(p_t + 1e-8)
            focal_loss = alpha * focal_weight * ce_loss

            return torch.mean(focal_loss)

        return cce

    def invert_bbox(self, corners, transform):
        """Invert the given transformation to bring the corners back to original image"""
        r, dw, dh = transform
        corners = corners.clone()
        corners[::2] = corners[::2] - dw
        corners[1::2] = corners[1::2] - dh
        corners = corners / r
        return corners

    def bbox_to_corners(self, bbox, device=None):
        x1, y1, x2, y2 = bbox
        corners = torch.tensor([[
            [x1, y1],  # top-left
            [x2, y1],  # top-right
            [x2, y2],  # bottom-right
            [x1, y2]   # bottom-left
        ]], device=device or self.device)
        return corners

    def corners_to_bbox(self, corners):
        min_x = torch.min(corners[:, 0])
        max_x = torch.max(corners[:, 0])
        min_y = torch.min(corners[:, 1])
        max_y = torch.max(corners[:, 1])
        return torch.stack([min_x, min_y, max_x, max_y])

    def boxes_IoU(self, boxes1, boxes2):
        area1 = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
        area2 = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

        boxes1 = boxes1.unsqueeze(1)  # [N, 1, 4]
        boxes2 = boxes2.unsqueeze(0)  # [1, M, 4]

        # Intersection coordinates
        inter_x1 = torch.max(boxes1[..., 0], boxes2[..., 0])
        inter_y1 = torch.max(boxes1[..., 1], boxes2[..., 1])
        inter_x2 = torch.min(boxes1[..., 2], boxes2[..., 2])
        inter_y2 = torch.min(boxes1[..., 3], boxes2[..., 3])

        # Intersection area
        inter_width = torch.clamp(inter_x2 - inter_x1, min=0)
        inter_height = torch.clamp(inter_y2 - inter_y1, min=0)
        inter_area = inter_width * inter_height

        # Union area
        union_area = area1 + area2 - inter_area

        return inter_area / (union_area + 1e-8)

    def partial_loss(self, batch, patch, use_ocr_baseline=True):
        # Load original image (no patch)
        prep_image = batch['prep_image'].to(self.device)
        corners = batch['new_corners'].to(self.device)

        if self.match_detection:
            target_box = self.get_patch_bounding_box(corners)
        else:
            target_box = self.corners_to_bbox(corners)

        # Run YOLO detection on full patched image
        model_output = self.model(prep_image.unsqueeze(0))

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

            cropped_plate = kornia.geometry.crop_and_resize(
                batch['orig_image'].unsqueeze(0),
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
        """
        Compute total variation (TV) regularization loss for the adversarial patch.

        Args:
            patch: [3, H, W]

        Returns:
            torch.Tensor: Scalar regularization loss
        """
        C, H, W = patch.shape

        # Horizontal total variation
        tv_h = torch.pow(patch[:, :, 1:] - patch[:, :, :-1], 2).sum()

        # Vertical total variation
        tv_v = torch.pow(patch[:, 1:, :] - patch[:, :-1, :], 2).sum()

        # Number of comparisons
        num_comparisons = C * (H * (W - 1) + (H - 1) * W)

        # Normalize and scale
        loss = (tv_h + tv_v) / num_comparisons
        loss = loss * 2.5

        return loss

    def calculate_baseline_loss(self) -> float:
        """
        Calculate baseline OCR loss across entire dataset using ground truth boxes.
        """
        total_ocr_loss = 0.0
        total_det_loss = 0.0
        total_plates = 0

        # Sample a single patch for baseline calculation
        z = self.sample_coefficients(1)
        patch = self.generate_patches(z)[0]  # [3, H, W]

        desc = "Calculating baseline OCR loss"
        with tqdm(self.train_loader, desc=desc, leave=False) as pbar:
            with torch.no_grad():
                for batch in pbar:
                    batch = {k: v[0] for k, v in batch.items()}
                    det_loss, ocr_loss = self.partial_loss(batch, patch, use_ocr_baseline=False)
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

    def compute_loss_full_image(self, batch: dict, patch: torch.Tensor, use_ocr_baseline=True) -> torch.Tensor:
        """Compute loss for full image detection"""

        batch = {k: v[0] for k, v in batch.items()}

        # Apply adversarial patch to YOLO input
        patched_image, _ = self.apply_patch_to_image(
            batch['prep_image'].to(self.device).unsqueeze(0),
            batch['new_corners'].to(self.device).unsqueeze(0),
            patch
        )
        batch['prep_image'] = patched_image.squeeze()

        # Apply adversarial patch to full original image
        patched_image, _ = self.apply_patch_to_image(
            batch['orig_image'].to(self.device).unsqueeze(0),
            batch['orig_corners'].to(self.device).unsqueeze(0),
            patch
        )
        batch['orig_image'] = patched_image.squeeze()

        det_loss, ocr_loss = self.partial_loss(batch, patch, use_ocr_baseline=use_ocr_baseline)

        # Add TV regularization loss if enabled
        if self.use_tv_loss:
            reg_loss = self.patch_reg_loss(patch)
        else:
            reg_loss = 0.0

        # Combine losses with conditional sqrt on OCR loss
        # Apply sqrt only if ocr_loss > 1 (compress large values)
        # Below 1, use linear (don't deprioritize reasonable losses)
        ocr_term = torch.where(
            ocr_loss > 1.0,
            torch.sqrt(ocr_loss),
            ocr_loss
        )
        return (det_loss + ocr_term) / 2 + reg_loss

    def train_epoch(self, optimizer: torch.optim.Optimizer, epoch: int) -> float:
        """Train for one epoch with gradient accumulation and diversity loss

        Unlike offensive_patch.py which optimizes a single patch parameter,
        basis optimization generates different patches per image (p = Uz).

        The diversity loss requires ALL patches from the accumulation window
        to compute the Gram matrix, so we must keep all patches in the
        computational graph until we compute combined loss and call backward() once.

        Memory requirements: Higher than offensive_patch.py because we keep
        batch_size full computational graphs in memory. If OOM, reduce grad_accumulate.
        """
        total_loss = 0.0
        step_count = 0
        num_updates = 0

        # Determine update frequency
        update_every = len(
            self.train_loader) if self.grad_accumulate is None else self.grad_accumulate

        # Storage for patches and losses in current accumulation window
        # Must keep in computational graph to compute diversity on same patches as adversarial
        accumulated_patches = []
        accumulated_losses = []
        last_diversity_loss = 0.0  # Track for display during accumulation

        desc = f"Epoch {epoch+1} - Training (AccumSteps={update_every})"
        with tqdm(enumerate(self.train_loader), desc=desc, leave=False,
                  total=len(self.train_loader)) as pbar:

            for idx, batch in pbar:
                # Sample coefficients and generate patch for this image
                z = self.sample_coefficients(1)
                patch = self.generate_patches(z)[0]  # [3, H, W]

                # Store patch for diversity loss - MUST keep in graph
                accumulated_patches.append(patch)

                # Compute adversarial loss - MUST keep in graph
                adv_loss = self.compute_loss_full_image(batch, patch)
                accumulated_losses.append(adv_loss)

                step_count += 1

                # Update model every update_every steps
                if step_count % update_every == 0:
                    # Compute mean adversarial loss over batch
                    mean_adv_loss = sum(accumulated_losses) / len(accumulated_losses)

                    # Compute diversity loss on SAME patches used for adversarial loss
                    patches_tensor = torch.stack(accumulated_patches, dim=0)  # [batch_size, 3, H, W]
                    diversity_score = self.compute_diversity_loss(patches_tensor)
                    diversity_loss = -self.diversity_weight * (1.0 / len(accumulated_patches)) * diversity_score
                    last_diversity_loss = diversity_loss.item()

                    # Combine losses and backward ONCE through entire graph
                    combined_loss = mean_adv_loss + diversity_loss
                    combined_loss.backward()

                    # Apply accumulated gradients
                    torch.nn.utils.clip_grad_norm_(self.generator.parameters(), max_norm=1.0)
                    optimizer.step()
                    optimizer.zero_grad()

                    # Track total loss
                    total_loss += combined_loss.item()
                    num_updates += 1

                    # Update progress bar
                    avg_loss = total_loss / num_updates
                    # Display diversity loss scaled by batch size to have consistent magnitude
                    div_loss_scaled = last_diversity_loss * (update_every / len(accumulated_patches))
                    pbar.set_postfix({
                        'Loss': f"{avg_loss:.4f}",
                        'AdvLoss': f"{mean_adv_loss.item():.4f}",
                        'DivLoss': f"{div_loss_scaled:.4f}",
                        'Updates': num_updates
                    })

                    # Memory cleanup after update - NOW graphs are freed
                    del combined_loss, mean_adv_loss, patches_tensor, diversity_score, diversity_loss
                    accumulated_patches = []
                    accumulated_losses = []

                    if self.device == 'cuda':
                        torch.cuda.empty_cache()
                    elif self.device == 'mps':
                        torch.mps.empty_cache()

                else:
                    # Show accumulation progress with current average loss
                    current_batch_avg = sum(accumulated_losses) / len(accumulated_losses) if accumulated_losses else 0

                    # Compute diversity loss on patches accumulated so far (for display only)
                    if len(accumulated_patches) > 1:  # Need at least 2 patches for meaningful diversity
                        with torch.no_grad():
                            patches_tensor = torch.stack(accumulated_patches, dim=0)
                            diversity_score = self.compute_diversity_loss(patches_tensor)
                            current_div_loss = -self.diversity_weight * (1.0 / len(accumulated_patches)) * diversity_score
                            last_diversity_loss = current_div_loss.item()

                    # Display diversity loss scaled by batch size to have consistent magnitude
                    div_loss_scaled = last_diversity_loss * (update_every / len(accumulated_patches))
                    pbar.set_postfix({
                        'AccumLoss': f"{current_batch_avg.item():.4f}" if hasattr(current_batch_avg, 'item') else f"{current_batch_avg:.4f}",
                        'DivLoss': f"{div_loss_scaled:.4f}",
                        'Progress': f"{step_count % update_every}/{update_every}"
                    })

            # Handle remaining accumulated samples
            if step_count % update_every != 0 and self.grad_accumulate is not None:
                if len(accumulated_patches) > 0:
                    mean_adv_loss = sum(accumulated_losses) / len(accumulated_losses)
                    patches_tensor = torch.stack(accumulated_patches, dim=0)
                    diversity_score = self.compute_diversity_loss(patches_tensor)
                    diversity_loss = -self.diversity_weight * (1.0 / len(accumulated_patches)) * diversity_score

                    combined_loss = mean_adv_loss + diversity_loss
                    combined_loss.backward()

                    torch.nn.utils.clip_grad_norm_(self.generator.parameters(), max_norm=1.0)
                    optimizer.step()
                    optimizer.zero_grad()

                    total_loss += combined_loss.item()
                    num_updates += 1

                if self.device == 'cuda':
                    torch.cuda.empty_cache()
                elif self.device == 'mps':
                    torch.mps.empty_cache()

        # Return average loss per update
        return total_loss / max(num_updates, 1)

    def validate(self) -> float:
        """Validation pass on held-out data"""
        losses = []

        with torch.no_grad():
            for batch in self.val_loader:
                # Sample a patch for validation
                z = self.sample_coefficients(1)
                patch = self.generate_patches(z)[0]

                loss = self.compute_loss_full_image(batch, patch)
                losses.append(loss.detach().cpu().item())

        return np.mean(losses)

    def save_basis(self, epoch: int, save_dir: str = "neural_basis_patches"):
        """Save current generator state and sample patches"""
        Path(save_dir).mkdir(exist_ok=True)

        with torch.no_grad():
            # Save generator network
            torch.save({
                'generator_state_dict': self.generator.state_dict(),
                'epoch': epoch,
                'basis_dim': self.basis_dim,
                'patch_size': (self.patch_height, self.patch_width)
            }, f"{save_dir}/generator_epoch_{epoch:04d}.pt")

            # Sample and save a few example patches
            num_samples = 5
            z_samples = self.sample_coefficients(num_samples)
            sample_patches = self.generate_patches(z_samples)

            for i, patch in enumerate(sample_patches):
                patch_pil = T.ToPILImage()(patch.cpu())
                patch_pil.save(f"{save_dir}/sample_{i}_epoch_{epoch:04d}.png")

    def train(self, num_epochs: int = 100, learning_rate: float = 0.01,
              save_interval: int = 10, early_stop_patience: int = 15,
              warmup_epochs: int = 5, lr_min: float = 1e-5):
        """Main training loop with linear warmup + cosine annealing LR schedule"""

        # Initialize optimizer with peak learning rate
        # LinearLR will scale it down during warmup, then back up
        optimizer = optim.AdamW(self.generator.parameters(), lr=learning_rate, weight_decay=1e-4)

        # Linear warmup scheduler (epochs 0-4: lr_min -> learning_rate)
        # start_factor scales base_lr to lr_min, end_factor is 1.0 (full base_lr)
        warmup_scheduler = optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=lr_min / learning_rate,
            end_factor=1.0,
            total_iters=warmup_epochs
        )

        # Cosine annealing scheduler (epochs 5-99: learning_rate -> lr_min)
        cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=num_epochs - warmup_epochs,
            eta_min=lr_min
        )

        # Sequential scheduler: warmup then cosine
        scheduler = optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_epochs]
        )

        history = {'loss': [], 'val_score': [], 'learning_rate': []}

        best_loss = float('inf')
        patience_counter = 0

        print("\nStarting neural basis patch training")
        print(f"   Dataset: {len(self.train_loader) + len(self.val_loader)} images")
        print(f"   Patch size: {self.patch_height}×{self.patch_width}")
        print(f"   Latent dimensions: {self.basis_dim}")
        vae_latent_dim = self.generator.vae_latent_dim
        print(f"   Generator architecture:")
        print(f"     Main path:")
        print(f"       Adapter (trainable): z[{self.basis_dim}] -> 512 -> 1024 -> 2048 -> VAE latent[{vae_latent_dim}]")
        print(f"       VAE decoder (trainable): latent[4×32×64] -> features[3×{self.patch_height}×{self.patch_width}]")
        print(f"     Skip connection:")
        print(f"       Skip projection (trainable): z[{self.basis_dim}] -> 512 -> spatial[1×{self.patch_height}×{self.patch_width}]")
        print(f"     CNN refiner (trainable):")
        print(f"       Input: concat(VAE output, skip)[4 channels]")
        print(f"       Block1: Conv[4→64] + Conv[64→64]")
        print(f"       Block2: Conv[64→128] + Conv[128→128]")
        print(f"       Block3: Conv[128→64] + Conv[64→64]")
        print(f"       Output: [64 channels]")
        print(f"     DNN block (trainable):")
        print(f"       Input: concat(CNN output[64], skip[1])[65 channels]")
        print(f"       Global pool: 65×{self.patch_height}×{self.patch_width} → 65×8×16")
        print(f"       Dense: {self.generator.dnn_input_dim} → 2048 → 2048 → 1024 → {3 * self.patch_height * self.patch_width}")
        print(f"       Output: patch[3×{self.patch_height}×{self.patch_width}]")
        print(f"   Diversity weight: {self.diversity_weight}")
        print(f"   Device: {self.device}")
        print(f"   Epochs: {num_epochs}")
        print(f"   LR schedule: Warmup to {learning_rate} over {warmup_epochs} epochs, then cosine anneal to {lr_min}")
        print(f"   Match detection: {self.match_detection}")
        print(
            f"   Impersonation target: {self.impersonation_target or 'None (penalize correct reading)'}")
        print(f"   TV loss: {'Enabled' if self.use_tv_loss else 'Disabled'}")
        print(f"   Homography: {'Enabled' if self.use_homography else 'Disabled'}")
        print("   Processing: Full 384x384 images only")
        print("-" * 60)

        for epoch in range(num_epochs):
            # Training and validation
            train_loss = self.train_epoch(optimizer, epoch)
            val_detection_score = self.validate()

            # Learning rate scheduling (step at end of each epoch)
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']

            # Record history
            history['loss'].append(train_loss)
            history['val_score'].append(val_detection_score)
            history['learning_rate'].append(current_lr)

            # Calculate loss change from initial
            initial_loss = history['val_score'][0] if len(history['val_score']) > 0 else 1.0
            loss_change = (val_detection_score / initial_loss - 1) * 100

            # Print epoch summary
            print(f"Epoch {epoch+1:3d}/{num_epochs} | "
                  f"Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_detection_score:.3f} | "
                  f"Change: {loss_change:+.1f}% | "
                  f"LR: {current_lr:.2e} | ")

            # Save best model
            if val_detection_score < best_loss:
                best_loss = val_detection_score
                patience_counter = 0
                self.save_basis(epoch, "best_neural_patches")
                print(f"   New best loss: {best_loss:.4f}")
            else:
                patience_counter += 1

            # Periodic saves
            if (epoch + 1) % save_interval == 0:
                self.save_basis(epoch, "checkpoint_neural_patches")

            # Early stopping
            if patience_counter >= early_stop_patience:
                print(f"   Early stopping: No improvement for {early_stop_patience} epochs")
                break

            # Check for convergence
            if len(history['loss']) >= 20:
                recent_losses = history['loss'][-20:]
                if (max(recent_losses) - min(recent_losses)) < 0.0001:
                    print("   Converged: Loss stabilized")
                    break

        print("\nTraining completed!")
        print(f"   Best loss: {best_loss:.4f}")
        final_change = (history['val_score'][-1] / history['val_score'][0] - 1) * 100
        print(f"   Final loss change: {final_change:+.1f}%")

        return history


def main():
    parser = argparse.ArgumentParser(description='Neural Basis Adversarial Patch Training')
    parser.add_argument('--match-detection', action='store_true',
                        help='Maximize IoU with patch bounding box instead of minimizing with ground truth')
    parser.add_argument('--impersonation-target', type=str, default=None,
                        help='Target plate text for impersonation (e.g., "ABC123"). If not provided, '
                        'uses disruption mode to prevent correct reading of VRJ7774')
    parser.add_argument('--disable-tv-loss', action='store_true',
                        help='Disable total variation (TV) regularization loss during training')
    parser.add_argument('--disable-homography', action='store_true',
                        help='Disable homography-based patch application (use simple rectangle overlay instead)')
    parser.add_argument('--basis-dim', type=int, default=16,
                        help='Dimensionality of basis (default: 16)')
    parser.add_argument('--diversity-weight', type=float, default=1.0,
                        help='Weight for diversity loss (default: 1.0)')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Gradient accumulation steps / effective batch size (default: 16). '
                        'Reduce if OOM, increase if you have more VRAM.')
    parser.add_argument('--num-epochs', type=int, default=100,
                        help='Number of training epochs (default: 100)')
    parser.add_argument('--early-stop-patience', type=int, default=20,
                        help='Early stopping patience: number of epochs without improvement before stopping (default: 20)')
    args = parser.parse_args()

    # Configuration
    CSV_PATH = "preproc_labels.csv"
    LEARNING_RATE = 5e-3  # Peak LR after warmup

    # Trainer kwargs
    trainer_kwargs = {
        'device': 'cuda',
        'grad_accumulate': args.batch_size,
        'match_detection': args.match_detection,
        'impersonation_target': args.impersonation_target,
        'use_tv_loss': not args.disable_tv_loss,
        'use_homography': not args.disable_homography,
        'basis_dim': args.basis_dim,
        'diversity_weight': args.diversity_weight
    }

    # Training mode
    try:
        trainer = FoundationBasisPatchTrainer(CSV_PATH, training=True, **trainer_kwargs)

        history = trainer.train(
            num_epochs=args.num_epochs,
            learning_rate=LEARNING_RATE,
            save_interval=1,
            early_stop_patience=args.early_stop_patience
        )

        # Save training history as CSV
        import pandas as pd
        history_df = pd.DataFrame(history)
        history_df.insert(0, 'epoch', range(1, len(history_df) + 1))
        history_df.to_csv('neural_basis_training_history.csv', index=False)
        print(f"\nTraining history saved to: neural_basis_training_history.csv")

        # Plot training results
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Loss curve
        ax1.plot(history['loss'], 'b-', label='Training Loss')
        ax1.plot(history['val_score'], 'r-', label='Validation Loss')
        ax1.set_title('Loss Over Time')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # Learning rate
        ax2.semilogy(history['learning_rate'], 'purple', label='Learning Rate')
        ax2.set_title('Learning Rate Schedule')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Learning Rate (log scale)')
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        plt.tight_layout()
        plt.savefig('neural_basis_training_curves.png', dpi=300, bbox_inches='tight')

        # Create separate figure for sample patches
        fig2 = plt.figure(figsize=(12, 12))
        with torch.no_grad():
            z_samples = trainer.sample_coefficients(9)
            sample_patches = trainer.generate_patches(z_samples)

            for i in range(9):
                ax = plt.subplot(3, 3, i + 1)
                patch_np = sample_patches[i].detach().cpu().permute(1, 2, 0).numpy()
                ax.imshow(patch_np)
                ax.set_title(f'Sample {i+1}')
                ax.axis('off')

        plt.tight_layout()
        plt.savefig('neural_basis_sample_patches.png', dpi=300, bbox_inches='tight')

        print("\nResults saved to 'neural_basis_training_curves.png' and 'neural_basis_sample_patches.png'")
        print("Generator checkpoints saved in 'neural_basis_patches/' directory")

    except Exception as e:
        print(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    main()
