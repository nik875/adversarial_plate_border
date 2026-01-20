#!/usr/bin/env python3
"""
Progressive Layer Attack: Train adversarial patches by progressively targeting
deeper layers of the OCR model, starting from early CNN features and moving
towards final outputs.
"""
import os
from typing import Tuple, List, Optional, Dict, Any
from dataclasses import dataclass
import logging
import warnings
import argparse
from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
from torch.utils.checkpoint import checkpoint
import numpy as np
from tqdm import tqdm
import kornia
import kornia.geometry as K
import torchvision.transforms as T
import matplotlib.pyplot as plt
from matplotlib import patches
import onnx
import onnx2torch
from dataset import create_dataloaders
from diffusers import AutoencoderKL
warnings.filterwarnings("ignore")


PATCH_WIDTH = 512
PATCH_HEIGHT = 256


@dataclass
class LayerConfig:
    """Configuration for a target layer in progressive attack"""
    name: str  # Layer name in the model
    description: str  # Human-readable description
    max_epochs: int = 50  # Maximum epochs to train on this layer
    convergence_threshold: float = 1.0  # Diversity score threshold for early stopping

    def __repr__(self):
        return f"{self.description} ({self.name})"


def get_ocr_layer_progression(max_epochs: int = 50, convergence_threshold: float = 1.0,
                              final_layer_epochs: Optional[int] = None) -> List[LayerConfig]:
    """
    Define the layer progression for the OCR model.

    Progression:
    1-4: Conv stem layers (4 convolutional layers with increasing channels)
    5: Patch extractor (visual→sequence transformation)
    6-9: Transformer blocks (4 transformer encoder blocks)
    10: Final output (vocab projection softmax)

    Args:
        max_epochs: Maximum epochs per layer (default 50)
        convergence_threshold: Diversity threshold for convergence. Use 0 or negative to disable (default 1.0)
        final_layer_epochs: Maximum epochs for final layer. If None, defaults to 2x max_epochs (default None)

    Returns:
        List of LayerConfig objects defining the attack progression
    """
    # Final layer gets more epochs and stricter convergence if convergence is enabled
    if final_layer_epochs is None:
        final_layer_epochs = max_epochs * 2 if max_epochs < 100 else max_epochs
    final_convergence = convergence_threshold * 0.5 if convergence_threshold > 0 else convergence_threshold

    return [
        # Conv stem layers (32 → 48 → 64 → 80 → 96 channels)
        LayerConfig(
            name="CCT_OCR_1/conv_stem_1/conv2d_1/BiasAdd",
            description="Conv Layer 1 (32ch)",
            max_epochs=max_epochs,
            convergence_threshold=convergence_threshold
        ),
        LayerConfig(
            name="CCT_OCR_1/conv_stem_1/conv2d_1_2/BiasAdd",
            description="Conv Layer 2 (48ch)",
            max_epochs=max_epochs,
            convergence_threshold=convergence_threshold
        ),
        LayerConfig(
            name="CCT_OCR_1/conv_stem_1/conv2d_2_1/BiasAdd",
            description="Conv Layer 3 (64ch)",
            max_epochs=max_epochs,
            convergence_threshold=convergence_threshold
        ),
        LayerConfig(
            name="CCT_OCR_1/conv_stem_1/conv2d_3_1/BiasAdd",
            description="Conv Layer 4 (80ch)",
            max_epochs=max_epochs,
            convergence_threshold=convergence_threshold
        ),
        LayerConfig(
            name="CCT_OCR_1/conv_stem_1/conv2d_4_1/BiasAdd",
            description="Conv Layer 5 (96ch)",
            max_epochs=max_epochs,
            convergence_threshold=convergence_threshold
        ),
        # Patch extractor (visual→sequence)
        LayerConfig(
            name="CCT_OCR_1/patch_extractor_1/convolution",
            description="Patch Extractor (384ch)",
            max_epochs=max_epochs,
            convergence_threshold=convergence_threshold
        ),
        # Transformer blocks (4 blocks)
        LayerConfig(
            name="CCT_OCR_1/transformer_block_1_1/add_9_1/Add",
            description="Transformer Block 1 Output",
            max_epochs=max_epochs,
            convergence_threshold=convergence_threshold
        ),
        LayerConfig(
            name="CCT_OCR_1/transformer_block_2_1/add_11_1/Add",
            description="Transformer Block 2 Output",
            max_epochs=max_epochs,
            convergence_threshold=convergence_threshold
        ),
        LayerConfig(
            name="CCT_OCR_1/transformer_block_3_1/add_13_1/Add",
            description="Transformer Block 3 Output",
            max_epochs=max_epochs,
            convergence_threshold=convergence_threshold
        ),
        LayerConfig(
            name="CCT_OCR_1/transformer_block_4_1/add_15_1/Add",
            description="Transformer Block 4 Output",
            max_epochs=max_epochs,
            convergence_threshold=convergence_threshold
        ),
        # Final output
        LayerConfig(
            name="CCT_OCR_1/vocab_projection_1/dense_9_1/Softmax",
            description="Final Output (Vocab Softmax)",
            max_epochs=final_layer_epochs,
            convergence_threshold=final_convergence
        ),
    ]


class SimplePatchGenerator(nn.Module):
    """Simple MLP patch generator (memory-efficient alternative to FoundationPatchGenerator)"""
    def __init__(self, latent_dim: int, patch_height: int = 256, patch_width: int = 512,
                 hidden_dims: List[int] = None):
        super().__init__()

        self.latent_dim = latent_dim
        self.patch_height = patch_height
        self.patch_width = patch_width
        self.patch_dim = 3 * patch_height * patch_width

        # Default hidden dimensions if not specified
        if hidden_dims is None:
            hidden_dims = [256, 512, 1024]

        layers = []
        prev_dim = latent_dim

        # Build hidden layers
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            prev_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(prev_dim, self.patch_dim))
        layers.append(nn.Sigmoid())  # Output in [0, 1]

        self.network = nn.Sequential(*layers)

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        print(f"Simple generator initialized: {latent_dim} → {' → '.join(map(str, hidden_dims))} → {self.patch_dim}")

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: [batch_size, latent_dim]
        Returns:
            patches: [batch_size, 3, patch_height, patch_width]
        """
        patches_flat = self.network(z)  # [batch_size, patch_dim]
        patches = patches_flat.view(-1, 3, self.patch_height, self.patch_width)
        return patches


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


class ProgressivePatchTrainer:
    """
    Progressive layer attack trainer.

    Trains adversarial patches by progressively targeting deeper layers,
    starting from early CNN features and moving towards final outputs.
    """
    def __init__(self,
                 csv_path: str,
                 device: str = None,
                 grad_accumulate: int = None,
                 basis_dim: int = 16,
                 diversity_weight: float = 1.0,
                 tv_weight: float = 250,
                 max_epochs_per_layer: int = 50,
                 final_layer_epochs: Optional[int] = None,
                 convergence_threshold: float = 1.0,
                 target_layer: Optional[List[int]] = None,
                 layer_configs: Optional[List[LayerConfig]] = None,
                 eval_depth: Optional[int] = None,
                 use_simple_generator: bool = False,
                 use_all_for_train: bool = True):
        self.basis_dim = basis_dim
        self.diversity_weight = diversity_weight
        self.tv_weight = tv_weight
        self.max_epochs_per_layer = max_epochs_per_layer
        self.final_layer_epochs = final_layer_epochs
        self.convergence_threshold = convergence_threshold
        self.target_layer = target_layer
        self.eval_depth = eval_depth
        self.use_simple_generator = use_simple_generator
        self.use_all_for_train = use_all_for_train

        # Progressive layer configuration
        all_layer_configs = layer_configs or get_ocr_layer_progression(
            max_epochs=max_epochs_per_layer,
            convergence_threshold=convergence_threshold,
            final_layer_epochs=final_layer_epochs
        )

        # If target_layer specified, filter to only those layers
        if target_layer is not None:
            # Validate all indices
            for idx in target_layer:
                if idx < 0 or idx >= len(all_layer_configs):
                    raise ValueError(f"target_layer index {idx} out of range (must be 0-{len(all_layer_configs)-1})")

            # Filter layer configs to only specified indices (in order)
            sorted_indices = sorted(target_layer)
            self.layer_configs = [all_layer_configs[i] for i in sorted_indices]
            self.original_layer_indices = sorted_indices  # Track original indices for display
            self.current_layer_idx = 0
            print(f"Progressive training queued for {len(self.layer_configs)} layers: {sorted_indices}")
        else:
            self.layer_configs = all_layer_configs
            self.original_layer_indices = None  # All layers, use natural indices
            self.current_layer_idx = 0

        self.current_layer_epoch = 0
        self.layer_history = []  # Track training history for each layer

        # Image preprocessing
        self.transform = T.Compose([T.ToTensor()])

        self.grad_accumulate = grad_accumulate

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
                                                                n_jobs=0, use_all_for_train=use_all_for_train)

        # Initialize generator (simple or foundation model)
        if use_simple_generator:
            # Simple MLP generator (memory-efficient)
            self.generator = SimplePatchGenerator(
                latent_dim=basis_dim,
                patch_height=self.patch_height,
                patch_width=self.patch_width
            ).to(self.device)
        else:
            # Foundation model generator with trainable VAE
            # Create generator with trainable adapter + trainable SD VAE decoder
            self.generator = FoundationPatchGenerator(
                latent_dim=basis_dim,
                patch_height=self.patch_height,
                patch_width=self.patch_width
            ).to(self.device)

        # Activation capture for diversity metric
        self.ocr_activations = None  # Current activations from forward hook
        self.baseline_ocr_activations = {}  # Dict mapping image index to baseline activations
        self.activation_hook = None  # Handle for the forward hook

        # Load OCR model for diversity computation
        self.load_ocr_model()

        # Track statistics
        self.epoch_stats = []

    def load_ocr_model(self):
        """Load OCR model for diversity computation only"""
        print("Loading OCR model for diversity computation...")
        ocr_path = Path.home() / ".cache/fast-plate-ocr/cct-xs-v1-global-model/cct_xs_v1_global.onnx"

        if not ocr_path.exists():
            raise FileNotFoundError(f"OCR model not found at: {ocr_path}")

        ocr_model = onnx.load(str(ocr_path))
        self.ocr_input_shape = (64, 128, 3)

        self.ocr = onnx2torch.convert(ocr_model).to(self.device)
        self.ocr.eval()

        # Disable gradients for OCR model parameters to save memory
        for param in self.ocr.parameters():
            param.requires_grad = False

        # Setup activation capture
        self.setup_activation_hook()

        # Calculate baseline activations for diversity computation
        self.calculate_baseline_activations()

    def setup_activation_hook(self, layer_name: Optional[str] = None):
        """
        Register forward hook to capture activations from specified layer.

        Args:
            layer_name: Name of the layer to hook. If None, uses current layer from progression.
        """
        # Remove existing hook if present
        if self.activation_hook is not None:
            self.activation_hook.remove()
            self.activation_hook = None

        # Determine target layer
        if layer_name is None:
            layer_name = self.layer_configs[self.current_layer_idx].name

        # Find the target layer
        target_layer = None
        for name, module in self.ocr.named_modules():
            if name == layer_name:
                target_layer = module
                break

        if target_layer is None:
            raise RuntimeError(f"Could not find layer: {layer_name}")

        # Define hook function that captures activations
        # This needs to handle arbitrary output shapes
        def hook_fn(module, input, output):
            # Store activations - shape depends on the layer:
            # Conv layers: [batch, channels, H, W] → permute to [batch, H, W, channels]
            # Transformer/Linear: [batch, seq_len, channels] → keep as is
            # Final softmax: [batch, seq_len, vocab_size] → keep as is

            if len(output.shape) == 4:
                # Conv layer output: [batch, C, H, W] → [batch, H, W, C]
                self.ocr_activations = output.permute(0, 2, 3, 1)
            elif len(output.shape) == 3:
                # Transformer/sequence output: [batch, seq_len, features] → keep
                self.ocr_activations = output
            elif len(output.shape) == 2:
                # Dense output: [batch, features] → keep
                self.ocr_activations = output
            else:
                # Fallback: store as-is
                self.ocr_activations = output

        # Register the hook
        self.activation_hook = target_layer.register_forward_hook(hook_fn)
        layer_desc = self.layer_configs[self.current_layer_idx].description
        print(f"✓ Registered activation hook on: {layer_desc} ({layer_name})")

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

    def total_variation_loss(self, patches: torch.Tensor) -> torch.Tensor:
        """
        Compute total variation (TV) regularization loss for the adversarial patches.

        Encourages smooth, natural-looking patches by penalizing large differences
        between adjacent pixels. Uses L2 norm of gradients in both horizontal and
        vertical directions.

        Args:
            patches: [batch_size, 3, H, W] patch tensor

        Returns:
            torch.Tensor: Scalar regularization loss (normalized)
        """
        # Average over batch dimension: compute TV for each patch and then average
        batch_size = patches.shape[0]
        total_tv_loss = 0.0

        for i in range(batch_size):
            patch = patches[i]  # [3, H, W]
            C, H, W = patch.shape

            # Horizontal total variation: differences along width dimension
            tv_h = torch.pow(patch[:, :, 1:] - patch[:, :, :-1], 2).sum()

            # Vertical total variation: differences along height dimension
            tv_v = torch.pow(patch[:, 1:, :] - patch[:, :-1, :], 2).sum()

            # Number of comparisons: C × (H×(W-1) + (H-1)×W)
            num_comparisons = C * (H * (W - 1) + (H - 1) * W)

            # Normalize by number of comparisons
            patch_tv_loss = (tv_h + tv_v) / num_comparisons

            # Scale by 2.5x to keep loss in reasonable range
            patch_tv_loss = patch_tv_loss * 2.5

            total_tv_loss += patch_tv_loss

        # Average over batch
        avg_tv_loss = total_tv_loss / batch_size

        return avg_tv_loss

    def apply_patch_to_image(self, image: torch.Tensor,
                             corners: torch.Tensor,
                             patch: torch.Tensor,
                             border_scale: float = 1.4) \
            -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Apply adversarial patch as border around license plate using homography"""
        batch_size = image.shape[0]

        # Extract image dimensions dynamically
        image_height, image_width = image.shape[2], image.shape[3]
        dsize = (image_height, image_width)

        # Patch is already in [0, 1] range (no tanh normalization)

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

    def _get_activations_for_patch_image(self, batch: dict, patch: torch.Tensor,
                                          use_grad: bool = False, skip_detection: bool = False) -> torch.Tensor:
        """
        Apply patch to a single image and return OCR activations from that forward pass.

        Args:
            batch: Single unbatched batch item from dataloader
            patch: [3, H, W] patch tensor
            use_grad: If True, compute with gradients (needed for diversity-only mode)
            skip_detection: If True, use known plate corners instead of running YOLO detection

        Returns:
            activations: [H, W, C] from conv stem (shape depends on input size)
        """
        # Use context manager conditionally
        context = torch.no_grad() if not use_grad else torch.enable_grad()
        with context:
            # Apply patch to original image
            patched_image, _ = self.apply_patch_to_image(
                batch['orig_image'].to(self.device).unsqueeze(0),
                batch['orig_corners'].to(self.device).unsqueeze(0),
                patch
            )

            orig_image = batch['orig_image'].unsqueeze(0).to(self.device)

            # Use BORDER corners (1.4x scaled) - this is where the patch actually is!
            # Plate corners would miss the patch since it's applied as a border
            # Note: skip_detection parameter is kept for API compatibility but always uses corners
            plate_corners = batch['orig_corners'].to(self.device)
            border_corners = self.get_border_corners(plate_corners, border_scale=1.4)
            corners_box = border_corners.unsqueeze(0)  # [1, 4, 2]

            # Crop plate from patched image and run OCR
            cropped_plate = K.crop_and_resize(
                patched_image,
                corners_box,
                self.ocr_input_shape[:2]
            )

            ocr_input = cropped_plate.permute(0, 2, 3, 1) * 255
            self.ocr(ocr_input)  # Forward pass captures activations via hook

            if self.ocr_activations is not None:
                result = self.ocr_activations.squeeze(0)  # [H, W, C]
                result = result if use_grad else result.detach()
                # Clear hook reference to allow garbage collection
                self.ocr_activations = None
                return result

            # Fallback if OCR didn't produce activations
            if hasattr(self, 'activation_shape'):
                return torch.zeros(self.activation_shape, device=self.device, requires_grad=use_grad)
            else:
                return torch.zeros(1, 1, 48, device=self.device, requires_grad=use_grad)

    def compute_activation_diversity(self, patches_list: List[torch.Tensor],
                                      batches_list: List[dict],
                                      baseline_indices: List[int],
                                      diagonal_activations: List[torch.Tensor],
                                      use_grad: bool = False) -> torch.Tensor:
        """
        Compute diversity via average activation impact across sampled images.

        For each patch, evaluate it on a subset of images (always diagonal,
        randomly sample off-diagonal) and compute the average activation delta.

        The eval_depth parameter controls the total number of (patch, image)
        evaluations: always includes batch_size diagonal, randomly samples
        remaining off-diagonal pairs from the budget.

        Args:
            patches_list: List of [3, H, W] patches (one per image)
            batches_list: List of batch dicts (one per image)
            baseline_indices: List of dataset indices for baseline activations
            diagonal_activations: List of [H, W, C] activations from (patch_i, image_i) pairs
            use_grad: If True, compute with gradients (needed for diversity-only mode)

        Returns:
            log_det: scalar diversity score
        """
        batch_size = len(patches_list)

        # Determine actual eval_depth (default to full matrix if not specified)
        actual_eval_depth = self.eval_depth if self.eval_depth is not None else batch_size ** 2
        actual_eval_depth = min(actual_eval_depth, batch_size ** 2)  # Cap at max possible

        # Build set of off-diagonal pairs
        off_diag_pairs = []
        for patch_idx in range(batch_size):
            for img_idx in range(batch_size):
                if patch_idx != img_idx:
                    off_diag_pairs.append((patch_idx, img_idx))

        # Calculate off-diagonal budget (eval_depth - diagonal evaluations)
        diagonal_count = batch_size
        off_diag_budget = max(0, actual_eval_depth - diagonal_count)

        # Randomly sample off-diagonal pairs
        if off_diag_budget > 0 and len(off_diag_pairs) > 0:
            num_to_sample = min(off_diag_budget, len(off_diag_pairs))
            sampled_indices = np.random.choice(len(off_diag_pairs), size=num_to_sample, replace=False)
            sampled_pairs = set((off_diag_pairs[i][0], off_diag_pairs[i][1]) for i in sampled_indices)
        else:
            sampled_pairs = set()

        # For each patch, compute average activation delta across sampled images
        patch_avg_deltas = []

        for patch_idx, patch in enumerate(patches_list):
            activation_deltas = []

            # Always include diagonal (patch_i on image_i)
            activations = diagonal_activations[patch_idx]  # [H, W, C]
            baseline = self.baseline_ocr_activations[baseline_indices[patch_idx]]  # [H, W, C]
            delta = activations - baseline  # [H, W, C]
            activation_deltas.append(delta)

            # Include sampled off-diagonal pairs involving this patch
            for img_idx in range(batch_size):
                if img_idx != patch_idx and (patch_idx, img_idx) in sampled_pairs:
                    batch = batches_list[img_idx]
                    activations = self._get_activations_for_patch_image(batch, patch, use_grad=use_grad, skip_detection=True)  # [H, W, C]
                    baseline = self.baseline_ocr_activations[baseline_indices[img_idx]]  # [H, W, C]
                    delta = activations - baseline  # [H, W, C]
                    activation_deltas.append(delta)

            # Average deltas for this patch over sampled images
            stacked_deltas = torch.stack(activation_deltas, dim=0)
            avg_delta = stacked_deltas.mean(dim=0)  # [H, W, C]
            patch_avg_deltas.append(avg_delta)
            # Clean up intermediate tensors
            del stacked_deltas, activation_deltas

        # Stack averaged deltas: [batch_size, H, W, C]
        avg_deltas_stacked = torch.stack(patch_avg_deltas, dim=0)
        # Clean up intermediate list
        del patch_avg_deltas
        avg_deltas_flat = avg_deltas_stacked.reshape(batch_size, -1)  # [batch_size, H*W*C]

        # L2 normalize each patch's average delta vector
        normalized = F.normalize(avg_deltas_flat, p=2, dim=1)  # [batch_size, H*W*C]

        # Compute Gram matrix: [batch_size, batch_size]
        gram = normalized @ normalized.t()

        # Add epsilon for numerical stability
        epsilon = max(1e-6, 1e-2 / batch_size)
        gram = gram + epsilon * torch.eye(batch_size, device=self.device)

        # Use slogdet for numerical stability
        sign, log_det = torch.slogdet(gram)

        # Handle numerical issues
        if torch.isnan(log_det):
            log_det = torch.tensor(-20.0, device=self.device, dtype=log_det.dtype, requires_grad=use_grad)
        elif sign <= 0:
            log_det = torch.tensor(-20.0, device=self.device, dtype=log_det.dtype, requires_grad=use_grad)

        return log_det

    def get_border_corners(self, corners: torch.Tensor,
                            border_scale: float = 1.4) -> torch.Tensor:
        """Calculate the 4 corners of the border area (scaled from plate corners)"""
        plate_corners = corners  # [4, 2]

        # Calculate center and create larger border quad
        center_x = plate_corners[:, 0].mean()
        center_y = plate_corners[:, 1].mean()
        center = torch.tensor([center_x, center_y], device=plate_corners.device)

        border_corners = center.unsqueeze(0) + (plate_corners - center.unsqueeze(0)) * border_scale
        return border_corners  # [4, 2]

    def calculate_baseline_activations(self):
        """
        Capture baseline activations for each image (without patches).
        """
        desc = "Calculating baseline activations"
        with tqdm(self.train_loader, desc=desc, leave=False) as pbar:
            with torch.no_grad():
                for idx, batch in enumerate(pbar):
                    batch = {k: v[0] for k, v in batch.items()}

                    # Use BORDER corners (1.4x scaled) to match where patches will be applied
                    plate_corners = batch['orig_corners'].to(self.device)
                    border_corners = self.get_border_corners(plate_corners, border_scale=1.4)
                    corners_box = border_corners.unsqueeze(0)  # [1, 4, 2]

                    # Crop plate area from original image (no patch)
                    orig_image = batch['orig_image'].unsqueeze(0).to(self.device)
                    cropped_plate = K.crop_and_resize(
                        orig_image,
                        corners_box,
                        self.ocr_input_shape[:2]
                    )

                    ocr_input = cropped_plate.permute(0, 2, 3, 1) * 255
                    self.ocr(ocr_input)  # Forward pass captures activations via hook

                    # Capture the baseline activations from this forward pass
                    if self.ocr_activations is not None:
                        # Store per-image baseline: [H, W, C]
                        baseline_act = self.ocr_activations.squeeze(0).detach().clone()
                        self.baseline_ocr_activations[idx] = baseline_act

                        # Store activation shape from first sample for fallback zeros
                        if not hasattr(self, 'activation_shape'):
                            self.activation_shape = baseline_act.shape

                        # Clear hook reference
                        self.ocr_activations = None

        print(f"✓ Stored baseline activations for {len(self.baseline_ocr_activations)} images")

    def check_convergence(self, diversity_score: float) -> bool:
        """
        Check if current layer has converged based on diversity score.

        Args:
            diversity_score: Current diversity score (lower is better)

        Returns:
            True if converged (diversity below threshold). Returns False if convergence checking is disabled.
        """
        current_config = self.layer_configs[self.current_layer_idx]
        # If threshold is 0 or negative, convergence checking is disabled
        if current_config.convergence_threshold <= 0:
            return False
        return diversity_score < current_config.convergence_threshold

    def advance_to_next_layer(self) -> bool:
        """
        Advance to the next layer in the progression.

        Returns:
            True if advanced successfully, False if already at final layer
        """
        if self.current_layer_idx >= len(self.layer_configs) - 1:
            print("\n" + "="*80)
            print("✓ Reached final layer - training complete!")
            print("="*80 + "\n")
            return False

        # Record completion of current layer
        current_config = self.layer_configs[self.current_layer_idx]
        # Get original layer index if using queued layers
        original_idx = (self.original_layer_indices[self.current_layer_idx]
                       if self.original_layer_indices is not None
                       else self.current_layer_idx)
        self.layer_history.append({
            'layer_idx': self.current_layer_idx,
            'original_layer_idx': original_idx,
            'layer_name': current_config.name,
            'description': current_config.description,
            'epochs_trained': self.current_layer_epoch
        })

        # Save checkpoint after completing this layer (with 10 sample patches)
        layer_checkpoint_name = f"layer{original_idx + 1}_complete_{current_config.description.replace(' ', '_').replace('(', '').replace(')', '')}"
        self.save_basis(self.current_layer_epoch, layer_checkpoint_name, num_samples=10)
        print(f"\n✓ Saved checkpoint after layer {original_idx + 1} completion (with 10 sample patches)")

        # Move to next layer
        self.current_layer_idx += 1
        self.current_layer_epoch = 0

        # Setup hook for new layer
        next_config = self.layer_configs[self.current_layer_idx]
        print("\n" + "="*80)
        print(f"ADVANCING TO NEXT LAYER: {next_config.description}")
        print(f"Layer {self.current_layer_idx + 1}/{len(self.layer_configs)}")
        print("="*80 + "\n")

        # Re-register activation hook for new layer
        self.setup_activation_hook()

        # Re-calculate baseline activations for new layer
        print("Recalculating baseline activations for new layer...")
        self.baseline_ocr_activations.clear()
        self.activation_shape = None  # Reset shape tracking
        self.calculate_baseline_activations()

        return True

    def should_continue_current_layer(self, diversity_score: float) -> bool:
        """
        Determine if training should continue on current layer.

        Args:
            diversity_score: Current diversity score

        Returns:
            True if should continue, False if should advance to next layer
        """
        current_config = self.layer_configs[self.current_layer_idx]
        is_final_layer = self.current_layer_idx == len(self.layer_configs) - 1

        # Check max epochs
        if self.current_layer_epoch >= current_config.max_epochs:
            print(f"\n→ Reached max epochs ({current_config.max_epochs}) for {current_config.description}")
            return False

        # For final layer, ignore convergence threshold and always train full epochs
        if is_final_layer:
            return True

        # Check convergence (skipped for final layer)
        if self.check_convergence(diversity_score):
            print(f"\n→ Converged (diversity={diversity_score:.4f} < {current_config.convergence_threshold:.2f}) on {current_config.description}")
            return False

        return True

    def train_epoch(self, optimizer: torch.optim.Optimizer, epoch: int) -> float:
        """Train for one epoch with gradient accumulation and activation-based diversity loss

        Unlike offensive_patch.py which optimizes a single patch parameter,
        basis optimization generates different patches per image (p = Uz).

        Activation-based diversity: measures how differently each patch affects
        OCR internal representations (at the patch_extractor layer) compared to baseline.
        """
        total_diversity_loss = 0.0
        total_tv_loss = 0.0
        step_count = 0
        num_updates = 0

        # Determine update frequency
        update_every = len(
            self.train_loader) if self.grad_accumulate is None else self.grad_accumulate

        # Storage for patches and activations in current accumulation window
        accumulated_patches = []
        accumulated_batches = []
        accumulated_activations = []
        accumulated_indices = []
        last_diversity_loss = 0.0  # Track for display during accumulation
        last_tv_loss = 0.0  # Track for display during accumulation

        desc = f"Epoch {epoch+1} - Training (AccumSteps={update_every})"
        with tqdm(enumerate(self.train_loader), desc=desc, leave=False,
                  total=len(self.train_loader)) as pbar:

            for idx, batch in pbar:
                # Sample coefficients and generate patch for this image
                z = self.sample_coefficients(1)
                patch = self.generate_patches(z)[0]  # [3, H, W]

                # Store patch and batch for diversity evaluation
                # Keep gradients for diversity computation
                accumulated_patches.append(patch)
                # Detach and clone batch tensors to prevent holding references to dataloader tensors
                accumulated_batches.append({k: v[0].detach().clone() if torch.is_tensor(v[0]) else v[0]
                                           for k, v in batch.items()})

                # Compute diagonal activation (patch_i on image_i) with gradients
                batch_unbatched = {k: v[0] for k, v in batch.items()}
                diagonal_activation = self._get_activations_for_patch_image(
                    batch_unbatched, patch, use_grad=True, skip_detection=True
                )  # [H, W, C]
                accumulated_activations.append(diagonal_activation)

                # Track dataset index for baseline lookup
                accumulated_indices.append(idx)

                step_count += 1

                # Update model every update_every steps
                if step_count % update_every == 0:
                    # Compute activation-based diversity score
                    # Reuse diagonal activations, only compute off-diagonal
                    diversity_score = self.compute_activation_diversity(
                        accumulated_patches,
                        accumulated_batches,
                        accumulated_indices,
                        accumulated_activations,
                        use_grad=True
                    )
                    diversity_loss = -self.diversity_weight * (1.0 / len(accumulated_patches)) * diversity_score

                    # Compute total variation loss on generated patches
                    patches_stacked = torch.stack(accumulated_patches, dim=0)  # [batch_size, 3, H, W]
                    tv_loss = self.total_variation_loss(patches_stacked)
                    tv_loss_weighted = self.tv_weight * tv_loss

                    # Combined loss
                    total_loss = diversity_loss + tv_loss_weighted
                    last_diversity_loss = diversity_loss.item()
                    last_tv_loss = tv_loss_weighted.item()  # Display weighted version

                    # Train on combined loss
                    total_loss.backward()

                    # Apply accumulated gradients
                    torch.nn.utils.clip_grad_norm_(self.generator.parameters(), max_norm=1.0)
                    optimizer.step()
                    optimizer.zero_grad()

                    # Track losses (store weighted version for display)
                    total_diversity_loss += diversity_loss.item()
                    total_tv_loss += tv_loss_weighted.item()
                    num_updates += 1

                    # Update progress bar
                    avg_diversity_loss = total_diversity_loss / num_updates
                    avg_tv_loss = total_tv_loss / num_updates
                    pbar.set_postfix({
                        'DivLoss': f"{avg_diversity_loss:.4f}",
                        'TVLoss': f"{avg_tv_loss:.4f}",
                        'Updates': num_updates
                    })

                    # Memory cleanup after update
                    del diversity_score, diversity_loss, tv_loss, tv_loss_weighted, total_loss, patches_stacked
                    # Clear accumulated lists and their contents
                    for patch in accumulated_patches:
                        del patch
                    for act in accumulated_activations:
                        del act
                    accumulated_patches = []
                    accumulated_batches = []
                    accumulated_activations = []
                    accumulated_indices = []
                    # Clear hook-stored activations
                    self.ocr_activations = None

                    if self.device == 'cuda':
                        torch.cuda.empty_cache()
                    elif self.device == 'mps':
                        torch.mps.empty_cache()

                else:
                    # Show accumulation progress
                    pbar.set_postfix({
                        'DivLoss': f"{last_diversity_loss:.4f}",
                        'TVLoss': f"{last_tv_loss:.4f}",
                        'Progress': f"{step_count % update_every}/{update_every}"
                    })

            # Handle remaining accumulated samples
            if step_count % update_every != 0 and self.grad_accumulate is not None:
                if len(accumulated_patches) > 0:
                    # Compute activation-based diversity score
                    # Reuse diagonal activations, only compute off-diagonal
                    diversity_score = self.compute_activation_diversity(
                        accumulated_patches,
                        accumulated_batches,
                        accumulated_indices,
                        accumulated_activations,
                        use_grad=True
                    )
                    diversity_loss = -self.diversity_weight * (1.0 / len(accumulated_patches)) * diversity_score

                    # Compute total variation loss on generated patches
                    patches_stacked = torch.stack(accumulated_patches, dim=0)  # [batch_size, 3, H, W]
                    tv_loss = self.total_variation_loss(patches_stacked)
                    tv_loss_weighted = self.tv_weight * tv_loss

                    # Combined loss
                    total_loss = diversity_loss + tv_loss_weighted
                    total_loss.backward()

                    torch.nn.utils.clip_grad_norm_(self.generator.parameters(), max_norm=1.0)
                    optimizer.step()
                    optimizer.zero_grad()

                    total_diversity_loss += diversity_loss.item()
                    total_tv_loss += tv_loss_weighted.item()
                    num_updates += 1

                    # Memory cleanup for remaining samples
                    del diversity_score, diversity_loss, tv_loss, tv_loss_weighted, total_loss, patches_stacked
                    for patch in accumulated_patches:
                        del patch
                    for act in accumulated_activations:
                        del act
                    self.ocr_activations = None

                if self.device == 'cuda':
                    torch.cuda.empty_cache()
                elif self.device == 'mps':
                    torch.mps.empty_cache()

        # Return average losses per update
        avg_diversity_loss = total_diversity_loss / max(num_updates, 1)
        avg_tv_loss = total_tv_loss / max(num_updates, 1)
        return avg_diversity_loss, avg_tv_loss

    def validate(self) -> float:
        """Validation pass using diversity score"""
        diversity_scores = []

        with torch.no_grad():
            # Sample multiple patches for validation diversity
            num_val_samples = min(16, len(self.val_loader))
            patches = []
            batches = []
            activations = []
            indices = []

            for idx, batch in enumerate(self.val_loader):
                if idx >= num_val_samples:
                    break

                # Sample a patch for validation
                z = self.sample_coefficients(1)
                patch = self.generate_patches(z)[0]
                patches.append(patch)
                batches.append({k: v[0] for k, v in batch.items()})

                # Get activations
                act = self._get_activations_for_patch_image(
                    {k: v[0] for k, v in batch.items()}, patch, use_grad=False, skip_detection=True
                )
                activations.append(act)
                indices.append(idx)

            # Compute diversity score
            if len(patches) > 1:
                diversity_score = self.compute_activation_diversity(
                    patches, batches, indices, activations, use_grad=False
                )
                return diversity_score.item()
            else:
                return 0.0

    def save_basis(self, epoch: int, save_dir: str = "foundation_basis_activation_patches", num_samples: int = 5):
        """Save current generator state and sample patches

        Args:
            epoch: Current epoch for naming
            save_dir: Directory to save to
            num_samples: Number of sample patches to generate and save (default 5)
        """
        Path(save_dir).mkdir(exist_ok=True)

        with torch.no_grad():
            # Save generator network
            torch.save({
                'generator_state_dict': self.generator.state_dict(),
                'epoch': epoch,
                'basis_dim': self.basis_dim,
                'patch_size': (self.patch_height, self.patch_width)
            }, f"{save_dir}/generator_epoch_{epoch:04d}.pt")

            # Sample and save example patches
            z_samples = self.sample_coefficients(num_samples)
            sample_patches = self.generate_patches(z_samples)

            for i, patch in enumerate(sample_patches):
                patch_pil = T.ToPILImage()(patch.cpu())
                patch_pil.save(f"{save_dir}/sample_{i}_epoch_{epoch:04d}.png")

    def train(self, learning_rate: float = 0.01,
              warmup_epochs: int = 5, lr_min: float = 1e-5):
        """
        Progressive layer training loop.

        Trains by progressively targeting deeper layers, starting from early CNN
        features and moving towards final outputs. Each layer is trained until
        convergence or max epochs.

        Saves:
        - best_progressive_patch.tar: Best model across all training
        - layer*_complete_*.tar: Model after completing each layer
        """

        # Initialize optimizer
        optimizer = optim.AdamW(self.generator.parameters(), lr=learning_rate, weight_decay=1e-4)

        # Global training history
        global_history = {
            'layer_idx': [],
            'layer_name': [],
            'epoch': [],
            'diversity_loss': [],
            'tv_loss': [],
            'val_diversity': [],
            'learning_rate': []
        }

        best_diversity = -float('inf')  # Higher diversity is better
        global_epoch = 0

        print("\n" + "="*80)
        if self.target_layer is not None:
            print("PROGRESSIVE LAYER ATTACK (QUEUED LAYERS)")
            print(f"Training layers: {self.target_layer}")
        else:
            print("PROGRESSIVE LAYER ATTACK (ALL LAYERS)")
        print("="*80)
        print(f"   Dataset: {len(self.train_loader) + len(self.val_loader)} images")
        print(f"   Patch size: {self.patch_height}×{self.patch_width}")
        print(f"   Latent dimensions: {self.basis_dim}")

        # Print generator architecture based on type
        if self.use_simple_generator:
            print(f"   Generator: SimplePatchGenerator (MLP-based, memory-efficient)")
            print(f"     Architecture: z[{self.basis_dim}] → 256 → 512 → 1024 → {3 * self.patch_height * self.patch_width} → patch[3×{self.patch_height}×{self.patch_width}]")
        else:
            vae_latent_dim = self.generator.vae_latent_dim
            print(f"   Generator: FoundationPatchGenerator (VAE-based, high quality)")
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
        print(f"   TV weight: {self.tv_weight}")
        print(f"   Device: {self.device}")
        print(f"   LR: {learning_rate} (warmup {warmup_epochs} epochs, min {lr_min})")

        if self.target_layer is None:
            print(f"\nLayer Progression ({len(self.layer_configs)} layers total):")
            for i, config in enumerate(self.layer_configs, 1):
                print(f"   {i:2d}. {config.description:35s} (max {config.max_epochs} epochs)")
        print("="*80 + "\n")

        # Progressive layer training loop
        while self.current_layer_idx < len(self.layer_configs):
            current_config = self.layer_configs[self.current_layer_idx]
            # Get original layer index for display
            display_layer_idx = (self.original_layer_indices[self.current_layer_idx]
                                if self.original_layer_indices is not None
                                else self.current_layer_idx)

            print(f"\n{'='*80}")
            print(f"LAYER {display_layer_idx + 1}/{len(self.layer_configs)} ({self.current_layer_idx + 1}/{len(self.layer_configs)} in queue): {current_config.description}")
            if self.current_layer_idx == len(self.layer_configs) - 1:
                print(f"(Final layer - convergence threshold disabled, will train full {current_config.max_epochs} epochs)")
            print(f"{'='*80}\n")

            # Create schedulers for this layer
            warmup_scheduler = optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=lr_min / learning_rate,
                end_factor=1.0,
                total_iters=warmup_epochs
            )
            cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=current_config.max_epochs - warmup_epochs,
                eta_min=lr_min
            )
            scheduler = optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, cosine_scheduler],
                milestones=[warmup_epochs]
            )

            # Train on current layer until convergence or max epochs
            while True:
                self.current_layer_epoch += 1
                global_epoch += 1

                # Training and validation
                train_diversity_loss, train_tv_loss = self.train_epoch(optimizer, global_epoch)
                val_diversity_score = self.validate()

                # Learning rate scheduling
                scheduler.step()
                current_lr = optimizer.param_groups[0]['lr']

                # Record history
                global_history['layer_idx'].append(self.current_layer_idx)
                global_history['layer_name'].append(current_config.name)
                global_history['epoch'].append(global_epoch)
                global_history['diversity_loss'].append(train_diversity_loss)
                global_history['tv_loss'].append(train_tv_loss)
                global_history['val_diversity'].append(val_diversity_score)
                global_history['learning_rate'].append(current_lr)

                # Print epoch summary
                print(f"[L{display_layer_idx+1}] Epoch {self.current_layer_epoch:3d}/{current_config.max_epochs} | "
                      f"DivLoss: {train_diversity_loss:.4f} | "
                      f"TVLoss: {train_tv_loss:.4f} | "
                      f"Val: {val_diversity_score:.3f} | "
                      f"LR: {current_lr:.2e}")

                # Save best model globally (higher diversity is better)
                if val_diversity_score > best_diversity:
                    best_diversity = val_diversity_score
                    self.save_basis(global_epoch, "best_progressive_patch")
                    print(f"   ✓ New best diversity: {best_diversity:.4f}")

                # Check if should continue on current layer
                if not self.should_continue_current_layer(train_diversity_loss):
                    break

            # Advance to next layer (or finish if at final layer)
            if not self.advance_to_next_layer():
                break

        print("\n" + "="*80)
        if self.target_layer is not None:
            print("PROGRESSIVE TRAINING COMPLETED (QUEUED LAYERS)!")
        else:
            print("PROGRESSIVE TRAINING COMPLETED (ALL LAYERS)!")
        print("="*80)
        print(f"   Best diversity: {best_diversity:.4f}")
        print(f"   Total epochs: {global_epoch}")

        if len(self.layer_history) > 0:
            print(f"\nLayer progression summary:")
            for record in self.layer_history:
                layer_num = record['original_layer_idx'] + 1
                print(f"   Layer {layer_num:2d}: {record['description']:35s} - {record['epochs_trained']} epochs")
        print("="*80 + "\n")

        return global_history


def main():
    parser = argparse.ArgumentParser(description='Progressive Layer Diversity Training for Patch Generation')
    parser.add_argument('--basis-dim', type=int, default=16,
                        help='Dimensionality of latent basis (default: 16)')
    parser.add_argument('--diversity-weight', type=float, default=1.0,
                        help='Weight for diversity loss (default: 1.0)')
    parser.add_argument('--tv-weight', type=float, default=250,
                        help='Weight for total variation loss to encourage spatial smoothness (default: 250)')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Gradient accumulation steps / effective batch size (default: 16). '
                        'Reduce if OOM, increase if you have more VRAM.')
    parser.add_argument('--learning-rate', type=float, default=5e-3,
                        help='Peak learning rate after warmup (default: 5e-3)')
    parser.add_argument('--lr-min', type=float, default=1e-5,
                        help='Minimum learning rate (initial and final, default: 1e-5). '
                        'Used as start of warmup and end of cosine annealing.')
    parser.add_argument('--warmup-epochs', type=int, default=5,
                        help='Number of epochs for linear warmup (default: 5)')
    parser.add_argument('--max-epochs-per-layer', type=int, default=50,
                        help='Maximum epochs to train on each layer (default: 50). '
                        'Set to high value like 1000 to disable max epoch stopping.')
    parser.add_argument('--final-layer-epochs', type=int, default=None,
                        help='Maximum epochs for the final layer in the progression (default: 2x max-epochs-per-layer). '
                        'The final layer typically gets more training time for refinement.')
    parser.add_argument('--convergence-threshold', type=float, default=1.0,
                        help='Diversity score threshold for convergence. Training on a layer stops when diversity < threshold. '
                        '(default: 1.0). Set to 0 or negative to disable convergence checking and train full max-epochs.')
    parser.add_argument('--target-layer', type=str, default=None,
                        help='Queue specific layers for progressive training (comma-separated, e.g., "0,3,5,10"). '
                        'Training will progress through only these layers. '
                        '0=Conv1(32ch), 1=Conv2(48ch), 2=Conv3(64ch), 3=Conv4(80ch), 4=Conv5(96ch), '
                        '5=PatchExtractor(384ch), 6=Transformer1, 7=Transformer2, 8=Transformer3, '
                        '9=Transformer4, 10=FinalOutput. If not specified, trains all layers progressively.')
    parser.add_argument('--eval-depth', type=int, default=None,
                        help='Maximum number of (patch, image) evaluations for diversity computation. '
                        'Default: batch_size^2 (evaluate all pairs). '
                        'Always includes batch_size diagonal evaluations, randomly samples remaining off-diagonal. '
                        'Upper bound: batch_size^2. Use to reduce memory usage with large batch sizes.')
    parser.add_argument('--no-use-all-for-train', action='store_true',
                        help='Disable using all data for training (use 80%% train / 20%% validation split). '
                        'Default: uses 100%% of data for training.')
    parser.add_argument('--simple-generator', action='store_true',
                        help='Use simple MLP generator instead of foundation model (VAE-based). '
                        'Simple generator: z → MLP[256→512→1024] → patch. '
                        'Foundation model: z → adapter → VAE decoder → CNN refiner → DNN → patch. '
                        'Simple generator uses ~10x less memory but may produce lower quality patches.')
    args = parser.parse_args()

    # Configuration
    CSV_PATH = "preproc_labels.csv"

    # Parse target layers if specified
    target_layers = None
    if args.target_layer is not None:
        try:
            target_layers = [int(x.strip()) for x in args.target_layer.split(',')]
        except ValueError:
            raise ValueError(f"Invalid target-layer format: '{args.target_layer}'. Expected comma-separated integers (e.g., '0,3,5,10')")

    # Trainer kwargs
    trainer_kwargs = {
        'device': 'cuda',
        'grad_accumulate': args.batch_size,
        'basis_dim': args.basis_dim,
        'diversity_weight': args.diversity_weight,
        'tv_weight': args.tv_weight,
        'max_epochs_per_layer': args.max_epochs_per_layer,
        'final_layer_epochs': args.final_layer_epochs,
        'convergence_threshold': args.convergence_threshold,
        'target_layer': target_layers,
        'eval_depth': args.eval_depth,
        'use_simple_generator': args.simple_generator,
        'use_all_for_train': not args.no_use_all_for_train
    }

    # Training mode
    try:
        trainer = ProgressivePatchTrainer(CSV_PATH, **trainer_kwargs)

        history = trainer.train(
            learning_rate=args.learning_rate,
            warmup_epochs=args.warmup_epochs,
            lr_min=args.lr_min
        )

        # Save training history as CSV
        import pandas as pd
        history_df = pd.DataFrame(history)
        history_df.to_csv('progressive_patch_training_history.csv', index=False)
        print(f"\nTraining history saved to: progressive_patch_training_history.csv")

        # Plot training results
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

        # Loss curve
        ax1.plot(history['epoch'], history['loss'], 'b-', label='Training Loss', alpha=0.7)
        ax1.plot(history['epoch'], history['val_score'], 'r-', label='Validation Loss', alpha=0.7)
        # Add vertical lines for layer transitions
        for i, record in enumerate(trainer.layer_history):
            if i > 0:  # Skip first layer (starts at epoch 0)
                transition_epoch = sum(r['epochs_trained'] for r in trainer.layer_history[:i])
                ax1.axvline(x=transition_epoch, color='gray', linestyle='--', alpha=0.5)
        ax1.set_title('Loss Over Time (Progressive Layers)')
        ax1.set_xlabel('Global Epoch')
        ax1.set_ylabel('Loss')
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # Diversity loss
        ax2.plot(history['epoch'], history['diversity_loss'], 'g-', label='Diversity Loss')
        for i, record in enumerate(trainer.layer_history):
            if i > 0:
                transition_epoch = sum(r['epochs_trained'] for r in trainer.layer_history[:i])
                ax2.axvline(x=transition_epoch, color='gray', linestyle='--', alpha=0.5)
        ax2.set_title('Diversity Loss (Progressive Layers)')
        ax2.set_xlabel('Global Epoch')
        ax2.set_ylabel('Diversity')
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        # Learning rate
        ax3.semilogy(history['epoch'], history['learning_rate'], 'purple', label='Learning Rate')
        ax3.set_title('Learning Rate Schedule')
        ax3.set_xlabel('Global Epoch')
        ax3.set_ylabel('Learning Rate (log scale)')
        ax3.grid(True, alpha=0.3)
        ax3.legend()

        plt.tight_layout()
        plt.savefig('progressive_patch_training_curves.png', dpi=300, bbox_inches='tight')

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
        plt.savefig('progressive_patch_sample_patches.png', dpi=300, bbox_inches='tight')

        print("\nResults saved to:")
        print("  - progressive_patch_training_curves.png")
        print("  - progressive_patch_sample_patches.png")
        print("  - progressive_patch_training_history.csv")
        print("\nGenerator checkpoints saved:")
        print("  - best_progressive_patch.tar - Best model across all layers")
        print("  - layer*_complete_*.tar - Checkpoint after completing each layer")

    except Exception as e:
        print(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    main()


"""
USAGE:
======

Basic usage:
    python progressive_patch.py

With custom settings:
    python progressive_patch.py --learning-rate 0.01 --diversity-weight 2.0 --basis-dim 32

Memory-efficient training with eval_depth (reduces diversity computation):
    python progressive_patch.py --batch-size 32 --eval-depth 256
    # Evaluates up to 256 (patch, image) pairs instead of 32^2=1024
    # Always includes 32 diagonal evaluations, randomly samples 224 off-diagonal

    python progressive_patch.py --batch-size 64 --eval-depth 512
    # Evaluates up to 512 pairs instead of 64^2=4096 (8x memory reduction)

Memory-efficient training with simple generator (reduces model size):
    python progressive_patch.py --simple-generator
    # Uses SimplePatchGenerator (MLP) instead of FoundationPatchGenerator (VAE-based)
    # ~10x less memory usage, faster training, but potentially lower quality patches

    # Combine with eval_depth for maximum memory efficiency:
    python progressive_patch.py --simple-generator --batch-size 64 --eval-depth 512

Queue specific layers for progressive training:
    python progressive_patch.py --target-layer "0,5,10"
    # Train progressively through only layers 0 (Conv1), 5 (PatchExtractor), and 10 (FinalOutput)
    # Skips intermediate layers 1-4, 6-9

    python progressive_patch.py --target-layer "5,6,7,8,9,10"
    # Start from PatchExtractor and train through all remaining layers

    python progressive_patch.py --target-layer "10"
    # Train only the final output layer (single layer, no progression)

Custom final layer training epochs:
    python progressive_patch.py --final-layer-epochs 200
    # Give the final layer more training time (default: 2x max-epochs-per-layer)

    python progressive_patch.py --target-layer "0,5,10" --max-epochs-per-layer 30 --final-layer-epochs 150
    # Train layers 0 and 5 for max 30 epochs each, but train layer 10 for up to 150 epochs

Custom layer progression:
    You can define your own layer progression by modifying get_ocr_layer_progression()
    or passing a custom list of LayerConfig objects to ProgressivePatchTrainer.

Example custom layer configuration:
    from progressive_patch import LayerConfig, ProgressivePatchTrainer

    custom_layers = [
        LayerConfig("CCT_OCR_1/conv_stem_1/conv2d_1/BiasAdd", "First Conv", max_epochs=30, convergence_threshold=0.8),
        LayerConfig("CCT_OCR_1/patch_extractor_1/convolution", "Patch Extractor", max_epochs=50, convergence_threshold=1.0),
        LayerConfig("CCT_OCR_1/vocab_projection_1/dense_9_1/Softmax", "Final Output", max_epochs=100, convergence_threshold=0.5),
    ]

    trainer = ProgressivePatchTrainer(
        "preproc_labels.csv",
        device='cuda',
        layer_configs=custom_layers
    )
    history = trainer.train(learning_rate=0.01)

The progressive attack strategy:
- Starts with early CNN layers (low-level features)
- Progresses to deeper layers (higher-level representations)
- Ends with final output (direct optimization of predictions)
- Each layer is trained until convergence or max epochs
- Generator weights are preserved across layer transitions
- Saves checkpoints at each layer transition

Convergence criteria:
- Diversity score falls below threshold (unless disabled), OR
- Maximum epochs reached for current layer

Then automatically advances to next layer in progression.

Generator architectures:
- SimplePatchGenerator (--simple-generator flag):
  * Simple MLP: z[basis_dim] → 256 → 512 → 1024 → patch
  * Memory: ~10x less than FoundationPatchGenerator
  * Speed: Faster forward/backward passes
  * Quality: Lower quality patches, less realistic textures
  * Use case: Quick prototyping, limited GPU memory, faster iterations

- FoundationPatchGenerator (default):
  * Complex architecture: Adapter → VAE decoder → CNN refiner → DNN block
  * Memory: High (requires SD VAE decoder with many parameters)
  * Speed: Slower due to VAE and multiple refinement stages
  * Quality: Higher quality, more realistic patches
  * Use case: Final runs, high-quality patch generation

Diversity computation and memory:
- eval_depth controls the total number of (patch, image) evaluations for diversity:
  * Default (None): Evaluates all batch_size^2 pairs (full matrix)
  * Specified value: Evaluates up to eval_depth pairs total
  * Always includes batch_size diagonal evaluations (patch_i on image_i)
  * Randomly samples off-diagonal pairs from budget (eval_depth - batch_size)
- Benefits: Reduces memory without sacrificing diversity quality, enables larger batch sizes
- Example: batch_size=32, eval_depth=256 reduces evals from 1024 to 256 (75% reduction)
- Combine with --simple-generator for maximum memory efficiency

Checkpoint structure:
- best_progressive_patch.tar: Best loss across all training
- layer*_complete_*.tar: Model state after completing each layer
  (e.g., layer1_complete_Conv_Layer_1_32ch.tar, layer2_complete_Conv_Layer_2_48ch.tar, etc.)

You can resume from any checkpoint by loading it with the trainer's load_basis() method.
"""
