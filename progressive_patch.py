#!/usr/bin/env python3
"""
Progressive Layer Attack: Train adversarial patches by progressively targeting
deeper layers of the OCR model, starting from early CNN features and moving
towards final outputs.
"""
import os
import sys
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
from torch.utils.data import Dataset, DataLoader, random_split, ConcatDataset
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
from PIL import Image
from dataset import create_dataloaders
from diffusers import AutoencoderKL

# Import load_datasets module from foundationmodel
try:
    import importlib.util
    load_datasets_path = Path(__file__).parent / "foundationmodel" / "dataset" / "load_datasets.py"
    spec = importlib.util.spec_from_file_location("load_datasets", load_datasets_path)
    load_datasets = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(load_datasets)
    iter_dataset = load_datasets.iter_dataset
    DATASETS = load_datasets.DATASETS
except Exception as e:
    print(f"Warning: Could not import load_datasets: {e}")
    print("OCR mode will not be available without this module.")
    # Define empty placeholders to allow script to load
    def iter_dataset(*args, **kwargs):
        raise NotImplementedError("load_datasets not available")
    DATASETS = {}

warnings.filterwarnings("ignore")

import math


# LoRA Classes for VAE Decoder
class LoRALinear(nn.Module):
    """Linear layer with LoRA adaptation"""
    def __init__(self, linear_layer: nn.Linear, r: int = 8, lora_alpha: int = 16):
        super().__init__()
        self.r = r
        self.lora_alpha = lora_alpha

        # Store base weights as buffers (not as submodule to avoid parameter registration)
        self.register_buffer('weight', linear_layer.weight.detach())
        if linear_layer.bias is not None:
            self.register_buffer('bias', linear_layer.bias.detach())
        else:
            self.register_buffer('bias', None)

        # LoRA matrices
        self.lora_A = nn.Parameter(torch.zeros(r, linear_layer.in_features))
        self.lora_B = nn.Parameter(torch.zeros(linear_layer.out_features, r))
        self.scaling = lora_alpha / r

        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = F.linear(x, self.weight, self.bias)
        lora_out = (x @ self.lora_A.T @ self.lora_B.T) * self.scaling
        return result + lora_out


class LoRAConv2d(nn.Module):
    """Conv2d with LoRA using 1×1 convolutions"""
    def __init__(self, conv_layer: nn.Conv2d, r: int = 8, lora_alpha: int = 16):
        super().__init__()
        self.r = r
        self.lora_alpha = lora_alpha
        self.padding = conv_layer.padding
        self.stride = conv_layer.stride
        self.dilation = conv_layer.dilation
        self.groups = conv_layer.groups

        # Store base weights as buffers (not as submodule to avoid parameter registration)
        self.register_buffer('weight', conv_layer.weight.detach())
        if conv_layer.bias is not None:
            self.register_buffer('bias', conv_layer.bias.detach())
        else:
            self.register_buffer('bias', None)

        # LoRA 1×1 convs
        self.lora_down = nn.Conv2d(conv_layer.in_channels, r, kernel_size=1, bias=False)
        self.lora_up = nn.Conv2d(r, conv_layer.out_channels, kernel_size=1, bias=False)
        self.scaling = lora_alpha / r

        nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_up.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = F.conv2d(x, self.weight, self.bias, stride=self.stride, padding=self.padding,
                         dilation=self.dilation, groups=self.groups)
        lora_out = self.lora_up(self.lora_down(x)) * self.scaling
        return result + lora_out


def inject_lora_into_vae_decoder(vae, r: int = 8, lora_alpha: int = 16):
    """Inject LoRA into ALL Conv2d and attention Linear layers in VAE decoder"""
    lora_modules = {}

    # Recursively wrap all Conv2d and attention Linear layers in the entire decoder
    def wrap_all_conv_and_attention(module, prefix):
        """Recursively wrap Conv2d and attention Linear layers"""
        for name, child in module.named_children():
            full_name = f"{prefix}.{name}" if prefix else name

            if isinstance(child, nn.Conv2d):
                # Wrap all Conv2d layers
                wrapped = LoRAConv2d(child, r, lora_alpha)
                setattr(module, name, wrapped)
                lora_modules[full_name] = wrapped
            elif isinstance(child, nn.Linear):
                # Wrap all Linear layers (not just attention)
                wrapped = LoRALinear(child, r, lora_alpha)
                setattr(module, name, wrapped)
                lora_modules[full_name] = wrapped
            else:
                # Recurse into child modules
                wrap_all_conv_and_attention(child, full_name)

    # Start wrapping from decoder root
    wrap_all_conv_and_attention(vae.decoder, 'decoder')

    return lora_modules


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


class OCRDataset(Dataset):
    """
    PyTorch Dataset wrapper for public OCR datasets from load_datasets.py.

    Loads all samples into memory on initialization for consistent indexing.
    Suitable for datasets like IIIT5K, ICDAR, etc.
    """
    def __init__(self, dataset_name: str, split: str = 'train',
                 transform=None, max_samples: Optional[int] = None):
        """
        Args:
            dataset_name: Name of dataset (e.g., 'iiit5k', 'icdar2013')
            split: Dataset split ('train', 'test', 'val')
            transform: Optional torchvision transforms to apply
            max_samples: Optional limit on number of samples to load
        """
        self.dataset_name = dataset_name
        self.split = split
        self.transform = transform

        # Load all samples into memory
        print(f"Loading {dataset_name} ({split} split)...")
        self.samples = []
        self.sample_metadata = []  # Track dataset source and index for each sample
        for img, text, meta in tqdm(iter_dataset(dataset_name, split, max_samples),
                                     desc=f"Loading {dataset_name}"):
            self.samples.append((img, text, meta))
            # Store metadata for tracking: (dataset_name, sample_index, text_label)
            self.sample_metadata.append({
                'dataset': dataset_name,
                'global_idx': len(self.samples) - 1,
                'text': text
            })

        print(f"Loaded {len(self.samples)} samples from {dataset_name} ({split})")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img, text, meta = self.samples[idx]

        # Convert PIL image to tensor
        if self.transform:
            # Convert to numpy array first if needed
            if isinstance(img, Image.Image):
                img = np.array(img)
            img_tensor = self.transform(img)
        else:
            # Default: convert to tensor
            img_tensor = T.ToTensor()(img)

        # Return format compatible with progressive_patch expectations
        # Use prep_image since we're in OCR mode (cropped plates)
        return {
            'prep_image': img_tensor,
            'text': text,
            'dataset': meta['dataset'],
            'split': meta['split']
        }


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


class BottleneckDenseRefiner(nn.Module):
    """
    Bottleneck dense refiner for patch refinement with minimal parameters.

    Architecture: Compress spatial dims with conv → dense bottleneck → expand back
    Total parameters: ~50-100K (vs 300M+ for full dense)

    Processes patches through:
    256x512x3 → Conv stride 4 → 64x128x64 → Conv stride 2 → 64x128x128
    → GlobalAvgPool → Dense layers → Upsample back to 256x512x3

    Seed conditioning: Latent seed z is projected and concatenated at bottleneck
    to provide learned guidance to the refinement process.
    """
    def __init__(self, patch_height: int = 256, patch_width: int = 512, latent_dim: int = 16, bottleneck_dim: int = 256):
        super().__init__()

        self.patch_height = patch_height
        self.patch_width = patch_width
        self.latent_dim = latent_dim
        self.bottleneck_dim = bottleneck_dim

        # Compress spatial dimensions with strided convolutions
        self.compress = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=4, padding=1),  # 256x512 → 64x128
            nn.SiLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # 64x128 → 32x64
            nn.SiLU(inplace=True),
        )

        # Global context and dense processing
        self.bottleneck = nn.Sequential(
            nn.AdaptiveAvgPool2d((4, 8)),  # 32x64x128 → 4x8x128
        )

        # Project latent seed to bottleneck embedding
        # Maps from latent_dim to seed_embed_dim for concatenation
        self.seed_embed_dim = 64
        self.seed_projection = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.SiLU(inplace=True),
            nn.Linear(128, self.seed_embed_dim),
        )

        # Dense layers for feature refinement
        # Bottleneck size: 128 * 4 * 8 = 4096
        # With seed: 4096 + seed_embed_dim = 4160
        bottleneck_with_seed_dim = 4096 + self.seed_embed_dim
        self.dense = nn.Sequential(
            nn.Linear(bottleneck_with_seed_dim, 512),
            nn.SiLU(inplace=True),
            nn.Linear(512, self.bottleneck_dim),
            nn.SiLU(inplace=True),
            nn.Linear(self.bottleneck_dim, 4096),
            nn.SiLU(inplace=True),
        )

        # Expand back to full spatial resolution
        self.expand = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 4x8 → 8x16
            nn.SiLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=4, padding=0),  # 8x16 → 64x128
            nn.SiLU(inplace=True),
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=4, padding=0),  # 64x128 → 256x512
            nn.Tanh()  # Output symmetric refinement in [-1, 1]
        )

        # Post-expansion smoothing with progressive kernel sizes
        # Progressively larger kernels (3 → 7 → 9) to smooth the output
        self.post_expansion_smooth = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),  # kernel 3, maintain size
            nn.SiLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=7, padding=3),  # kernel 7, maintain size
            nn.SiLU(inplace=True),
            nn.Conv2d(16, 3, kernel_size=9, padding=4),  # kernel 9, maintain size
        )

        # Spatial propagation layers (same padding to preserve size)
        # Input: [original_patch (3ch), refined_patch (3ch), char_scale_8 (1ch), char_scale_16 (1ch), char_scale_32 (1ch), char_scale_64 (1ch)] = 10 channels
        self.spatial_layers = nn.Sequential(
            nn.Conv2d(10, 32, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
        )

        # Multi-scale regional decomposition
        # Different scales capture different contextual information
        self.scales = [8, 16, 32, 64]  # Patch sizes

        # Load Omniglot decoder for character generation
        print("Loading Omniglot decoder for character conditioning...")
        omniglot_decoder_path = Path(__file__).parent / "omniglot_ae_export" / "decoder_traced.pt"
        if omniglot_decoder_path.exists():
            self.omniglot_decoder = torch.jit.load(str(omniglot_decoder_path), map_location="cpu")
            self.omniglot_decoder.eval()
            for param in self.omniglot_decoder.parameters():
                param.requires_grad = False
            print(f"✓ Omniglot decoder loaded from {omniglot_decoder_path}")
        else:
            raise FileNotFoundError(f"Omniglot decoder not found at {omniglot_decoder_path}. "
                                  f"Please ensure omniglot_ae_export/decoder_traced.pt exists.")

        # Create scale-specific dense MLPs for seed-conditioned feature generation
        # Each MLP takes z and outputs a spatial feature map (channel dim encodes seed)
        self.scale_mlps = nn.ModuleDict()
        self.char_embed_dim = 32  # Channel dimension that encodes seed info

        for scale in self.scales:
            num_patches_h = patch_height // scale
            num_patches_w = patch_width // scale
            output_size = self.char_embed_dim * num_patches_h * num_patches_w

            # Dense MLP: z[16] → 64 → 64 → char_embed_dim * num_patches
            mlp = nn.Sequential(
                nn.Linear(latent_dim, 64),
                nn.SiLU(inplace=True),
                nn.Linear(64, 64),
                nn.SiLU(inplace=True),
                nn.Linear(64, output_size),
            )
            self.scale_mlps[str(scale)] = mlp

        self.scale_convs = nn.ModuleDict()
        for scale in self.scales:
            # Each scale gets its own 1x1 conv to compress 32 → 3 channels
            self.scale_convs[str(scale)] = nn.Conv2d(32, 3, kernel_size=1)

        # Per-pixel scale attention: learns spatially-varying weights for each scale
        # Takes spatial_features [B, 32, H, W] and outputs [B, num_scales, H, W]
        self.scale_attention = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(64, len(self.scales), kernel_size=1),
        )

        # Initialize weights
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        # Count parameters
        total_params = sum(p.numel() for p in self.parameters())
        print(f"BottleneckDenseRefiner initialized: ~{total_params:,} parameters")

    def forward(self, patches: torch.Tensor, z: torch.Tensor = None) -> torch.Tensor:
        """
        Refine patches through bottleneck dense layers with optional seed conditioning.

        Args:
            patches: [batch_size, 3, patch_height, patch_width]
            z: Optional latent seed [batch_size, latent_dim] for conditioning
        Returns:
            refined_patches: [batch_size, 3, patch_height, patch_width]
        """
        batch_size = patches.shape[0]

        # Compress spatial dimensions
        compressed = self.compress(patches)  # [B, 128, 32, 64]

        # Pool to bottleneck
        pooled = self.bottleneck(compressed)  # [B, 128, 4, 8]

        # Flatten for dense processing
        bottleneck_flat = pooled.view(batch_size, -1)  # [B, 4096]

        # Concatenate seed conditioning if provided
        if z is not None:
            seed_embed = self.seed_projection(z)  # [B, seed_embed_dim]
            bottleneck_with_seed = torch.cat([bottleneck_flat, seed_embed], dim=1)  # [B, 4096 + seed_embed_dim]
        else:
            bottleneck_with_seed = bottleneck_flat  # [B, 4096]

        # Process through dense layers
        refined_features = self.dense(bottleneck_with_seed)  # [B, 4096]

        # Reshape back for expansion
        refined_features = refined_features.view(batch_size, 128, 4, 8)  # [B, 128, 4, 8]

        # Expand back to original resolution
        refined = self.expand(refined_features)  # [B, 3, 256, 512]

        # Ensure correct size (in case of rounding errors)
        if refined.shape[2] != self.patch_height or refined.shape[3] != self.patch_width:
            refined = F.interpolate(
                refined,
                size=(self.patch_height, self.patch_width),
                mode='bilinear',
                align_corners=True
            )

        # Generate seed-conditioned character features for each scale
        # MLP: z[B, 16] → [B, num_patches * 32]
        # Reshape: [B, num_patches, 32]
        # Flatten: [B*num_patches, 32]
        # Decoder: → [B*num_patches, 1, 56, 56]
        # Downscale and reshape: [B, 1, H, W]
        char_features_list = []
        for scale in self.scales:
            mlp = self.scale_mlps[str(scale)]
            num_patches_h = self.patch_height // scale
            num_patches_w = self.patch_width // scale
            num_patches = num_patches_h * num_patches_w

            # MLP output: [B, num_patches * 32]
            mlp_output = mlp(z)

            # Reshape to [B, num_patches, 32]
            char_embeddings = mlp_output.view(batch_size, num_patches, self.char_embed_dim)

            # Flatten to [B*num_patches, 32]
            char_embeddings_flat = char_embeddings.view(batch_size * num_patches, self.char_embed_dim)

            # Pass through Omniglot decoder: [B*num_patches, 32] → [B*num_patches, 1, 56, 56]
            with torch.no_grad():
                self.omniglot_decoder = self.omniglot_decoder.to(char_embeddings_flat.device)
                characters = self.omniglot_decoder(char_embeddings_flat)

            # Downscale characters to scale size: [B*num_patches, 1, scale, scale]
            characters_resized = F.interpolate(
                characters,
                size=(scale, scale),
                mode='bilinear',
                align_corners=True
            )

            # Reshape back to [B, 1, H, W]
            characters_resized = characters_resized.view(batch_size, num_patches_h, num_patches_w, 1, scale, scale)
            characters_resized = characters_resized.permute(0, 3, 1, 4, 2, 5).contiguous()
            char_scale = characters_resized.view(batch_size, 1, self.patch_height, self.patch_width)

            char_features_list.append(char_scale)

        # Multi-scale regional decomposition
        # Concatenate original patch, refined patch, and character features from all scales
        # [B, 3] + [B, 3] + [B, 1] + [B, 1] + [B, 1] + [B, 1] = [B, 10]
        combined = torch.cat([patches, refined] + char_features_list, dim=1)  # [B, 10, H, W]

        # Spatial propagation: propagate information without changing size
        spatial_features = self.spatial_layers(combined)  # [B, 32, H, W]

        # Process at multiple scales
        scale_outputs = []

        for scale_idx, scale in enumerate(self.scales):
            # Extract spatial patches of this scale
            scale_output = self._process_scale(spatial_features, scale, batch_size)
            scale_outputs.append(scale_output)

        # Compute per-pixel scale weights
        spatial_scale_weights = self.scale_attention(spatial_features)  # [B, num_scales, H, W]
        spatial_scale_weights = torch.softmax(spatial_scale_weights, dim=1)  # Normalize over scales per pixel

        # Weighted average of all scales with per-pixel weights
        refined_patches = torch.zeros_like(patches)  # [B, 3, H, W]
        for scale_idx, scale_output in enumerate(scale_outputs):
            # scale_output: [B, 3, H, W]
            # spatial_scale_weights[:, scale_idx:scale_idx+1]: [B, 1, H, W]
            refined_patches = refined_patches + spatial_scale_weights[:, scale_idx:scale_idx+1] * scale_output

        # Apply sigmoid to bound to [0, 1]
        refined_patches = torch.sigmoid(refined_patches)

        # Apply post-multi-scale smoothing with progressive kernel sizes
        refined_patches = self.post_expansion_smooth(refined_patches)  # [B, 3, H, W]

        return refined_patches

    def _process_scale(self, spatial_features: torch.Tensor, scale: int, batch_size: int) -> torch.Tensor:
        """
        Process features at a specific scale.

        Args:
            spatial_features: [B, 32, H, W] propagated features
            scale: Patch size (8, 16, 32, or 64)
            batch_size: Batch size

        Returns:
            output: [B, 3, H, W] refined patch at this scale
        """
        B, C, H, W = spatial_features.shape
        device = spatial_features.device

        # Calculate patch grid dimensions
        num_patches_h = H // scale
        num_patches_w = W // scale

        # Unfold into patches: [B, C, num_patches_h, scale, num_patches_w, scale]
        patches = spatial_features.unfold(2, scale, scale).unfold(3, scale, scale)
        # Reshape to [B, C, num_patches_h, num_patches_w, scale, scale]
        patches = patches.permute(0, 1, 2, 4, 3, 5).contiguous()
        # Reshape to [B*num_patches_h*num_patches_w, C, scale, scale]
        num_total_patches = num_patches_h * num_patches_w
        patches = patches.view(B * num_total_patches, C, scale, scale)

        # Apply scale-specific 1x1 conv to compress to 3 channels
        conv_layer = self.scale_convs[str(scale)]
        patch_output = conv_layer(patches)  # [B*num_patches, 3, scale, scale]

        # Reshape back to full image
        patch_output = patch_output.view(B, num_patches_h, num_patches_w, 3, scale, scale)
        patch_output = patch_output.permute(0, 3, 1, 4, 2, 5).contiguous()
        patch_output = patch_output.view(B, 3, H, W)

        return patch_output


class FoundationPatchGenerator(nn.Module):
    """Patch generator using Stable Diffusion VAE decoder with trainable adapter and CNN refinement"""
    def __init__(self, latent_dim: int, patch_height: int = 256, patch_width: int = 512, num_layers: int = 11,
                 use_vae_lora: bool = True, lora_rank: int = 8, lora_alpha: int = 16,
                 use_bottleneck_refiner: bool = False, bottleneck_dim: int = 256):
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
            "stabilityai/sdxl-vae",  # Official SDXL VAE (no fp16 modifications needed)
            torch_dtype=torch.float32
        )

        # We only use the decoder, delete encoder to save memory
        del self.vae.encoder
        self.vae.encoder = None

        # LoRA configuration
        self.use_vae_lora = use_vae_lora
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha

        if self.use_vae_lora:
            print(f"Injecting LoRA (rank={self.lora_rank}, alpha={self.lora_alpha})...")
            self.vae_lora_modules = inject_lora_into_vae_decoder(self.vae, r=self.lora_rank, lora_alpha=self.lora_alpha)

            # Count LoRA trainable parameters
            vae_lora_params = sum(p.numel() for p in self.vae.parameters() if p.requires_grad)
            # Count base weights stored as buffers (frozen)
            vae_base_params = sum(b.numel() for b in self.vae.buffers())
            print(f"  LoRA injected into {len(self.vae_lora_modules)} modules")
            print(f"  VAE base (frozen): {vae_base_params:,}")
            print(f"  VAE LoRA (trainable): {vae_lora_params:,}")
            if vae_base_params > 0:
                print(f"  Reduction: {100 * (1 - vae_lora_params / vae_base_params):.2f}%")
        else:
            print("VAE full fine-tuning enabled")
            self.vae_lora_modules = None

        self.vae.train()

        print(f"VAE loaded. Latent space: [{self.vae_latent_channels}, {self.vae_latent_h}, {self.vae_latent_w}]")

        # Trainable adapter: z → VAE latent space
        # Simple linear projection to transform latent codes to VAE latent space
        self.adapter = nn.Linear(latent_dim, self.vae_latent_dim)

        # Initialize adapter weights
        nn.init.kaiming_normal_(self.adapter.weight, mode='fan_out', nonlinearity='relu')
        if self.adapter.bias is not None:
            nn.init.constant_(self.adapter.bias, 0)

        # Skip connection: per-channel scaling modulation
        # Simple linear layer to learn a scalar scale factor from latent code
        # This provides a learned modulation that influences the CNN refiner and patch projector
        self.skip_projection = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.SiLU(inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid()  # Keep scaling factor in [0, 1] for stability
        )

        # CNN refinement module - deeper conv architecture
        # Input: VAE output (3 channels) + skip connection (1 channel) = 4 channels
        # Output: Feature maps (64 channels)
        self.cnn_refiner = nn.Sequential(
            # Block 1
            nn.Conv2d(4, 64, kernel_size=3, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(inplace=True),

            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.GroupNorm(8, 128),
            nn.SiLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.GroupNorm(8, 128),
            nn.SiLU(inplace=True),

            # Block 3
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(inplace=True),
        )

        # Initialize CNN weights
        for m in self.cnn_refiner.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        # Patch projection: simple 1x1 convolutions
        # Input: CNN output (64 channels) + skip features (1 channel) = 65 channels
        # Output: 3 RGB channels
        self.patch_projector = nn.Sequential(
            nn.Conv2d(65, 32, kernel_size=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(32, 3, kernel_size=1),
            nn.Sigmoid()  # Ensure output is in [0, 1]
        )

        # Initialize projector weights
        for m in self.patch_projector.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        # Bottleneck dense refiner for final refinement
        self.bottleneck_refiner = BottleneckDenseRefiner(patch_height, patch_width, latent_dim, bottleneck_dim)
        print(f"Bottleneck dense refiner enabled for final patch refinement (with seed conditioning, bottleneck_dim={bottleneck_dim})")

        print(f"CNN refiner initialized: 4 → 64 → 64 → 128 → 128 → 64 → 64 channels")
        print(f"Patch projector: 65 → 32 → 3 channels (1x1 convolutions, ~2.2K parameters)")

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

        # Skip connection: per-channel scaling modulation
        skip_scale = self.skip_projection(z)  # [B, 1] - learned scale factor
        skip_scale = skip_scale.view(batch_size, 1, 1, 1)  # [B, 1, 1, 1] for broadcasting

        # Expand skip_scale to match spatial dimensions for concatenation
        skip_features = skip_scale.expand(batch_size, 1, self.patch_height, self.patch_width)  # [B, 1, H, W]

        # Concatenate VAE output with skip connection for CNN input
        cnn_input = torch.cat([vae_output, skip_features], dim=1)  # [B, 4, H, W]

        # Process through CNN refiner
        cnn_output = self.cnn_refiner(cnn_input)  # [B, 64, H, W]

        # Concatenate CNN output with skip features (now just the broadcast scalar)
        projector_input = torch.cat([cnn_output, skip_features], dim=1)  # [B, 65, H, W]

        # Project to patch using 1x1 convolutions
        patches = self.patch_projector(projector_input)  # [B, 3, H, W]

        # Apply bottleneck dense refiner (with seed conditioning)
        patches = self.bottleneck_refiner(patches, z)

        return patches


class ProgressivePatchTrainer:
    """
    Progressive layer attack trainer.

    Trains adversarial patches by progressively targeting deeper layers,
    starting from early CNN features and moving towards final outputs.
    """
    def __init__(self,
                 ocr_dataset: str,
                 device: str = None,
                 grad_accumulate: int = None,
                 basis_dim: int = 16,
                 diversity_weight: float = 1.0,
                 quality_weight: float = 1.0,
                 performance_weight: float = 1.0,
                 tv_weight: float = 2.5,
                 spectrum_weight: float = 1.0,
                 layer_configs: Optional[List[LayerConfig]] = None,
                 ocr_dataset_split: str = 'train',
                 ocr_max_samples: Optional[int] = None,
                 ocr_images_per_batch: int = 1,
                 ocr_patches_per_image: int = None,
                 use_vae_lora: bool = True,
                 lora_rank: int = 8,
                 lora_alpha: int = 16,
                 bottleneck_dim: int = 256,
                 save_examples_every: Optional[int] = None):
        self.basis_dim = basis_dim
        self.diversity_weight = diversity_weight
        self.quality_weight = quality_weight
        self.performance_weight = performance_weight
        self.tv_weight = tv_weight
        self.spectrum_weight = spectrum_weight
        self.save_examples_every = save_examples_every
        self.ocr_images_per_batch = ocr_images_per_batch
        self.use_vae_lora = use_vae_lora
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.bottleneck_dim = bottleneck_dim

        # Require ocr_patches_per_image
        if ocr_patches_per_image is None:
            raise ValueError("ocr_patches_per_image is required")
        self.ocr_patches_per_image = ocr_patches_per_image
        print(f"OCR batch config: {ocr_images_per_batch} images × {self.ocr_patches_per_image} patches = {ocr_images_per_batch * self.ocr_patches_per_image} total")

        # Always use 80/20 validation split in OCR mode
        self.use_all_for_train = False

        # Progressive layer configuration
        # Always target the last layer (FinalOutput)
        self.layer_configs = layer_configs or get_ocr_layer_progression()

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

        # Load OCR datasets
        dataset_names = [name.strip() for name in ocr_dataset.split(',')]

        print(f"\nUsing public OCR dataset(s): {', '.join(dataset_names)} (split: {ocr_dataset_split})")

        # Validate all datasets exist
        for dataset_name in dataset_names:
            if dataset_name not in DATASETS:
                available = ', '.join(sorted(DATASETS.keys()))
                raise ValueError(
                    f"Unknown dataset '{dataset_name}'. Available: {available}"
                )

        # Load all datasets
        datasets_to_combine = []
        for dataset_name in dataset_names:
            print(f"\nLoading {dataset_name}...")
            dataset = OCRDataset(
                dataset_name=dataset_name,
                split=ocr_dataset_split,
                transform=self.transform,
                max_samples=ocr_max_samples
            )
            datasets_to_combine.append(dataset)
            print(f"  Loaded {len(dataset)} samples from {dataset_name}")

        # Combine datasets if multiple, otherwise use single dataset
        if len(datasets_to_combine) > 1:
            full_dataset = ConcatDataset(datasets_to_combine)
            print(f"\nCombined {len(dataset_names)} datasets: {len(full_dataset)} total samples")
        else:
            full_dataset = datasets_to_combine[0]

        # Store for later use in train()
        self.full_dataset = full_dataset

        # Split into train (80%) and val (20%)
        train_size = int(0.8 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        train_dataset, val_dataset = random_split(
            full_dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )

        # Save train/val split mapping to CSV (immediately)
        self._save_train_val_split(
            full_dataset,
            train_dataset,
            val_dataset,
            datasets_to_combine
        )

        # Create dataloaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=1,
            shuffle=True,
            num_workers=0,
            pin_memory=True if self.device == 'cuda' else False
        )
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            pin_memory=True if self.device == 'cuda' else False
        )

        print(f"Split: {len(train_dataset)} train, {len(val_dataset)} val")

        # Initialize generator (foundation model with trainable VAE)
        # num_layers is fixed at 11 for the OCR model
        num_layers = len(self.layer_configs)

        self.generator = FoundationPatchGenerator(
            latent_dim=basis_dim,
            patch_height=self.patch_height,
            patch_width=self.patch_width,
            num_layers=num_layers,
            use_vae_lora=use_vae_lora,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            use_bottleneck_refiner=True,
            bottleneck_dim=bottleneck_dim
        ).to(self.device)

        # Activation capture for diversity metric
        self.ocr_activations = None  # Current activations from forward hook
        self.baseline_ocr_activations = {}  # Dict mapping image index to baseline activations
        self.activation_hook = None  # Handle for the forward hook
        self.layer_activation_stddev = {}  # Dict mapping layer_idx to std_dev of activations

        # Load OCR model for diversity computation
        self.load_ocr_model()

        # Track statistics
        self.epoch_stats = []

    def load_ocr_model(self):
        """Load OCR model for diversity computation only. Auto-initializes if missing."""
        print("Loading OCR model for diversity computation...")
        ocr_path = Path.home() / ".cache/fast-plate-ocr/cct-xs-v1-global-model/cct_xs_v1_global.onnx"

        if not ocr_path.exists():
            print(f"⚠️  OCR model not found at: {ocr_path}")
            print(f"📥 Auto-initializing OCR model (downloading models)...")
            try:
                # Import and run init_alpr.py logic to download models
                import sys
                current_dir = Path(__file__).parent
                sys.path.insert(0, str(current_dir))
                from fast_alpr import ALPR
                # Creating ALPR triggers model downloads
                alpr = ALPR(
                    detector_model="yolo-v9-t-384-license-plate-end2end",
                    ocr_model="cct-xs-v1-global-model",
                )
                print(f"✓ OCR model downloaded successfully")
            except Exception as e:
                raise RuntimeError(
                    f"Failed to auto-initialize OCR model: {str(e)}\n"
                    f"Please ensure fast_alpr is installed: pip install fast-alpr"
                )

        # Verify model exists after initialization attempt
        if not ocr_path.exists():
            raise FileNotFoundError(
                f"OCR model still not found at: {ocr_path}\n"
                f"Try manually running: python init_alpr.py"
            )

        ocr_model = onnx.load(str(ocr_path))
        self.ocr_input_shape = (64, 128, 3)

        self.ocr = onnx2torch.convert(ocr_model).to(self.device)
        self.ocr.eval()

        # Disable gradients for OCR model parameters to save memory
        for param in self.ocr.parameters():
            param.requires_grad = False

        # Calculate baseline activations for diversity computation
        # (Dynamic hooks are registered per-layer during training via _get_multi_layer_activations)
        self.calculate_baseline_activations()

    def _save_train_val_split(self, full_dataset, train_dataset, val_dataset, datasets_list, save_dir=None):
        """
        Save train/val split mapping to CSV file for tracking which images are used.

        Args:
            full_dataset: Combined dataset (ConcatDataset or single dataset)
            train_dataset: Training subset (result of random_split)
            val_dataset: Validation subset (result of random_split)
            datasets_list: List of OCRDataset objects that were combined
            save_dir: Optional directory to save CSV to (default: current directory)
        """
        import csv
        from datetime import datetime

        # Get indices from the random_split
        train_indices = set(train_dataset.indices) if hasattr(train_dataset, 'indices') else set()
        val_indices = set(val_dataset.indices) if hasattr(val_dataset, 'indices') else set()

        # Build metadata for all samples
        split_data = []

        for idx in range(len(full_dataset)):
            # Determine which split this index belongs to
            if idx in train_indices:
                split_type = 'train'
            elif idx in val_indices:
                split_type = 'val'
            else:
                split_type = 'unknown'

            # Get sample information
            sample = full_dataset[idx]
            text = sample.get('text', '')
            dataset_name = sample.get('dataset', 'unknown')

            split_data.append({
                'index': idx,
                'split': split_type,
                'dataset': dataset_name,
                'text': text
            })

        # Save to CSV
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        csv_filename = f'train_val_split_{timestamp}.csv'

        # If save_dir provided, use it; otherwise use current directory
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            csv_filepath = os.path.join(save_dir, csv_filename)
        else:
            csv_filepath = csv_filename

        with open(csv_filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['index', 'split', 'dataset', 'text'])
            writer.writeheader()
            writer.writerows(split_data)

        print(f"\nTrain/Val split mapping saved to: {csv_filepath}")
        print(f"  Train samples: {len(train_indices)}")
        print(f"  Val samples: {len(val_indices)}")

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
        """Sample z ~ Uniform(0, 1) for generating patches"""
        return torch.rand(batch_size, self.basis_dim, device=self.device)

    def generate_patches(self, z: torch.Tensor) -> torch.Tensor:
        """
        Generate patches from latent codes using foundation model:
        z → adapter → frozen VAE decoder → CNN refiner → patch projector (1×1 convs) → patch

        Args:
            z: Latent codes [batch_size, basis_dim]

        Returns:
            patches: [batch_size, 3, H, W] in [0, 1]
        """
        # Pass through full generator pipeline
        patches = self.generator(z)  # [batch_size, 3, H, W], already in [0, 1]

        return patches

    def create_border_mask(self, height: int, width: int, border_scale: float = 1.4) -> torch.Tensor:
        """
        Create a mask for the visible border region of the patch.

        When applied, the patch fills a border_scale region (1.4x) around the plate,
        but the center plate region (1.0x) gets cut out. This mask identifies which
        parts of the patch are actually visible (border) vs obscured (center plate).

        Args:
            height: Patch height
            width: Patch width
            border_scale: Scale factor for border (default 1.4)

        Returns:
            mask: [1, 1, H, W] tensor, 1 = visible border, 0 = obscured plate region
        """
        # The center region (1.0x / border_scale of the patch) is obscured
        plate_scale_ratio = 1.0 / border_scale  # ~0.714 for border_scale=1.4

        # Calculate the size of the obscured center region
        center_h = int(height * plate_scale_ratio)
        center_w = int(width * plate_scale_ratio)

        # Create mask: all ones (visible) except center region (obscured)
        mask = torch.ones(1, 1, height, width)

        # Calculate bounds of center region to mask out
        h_start = (height - center_h) // 2
        h_end = h_start + center_h
        w_start = (width - center_w) // 2
        w_end = w_start + center_w

        # Set center region to 0 (obscured by plate)
        mask[:, :, h_start:h_end, w_start:w_end] = 0.0

        return mask

    def compute_spectrum_loss(self, patches: torch.Tensor) -> torch.Tensor:
        """
        Compute SSIM-based structural similarity penalty to encourage patch diversity.

        SSIM (Structural Similarity Index) measures similarity based on luminance,
        contrast, and structure. We penalize high SSIM between patch pairs to force
        genuinely different structures across patches in the batch.

        High SSIM penalty = patches have similar structures
        Low SSIM penalty = patches have different structures (good)

        Process:
        1. Apply border mask to focus on visible region
        2. Compute pairwise SSIM between all patch pairs
        3. Average SSIM across visible regions
        4. Return mean pairwise SSIM as the penalty

        Args:
            patches: [batch_size, 3, H, W] patch tensor in [0, 1]

        Returns:
            torch.Tensor: Scalar loss (mean pairwise SSIM penalty)
        """
        batch_size = patches.shape[0]
        if batch_size < 2:
            return torch.tensor(0.0, device=patches.device, dtype=patches.dtype)

        _, _, H, W = patches.shape
        border_mask = self.create_border_mask(H, W, border_scale=1.4).to(patches.device)  # [1, 1, H, W]

        # Compute pairwise SSIM between all patches
        ssim_sum = 0.0
        pair_count = 0

        for i in range(batch_size):
            for j in range(i + 1, batch_size):
                # SSIM expects [B, C, H, W], so add batch dim to each patch
                patch_i = patches[i:i+1]  # [1, 3, H, W]
                patch_j = patches[j:j+1]  # [1, 3, H, W]

                # Compute SSIM using kornia (returns per-pixel map [B, 3, H, W])
                # window_size=11 is standard for SSIM
                ssim_map = ssim(patch_i, patch_j, window_size=11)  # [1, 3, H, W]

                # Average across channels: [1, 3, H, W] -> [1, 1, H, W]
                ssim_map_avg = ssim_map.mean(dim=1, keepdim=True)

                # Apply border mask: only consider visible regions
                masked_ssim = ssim_map_avg * border_mask  # [1, 1, H, W]

                # Compute mean only over visible border region
                num_visible_pixels = border_mask.sum()
                if num_visible_pixels > 0:
                    ssim_val = masked_ssim.sum() / num_visible_pixels
                else:
                    ssim_val = 0.0

                ssim_sum += ssim_val
                pair_count += 1

        # Return mean SSIM across all pairs
        # Minimizing this penalty maximizes structural diversity
        if pair_count > 0:
            return ssim_sum / pair_count
        else:
            return torch.tensor(0.0, device=patches.device, dtype=patches.dtype)

    def total_variation_loss(self, patches: torch.Tensor) -> torch.Tensor:
        """
        Compute total variation (TV) regularization loss for the adversarial patches.

        Encourages smooth, natural-looking patches by penalizing large differences
        between adjacent pixels. Uses L2 norm of gradients in both horizontal and
        vertical directions.

        Only computes TV on the visible border region (excludes center plate region
        that gets obscured when patch is applied).

        Args:
            patches: [batch_size, 3, H, W] patch tensor

        Returns:
            torch.Tensor: Scalar regularization loss (normalized)
        """
        # Average over batch dimension: compute TV for each patch and then average
        batch_size = patches.shape[0]
        total_tv_loss = 0.0

        # Create border mask (same for all patches)
        _, _, H, W = patches.shape
        border_mask = self.create_border_mask(H, W, border_scale=1.4).to(patches.device)  # [1, 1, H, W]

        for i in range(batch_size):
            patch = patches[i:i+1]  # [1, 3, H, W]
            C, H, W = patch.shape[1:]

            # Horizontal total variation: differences along width dimension
            tv_h = torch.pow(patch[:, :, :, 1:] - patch[:, :, :, :-1], 2)

            # Vertical total variation: differences along height dimension
            tv_v = torch.pow(patch[:, :, 1:, :] - patch[:, :, :-1, :], 2)

            # Apply border mask to both TV components
            # Mask is [1, 1, H, W], need to match dimensions for masking
            mask_h = border_mask[:, :, :, :-1]  # [1, 1, H, W-1]
            mask_v = border_mask[:, :, :-1, :]  # [1, 1, H-1, W]

            tv_h_masked = (tv_h * mask_h).sum()
            tv_v_masked = (tv_v * mask_v).sum()

            # Count only visible pixels in the border region
            num_visible_h = mask_h.sum()
            num_visible_v = mask_v.sum()

            # Normalize by number of visible comparisons
            patch_tv_loss = (tv_h_masked + tv_v_masked) / (num_visible_h + num_visible_v) if (num_visible_h + num_visible_v) > 0 else torch.tensor(0.0, device=patches.device)

            # Scale by 2.5x to keep loss in reasonable range
            patch_tv_loss = patch_tv_loss * 2.5

            total_tv_loss += patch_tv_loss

        # Average over batch
        avg_tv_loss = total_tv_loss / batch_size

        return avg_tv_loss

    def apply_patch_ocr_mode(self, image: torch.Tensor, patch: torch.Tensor,
                           center_ratio: float = 0.6) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Apply adversarial patch in OCR mode (simplified for cropped license plates).

        In OCR mode, we work with cropped license plate images directly.
        The patch is resized to match the image dimensions, then the center region
        is cut out and replaced with the original plate content.

        Args:
            image: [B, 3, H, W] - cropped license plate images
            patch: [3, patch_h, patch_w] - generated adversarial patch (256x512)
            center_ratio: Fraction of image to preserve in center (default: 0.6)

        Returns:
            result_image: [B, 3, H, W] - image with patch applied as border
            center_mask: [B, 3, H, W] - mask showing preserved center region
        """
        batch_size = image.shape[0]
        image_height, image_width = image.shape[2], image.shape[3]

        # Resize patch to match image dimensions
        # patch is [3, patch_h, patch_w], need to add batch dim and resize
        patch_resized = F.interpolate(
            patch.unsqueeze(0),  # [1, 3, patch_h, patch_w]
            size=(image_height, image_width),
            mode='bilinear',
            align_corners=False
        )  # [1, 3, H, W]

        # Expand to batch size
        patch_batch = patch_resized.repeat(batch_size, 1, 1, 1)  # [B, 3, H, W]

        # Create center mask (1 in center, 0 on borders)
        center_h = int(image_height * center_ratio)
        center_w = int(image_width * center_ratio)

        # Calculate padding to center the mask
        pad_h = (image_height - center_h) // 2
        pad_w = (image_width - center_w) // 2

        # Create mask: 1 in center region, 0 elsewhere
        center_mask = torch.zeros(batch_size, 1, image_height, image_width,
                                 dtype=torch.float32, device=self.device)
        center_mask[:, :, pad_h:pad_h+center_h, pad_w:pad_w+center_w] = 1.0
        center_mask = center_mask.expand(-1, 3, -1, -1)  # [B, 3, H, W]

        # Blend: keep original image in center, use patch on borders
        result_image = image * center_mask + patch_batch * (1 - center_mask)
        result_image = torch.clamp(result_image, 0, 1)

        return result_image, center_mask

    def apply_neutral_border_ocr_mode(self, image: torch.Tensor, center_ratio: float = 0.6,
                                       border_color: float = 0.5) -> torch.Tensor:
        """
        Apply a neutral gray/black border in OCR mode for fair baseline comparison.

        This matches the spatial structure of apply_patch_ocr_mode but uses a neutral color
        instead of adversarial content. Used for baseline computation so the only difference
        between baseline and patched images is the adversarial patch content, not the presence
        of a border region.

        Args:
            image: [B, 3, H, W] - cropped license plate images
            center_ratio: Fraction of image to preserve in center (default: 0.6)
            border_color: Value for neutral border (default: 0.5 = gray)

        Returns:
            result_image: [B, 3, H, W] - image with neutral border applied
        """
        batch_size = image.shape[0]
        image_height, image_width = image.shape[2], image.shape[3]

        # Create center mask (1 in center, 0 on borders)
        center_h = int(image_height * center_ratio)
        center_w = int(image_width * center_ratio)

        # Calculate padding to center the mask
        pad_h = (image_height - center_h) // 2
        pad_w = (image_width - center_w) // 2

        # Create mask: 1 in center region, 0 elsewhere
        center_mask = torch.zeros(batch_size, 1, image_height, image_width,
                                 dtype=torch.float32, device=self.device)
        center_mask[:, :, pad_h:pad_h+center_h, pad_w:pad_w+center_w] = 1.0
        center_mask = center_mask.expand(-1, 3, -1, -1)  # [B, 3, H, W]

        # Create neutral border (gray by default)
        neutral_border = torch.full_like(image, border_color)

        # Blend: keep original image in center, use neutral border on borders
        result_image = image * center_mask + neutral_border * (1 - center_mask)
        result_image = torch.clamp(result_image, 0, 1)

        return result_image

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
            # OCR mode: work with cropped license plates directly
            # Use prep_image (already cropped plate) instead of orig_image
            prep_image = batch['prep_image'].to(self.device).unsqueeze(0)  # [1, 3, H, W]

            # Apply patch in OCR mode (simplified: patch as border, keep center)
            patched_image, _ = self.apply_patch_ocr_mode(prep_image, patch)

            # Crop and resize to OCR input shape
            cropped_plate = F.interpolate(
                patched_image,
                size=self.ocr_input_shape[:2],
                mode='bilinear',
                align_corners=False
            )

            # Run OCR on cropped plate
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
                                      baseline_activations: List[torch.Tensor],
                                      diagonal_activations: List[torch.Tensor],
                                      use_grad: bool = False) -> torch.Tensor:
        """
        Compute diversity as the determinant of the gram matrix of patch activation deltas.

        For each patch, compute the activation delta (activation with patch - baseline without patch)
        on its corresponding image. Then measure diversity as how different these deltas are
        in activation space via the log determinant of the gram matrix.

        Args:
            patches_list: List of [3, H, W] patches (one per image)
            batches_list: List of batch dicts (one per image)
            baseline_activations: List of [H, W, C] baseline activations (pre-computed)
            diagonal_activations: List of [H, W, C] activations from (patch_i, image_i) pairs
            use_grad: If True, compute with gradients (needed for diversity-only mode)

        Returns:
            log_det: scalar diversity score
        """
        batch_size = len(patches_list)

        # Compute activation delta for each patch: delta_i = activation_i - baseline
        deltas = []
        for patch_idx in range(batch_size):
            activations = diagonal_activations[patch_idx]  # [H, W, C]
            baseline = baseline_activations[patch_idx]  # [H, W, C]
            delta = activations - baseline  # [H, W, C]
            deltas.append(delta)

        # Stack deltas: [batch_size, H, W, C]
        deltas_stacked = torch.stack(deltas, dim=0)
        # Flatten: [batch_size, H*W*C]
        deltas_flat = deltas_stacked.reshape(batch_size, -1)

        # L2 normalize each delta vector to unit length
        normalized = F.normalize(deltas_flat, p=2, dim=1)  # [batch_size, H*W*C]

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

    def get_baseline_activation(self, batch: dict, idx: int) -> torch.Tensor:
        """
        Get baseline activation for an image (compute on-the-fly if needed).

        Applies a neutral gray border to match the spatial structure of adversarial patches,
        ensuring fair comparison between baseline and patched activations.

        In OCR mode: computes on-the-fly and caches (memory efficient).
        In standard mode: uses pre-computed baselines.

        Args:
            batch: Batch dict with image data
            idx: Dataset index

        Returns:
            baseline: [H, W, C] baseline activation
        """
        # Check if already cached
        if idx in self.baseline_ocr_activations:
            return self.baseline_ocr_activations[idx]

        # Compute on-the-fly
        with torch.no_grad():
            # OCR mode: use prep_image directly (already cropped)
            prep_image = batch['prep_image'].unsqueeze(0).to(self.device)

            # Apply neutral border to match patch structure
            prep_image_with_border = self.apply_neutral_border_ocr_mode(prep_image)

            cropped_plate = F.interpolate(
                prep_image_with_border,
                size=self.ocr_input_shape[:2],
                mode='bilinear',
                align_corners=False
            )

            ocr_input = cropped_plate.permute(0, 2, 3, 1) * 255
            self.ocr(ocr_input)  # Forward pass captures activations via hook

            # Extract baseline activation
            if self.ocr_activations is not None:
                baseline_act = self.ocr_activations.squeeze(0).detach().clone()
                # Cache it
                self.baseline_ocr_activations[idx] = baseline_act
                # Clear hook reference
                self.ocr_activations = None
                return baseline_act
            else:
                # Fallback
                if hasattr(self, 'activation_shape'):
                    return torch.zeros(self.activation_shape, device=self.device)
                else:
                    return torch.zeros(1, 1, 48, device=self.device)

    def profile_layer_activations(self, num_samples: int = 1024):
        """
        Profile activation deltas from uniform random pixel patches vs neutral baseline.

        For each image, computes activations with:
        1. Neutral border (baseline)
        2. Uniform random pixel patch applied as border (matching training spatial layout)

        Profiles only the final layer (FinalOutput) to match training focus.

        Uses torch.rand pixel patches (not generator output) to ensure maximum diversity
        and proper activation scale calibration. Generator-initialized patches would all
        be nearly identical, yielding near-zero std and blown-up normalized quality scores.

        Args:
            num_samples: Number of images to sample for profiling (default: 1024)
        """
        print(f"\nProfiling patch-induced activation deltas on {num_samples} random images (final layer only)...")

        num_samples = min(num_samples, len(self.train_loader))
        target_layer_idx = len(self.layer_configs) - 1  # Final layer only
        layer_deltas = []  # List of delta vectors for target layer

        # Sample random images from training set
        self.ocr.eval()
        with torch.no_grad():
            for sample_idx in range(num_samples):
                if sample_idx % 100 == 0:
                    print(f"  Profiling: {sample_idx}/{num_samples}")

                # Get random sample from training set
                batch_idx = np.random.randint(0, len(self.train_loader.dataset))
                batch_item = self.train_loader.dataset[batch_idx]
                batch_dict = batch_item
                prep_image = batch_dict['prep_image'].unsqueeze(0).to(self.device)

                # 1. Get baseline activations (neutral border)
                prep_image_baseline = self.apply_neutral_border_ocr_mode(prep_image)
                cropped_baseline = F.interpolate(
                    prep_image_baseline,
                    size=self.ocr_input_shape[:2],
                    mode='bilinear',
                    align_corners=False
                )
                ocr_input_baseline = cropped_baseline.permute(0, 2, 3, 1) * 255
                baseline_acts = self._capture_activations_target_layer(ocr_input_baseline, target_layer_idx)

                # 2. Generate uniform random pixel patch and apply as border
                # Use torch.rand (not generator) to ensure maximum patch diversity
                random_patch = torch.rand(3, self.patch_height, self.patch_width, device=self.device)

                # Apply patch as border using the same method as training
                prep_image_patched, _ = self.apply_patch_ocr_mode(prep_image, random_patch)

                cropped_patched = F.interpolate(
                    prep_image_patched,
                    size=self.ocr_input_shape[:2],
                    mode='bilinear',
                    align_corners=False
                )
                ocr_input_patched = cropped_patched.permute(0, 2, 3, 1) * 255
                patched_acts = self._capture_activations_target_layer(ocr_input_patched, target_layer_idx)

                # 3. Compute delta for target layer
                delta = patched_acts - baseline_acts
                layer_deltas.append(delta)

        # Compute mean and std_dev of deltas for the target layer
        self.layer_activation_stddev = {}
        deltas_stacked = torch.stack(layer_deltas, dim=0)
        mean_delta = deltas_stacked.mean(dim=0)
        std_delta = deltas_stacked.std(dim=0)
        # Use std_delta for normalization; clamp to avoid division issues
        std_delta = torch.clamp(std_delta, min=1e-6)
        self.layer_activation_stddev[target_layer_idx] = std_delta.to(self.device)

        layer_config = self.layer_configs[target_layer_idx]
        print(f"✓ Profiled final layer ({target_layer_idx}): {layer_config.description}")
        print(f"  Layer {target_layer_idx}: {std_delta.shape} activations, mean_delta_std={std_delta.mean():.6f}")

    def _capture_activations(self, ocr_input: torch.Tensor) -> Dict[int, torch.Tensor]:
        """Helper to capture activations at all layers for a given input."""
        activations_dict = {}
        hooks = []

        # Register hooks for all layers
        for layer_idx, layer_config in enumerate(self.layer_configs):
            def make_hook(idx):
                def hook(module, input, output):
                    if isinstance(output, torch.Tensor):
                        activations_dict[idx] = output.squeeze(0).reshape(-1).detach().cpu()
                return hook

            layer_module = self._find_layer_by_name(layer_config.name)
            if layer_module is not None:
                hooks.append(layer_module.register_forward_hook(make_hook(layer_idx)))

        # Forward pass
        self.ocr(ocr_input)

        # Remove hooks
        for hook in hooks:
            hook.remove()

        return activations_dict

    def _capture_activations_target_layer(self, ocr_input: torch.Tensor, target_layer_idx: int) -> torch.Tensor:
        """Helper to capture activations for only the target layer."""
        activations_dict = {}
        hooks = []

        # Register hook only for target layer
        def make_hook(idx):
            def hook(module, input, output):
                if isinstance(output, torch.Tensor):
                    activations_dict[idx] = output.squeeze(0).reshape(-1).detach().cpu()
            return hook

        layer_config = self.layer_configs[target_layer_idx]
        layer_module = self._find_layer_by_name(layer_config.name)
        if layer_module is not None:
            hooks.append(layer_module.register_forward_hook(make_hook(target_layer_idx)))

        # Forward pass
        self.ocr(ocr_input)

        # Remove hooks
        for hook in hooks:
            hook.remove()

        return activations_dict.get(target_layer_idx, torch.zeros(1, device=self.device))

    def _find_layer_by_name(self, layer_name: str):
        """Find a module in the OCR model by its name."""
        for name, module in self.ocr.named_modules():
            if name == layer_name or layer_name in name:
                return module
        return None

    def calculate_baseline_activations(self):
        """
        Baselines are computed on-the-fly in OCR mode (memory efficient).
        """
        print("Computing baselines on-the-fly (memory efficient)")

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
        layer_checkpoint_path = os.path.join(self.checkpoint_base, layer_checkpoint_name)
        self.save_basis(self.current_layer_epoch, layer_checkpoint_path, num_samples=10)
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

    def _get_multi_layer_activations(self, batch: dict, patch: torch.Tensor, layer_indices: List[int],
                                      use_grad: bool = False) -> Dict[int, torch.Tensor]:
        """
        Get activations from multiple layers in a single forward pass.

        Args:
            batch: Single unbatched batch item from dataloader
            patch: [3, H, W] patch tensor
            layer_indices: List of layer indices to capture activations from
            use_grad: If True, compute with gradients

        Returns:
            Dictionary mapping layer_idx -> activation tensor [H, W, C]
        """
        context = torch.no_grad() if not use_grad else torch.enable_grad()
        with context:
            # Apply patch
            prep_image = batch['prep_image'].to(self.device).unsqueeze(0)
            patched_image, _ = self.apply_patch_ocr_mode(prep_image, patch)

            # Crop and resize to OCR input shape
            cropped_plate = F.interpolate(
                patched_image,
                size=self.ocr_input_shape[:2],
                mode='bilinear',
                align_corners=False
            )

            # Register hooks on all requested layers
            activations_dict = {}
            hooks = []

            for layer_idx in layer_indices:
                if layer_idx >= len(self.layer_configs):
                    continue

                def make_hook(idx):
                    def hook(module, input, output):
                        if isinstance(output, torch.Tensor):
                            activations_dict[idx] = output.squeeze(0).detach().clone() if not use_grad else output.squeeze(0)
                    return hook

                layer_config = self.layer_configs[layer_idx]
                layer_module = self._find_layer_by_name(layer_config.name)
                if layer_module is not None:
                    hooks.append(layer_module.register_forward_hook(make_hook(layer_idx)))

            # Forward pass
            ocr_input = cropped_plate.permute(0, 2, 3, 1) * 255
            self.ocr(ocr_input)

            # Remove hooks
            for hook in hooks:
                hook.remove()

            # Clear global activation reference
            self.ocr_activations = None

            return activations_dict

    def train_epoch(self, optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler.LambdaLR, epoch: int) -> Tuple[float, float, float]:
        """Train for one epoch targeting the final layer.

        For each image, trains patches to:
        1. Maximize diversity at the final layer (diversity score)
        2. Achieve high quality score at the final layer (quality = normalized activation delta)

        Loss = -(performance_weight * (diversity_weight * diversity_score + quality_weight * quality_score))
        """
        total_diversity_loss = 0.0
        total_tv_loss = 0.0
        total_spectrum_loss = 0.0
        total_raw_diversity_score = 0.0
        total_raw_quality_score = 0.0
        num_updates = 0

        # OCR mode: Generate multiple patches per image, process multiple images per batch
        dataloader_iter = iter(self.train_loader)
        num_images = len(self.train_loader)
        images_per_batch = self.ocr_images_per_batch
        patches_per_image = self.ocr_patches_per_image

        # Calculate number of batches
        import math
        num_batches = math.ceil(num_images / images_per_batch)
        print(f"  Train epoch {epoch}: {num_images} images / {images_per_batch} per batch = {num_batches} batches")

        desc = f"Epoch {epoch} - Training"

        # Create example samples directory if needed
        example_samples_dir = None
        if self.save_examples_every is not None:
            example_samples_dir = os.path.join(self.checkpoint_base, "example_samples")
            os.makedirs(example_samples_dir, exist_ok=True)

        batch_count_global = 0
        with tqdm(total=num_batches, desc=desc, leave=False) as pbar:
            while True:
                try:
                    # Accumulate losses for this batch
                    batch_diversity_loss = 0.0
                    batch_tv_loss = 0.0
                    batch_spectrum_loss = 0.0
                    batch_raw_diversity_score = 0.0
                    batch_raw_quality_score = 0.0
                    batch_count = 0

                    # Process images_per_batch images
                    for img_in_batch in range(images_per_batch):
                        # Get one image
                        batch = next(dataloader_iter)
                        single_batch = {k: v[0] for k, v in batch.items()}

                        # Always target the last layer (FinalOutput)
                        sampled_layer_idx = len(self.layer_configs) - 1

                        # Generate patches_per_image patches for this image
                        accumulated_patches = []
                        accumulated_batches = []
                        layer_indices_to_capture = list(range(sampled_layer_idx + 1))  # 0 to sampled_layer_idx

                        for patch_num in range(patches_per_image):
                            z = self.sample_coefficients(1)
                            patch = self.generate_patches(z)[0]  # [3, H, W]
                            accumulated_patches.append(patch)
                            accumulated_batches.append({k: v.detach().clone() if torch.is_tensor(v) else v
                                                       for k, v in single_batch.items()})

                        # Compute baseline activations for target layer (with neutral border)
                        prep_image = single_batch['prep_image'].to(self.device).unsqueeze(0)

                        # Apply neutral border to match patch structure for fair comparison
                        prep_image_with_border = self.apply_neutral_border_ocr_mode(prep_image)

                        cropped_plate = F.interpolate(
                            prep_image_with_border,
                            size=self.ocr_input_shape[:2],
                            mode='bilinear',
                            align_corners=False
                        )
                        baseline_activations_dict = {}

                        # Register hook for target layer only
                        def make_hook(idx):
                            def hook(module, input, output):
                                if isinstance(output, torch.Tensor):
                                    baseline_activations_dict[idx] = output.squeeze(0).detach().clone()
                            return hook

                        layer_config = self.layer_configs[sampled_layer_idx]
                        layer_module = self._find_layer_by_name(layer_config.name)
                        baseline_hook = None
                        if layer_module is not None:
                            baseline_hook = layer_module.register_forward_hook(make_hook(sampled_layer_idx))

                        # Forward pass without patch
                        ocr_input = cropped_plate.permute(0, 2, 3, 1) * 255
                        self.ocr(ocr_input)
                        baseline_activation = baseline_activations_dict.get(sampled_layer_idx)

                        # Remove baseline hook
                        if baseline_hook is not None:
                            baseline_hook.remove()

                        # Get activations for all patches at target layer only
                        target_layer_activations = []
                        for patch in accumulated_patches:
                            activations_dict = self._get_multi_layer_activations(
                                single_batch, patch, layer_indices_to_capture, use_grad=True
                            )
                            target_layer_activations.append(activations_dict.get(sampled_layer_idx))

                        # Create baseline list for target layer (one per accumulated patch)
                        baseline_for_diversity = [baseline_activation] * len(accumulated_patches) if baseline_activation is not None else None

                        # Compute diversity score
                        diversity_score = self.compute_activation_diversity(
                            accumulated_patches,
                            accumulated_batches,
                            baseline_for_diversity,
                            target_layer_activations,
                            use_grad=True
                        )

                        # Compute quality score at target layer
                        quality_scores = []
                        if sampled_layer_idx in self.layer_activation_stddev and baseline_activation is not None:
                            std_dev = self.layer_activation_stddev[sampled_layer_idx]
                            for patch_act in target_layer_activations:
                                if patch_act is not None:
                                    # Compute delta
                                    delta = patch_act - baseline_activation
                                    delta_flat = delta.flatten()

                                    # Normalize by std_dev
                                    normalized_delta = delta_flat / (std_dev + 1e-8)
                                    # Use RMS (dimension-independent) instead of L2 norm
                                    normalized_delta_rms = (normalized_delta ** 2).mean().sqrt()

                                    # Quality score: normalized activation delta (dimension-independent)
                                    quality_scores.append(normalized_delta_rms)
                        else:
                            # If no stddev profiling, use ones
                            quality_scores = [torch.tensor(1.0, device=self.device) for _ in accumulated_patches]

                        quality_score = torch.stack(quality_scores).mean() if quality_scores else torch.tensor(1.0, device=self.device)

                        log_quality_score = torch.log(quality_score + 1e-8)
                        combined = self.diversity_weight * diversity_score + self.quality_weight * log_quality_score
                        total_loss = -(self.performance_weight * combined)

                        # Stack patches for batch operations
                        patches_stacked = torch.stack(accumulated_patches, dim=0)

                        # Compute TV loss
                        tv_loss = self.total_variation_loss(patches_stacked)
                        tv_loss_weighted = self.tv_weight * tv_loss

                        # Compute spectrum diversity loss
                        spectrum_loss = self.compute_spectrum_loss(patches_stacked)
                        spectrum_loss_weighted = self.spectrum_weight * spectrum_loss

                        # Final combined loss
                        final_loss = total_loss + tv_loss_weighted + spectrum_loss_weighted

                        # Backward (accumulate gradients)
                        final_loss.backward()

                        # Accumulate losses for batch
                        batch_diversity_loss += total_loss.item()
                        batch_tv_loss += tv_loss_weighted.item()
                        batch_spectrum_loss += spectrum_loss_weighted.item()
                        # Accumulate raw diversity and quality scores
                        batch_raw_diversity_score += diversity_score.item() if isinstance(diversity_score, torch.Tensor) else diversity_score
                        batch_raw_quality_score += log_quality_score.item() if isinstance(log_quality_score, torch.Tensor) else log_quality_score

                        batch_count += 1

                        # Memory cleanup
                        if self.device == 'cuda':
                            torch.cuda.empty_cache()
                        elif self.device == 'mps':
                            torch.mps.empty_cache()

                    # Update weights after processing all images in batch
                    if batch_count > 0:
                        torch.nn.utils.clip_grad_norm_(self.generator.parameters(), max_norm=1.0)
                        optimizer.step()
                        scheduler.step()  # Step-based learning rate scheduling (per batch)
                        optimizer.zero_grad()
                        num_updates += 1

                        # Track losses: divide by batch_count to get per-patch average
                        total_diversity_loss += batch_diversity_loss / batch_count
                        total_tv_loss += batch_tv_loss / batch_count
                        total_spectrum_loss += batch_spectrum_loss / batch_count
                        total_raw_diversity_score += batch_raw_diversity_score / batch_count
                        total_raw_quality_score += batch_raw_quality_score / batch_count

                except StopIteration:
                    # Dataloader exhausted - apply remaining gradients if any were accumulated
                    if batch_count > 0:
                        torch.nn.utils.clip_grad_norm_(self.generator.parameters(), max_norm=1.0)
                        optimizer.step()
                        scheduler.step()
                        optimizer.zero_grad()
                        num_updates += 1

                        # Track losses
                        total_diversity_loss += batch_diversity_loss / batch_count
                        total_tv_loss += batch_tv_loss / batch_count
                        total_spectrum_loss += batch_spectrum_loss / batch_count
                        total_raw_diversity_score += batch_raw_diversity_score / batch_count
                        total_raw_quality_score += batch_raw_quality_score / batch_count

                        # Update progress bar
                        batch_count_global += 1
                        pbar.update(1)
                    # Exit the main loop
                    break

                # Update progress bar: show average loss per patch (divided by num_patches_processed)
                total_patches_processed = num_updates * batch_count if num_updates > 0 else 1
                avg_diversity_loss = total_diversity_loss / num_updates if num_updates > 0 else 0
                avg_tv_loss = total_tv_loss / num_updates if num_updates > 0 else 0
                avg_spectrum_loss = total_spectrum_loss / num_updates if num_updates > 0 else 0
                avg_raw_diversity = total_raw_diversity_score / num_updates if num_updates > 0 else 0
                avg_log_quality = total_raw_quality_score / num_updates if num_updates > 0 else 0
                pbar.set_postfix({
                    'DivScore': f"{avg_raw_diversity:.4f}",
                    'LogQual': f"{avg_log_quality:.4f}",
                    'DivLoss': f"{avg_diversity_loss:.4f}",
                    'TVLoss': f"{avg_tv_loss:.4f}",
                    'SSIMLoss': f"{avg_spectrum_loss:.4f}",
                })
                pbar.update(1)

                # Save example patches periodically if configured
                batch_count_global += 1
                if self.save_examples_every is not None and batch_count_global % self.save_examples_every == 0:
                    save_subdir = os.path.join(example_samples_dir, f"epoch_{epoch:04d}_batch_{batch_count_global:06d}")
                    # Generate 10 samples for the last layer only
                    last_layer_idx = len(self.layer_configs) - 1
                    if self.save_basis_safe(epoch, save_subdir, num_samples=10, save_generator=False, layer_idx=last_layer_idx):
                        print(f"   ✓ Saved periodic samples for final layer")

        # Return average losses per update
        avg_diversity_loss = total_diversity_loss / max(num_updates, 1)
        avg_tv_loss = total_tv_loss / max(num_updates, 1)
        avg_spectrum_loss = total_spectrum_loss / max(num_updates, 1)
        return avg_diversity_loss, avg_tv_loss, avg_spectrum_loss

    def validate(self) -> float:
        """Validation pass using diversity score"""
        with torch.no_grad():
            # Sample multiple patches for validation diversity
            num_val_samples = min(16, len(self.val_loader))
            patches_by_image = {}
            batches_by_image = {}
            activations_by_image = {}

            for idx, batch in enumerate(self.val_loader):
                if idx >= num_val_samples:
                    break

                batch_dict = {k: v[0] for k, v in batch.items()}
                # Sample a single patch for this image (validation uses 1 patch per image)
                z = self.sample_coefficients(1)
                patch = self.generate_patches(z)[0]

                patches_by_image[idx] = [patch]
                batches_by_image[idx] = [batch_dict]

                # Compute activations
                act = self._get_activations_for_patch_image(
                    batch_dict, patch, use_grad=False, skip_detection=True
                )
                activations_by_image[idx] = [act]

            # Compute diversity score independently for each image, then average
            if len(patches_by_image) > 0:
                diversity_scores = []
                for img_idx in sorted(patches_by_image.keys()):
                    image_patches = patches_by_image[img_idx]
                    image_batches = batches_by_image[img_idx]
                    image_activations = activations_by_image[img_idx]
                    image_indices = list(range(len(image_patches)))

                    # Compute diversity for this image's patches
                    img_diversity_score = self.compute_activation_diversity(
                        image_patches,
                        image_batches,
                        image_indices,
                        image_activations,
                        use_grad=False
                    )
                    diversity_scores.append(img_diversity_score)

                # Average diversity scores across images
                diversity_score = torch.stack(diversity_scores).mean()
                return diversity_score.item()
            else:
                return 0.0

    def save_basis_safe(self, epoch: int, save_dir: str, num_samples: int = 5, save_generator: bool = True, layer_idx: Optional[int] = None) -> bool:
        """Safely save basis with error handling. Returns True if successful, False if failed.

        This wrapper prevents training from crashing due to save failures, allowing training
        to continue even if checkpointing fails. All errors are logged.
        """
        try:
            self.save_basis(epoch, save_dir, num_samples, save_generator, layer_idx)
            return True
        except Exception as e:
            print(f"\n⚠️  WARNING: Failed to save checkpoint to {save_dir}")
            print(f"   Error: {type(e).__name__}: {str(e)}")
            print(f"   Continuing training without saving this checkpoint...\n")
            return False

    def save_basis(self, epoch: int, save_dir: str = "foundation_basis_activation_patches", num_samples: int = 5, save_generator: bool = True, layer_idx: Optional[int] = None):
        """Save current generator state and sample patches

        Args:
            epoch: Current epoch for naming
            save_dir: Directory to save to
            num_samples: Number of sample patches to generate and save (default 5)
            save_generator: Whether to save the generator model (default True). Set to False to save only sample patches.
            layer_idx: Target layer index (optional). If provided, includes in filename and generates samples with that layer embedding.
        """
        Path(save_dir).mkdir(parents=True, exist_ok=True)

        with torch.no_grad():
            # Save generator network (if requested)
            if save_generator:
                checkpoint = {
                    'generator_state_dict': self.generator.state_dict(),
                    'epoch': epoch,
                    'basis_dim': self.basis_dim,
                    'patch_size': (self.patch_height, self.patch_width),
                    'use_vae_lora': getattr(self.generator, 'use_vae_lora', False),
                    'lora_rank': getattr(self.generator, 'lora_rank', None),
                    'lora_alpha': getattr(self.generator, 'lora_alpha', None),
                }
                torch.save(checkpoint, f"{save_dir}/generator_epoch_{epoch:04d}.pt")

            # Sample and save example patches
            z_samples = self.sample_coefficients(num_samples)
            sample_patches = self.generate_patches(z_samples)

            # Simple filename for patches
            filename_template = f"patch_epoch_{epoch:04d}_sample_{{i}}.png"

            for i, patch in enumerate(sample_patches):
                patch_pil = T.ToPILImage()(patch.cpu())
                patch_pil.save(f"{save_dir}/{filename_template.format(i=i)}")

    def _create_cosine_scheduler(self, optimizer, vae_lr, custom_lr, lr_min, max_epochs, total_steps):
        """
        Create a cosine annealing scheduler that handles different initial learning rates
        for VAE and custom layers, both decaying to the same minimum learning rate.

        Uses step-based scheduling (updates after each batch) for smoother learning rate decay.

        Args:
            optimizer: PyTorch optimizer with parameter groups
            vae_lr: Initial learning rate for VAE
            custom_lr: Initial learning rate for custom layers
            lr_min: Minimum learning rate (same for all groups)
            max_epochs: Maximum number of epochs for scheduling
            total_steps: Total number of training steps (batches) across all epochs

        Returns:
            Scheduler that applies cosine annealing to each group independently, updated per step
        """
        import math

        def cosine_decay(initial_lr, step):
            """Cosine annealing from initial_lr to lr_min, based on step count"""
            return (lr_min + (initial_lr - lr_min) * (1 + math.cos(math.pi * step / total_steps)) / 2) / initial_lr

        # Create lambda functions for each parameter group
        # Group 0: VAE
        # Groups 1+: Custom layers
        lambda_funcs = []
        for group in optimizer.param_groups:
            if group['name'] in ['vae_lora', 'vae_full']:
                lambda_funcs.append(lambda step, lr=vae_lr: cosine_decay(lr, step))
            else:
                lambda_funcs.append(lambda step, lr=custom_lr: cosine_decay(lr, step))

        return optim.lr_scheduler.LambdaLR(optimizer, lambda_funcs)

    def train(self, learning_rate: float = 0.01, vae_learning_rate: Optional[float] = None, lr_min: float = 1e-5, max_epochs: int = 50):
        """
        Train patches targeting the final OCR layer.

        Trains patches by:
        1. Profiling final layer activations for normalization
        2. Training for specified number of epochs
        3. Targeting the final layer (FinalOutput) exclusively

        Saves:
        - checkpoints/{run_id}/training_complete_final_model/: Final model after all training (20 samples)
        - checkpoints/{run_id}/best_progressive_patch/: Best model across all training
        - checkpoints/{run_id}/checkpoint_epoch_*/: Periodic checkpoints every 10 epochs
        """
        from datetime import datetime

        # Create unique run ID based on timestamp
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.checkpoint_base = os.path.join("checkpoints", self.run_id)
        os.makedirs(self.checkpoint_base, exist_ok=True)
        print(f"\nRun ID: {self.run_id}")
        print(f"Checkpoint directory: {self.checkpoint_base}\n")

        # Save train/val split CSV to checkpoint directory
        self._save_train_val_split(
            self.full_dataset,
            self.train_loader.dataset,
            self.val_loader.dataset,
            [],
            save_dir=self.checkpoint_base
        )

        # Profile layer activations upfront for normalization
        print("\n" + "="*80)
        print("LAYER ACTIVATION PROFILING")
        print("="*80)
        self.profile_layer_activations(num_samples=1024)
        print("="*80 + "\n")

        # Initialize optimizer
        # Use separate learning rate for VAE if specified, otherwise use custom learning rate
        if vae_learning_rate is None:
            vae_learning_rate = learning_rate

        # Collect trainable parameters
        trainable_params = []

        # VAE parameters (LoRA only if enabled)
        if hasattr(self.generator, 'use_vae_lora') and self.generator.use_vae_lora:
            vae_lora_params = [p for p in self.generator.vae.parameters() if p.requires_grad]
            trainable_params.append({'params': vae_lora_params, 'lr': vae_learning_rate, 'name': 'vae_lora'})
            print(f"Optimizer: VAE LoRA parameters: {sum(p.numel() for p in vae_lora_params):,} (lr={vae_learning_rate})")
        else:
            vae_params = list(self.generator.vae.parameters())
            trainable_params.append({'params': vae_params, 'lr': vae_learning_rate, 'name': 'vae_full'})
            print(f"Optimizer: VAE full parameters: {sum(p.numel() for p in vae_params):,} (lr={vae_learning_rate})")

        # Other components (always trainable) - use custom learning rate
        for name in ['adapter', 'skip_projection', 'cnn_refiner', 'patch_projector']:
            module = getattr(self.generator, name)
            params = list(module.parameters())
            trainable_params.append({'params': params, 'lr': learning_rate, 'name': name})
            print(f"Optimizer: {name} parameters: {sum(p.numel() for p in params):,} (lr={learning_rate})")

        # Bottleneck refiner - use custom learning rate
        module = self.generator.bottleneck_refiner
        params = list(module.parameters())
        trainable_params.append({'params': params, 'lr': learning_rate, 'name': 'bottleneck_refiner'})
        print(f"Optimizer: bottleneck_refiner parameters: {sum(p.numel() for p in params):,} (lr={learning_rate})")

        # Print total trainable parameters
        total_trainable = sum(p.numel() for group in trainable_params for p in group['params'])
        print(f"Optimizer: TOTAL trainable parameters: {total_trainable:,}")
        print(f"Optimizer: VAE learning rate: {vae_learning_rate}, Custom layers learning rate: {learning_rate}")
        print(f"Optimizer: Both decay to lr_min={lr_min} via cosine annealing")

        optimizer = optim.AdamW(trainable_params, weight_decay=1e-4)

        # Calculate total training steps for step-based scheduling
        num_images = len(self.train_loader)
        batches_per_epoch = num_images // self.ocr_images_per_batch
        if num_images % self.ocr_images_per_batch != 0:
            batches_per_epoch += 1
        total_steps = max_epochs * batches_per_epoch
        print(f"Step-based scheduling: {batches_per_epoch} batches/epoch × {max_epochs} epochs = {total_steps} total steps")

        # Create custom cosine annealing scheduler that handles different initial LRs
        # Both VAE and custom layers decay to the same lr_min via step-based scheduling
        scheduler = self._create_cosine_scheduler(
            optimizer,
            vae_learning_rate,
            learning_rate,
            lr_min,
            max_epochs,
            total_steps
        )

        # Training history
        history = {
            'epoch': [],
            'diversity_loss': [],
            'tv_loss': [],
            'spectrum_loss': [],
            'learning_rate': []
        }

        best_train_loss = float('inf')
        best_epoch = 0

        print("\n" + "="*80)
        print("RANDOM LAYER SAMPLING TRAINING WITH CASCADE PENALTY")
        print("="*80)
        print(f"   Mode: OCR (cropped plates)")
        print(f"   Dataset: {len(self.train_loader) + len(self.val_loader)} images")
        print(f"   Patch size: {self.patch_height}×{self.patch_width}")
        print(f"   Latent dimensions: {self.basis_dim}")

        # Print generator architecture
        vae_latent_dim = self.generator.vae_latent_dim
        print(f"   Generator: FoundationPatchGenerator (VAE-based, high quality)")
        print(f"     Architecture: z[{self.basis_dim}] → adapter → VAE → CNN refiner → patch[3×{self.patch_height}×{self.patch_width}]")

        print(f"   Diversity weight: {self.diversity_weight}")
        print(f"   Quality weight: {self.quality_weight}")
        print(f"   Performance weight: {self.performance_weight}")
        print(f"   TV weight: {self.tv_weight}")
        print(f"   SSIM weight: {self.spectrum_weight}")
        print(f"   Device: {self.device}")
        print(f"   LR: {learning_rate} (cosine annealing to {lr_min})")
        print(f"   Max epochs: {max_epochs}")
        print(f"   Layers available: {len(self.layer_configs)}")
        print("="*80 + "\n")

        # Main training loop over epochs
        for epoch in range(1, max_epochs + 1):
            current_lr = optimizer.param_groups[0]['lr']

            # Training (scheduler.step() is called per batch inside train_epoch)
            train_diversity_loss, train_tv_loss, train_spectrum_loss = self.train_epoch(optimizer, scheduler, epoch)

            # Record history
            history['epoch'].append(epoch)
            history['diversity_loss'].append(train_diversity_loss)
            history['tv_loss'].append(train_tv_loss)
            history['spectrum_loss'].append(train_spectrum_loss)
            history['learning_rate'].append(current_lr)

            # Print epoch summary
            epoch_summary = (f"Epoch {epoch:3d}/{max_epochs} | "
                            f"DivLoss: {train_diversity_loss:.4f} | "
                            f"TVLoss: {train_tv_loss:.4f} | "
                            f"SSIMLoss: {train_spectrum_loss:.4f} | "
                            f"LR: {current_lr:.2e}")
            print(epoch_summary)

            # Save best model
            if train_diversity_loss < best_train_loss:
                best_train_loss = train_diversity_loss
                best_epoch = epoch
                best_dir = os.path.join(self.checkpoint_base, "best_progressive_patch")
                if self.save_basis_safe(epoch, best_dir, num_samples=10, save_generator=True):
                    print(f"   ✓ New best training loss: {best_train_loss:.4f} (saved)")
                else:
                    print(f"   ✓ New best training loss: {best_train_loss:.4f} (save failed, continuing)")

            # Save samples periodically (every 10 epochs)
            if epoch % 10 == 0:
                last_layer_idx = len(self.layer_configs) - 1
                checkpoint_dir = os.path.join(self.checkpoint_base, f"checkpoint_epoch_{epoch:04d}")
                if self.save_basis_safe(epoch, checkpoint_dir, num_samples=10, save_generator=False, layer_idx=last_layer_idx):
                    print(f"   ✓ Saved checkpoint for epoch {epoch}")

        print("\n" + "="*80)
        print("TRAINING COMPLETED!")
        print("="*80)
        print(f"   Best training loss: {best_train_loss:.4f} (epoch {best_epoch})")
        print(f"   Total epochs: {max_epochs}")
        print("="*80 + "\n")

        # Save final model (critical save - retry if needed)
        print("\nSaving final trained model...")
        final_save_dir = os.path.join(self.checkpoint_base, "training_complete_final_model")
        max_retries = 3
        saved_successfully = False

        for attempt in range(1, max_retries + 1):
            if self.save_basis_safe(max_epochs, final_save_dir, num_samples=20, save_generator=True):
                saved_successfully = True
                break
            elif attempt < max_retries:
                print(f"   Retrying final save (attempt {attempt + 1}/{max_retries})...")

        if saved_successfully:
            print(f"\n{'='*80}")
            print(f"✓ FINAL MODEL SAVED SUCCESSFULLY TO: {final_save_dir}/")
            print(f"{'='*80}")
            print(f"  Generator checkpoint: {final_save_dir}/generator_epoch_{max_epochs:04d}.pt")
            print(f"  Sample patches: 20 PNG files in {final_save_dir}/")
            print(f"  All checkpoints: {self.checkpoint_base}/")
            print(f"{'='*80}\n")
        else:
            print(f"\n{'='*80}")
            print(f"⚠️  CRITICAL: Final model save failed after {max_retries} attempts!")
            print(f"{'='*80}")
            print(f"  Training completed but final checkpoint could not be saved.")
            print(f"  Check disk space and permissions at: {self.checkpoint_base}/")
            print(f"  Best model checkpoint available at: {self.checkpoint_base}/best_progressive_patch/")
            print(f"{'='*80}\n")

        return history


def main():
    parser = argparse.ArgumentParser(description='Progressive Layer Diversity Training for Patch Generation')
    parser.add_argument('--basis-dim', type=int, default=16,
                        help='Dimensionality of latent basis (default: 16)')
    parser.add_argument('--diversity-weight', type=float, default=1.0,
                        help='Weight for diversity term in combined loss (default: 1.0). '
                        'Multiplies diversity_score. Loss = -(performance_weight * (diversity_weight * diversity_score + quality_weight * quality_score))')
    parser.add_argument('--quality-weight', type=float, default=1.0,
                        help='Weight for quality term in combined loss (default: 1.0). '
                        'Multiplies quality_score. Loss = -(performance_weight * (diversity_weight * diversity_score + quality_weight * quality_score))')
    parser.add_argument('--performance-weight', type=float, default=1.0,
                        help='Overall weight for diversity-quality combined loss (default: 1.0). '
                        'Multiplies the entire combined term. Loss = -(performance_weight * (diversity_weight * diversity_score + quality_weight * quality_score))')
    parser.add_argument('--tv-weight', type=float, default=2.5,
                        help='Weight for total variation loss to encourage spatial smoothness (default: 2.5)')
    parser.add_argument('--ssim-weight', type=float, default=1.0, dest='spectrum_weight',
                        help='Weight for SSIM structural diversity loss to discourage patch similarity (default: 1.0). '
                        'Penalizes high SSIM between patches - higher value = force more different structures.')
    parser.add_argument('--save-examples-every', type=int, default=None,
                        help='Save example patches every N batches during training (default: disabled). '
                        'Saves 5 sample patches to checkpoints/{run_id}/example_samples/. '
                        'Example: --save-examples-every 100 saves samples every 100 batches.')
    parser.add_argument('--learning-rate', type=float, default=5e-3,
                        help='Learning rate for custom layers (adapter, CNN refiner, etc.) (default: 5e-3)')
    parser.add_argument('--vae-learning-rate', type=float, default=None,
                        help='Learning rate for VAE (default: same as --learning-rate). '
                        'Both decay to --lr-min via cosine annealing.')
    parser.add_argument('--lr-min', type=float, default=1e-5,
                        help='Minimum learning rate for cosine annealing (default: 1e-5). '
                        'Both VAE and custom layers decay to this value.')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs to train for (default: 50). '
                        'Learning rate decays over this many epochs via cosine annealing.')
    parser.add_argument('--no-use-all-for-train', action='store_true',
                        help='Disable using all data for training (use 80%% train / 20%% validation split). '
                        'Default: uses 100%% of data for training.')
    parser.add_argument('--bottleneck-dim', type=int, default=256,
                        help='Hidden dimension of middle dense layer in bottleneck refiner (default: 256). '
                        'Controls expressivity: 256 (baseline) → 512 (more capacity, +35%% params) → 1024 (more capacity, +100%% params).')
    parser.add_argument('--ocr-dataset', type=str, required=True,
                        help='Public OCR dataset(s) to use in OCR mode. '
                        'Supports single dataset (e.g., iiit5k) or multiple comma-separated datasets '
                        '(e.g., iiit5k,icdar2013,roboflow_lpr). Multiple datasets will be combined. '
                        'Available: iiit5k, icdar2013, icdar2015, cocotext, roboflow_lpr, kaggle_lp, '
                        'indian_plates_kaggle, ccpd2019_base, ccpd2019_blur, ccpd2019_challenge, '
                        'ccpd2019_db, ccpd2019_fn, ccpd2019_np, ccpd2019_rotate, ccpd2019_tilt, '
                        'ccpd2019_weather, mercosur, crpd. Only used when --ocr-mode is enabled.')
    parser.add_argument('--ocr-dataset-split', type=str, default='train',
                        help='Dataset split to use (default: train). Options: train, test, val '
                        '(availability depends on dataset). Only used when --ocr-mode is enabled.')
    parser.add_argument('--ocr-max-samples', type=int, default=None,
                        help='Maximum number of samples to load from OCR dataset (default: all). '
                        'Useful for quick testing.')
    parser.add_argument('--ocr-images-per-batch', type=int, default=1,
                        help='Number of images to process per gradient update in OCR mode (default: 1). '
                        'Total patches = images_per_batch × patches_per_image. '
                        'Higher values give more diverse gradients but use more memory.')
    parser.add_argument('--ocr-patches-per-image', type=int, required=True,
                        help='Number of patches to generate per image in OCR mode. '
                        'Total patches = images_per_batch × patches_per_image.')
    parser.add_argument('--use-vae-lora', action='store_true', default=True, dest='use_vae_lora',
                        help='Use LoRA for VAE decoder (default: True)')
    parser.add_argument('--no-vae-lora', action='store_false', dest='use_vae_lora',
                        help='Disable LoRA, use full VAE fine-tuning')
    parser.add_argument('--lora-rank', type=int, default=8,
                        help='LoRA rank (default: 8)')
    parser.add_argument('--lora-alpha', type=int, default=16,
                        help='LoRA alpha (default: 16)')
    args = parser.parse_args()

    # Validate dataset argument
    dataset_list = [d.strip() for d in args.ocr_dataset.split(',')]
    for dataset_name in dataset_list:
        if dataset_name not in DATASETS:
            available_datasets = ', '.join(sorted(DATASETS.keys()))
            raise ValueError(
                f"Unknown dataset '{dataset_name}'. Available: {available_datasets}"
            )

    print(f"\n{'='*80}")
    print(f"OCR MODE (cropped license plates)")
    print(f"{'='*80}")
    print(f"  Dataset: {args.ocr_dataset}")
    print(f"  Split: {args.ocr_dataset_split}")
    print(f"  Images per batch: {args.ocr_images_per_batch}")
    print(f"  Patches per image: {args.ocr_patches_per_image}")
    if args.ocr_max_samples:
        print(f"  Max samples: {args.ocr_max_samples}")
    print(f"{'='*80}\n")

    # Trainer kwargs
    trainer_kwargs = {
        'ocr_dataset': args.ocr_dataset,
        'device': 'cuda',
        'grad_accumulate': 16,  # Default gradient accumulation steps
        'basis_dim': args.basis_dim,
        'diversity_weight': args.diversity_weight,
        'quality_weight': args.quality_weight,
        'performance_weight': args.performance_weight,
        'tv_weight': args.tv_weight,
        'spectrum_weight': args.spectrum_weight,
        'ocr_dataset_split': args.ocr_dataset_split,
        'ocr_max_samples': args.ocr_max_samples,
        'ocr_images_per_batch': args.ocr_images_per_batch,
        'ocr_patches_per_image': args.ocr_patches_per_image,
        'use_vae_lora': args.use_vae_lora,
        'lora_rank': args.lora_rank,
        'lora_alpha': args.lora_alpha,
        'bottleneck_dim': args.bottleneck_dim,
        'save_examples_every': args.save_examples_every
    }

    # Training mode
    try:
        trainer = ProgressivePatchTrainer(**trainer_kwargs)

        history = trainer.train(
            learning_rate=args.learning_rate,
            vae_learning_rate=args.vae_learning_rate,
            lr_min=args.lr_min,
            max_epochs=args.epochs
        )

        # Save training history as CSV
        import pandas as pd
        history_df = pd.DataFrame(history)
        history_df.to_csv('progressive_patch_training_history.csv', index=False)
        print(f"\nTraining history saved to: progressive_patch_training_history.csv")

        # Plot training results
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

        # Diversity loss (training vs validation)
        ax1.plot(history['epoch'], history['diversity_loss'], 'b-', label='Train Diversity Loss', alpha=0.7)
        ax1.plot(history['epoch'], history['val_diversity'], 'r-', label='Val Diversity', alpha=0.7)
        # Add vertical lines for layer transitions
        for i, record in enumerate(trainer.layer_history):
            if i > 0:  # Skip first layer (starts at epoch 0)
                transition_epoch = sum(r['epochs_trained'] for r in trainer.layer_history[:i])
                ax1.axvline(x=transition_epoch, color='gray', linestyle='--', alpha=0.5)
        ax1.set_title('Diversity Over Time (Progressive Layers)')
        ax1.set_xlabel('Global Epoch')
        ax1.set_ylabel('Diversity Loss')
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # TV and spectrum regularization losses
        ax2.plot(history['epoch'], history['tv_loss'], 'g-', label='TV Loss', alpha=0.7)
        ax2.plot(history['epoch'], history['spectrum_loss'], 'orange', label='Spectrum Loss', alpha=0.7)
        for i, record in enumerate(trainer.layer_history):
            if i > 0:
                transition_epoch = sum(r['epochs_trained'] for r in trainer.layer_history[:i])
                ax2.axvline(x=transition_epoch, color='gray', linestyle='--', alpha=0.5)
        ax2.set_title('Regularization Losses (Progressive Layers)')
        ax2.set_xlabel('Global Epoch')
        ax2.set_ylabel('Loss')
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
        print("  - training_complete_final_model/ - FINAL trained model with 20 sample patches")
        print("  - best_progressive_patch/ - Best model during training (by loss or diversity)")
        print("  - final_layer_checkpoint_epoch_*/ - Checkpoints every 25 epochs on final layer")
        print("  - layer*_complete_*/ - Checkpoint after completing each layer")

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

    python progressive_patch.py --target-layer "0,5,10" --max-epochs-per-layer "30,50,100"
    # Train layer 0 for max 30 epochs, layer 5 for 50 epochs, layer 10 for 100 epochs

    python progressive_patch.py --target-layer "0,5,10" --convergence-threshold "1.0,0.5,0.0"
    # Layer 0 stops when diversity < 1.0, layer 5 when < 0.5, layer 10 trains full epochs

    python progressive_patch.py --target-layer "0,5,10" --max-epochs-per-layer "30,50,100" --convergence-threshold "1.0,0.5,0.0"
    # Combine both: different max epochs and convergence thresholds per layer

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

Generator architecture:
- FoundationPatchGenerator:
  * Complex architecture: Adapter → VAE decoder → CNN refiner → Bottleneck dense refiner → patch projector
  * Uses Stable Diffusion VAE decoder (trainable with LoRA support)
  * CNN refiner provides feature refinement
  * Bottleneck refiner applies dense processing via spatial compression/expansion
  * Quality: High-quality, realistic patches with multi-scale processing

Diversity computation and memory:
- eval_depth controls the total number of (patch, image) evaluations for diversity:
  * Default (None): Evaluates all batch_size^2 pairs (full matrix)
  * Specified value: Evaluates up to eval_depth pairs total
  * Always includes batch_size diagonal evaluations (patch_i on image_i)
  * Randomly samples off-diagonal pairs from budget (eval_depth - batch_size)
- Benefits: Reduces memory without sacrificing diversity quality, enables larger batch sizes
- Example: batch_size=32, eval_depth=256 reduces evals from 1024 to 256 (75% reduction)

Checkpoint structure:
- training_complete_final_model/: FINAL model after all training (generator + 20 sample patches)
- best_progressive_patch/: Best model during training by loss/diversity (generator + sample patches)
- final_layer_checkpoint_epoch_*/: Checkpoints every 25 epochs on final layer (generator + 10 samples)
  (e.g., final_layer_checkpoint_epoch_0025/, final_layer_checkpoint_epoch_0050/, etc.)
- layer*_complete_*/: Model state after completing each layer (generator + 10 sample patches)
  (e.g., layer1_complete_Conv_Layer_1_32ch/, layer2_complete_Conv_Layer_2_48ch/, etc.)

You can resume from any checkpoint by loading the generator_epoch_XXXX.pt file inside these directories.
Use load_and_generate_samples.py to generate patches from any saved checkpoint.
"""
