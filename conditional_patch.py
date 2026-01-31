#!/usr/bin/env python3
"""
Conditional Patch Generator: Train patches conditioned on layer profiles

Randomly samples model/layer per batch, uses layer profiles as condition,
trains generator to disrupt target layer while preserving prior layers.

Loss = CKA(target_layer) - mean(CKA(prior_layers))
"""

import os
import json
import argparse
import traceback
from pathlib import Path
from typing import Tuple, Dict, List
import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from tqdm import tqdm
import onnx
import onnx2torch
from transformers import VisionEncoderDecoderModel
from doctr.models import vitstr_small
import kornia.geometry as K
from diffusers import AutoencoderKL

from dataset import create_dataloaders
import h5py


PATCH_HEIGHT = 256
PATCH_WIDTH = 512


class LayerProfileEncoder(nn.Module):
    """
    Encodes layer profile + metadata into latent space.

    Input:
        - layer_profile: [8] (eigendecomposed features)
        - layer_depth: [1] (fraction of model depth, e.g., 5/11 = 0.45)
        - model_code: [3] (one-hot: vitstr, cct, trocr)
    Output:
        - latent: [32]
    """
    def __init__(self, input_dim: int = 12, latent_dim: int = 32):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 128),
            nn.LayerNorm(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, latent_dim),
        )

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, layer_profile: torch.Tensor, layer_depth: torch.Tensor,
                model_code: torch.Tensor) -> torch.Tensor:
        """
        Args:
            layer_profile: [batch, 8]
            layer_depth: [batch, 1]
            model_code: [batch, 3]
        Returns:
            latent: [batch, 32]
        """
        # Concatenate all inputs
        x = torch.cat([layer_profile, layer_depth, model_code], dim=1)  # [batch, 12]
        return self.network(x)


class SimplePatchGenerator(nn.Module):
    """CNN-based patch generator using transposed convolutions"""
    def __init__(self, latent_dim: int = 32, patch_height: int = 256,
                 patch_width: int = 512):
        super().__init__()

        self.latent_dim = latent_dim
        self.patch_height = patch_height
        self.patch_width = patch_width

        # Dense layer to expand latent to spatial tensor
        # Output: (512, 4, 8) = 16384 features
        self.fc = nn.Linear(latent_dim, 512 * 4 * 8)

        # Transposed convolution blocks (progressively upsample)
        self.conv_blocks = nn.Sequential(
            # 512x4x8 → 256x8x16
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            # 256x8x16 → 128x16x32
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            # 128x16x32 → 64x32x64
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # 64x32x64 → 32x64x128
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            # 32x64x128 → 16x128x256
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),

            # 16x128x256 → 3x256x512
            nn.ConvTranspose2d(16, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),  # Output in [0, 1]
        )

        # Initialize weights
        for m in self.modules():
            if isinstance(m, (nn.ConvTranspose2d, nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: [batch_size, latent_dim]
        Returns:
            patches: [batch_size, 3, patch_height, patch_width]
        """
        # Expand latent to spatial tensor
        x = self.fc(z)  # [batch, 512*4*8]
        x = x.view(-1, 512, 4, 8)  # [batch, 512, 4, 8]

        # Progressive upsampling with convolutions
        patches = self.conv_blocks(x)  # [batch, 3, 256, 512]

        return patches


class DiffusionPatchGenerator(nn.Module):
    """Stable Diffusion VAE-based patch generator"""
    def __init__(self, latent_dim: int = 32, patch_height: int = 256,
                 patch_width: int = 512, device: str = 'cuda'):
        super().__init__()

        self.latent_dim = latent_dim
        self.patch_height = patch_height
        self.patch_width = patch_width

        # VAE latent space dimensions (8x downsampling)
        self.vae_latent_h = patch_height // 8  # 32
        self.vae_latent_w = patch_width // 8   # 64
        self.vae_latent_channels = 4
        self.vae_latent_dim = self.vae_latent_channels * self.vae_latent_h * self.vae_latent_w

        # Load pretrained VAE from Stable Diffusion
        print("Loading Stable Diffusion VAE decoder...")
        self.vae = AutoencoderKL.from_pretrained(
            "madebyollin/sdxl-vae-fp16-fix",
            torch_dtype=torch.float32
        ).to(device)
        self.vae.train()
        print(f"VAE loaded. Latent space: [{self.vae_latent_channels}, {self.vae_latent_h}, {self.vae_latent_w}]")

        # Adapter network: map latent → VAE latent space
        self.adapter = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 512),
            nn.LayerNorm(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, self.vae_latent_dim),
        )

        # Initialize adapter weights
        for m in self.adapter.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: [batch_size, latent_dim]
        Returns:
            patches: [batch_size, 3, patch_height, patch_width]
        """
        # Map latent to VAE latent space
        vae_latent_flat = self.adapter(z)  # [batch, vae_latent_dim]
        vae_latent = vae_latent_flat.view(
            z.shape[0],
            self.vae_latent_channels,
            self.vae_latent_h,
            self.vae_latent_w
        )  # [batch, 4, 32, 64]

        # Decode with VAE
        patches = self.vae.decode(vae_latent).sample  # [batch, 3, 256, 512]
        patches = torch.clamp(patches, 0.0, 1.0)

        return patches


def load_activation_statistics(model_name: str, layer_name: str,
                              stats_dir: str = "layer_profiles") -> torch.Tensor:
    """
    Load activation statistics (std per neuron) for a specific layer.

    Args:
        model_name: Name of the model (e.g., 'vitstr_small')
        layer_name: Name of the layer (e.g., 'encoder.blocks.0.norm1')
        stats_dir: Directory containing activation statistics HDF5 files (default: layer_profiles)

    Returns:
        std: Tensor of shape [n_features] with std for each neuron
    """
    stats_path = Path(stats_dir) / f"{model_name}_activation_statistics.h5"

    if not stats_path.exists():
        # Return None if file doesn't exist - will use default value
        print(f"Warning: Activation statistics not found at {stats_path}")
        return None

    try:
        with h5py.File(stats_path, 'r') as f:
            if 'activation_statistics' not in f:
                return None

            stats_group = f['activation_statistics']
            if layer_name not in stats_group:
                return None

            layer_group = stats_group[layer_name]
            std_array = np.array(layer_group['std'][:])
            std_tensor = torch.from_numpy(std_array).float()

            return std_tensor
    except Exception as e:
        print(f"Error loading activation statistics: {e}")
        return None


def compute_normalized_delta(X: torch.Tensor, Y: torch.Tensor,
                            layer_std: torch.Tensor = None,
                            epsilon: float = 1e-8) -> torch.Tensor:
    """
    Compute normalized delta between two activation matrices.

    For each neuron: (|activation_clean - activation_patched|) / std_neuron
    Then take the mean across all neurons.

    Args:
        X: Clean activations [n_samples, n_features_x]
        Y: Patched activations [n_samples, n_features_y]
        layer_std: Standard deviation per neuron [n_features]. If None, uses std=1.0
        epsilon: Small value for numerical stability

    Returns:
        similarity: Scalar in [0, inf) - lower is more similar
    """
    # Flatten to 2D
    X_flat = X.reshape(X.shape[0], -1)
    Y_flat = Y.reshape(Y.shape[0], -1)

    # Truncate to common dimension
    min_features = min(X_flat.shape[1], Y_flat.shape[1])
    X_flat = X_flat[:, :min_features]
    Y_flat = Y_flat[:, :min_features]

    # Compute absolute delta
    delta = torch.abs(X_flat - Y_flat)  # [n_samples, n_features]

    # Normalize by neuron-wise std
    if layer_std is not None:
        # Ensure std is on the same device
        layer_std = layer_std.to(delta.device)
        # Use only the std values for the features we have
        layer_std = layer_std[:min_features]
        # Normalize delta by std
        normalized_delta = delta / (layer_std.unsqueeze(0) + epsilon)  # [n_samples, n_features]
    else:
        # If no std provided, just use the raw delta
        normalized_delta = delta

    # Mean across all neurons and samples
    similarity = normalized_delta.mean()

    return similarity


def load_layer_profiles(profile_dir: str = "layer_profiles") -> Dict:
    """
    Load all layer profiles from directory.

    Returns:
        profiles: Dict with structure:
            {model_name: {
                'profiles': np.ndarray [n_images, n_layers*8],
                'n_layers': int,
                'layer_names': List[str]
            }}
    """
    profile_dir = Path(profile_dir)
    profiles = {}

    model_files = list(profile_dir.glob("*_layer_profiles.npz"))

    for model_file in model_files:
        model_name = model_file.stem.replace("_layer_profiles", "")

        # Load profiles
        data = np.load(model_file)

        # Load metadata
        metadata_file = profile_dir / f"{model_name}_metadata.json"
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)

        profiles[model_name] = {
            'profiles': data['profiles'],  # [n_images, n_layers*8]
            'n_layers': metadata['n_layers'],
            'layer_names': metadata['layer_names'],
            'k': metadata['k']  # Should be 8
        }

    return profiles


def load_ocr_models(device: str = 'cuda') -> Dict:
    """
    Load all three OCR models.

    Returns:
        models: Dict {model_name: (model, input_format)}
            input_format: dict with 'channels_last' and 'scale_to_255' flags
    """
    models = {}

    # CCT model
    print("Loading CCT model...")
    cct_path = Path.home() / ".cache/fast-plate-ocr/cct-xs-v1-global-model/cct_xs_v1_global.onnx"
    if cct_path.exists():
        cct_onnx = onnx.load(str(cct_path))
        cct_model = onnx2torch.convert(cct_onnx).to(device)
        cct_model.eval()
        models['cct_xs_v1_global'] = (
            cct_model,
            {'channels_last': True, 'scale_to_255': True}
        )
        print("  CCT loaded")

    # ViTSTR model
    print("Loading ViTSTR model...")
    vitstr_model = vitstr_small(pretrained=True)
    vitstr_model.eval()
    vitstr_model.to(device)
    models['vitstr_small'] = (
        vitstr_model,
        {'channels_last': False, 'scale_to_255': False}
    )
    print("  ViTSTR loaded")

    # TrOCR model (encoder only)
    print("Loading TrOCR model...")
    from transformers import TrOCRProcessor

    trocr_full = VisionEncoderDecoderModel.from_pretrained(
        "microsoft/trocr-small-printed"
    ).to(device)
    trocr_encoder = trocr_full.encoder
    trocr_encoder.eval()
    trocr_processor = TrOCRProcessor.from_pretrained("microsoft/trocr-small-printed")

    models['trocr_small_printed_encoder'] = (
        trocr_encoder,
        {'channels_last': False, 'scale_to_255': False},
        trocr_processor
    )
    print("  TrOCR loaded")

    return models


class ConditionalPatchTrainer:
    """Trainer for conditional patch generation"""

    def __init__(self, csv_path: str, profile_dir: str = "layer_profiles",
                 device: str = 'cuda', learning_rate: float = 1e-3,
                 generator_type: str = 'simple'):
        self.device = device
        self.learning_rate = learning_rate
        self.generator_type = generator_type

        print(f"\nGPU Memory at start: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

        # Load layer profiles
        print("\nLoading layer profiles...")
        self.layer_profiles = load_layer_profiles(profile_dir)
        print(f"GPU Memory after loading profiles: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        print(f"Loaded profiles for {len(self.layer_profiles)} models:")
        for model_name, profile_data in self.layer_profiles.items():
            print(f"  - {model_name}: {profile_data['n_layers']} layers")

        # Load activation statistics for normalized delta computation
        print("\nLoading activation statistics...")
        self.activation_stats = {}
        for model_name in self.layer_profiles.keys():
            all_layer_names = self.layer_profiles[model_name]['layer_names']
            self.activation_stats[model_name] = {}
            for layer_name in all_layer_names:
                layer_std = load_activation_statistics(model_name, layer_name, profile_dir)
                if layer_std is not None:
                    self.activation_stats[model_name][layer_name] = layer_std
        print(f"Loaded activation statistics for {len(self.activation_stats)} models")

        # Load OCR models
        print("\nLoading OCR models...")
        self.ocr_models = load_ocr_models(device)
        print(f"GPU Memory after loading OCR models: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

        # Create model name to index mapping
        self.model_names = sorted(self.layer_profiles.keys())
        self.model_to_idx = {name: idx for idx, name in enumerate(self.model_names)}
        self.n_models = len(self.model_names)

        # Create encoder and generator
        self.encoder = LayerProfileEncoder(input_dim=12, latent_dim=32).to(device)

        if generator_type == 'diffusion':
            print(f"\nUsing Diffusion-based generator")
            self.generator = DiffusionPatchGenerator(latent_dim=32,
                                                     patch_height=PATCH_HEIGHT,
                                                     patch_width=PATCH_WIDTH,
                                                     device=device)
        else:
            print(f"\nUsing Simple CNN generator")
            self.generator = SimplePatchGenerator(latent_dim=32,
                                                 patch_height=PATCH_HEIGHT,
                                                 patch_width=PATCH_WIDTH).to(device)

        # Optimizer
        self.optimizer = optim.Adam(
            list(self.encoder.parameters()) + list(self.generator.parameters()),
            lr=learning_rate
        )

        # Load dataset
        print("\nLoading dataset...")
        transform = T.Compose([T.ToPILImage(), T.ToTensor()])
        self.train_loader, _ = create_dataloaders(
            csv_path,
            transform=transform,
            preload=False,
            batch_size=1,
            n_jobs=0,
            use_all_for_train=True
        )

        print(f"\nSetup complete!")
        print(f"  Device: {device}")
        print(f"  Models: {self.n_models}")
        print(f"  Dataset: {len(self.train_loader)} images")

    def sample_conditions(self, batch_size: int, image_indices: List[int]) -> Tuple:
        """
        Sample random model/layer conditions for each sample in batch.

        Returns:
            model_names: List[str] of length batch_size
            layer_indices: List[int] of length batch_size
            layer_profiles: torch.Tensor [batch_size, 8]
            layer_depths: torch.Tensor [batch_size, 1]
            model_codes: torch.Tensor [batch_size, 3] (one-hot)
        """
        model_names_list = []
        layer_indices_list = []
        layer_profiles_list = []
        layer_depths_list = []
        model_codes_list = []

        for img_idx in image_indices:
            # Randomly sample model
            model_name = np.random.choice(self.model_names)

            # Randomly sample layer
            n_layers = self.layer_profiles[model_name]['n_layers']
            layer_idx = np.random.randint(0, n_layers)

            # Get layer profile for this image and layer
            # profiles shape: [n_images, n_layers*8]
            # Extract 8 features for selected layer
            profile_matrix = self.layer_profiles[model_name]['profiles']
            layer_profile = profile_matrix[img_idx, layer_idx*8:(layer_idx+1)*8]

            # Compute layer depth
            layer_depth = layer_idx / n_layers

            # Create one-hot model code
            model_code = np.zeros(self.n_models)
            model_code[self.model_to_idx[model_name]] = 1

            model_names_list.append(model_name)
            layer_indices_list.append(layer_idx)
            layer_profiles_list.append(layer_profile)
            layer_depths_list.append(layer_depth)
            model_codes_list.append(model_code)

        # Convert to tensors
        layer_profiles = torch.from_numpy(np.array(layer_profiles_list)).float()
        layer_depths = torch.from_numpy(np.array(layer_depths_list)).float().unsqueeze(1)
        model_codes = torch.from_numpy(np.array(model_codes_list)).float()

        return (model_names_list, layer_indices_list,
                layer_profiles, layer_depths, model_codes)

    def apply_patch(self, image: torch.Tensor, corners: torch.Tensor,
                    patch: torch.Tensor) -> torch.Tensor:
        """Apply patch as border around license plate (same as progressive_patch.py)

        Args:
            image: [1, 3, H, W]
            corners: [1, 4, 2]
            patch: [1, 3, 256, 512] (already has batch dim from generator)
        """
        # Create patch corner coordinates
        patch_h, patch_w = self.generator.patch_height, self.generator.patch_width
        src_corners = torch.tensor([
            [0, 0], [patch_w, 0], [patch_w, patch_h], [0, patch_h]
        ], dtype=torch.float32, device=self.device).unsqueeze(0)

        # Compute perspective transformation
        M_plate = K.get_perspective_transform(src_corners, corners)

        # Warp patch (patch already has batch dim: [1, 3, 256, 512])
        dsize = (image.shape[2], image.shape[3])  # (H, W)
        warped_patch = K.warp_perspective(patch, M_plate, dsize=dsize)

        # Blend with image
        patched_image = image.clone()
        mask = (warped_patch.sum(dim=1, keepdim=True) > 0).float()
        patched_image = patched_image * (1 - mask) + warped_patch * mask

        return patched_image

    def extract_activations(self, model, images: torch.Tensor, layer_names: List[str],
                           input_format: Dict, processor=None, model_name: str = None) -> Dict[str, torch.Tensor]:
        """
        Run model and extract activations at specified layers.

        Args:
            model: The OCR model
            images: [1, 3, H, W] image tensor
            layer_names: List of layer names to extract
            input_format: Dict with 'channels_last' and 'scale_to_255' flags
            processor: Optional TrOCRProcessor for TrOCR model
            model_name: Name of the model for determining preprocessing

        Returns:
            activations: Dict {layer_name: tensor [batch, features]}
        """
        # Use processor for TrOCR (converts to model's expected input size)
        if processor is not None:
            from PIL import Image
            import numpy as np

            # TrOCR uses DeiT preprocessing:
            # 1. Resize to 384×384 (BICUBIC, no aspect ratio preservation)
            # 2. Rescale: divide by 255
            # 3. Normalize: (x - 0.5) / 0.5
            # Final formula: (x/255 - 0.5) / 0.5 = x/127.5 - 1
            height, width = 384, 384

            requires_grad = images.requires_grad

            if requires_grad:
                # For gradients: use torch operations
                # Resize using bilinear interpolation (equivalent to BICUBIC in behavior)
                images_resized = torch.nn.functional.interpolate(
                    images, size=(height, width), mode='bicubic', align_corners=False
                )
                # Rescale and normalize: (x/255 - 0.5) / 0.5
                images = images_resized / 127.5 - 1.0
            else:
                # For non-gradient tensors: use official processor
                img = images[0]  # [3, H, W]
                img_np = (img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                pil_img = Image.fromarray(img_np)

                processed = processor(images=pil_img, return_tensors="pt")
                images = processed.pixel_values.to(self.device)  # [1, 3, 384, 384]
        else:
            # Model-specific preprocessing for ViTSTR and CCT
            if model_name and 'vitstr' in model_name:
                # ViTSTR expects 32x128
                images = torch.nn.functional.interpolate(
                    images, size=(32, 128), mode='bicubic', align_corners=False
                )
            elif model_name and 'cct' in model_name:
                # CCT expects 64x128
                images = torch.nn.functional.interpolate(
                    images, size=(64, 128), mode='bicubic', align_corners=False
                )

            # Format images for model
            if input_format['channels_last']:
                # Permute to [batch, H, W, C]
                images = images.permute(0, 2, 3, 1)
            if input_format['scale_to_255']:
                images = images * 255

        # Register hooks
        activations = {}
        handles = []

        def make_hook(name):
            def hook(module, input, output):
                # Store activation, flatten to [batch, features]
                if isinstance(output, torch.Tensor):
                    act = output.detach()
                elif isinstance(output, tuple):
                    act = output[0].detach()
                else:
                    return

                # Flatten
                batch_size = act.shape[0]
                flattened = act.reshape(batch_size, -1)
                activations[name] = flattened
            return hook

        # Find and hook layers
        for layer_name in layer_names:
            # Navigate to layer
            module = model
            for part in layer_name.split('.'):
                if part:
                    module = getattr(module, part, None)
                    if module is None:
                        break

            if module is not None and module != model:
                handle = module.register_forward_hook(make_hook(layer_name))
                handles.append(handle)

        # Forward pass
        with torch.no_grad():
            _ = model(images)

        # Remove hooks
        for handle in handles:
            handle.remove()

        return activations

    def compute_loss(self, batches: List[Dict], conditions) -> Tuple[torch.Tensor, Dict]:
        """
        Compute loss for conditional patch generation.

        Loss = CKA(target_layer) - mean(CKA(prior_layers))

        Args:
            batches: List of individual batch dicts (each has single image)
            conditions: Tuple of (model_names, layer_indices, layer_profiles, layer_depths, model_codes)
        """
        model_names, layer_indices, layer_profiles, layer_depths, model_codes = conditions

        # Move to device
        layer_profiles = layer_profiles.to(self.device)
        layer_depths = layer_depths.to(self.device)
        model_codes = model_codes.to(self.device)

        # Encode conditions to latent
        z = self.encoder(layer_profiles, layer_depths, model_codes)

        # Generate patches
        patches = self.generator(z)

        # Process each sample individually (no stacking due to variable image sizes)
        batch_size = len(batches)
        total_cka_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        stats = {'target_cka': [], 'prior_cka': []}
        valid_samples = 0

        for i in range(batch_size):
            model_name = model_names[i]
            layer_idx = layer_indices[i]
            batch = batches[i]

            # Get model and input format (processor for TrOCR only)
            model_data = self.ocr_models[model_name]
            if len(model_data) == 3:
                ocr_model, input_format, processor = model_data
            else:
                ocr_model, input_format = model_data
                processor = None

            # Get layer names (target + all prior)
            all_layer_names = self.layer_profiles[model_name]['layer_names']
            target_layer_name = all_layer_names[layer_idx]
            prior_layer_names = all_layer_names[:layer_idx] if layer_idx > 0 else []

            # Load images and corners
            image_cpu = batch['prep_image'].unsqueeze(0)  # [1, 3, H, W]
            corners_cpu = batch['new_corners'].unsqueeze(0)  # [1, 4, 2]
            patch = patches[i:i+1]  # [1, 3, 256, 512]

            # Crop to border region on CPU (1.4x plate size) to reduce memory
            # The DEIT preprocessor will handle resizing to 384x384
            border_scale = 1.4
            plate_corners = corners_cpu[0]  # [4, 2]
            plate_min = plate_corners.min(dim=0)[0]
            plate_max = plate_corners.max(dim=0)[0]
            plate_center = (plate_min + plate_max) / 2
            plate_size = plate_max - plate_min

            # Expand corners for border (1.4x)
            border_size = plate_size * border_scale / 2
            border_min = plate_center - border_size
            border_max = plate_center + border_size

            # Clamp to image bounds
            H, W = image_cpu.shape[2], image_cpu.shape[3]
            border_min = torch.clamp(border_min, min=0)
            border_max = torch.clamp(border_max, max=torch.tensor([W, H], dtype=border_max.dtype))

            # Crop border region on CPU
            x_min, y_min = int(border_min[0].item()), int(border_min[1].item())
            x_max, y_max = int(border_max[0].item()), int(border_max[1].item())
            clean_border = image_cpu[:, :, y_min:y_max, x_min:x_max]

            # Adjust corners to be relative to cropped border region
            corners_in_region = corners_cpu[0].clone()
            corners_in_region[:, 0] -= x_min
            corners_in_region[:, 1] -= y_min
            corners_in_region = corners_in_region.unsqueeze(0)

            # Load border region and corners to GPU
            clean_border = clean_border.to(self.device)
            corners_in_region = corners_in_region.to(self.device)

            # Apply patch to border region
            patched_border = self.apply_patch(clean_border, corners_in_region, patch)

            # Extract activations from target layer (and prior if any)
            layers_to_extract = [target_layer_name] + prior_layer_names

            try:
                mem_before = torch.cuda.memory_allocated() / 1e9 if self.device == 'cuda' else 0

                clean_acts = self.extract_activations(ocr_model, clean_border,
                                                      layers_to_extract, input_format, processor, model_name)
                mem_after_clean = torch.cuda.memory_allocated() / 1e9 if self.device == 'cuda' else 0

                patched_acts = self.extract_activations(ocr_model, patched_border,
                                                        layers_to_extract, input_format, processor, model_name)
                mem_after_patched = torch.cuda.memory_allocated() / 1e9 if self.device == 'cuda' else 0

            except Exception as e:
                # Skip this sample if extraction fails
                print(f"  Sample {i} ({model_name}) extraction failed: {e}")
                traceback.print_exc()
                continue

            # Compute normalized delta for target layer
            if target_layer_name in clean_acts and target_layer_name in patched_acts:
                # Get activation statistics if available
                target_std = None
                if model_name in self.activation_stats and target_layer_name in self.activation_stats[model_name]:
                    target_std = self.activation_stats[model_name][target_layer_name].to(self.device)

                target_sim = compute_normalized_delta(clean_acts[target_layer_name],
                                                      patched_acts[target_layer_name],
                                                      layer_std=target_std)
                stats['target_cka'].append(target_sim.item())
            else:
                target_sim = torch.tensor(0.0, device=self.device, requires_grad=True)

            # Compute normalized delta for prior layers (if any)
            if prior_layer_names:
                prior_sims = []
                for prior_name in prior_layer_names:
                    if prior_name in clean_acts and prior_name in patched_acts:
                        # Get activation statistics if available
                        prior_std = None
                        if model_name in self.activation_stats and prior_name in self.activation_stats[model_name]:
                            prior_std = self.activation_stats[model_name][prior_name].to(self.device)

                        prior_sim = compute_normalized_delta(clean_acts[prior_name],
                                                             patched_acts[prior_name],
                                                             layer_std=prior_std)
                        prior_sims.append(prior_sim)

                if prior_sims:
                    mean_prior_sim = torch.stack(prior_sims).mean()
                    stats['prior_cka'].append(mean_prior_sim.item())
                else:
                    mean_prior_sim = torch.tensor(0.0, device=self.device, requires_grad=True)
            else:
                mean_prior_sim = torch.tensor(0.0, device=self.device, requires_grad=True)

            # Sample loss: maximize target delta, minimize prior delta
            # Loss = -target_delta + 0.5 * mean_prior_delta
            sample_loss = -1 * target_sim + 0.5 * mean_prior_sim
            total_cka_loss = total_cka_loss + sample_loss
            valid_samples += 1

        # Average over valid samples
        if valid_samples > 0:
            loss = total_cka_loss / valid_samples
        else:
            # If no valid samples, return zero loss (will skip update)
            loss = torch.tensor(0.0, device=self.device, requires_grad=True)

        # Aggregate stats
        stats['target_cka'] = np.mean(stats['target_cka']) if stats['target_cka'] else 0
        stats['prior_cka'] = np.mean(stats['prior_cka']) if stats['prior_cka'] else 0

        return loss, stats

    def train(self, num_epochs: int = 100, batch_size: int = 16):
        """Train conditional patch generator"""
        import datetime

        # Create unique run ID based on timestamp
        run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        print("\n" + "="*80)
        print("CONDITIONAL PATCH TRAINING")
        print("="*80)
        print(f"Run ID: {run_id}")
        print(f"Epochs: {num_epochs}")
        print(f"Batch size: {batch_size}")
        print(f"Learning rate: {self.learning_rate}")
        print()

        # Track full training history
        training_history = {
            'run_id': run_id,
            'num_epochs': num_epochs,
            'batch_size': batch_size,
            'learning_rate': self.learning_rate,
            'epochs': []
        }

        try:
            for epoch in range(num_epochs):
                self.encoder.train()
                self.generator.train()

                epoch_losses = []
                epoch_target_cka = []
                epoch_prior_cka = []

                # Manually accumulate batches
                accumulated_batches = []
                accumulated_indices = []

                pbar = tqdm(enumerate(self.train_loader), desc=f"Epoch {epoch+1}/{num_epochs}",
                           total=len(self.train_loader))

                for idx, batch in pbar:
                    # Extract single image from batch (batch_size=1 from dataloader)
                    single_batch = {k: v[0] for k, v in batch.items()}
                    accumulated_batches.append(single_batch)
                    accumulated_indices.append(idx)

                    # Process when we have enough samples
                    if len(accumulated_batches) == batch_size or idx == len(self.train_loader) - 1:
                        if len(accumulated_batches) == 0:
                            continue

                        # Sample conditions
                        conditions = self.sample_conditions(len(accumulated_batches),
                                                           accumulated_indices)

                        # Compute loss (pass as list of individual batches, not merged)
                        loss, stats = self.compute_loss(accumulated_batches, conditions)

                        # Backward
                        self.optimizer.zero_grad()
                        loss.backward()
                        self.optimizer.step()

                        # Update progress bar
                        pbar.set_postfix({
                            'loss': f"{loss.item():.4f}",
                            'target_delta': f"{stats['target_cka']:.3f}",
                            'prior_delta': f"{stats['prior_cka']:.3f}"
                        })

                        # Track stats
                        epoch_losses.append(loss.item())
                        epoch_target_cka.append(stats['target_cka'])
                        epoch_prior_cka.append(stats['prior_cka'])

                        # Clear accumulated
                        accumulated_batches = []
                        accumulated_indices = []

                # Epoch summary
                mean_loss = np.mean(epoch_losses) if epoch_losses else 0
                mean_target_sim = np.mean(epoch_target_cka) if epoch_target_cka else 0
                mean_prior_sim = np.mean(epoch_prior_cka) if epoch_prior_cka else 0

                print(f"Epoch {epoch+1} Summary:")
                print(f"  Loss: {mean_loss:.4f}")
                print(f"  Target Normalized Delta: {mean_target_sim:.3f}")
                print(f"  Prior Normalized Delta: {mean_prior_sim:.3f}")
                print()

                # Save to training history
                training_history['epochs'].append({
                    'epoch': epoch + 1,
                    'loss': float(mean_loss),
                    'target_sim': float(mean_target_sim),
                    'prior_sim': float(mean_prior_sim),
                    'num_batches': len(epoch_losses)
                })

                # Save checkpoint every epoch
                self.save_checkpoint(epoch + 1, run_id, training_history)

        except KeyboardInterrupt:
            print("\n\nTraining interrupted by user")
            # Save history even on interrupt
            self.save_training_history(run_id, training_history)
            raise

        # Final save
        self.save_training_history(run_id, training_history)
        print(f"\nTraining complete! Run ID: {run_id}")

    def save_checkpoint(self, epoch: int, run_id: str, training_history: dict):
        """Save model checkpoint and training history"""
        checkpoint = {
            'epoch': epoch,
            'run_id': run_id,
            'encoder': self.encoder.state_dict(),
            'generator': self.generator.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }
        checkpoint_path = f"conditional_patch_{run_id}_epoch{epoch}.pt"
        torch.save(checkpoint, checkpoint_path)
        print(f"Saved checkpoint: {checkpoint_path}")

    def save_training_history(self, run_id: str, training_history: dict):
        """Save full training history to JSON"""
        history_path = f"training_history_{run_id}.json"
        import json
        with open(history_path, 'w') as f:
            json.dump(training_history, f, indent=2)
        print(f"Saved training history: {history_path}")


def main():
    parser = argparse.ArgumentParser(description='Train conditional patch generator')
    parser.add_argument('--csv-path', type=str, default='preproc_labels.csv',
                       help='Path to dataset CSV')
    parser.add_argument('--profile-dir', type=str, default='layer_profiles',
                       help='Path to layer profiles directory')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--generator', type=str, default='simple', choices=['simple', 'diffusion'],
                       help='Generator type: simple (CNN) or diffusion (Stable Diffusion VAE)')
    args = parser.parse_args()

    trainer = ConditionalPatchTrainer(
        csv_path=args.csv_path,
        profile_dir=args.profile_dir,
        learning_rate=args.lr,
        generator_type=args.generator
    )

    trainer.train(num_epochs=args.epochs, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
