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

from dataset import create_dataloaders


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
    """Simple MLP patch generator"""
    def __init__(self, latent_dim: int = 32, patch_height: int = 256,
                 patch_width: int = 512):
        super().__init__()

        self.latent_dim = latent_dim
        self.patch_height = patch_height
        self.patch_width = patch_width
        self.patch_dim = 3 * patch_height * patch_width

        self.network = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 512),
            nn.LayerNorm(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, self.patch_dim),
            nn.Sigmoid(),  # Output in [0, 1]
        )

        # Initialize weights
        for m in self.modules():
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
        patches_flat = self.network(z)
        patches = patches_flat.view(-1, 3, self.patch_height, self.patch_width)
        return patches


def compute_cosine_similarity(X: torch.Tensor, Y: torch.Tensor, epsilon: float = 1e-8) -> torch.Tensor:
    """
    Compute mean cosine similarity between two activation matrices.

    Handles variable-sized activations by truncating to common dimension.

    Args:
        X: [n_samples, n_features_x]
        Y: [n_samples, n_features_y]
        epsilon: Small value for numerical stability

    Returns:
        similarity: Scalar cosine similarity in [0, 1]
    """
    # Flatten to 2D
    X_flat = X.reshape(X.shape[0], -1)
    Y_flat = Y.reshape(Y.shape[0], -1)

    # Truncate to common dimension (handles different activation sizes)
    min_features = min(X_flat.shape[1], Y_flat.shape[1])
    X_flat = X_flat[:, :min_features]
    Y_flat = Y_flat[:, :min_features]

    # Normalize vectors
    X_norm = F.normalize(X_flat, p=2, dim=1)
    Y_norm = F.normalize(Y_flat, p=2, dim=1)

    # Cosine similarity: mean of dot products between normalized vectors
    similarity = (X_norm * Y_norm).sum(dim=1).mean()

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
                 device: str = 'cuda', learning_rate: float = 1e-3):
        self.device = device
        self.learning_rate = learning_rate

        print(f"\nGPU Memory at start: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

        # Load layer profiles
        print("\nLoading layer profiles...")
        self.layer_profiles = load_layer_profiles(profile_dir)
        print(f"GPU Memory after loading profiles: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        print(f"Loaded profiles for {len(self.layer_profiles)} models:")
        for model_name, profile_data in self.layer_profiles.items():
            print(f"  - {model_name}: {profile_data['n_layers']} layers")

        # Load OCR models
        print("\nLoading OCR models...")
        self.ocr_models = load_ocr_models(device)
        print(f"GPU Memory after loading OCR models: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

        # Create model name to index mapping
        self.model_names = sorted(self.layer_profiles.keys())
        self.model_to_idx = {name: idx for idx, name in enumerate(self.model_names)}
        self.n_models = len(self.model_names)

        # Define model-specific input shapes (from profile_ocr_models.py)
        self.model_input_shapes = {
            'vitstr_small': (32, 128),  # ViTSTR expects 32x128
            'cct_xs_v1_global': (64, 128),  # CCT expects 64x128
            'trocr_small_printed_encoder': (64, 128),  # TrOCR initial crop 64x128
        }

        # Create encoder and generator
        self.encoder = LayerProfileEncoder(input_dim=12, latent_dim=32).to(device)
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
                           input_format: Dict, processor=None) -> Dict[str, torch.Tensor]:
        """
        Run model and extract activations at specified layers.

        Args:
            model: The OCR model
            images: [1, 3, H, W] image tensor
            layer_names: List of layer names to extract
            input_format: Dict with 'channels_last' and 'scale_to_255' flags
            processor: Optional TrOCRProcessor for TrOCR model

        Returns:
            activations: Dict {layer_name: tensor [batch, features]}
        """
        # Use processor for TrOCR (converts to model's expected input size)
        if processor is not None:
            from PIL import Image
            import numpy as np

            # For tensors that require gradients, we need to keep them in torch
            # For gradients to flow, use torch operations instead of numpy
            requires_grad = images.requires_grad

            if requires_grad:
                # Keep gradients: use torch-based preprocessing
                # Get the expected input size from processor
                expected_size = processor.image_processor.size
                if isinstance(expected_size, dict):
                    height, width = expected_size.get('height', 384), expected_size.get('width', 384)
                else:
                    height, width = expected_size, expected_size

                # Resize using torch (bilinear interpolation preserves gradients)
                images_resized = torch.nn.functional.interpolate(
                    images, size=(height, width), mode='bilinear', align_corners=False
                )
                # Normalize to [-1, 1] (standard for vision transformers)
                images = images_resized / 255.0 * 2.0 - 1.0
            else:
                # No gradients: use PIL-based processor (more accurate)
                img = images[0]  # [3, H, W]
                img_np = (img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                pil_img = Image.fromarray(img_np)

                # Process with TrOCRProcessor
                processed = processor(images=pil_img, return_tensors="pt")
                images = processed.pixel_values.to(self.device)  # [1, 3, H_model, W_model]
        else:
            # Format images for other models
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

            # Crop plate region on CPU before loading to GPU (reduces memory usage)
            image_cpu = batch['prep_image'].unsqueeze(0)  # [1, 3, H, W]
            corners_cpu = batch['new_corners'].unsqueeze(0)  # [1, 4, 2]

            # Get model-specific input shape
            model_input_shape = self.model_input_shapes[model_name]

            # Crop to license plate area using model-specific shape
            cropped_clean = K.crop_and_resize(image_cpu, corners_cpu, model_input_shape)

            # Debug: verify cropping worked as expected
            if cropped_clean.shape != (1, 3, model_input_shape[0], model_input_shape[1]):
                print(f"  WARNING Sample {i} ({model_name}): Expected shape (1, 3, {model_input_shape[0]}, {model_input_shape[1]}), got {cropped_clean.shape}")

            # Load only the cropped image to GPU
            cropped_clean = cropped_clean.to(self.device)
            corners = corners_cpu.to(self.device)  # [1, 4, 2]
            patch = patches[i:i+1]  # [1, 3, 256, 512]

            # For patching, we need to apply patch to cropped region
            # Crop the original image for patching with border
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
            border_region = image_cpu[:, :, y_min:y_max, x_min:x_max]

            # Adjust corners to be relative to cropped border region
            corners_in_region = corners_cpu[0].clone()
            corners_in_region[:, 0] -= x_min
            corners_in_region[:, 1] -= y_min
            corners_in_region = corners_in_region.unsqueeze(0)

            # Load border region and corners to GPU
            border_region = border_region.to(self.device)
            corners_in_region = corners_in_region.to(self.device)

            # Apply patch to border region
            patched_border = self.apply_patch(border_region, corners_in_region, patch)

            # Crop patched region to model-specific input shape
            cropped_patched = K.crop_and_resize(patched_border, corners_in_region, model_input_shape)

            # Debug: verify patched cropping worked as expected
            if cropped_patched.shape != (1, 3, model_input_shape[0], model_input_shape[1]):
                print(f"  WARNING Sample {i} ({model_name}): Patched shape mismatch - Expected (1, 3, {model_input_shape[0]}, {model_input_shape[1]}), got {cropped_patched.shape}")
                print(f"    Border region shape: {border_region.shape}, Corners in region: {corners_in_region}")

            # Extract activations from target layer (and prior if any)
            layers_to_extract = [target_layer_name] + prior_layer_names

            try:
                mem_before = torch.cuda.memory_allocated() / 1e9 if self.device == 'cuda' else 0

                clean_acts = self.extract_activations(ocr_model, cropped_clean,
                                                      layers_to_extract, input_format, processor)
                mem_after_clean = torch.cuda.memory_allocated() / 1e9 if self.device == 'cuda' else 0

                patched_acts = self.extract_activations(ocr_model, cropped_patched,
                                                        layers_to_extract, input_format, processor)
                mem_after_patched = torch.cuda.memory_allocated() / 1e9 if self.device == 'cuda' else 0

                if i == 0:  # Print memory info for first sample only
                    print(f"  Sample {i}: {mem_before:.2f}GB → {mem_after_clean:.2f}GB (clean) → {mem_after_patched:.2f}GB (patched)")

            except Exception as e:
                # Skip this sample if extraction fails
                print(f"  Sample {i} ({model_name}) extraction failed: {e}")
                print(f"    Clean shape: {cropped_clean.shape}, Patched shape: {cropped_patched.shape}")
                traceback.print_exc()
                continue

            # Compute cosine similarity for target layer
            if target_layer_name in clean_acts and target_layer_name in patched_acts:
                target_sim = compute_cosine_similarity(clean_acts[target_layer_name],
                                                       patched_acts[target_layer_name])
                stats['target_cka'].append(target_sim.item())
            else:
                target_sim = torch.tensor(0.0, device=self.device, requires_grad=True)

            # Compute cosine similarity for prior layers (if any)
            if prior_layer_names:
                prior_sims = []
                for prior_name in prior_layer_names:
                    if prior_name in clean_acts and prior_name in patched_acts:
                        prior_sim = compute_cosine_similarity(clean_acts[prior_name],
                                                              patched_acts[prior_name])
                        prior_sims.append(prior_sim)

                if prior_sims:
                    mean_prior_sim = torch.stack(prior_sims).mean()
                    stats['prior_cka'].append(mean_prior_sim.item())
                else:
                    mean_prior_sim = torch.tensor(0.0, device=self.device, requires_grad=True)
            else:
                mean_prior_sim = torch.tensor(0.0, device=self.device, requires_grad=True)

            # Sample loss: minimize target similarity, maximize prior similarity
            # Loss = target_sim - mean_prior_sim
            sample_loss = target_sim - mean_prior_sim
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
        print("\n" + "="*80)
        print("CONDITIONAL PATCH TRAINING")
        print("="*80)
        print(f"Epochs: {num_epochs}")
        print(f"Batch size: {batch_size}")
        print(f"Learning rate: {self.learning_rate}")
        print()

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

                    # Track stats
                    if loss.item() > 0:  # Only track if loss was computed
                        epoch_losses.append(loss.item())
                        epoch_target_cka.append(stats['target_cka'])
                        epoch_prior_cka.append(stats['prior_cka'])

                        # Update progress
                        pbar.set_postfix({
                            'loss': f"{loss.item():.4f}",
                            'target_sim': f"{stats['target_cka']:.3f}",
                            'prior_sim': f"{stats['prior_cka']:.3f}"
                        })

                    # Clear accumulated
                    accumulated_batches = []
                    accumulated_indices = []

            # Epoch summary
            print(f"Epoch {epoch+1} Summary:")
            print(f"  Loss: {np.mean(epoch_losses):.4f}")
            print(f"  Target Cosine Similarity: {np.mean(epoch_target_cka):.3f}")
            print(f"  Prior Cosine Similarity: {np.mean(epoch_prior_cka):.3f}")
            print()

            # Save checkpoint every 10 epochs
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(f"conditional_patch_epoch{epoch+1}.pt")

    def save_checkpoint(self, filename: str):
        """Save model checkpoint"""
        checkpoint = {
            'encoder': self.encoder.state_dict(),
            'generator': self.generator.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }
        torch.save(checkpoint, filename)
        print(f"Saved checkpoint: {filename}")


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
    args = parser.parse_args()

    trainer = ConditionalPatchTrainer(
        csv_path=args.csv_path,
        profile_dir=args.profile_dir,
        learning_rate=args.lr
    )

    trainer.train(num_epochs=args.epochs, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
