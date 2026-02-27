#!/usr/bin/env python3
"""
Apply all attack strategies to validation images using a trained patch generator.

This script:
1. Loads validation images from a manifest or directory
2. Loads a trained generator checkpoint
3. Generates adversarial patches
4. Applies all attack strategies (border, sticker, perturbation)
5. Saves results in a tarred format at project root

Usage:
    python apply_attacks_to_images.py \\
        --manifest /path/to/manifest.csv \\
        --checkpoint /path/to/generator_epoch_0100.pt \\
        --num-samples 5 \\
        --num-test-images 10

Or with directory input:
    python apply_attacks_to_images.py \\
        --image-dir /path/to/images \\
        --checkpoint /path/to/generator_epoch_0100.pt
"""

import argparse
import csv
import os
import tarfile
import torch
import numpy as np
import torchvision.transforms as T
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from typing import Optional, List, Dict, Tuple

# Framework imports
from framework.generator import FoundationPatchGenerator
from framework.base.attack_strategy import (
    BorderStrategy,
    StickerStrategy,
    PerturbationStrategy,
)


def load_image_from_path(image_path: str) -> Tuple[torch.Tensor, str]:
    """
    Load an image and return as [3, H, W] tensor in [0, 1] range.

    Args:
        image_path: Path to image file

    Returns:
        Tuple of (tensor, filename)
    """
    try:
        img = Image.open(image_path).convert('RGB')
        img_tensor = T.ToTensor()(img)  # [3, H, W] in [0, 1]
        filename = Path(image_path).name
        return img_tensor, filename
    except Exception as e:
        print(f"Warning: could not load {image_path}: {e}")
        return None, None


def load_generator_checkpoint(checkpoint_path: str, device: str = 'cuda') -> FoundationPatchGenerator:
    """Load a trained generator from checkpoint."""
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    basis_dim = checkpoint['basis_dim']
    patch_size = checkpoint.get('patch_size', [224, 224])
    patch_height, patch_width = patch_size
    num_taesd = checkpoint.get('num_taesd', 1)

    # Get transformer params
    transformer_d_model = checkpoint.get('transformer_d_model', 256)
    transformer_nhead = checkpoint.get('transformer_nhead', 4)
    transformer_d_ff = checkpoint.get('transformer_d_ff', 1024)
    transformer_enc_layers = checkpoint.get('transformer_enc_layers', 2)
    transformer_dec_layers = checkpoint.get('transformer_dec_layers', 2)

    print(f"  Basis dim: {basis_dim}")
    print(f"  Patch size: {patch_height} × {patch_width}")
    print(f"  Num TAESD: {num_taesd}")

    generator = FoundationPatchGenerator(
        latent_dim=basis_dim,
        patch_height=patch_height,
        patch_width=patch_width,
        num_taesd=num_taesd,
        transformer_d_model=transformer_d_model,
        transformer_nhead=transformer_nhead,
        transformer_d_ff=transformer_d_ff,
        transformer_enc_layers=transformer_enc_layers,
        transformer_dec_layers=transformer_dec_layers,
    ).to(device)

    generator.load_state_dict(checkpoint['generator_state_dict'])
    generator.eval()
    print(f"✓ Generator loaded on {device}")

    return generator, patch_height, patch_width


def load_validation_images(
    manifest_path: Optional[str] = None,
    image_dir: Optional[str] = None,
    num_samples: int = 10,
    device: str = 'cuda',
) -> List[Tuple[torch.Tensor, str]]:
    """
    Load validation images from either a manifest CSV or a directory.

    Args:
        manifest_path: Path to CSV manifest with 'path' column
        image_dir: Directory containing images
        num_samples: Number of images to load
        device: Device to load images to

    Returns:
        List of (image_tensor, filename) tuples
    """
    images = []

    if manifest_path:
        print(f"Loading images from manifest: {manifest_path}")
        with open(manifest_path, 'r') as f:
            reader = csv.DictReader(f)
            paths = [row['path'] for i, row in enumerate(reader) if i < num_samples]
    elif image_dir:
        print(f"Loading images from directory: {image_dir}")
        image_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp'}
        paths = [
            str(p) for p in Path(image_dir).rglob('*')
            if p.suffix.lower() in image_exts
        ][:num_samples]
    else:
        raise ValueError("Must provide either --manifest or --image-dir")

    print(f"Found {len(paths)} images")

    for img_path in tqdm(paths, desc="Loading images"):
        img_tensor, filename = load_image_from_path(img_path)
        if img_tensor is not None:
            images.append((img_tensor.to(device), filename))

    print(f"✓ Loaded {len(images)} images")
    return images


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """Convert [3, H, W] tensor in [0, 1] to PIL Image."""
    # Clamp to [0, 1] to be safe
    tensor = torch.clamp(tensor, 0, 1)
    pil = T.ToPILImage()(tensor.cpu())
    return pil


def save_composited_images(
    output_dir: Path,
    image_name: str,
    original: torch.Tensor,
    border_img: torch.Tensor,
    sticker_img: torch.Tensor,
    perturbation_img: torch.Tensor,
    patch_img: torch.Tensor,
):
    """Save all variants of an image."""
    output_dir.mkdir(parents=True, exist_ok=True)

    base_name = Path(image_name).stem

    # Save original
    tensor_to_pil(original).save(output_dir / f"{base_name}_00_original.png")

    # Save border attack
    tensor_to_pil(border_img).save(output_dir / f"{base_name}_01_border_attack.png")

    # Save sticker attack
    tensor_to_pil(sticker_img).save(output_dir / f"{base_name}_02_sticker_attack.png")

    # Save perturbation attack
    tensor_to_pil(perturbation_img).save(output_dir / f"{base_name}_03_perturbation_attack.png")

    # Save patch
    tensor_to_pil(patch_img).save(output_dir / f"{base_name}_patch.png")


def apply_attacks(
    generator: FoundationPatchGenerator,
    images: List[Tuple[torch.Tensor, str]],
    patch_height: int,
    patch_width: int,
    num_patch_samples: int = 3,
    device: str = 'cuda',
) -> Path:
    """
    Apply all attack strategies to each image and save results.

    Args:
        generator: Trained patch generator
        images: List of (image_tensor, filename) tuples
        patch_height: Height of generated patches
        patch_width: Width of generated patches
        num_patch_samples: Number of different patches to generate per image
        device: Torch device

    Returns:
        Path to output directory
    """
    output_dir = Path("attack_outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize strategies
    border_strategy = BorderStrategy(center_ratio=0.91)
    sticker_strategy = StickerStrategy(area_fraction=0.2)
    perturbation_strategy = PerturbationStrategy(budget=0.1, norm='linf')

    print(f"\nApplying attacks to {len(images)} images...")
    print(f"  Border strategy: center_ratio=0.91")
    print(f"  Sticker strategy: area_fraction=0.2")
    print(f"  Perturbation strategy: budget=0.1")

    with torch.no_grad():
        for img_idx, (image, filename) in enumerate(tqdm(images, desc="Processing images")):
            # Generate patches
            z_samples = torch.rand(num_patch_samples, generator.latent_dim, device=device)
            patches = generator(z_samples)  # [num_samples, 3, H, W]

            # Use first patch for all strategies
            patch = patches[0]

            # Prepare image as batch [1, 3, H, W]
            image_batch = image.unsqueeze(0)

            # Resize image to patch size for border strategy
            image_resized = torch.nn.functional.interpolate(
                image_batch,
                size=(patch_height, patch_width),
                mode='bilinear',
                align_corners=False
            )

            # Apply border attack
            border_composited, _ = border_strategy.apply(image_resized, patch)
            border_composited = border_composited.squeeze(0)

            # Apply sticker attack (use original image size)
            sticker_composited, _ = sticker_strategy.apply(image_batch, patch)
            sticker_composited = sticker_composited.squeeze(0)

            # Apply perturbation attack (use original image size)
            perturbation_composited, _ = perturbation_strategy.apply(image_batch, patch)
            perturbation_composited = perturbation_composited.squeeze(0)

            # Save all variants
            image_output_dir = output_dir / f"image_{img_idx:03d}"
            save_composited_images(
                image_output_dir,
                filename,
                image.squeeze(0) if image.dim() == 4 else image,
                border_composited,
                sticker_composited,
                perturbation_composited,
                patch,
            )

    print(f"✓ Attack outputs saved to: {output_dir}")
    return output_dir


def create_tar_archive(output_dir: Path, tar_path: Path) -> None:
    """Create a tarred archive of the output directory."""
    print(f"\nCreating tar archive: {tar_path}")
    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(output_dir, arcname=output_dir.name)
    print(f"✓ Archive created: {tar_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Apply all attack strategies to validation images"
    )

    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        '--manifest',
        type=str,
        help='Path to manifest CSV with image paths'
    )
    input_group.add_argument(
        '--image-dir',
        type=str,
        help='Directory containing images'
    )

    # Checkpoint
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to trained generator checkpoint'
    )

    # Options
    parser.add_argument(
        '--num-samples',
        type=int,
        default=10,
        help='Number of validation images to process (default: 10)'
    )
    parser.add_argument(
        '--num-patch-samples',
        type=int,
        default=3,
        help='Number of different patches to generate per image (default: 3)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to use (cuda/cpu, default: cuda if available)'
    )
    parser.add_argument(
        '--output-tar',
        type=str,
        default='attack_outputs.tar.gz',
        help='Output tar.gz filename at project root (default: attack_outputs.tar.gz)'
    )

    args = parser.parse_args()

    # Validate checkpoint exists
    if not Path(args.checkpoint).exists():
        print(f"Error: checkpoint not found: {args.checkpoint}")
        return 1

    # Load generator
    try:
        generator, patch_h, patch_w = load_generator_checkpoint(args.checkpoint, args.device)
    except Exception as e:
        print(f"Error loading generator: {e}")
        return 1

    # Load images
    try:
        images = load_validation_images(
            manifest_path=args.manifest,
            image_dir=args.image_dir,
            num_samples=args.num_samples,
            device=args.device,
        )
    except Exception as e:
        print(f"Error loading images: {e}")
        return 1

    if not images:
        print("Error: no images loaded")
        return 1

    # Apply attacks
    try:
        output_dir = apply_attacks(
            generator,
            images,
            patch_h,
            patch_w,
            num_patch_samples=args.num_patch_samples,
            device=args.device,
        )
    except Exception as e:
        print(f"Error applying attacks: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Create tar archive
    try:
        tar_path = Path(args.output_tar)
        if tar_path.is_absolute():
            tar_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            tar_path = Path.cwd() / tar_path

        create_tar_archive(output_dir, tar_path)
    except Exception as e:
        print(f"Error creating tar archive: {e}")
        return 1

    print(f"\n✓ Done! Results available at:")
    print(f"  Directory: {output_dir}")
    print(f"  Archive: {tar_path}")

    return 0


if __name__ == '__main__':
    exit(main())
