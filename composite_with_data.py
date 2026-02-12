#!/usr/bin/env python3
"""
Composite adversarial patches with validation data.

Takes the latest epoch/batch patches from a progressive_patch.py run
and applies them to random validation samples for testing.

Usage:
  python composite_with_data.py checkpoints/20260211_174724 -o output_composites
"""

import argparse
import sys
from pathlib import Path
from glob import glob
import csv
import random
import importlib.util

import cv2
import torch
import torch.nn.functional as F
import numpy as np
from torchvision import transforms as T


def load_patch_from_png(patch_path):
    """Load patch from PNG file as [3, H, W] tensor in [0, 1]."""
    img = cv2.imread(str(patch_path))
    if img is None:
        raise FileNotFoundError(f"Could not load patch: {patch_path}")
    # Convert BGR to RGB and normalize
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    # Convert to tensor [3, H, W]
    tensor = torch.from_numpy(np.transpose(img_rgb, (2, 0, 1)))
    return tensor


def load_validation_samples_from_csv(csv_path, num_samples):
    """Load random validation samples from CSV using public dataset loaders.

    Args:
        csv_path: Path to train_val_split CSV
        num_samples: Number of samples to load

    Returns:
        List of loaded images as tensors [3, H, W] in [0, 1]
    """
    # Import load_datasets from foundationmodel/dataset/
    script_dir = Path(__file__).parent
    load_datasets_path = script_dir / "foundationmodel" / "dataset" / "load_datasets.py"

    if not load_datasets_path.exists():
        raise FileNotFoundError(f"Could not find load_datasets.py at {load_datasets_path}")

    spec = importlib.util.spec_from_file_location("load_datasets", load_datasets_path)
    load_datasets = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(load_datasets)
    iter_dataset = load_datasets.iter_dataset

    # Read CSV to find validation samples
    val_samples = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['split'].lower() == 'val':
                val_samples.append({
                    'dataset': row['dataset'],
                    'index': int(row['index']),
                    'text': row['text']
                })

    if not val_samples:
        raise ValueError(f"No validation samples found in {csv_path}")

    print(f"Found {len(val_samples)} validation samples in CSV")

    # Load images using iter_dataset
    images = []
    loaded_datasets = {}  # Cache dataset iterators

    # Try to load samples until we get the requested number
    attempts = 0
    max_attempts = len(val_samples)

    while len(images) < num_samples and attempts < max_attempts:
        # Randomly select one sample
        sample_info = random.choice(val_samples)
        dataset_name = sample_info['dataset']
        sample_idx = sample_info['index']

        # Load dataset if not already cached
        if dataset_name not in loaded_datasets:
            print(f"  Loading {dataset_name}...")
            all_samples = []
            for img, text, meta in iter_dataset(dataset_name, 'train'):
                all_samples.append((img, text, meta))
            loaded_datasets[dataset_name] = all_samples

        # Get the sample
        dataset_samples = loaded_datasets[dataset_name]
        if sample_idx < len(dataset_samples):
            try:
                img, text, meta = dataset_samples[sample_idx]

                # Convert PIL image to tensor
                if isinstance(img, np.ndarray):
                    img_array = img
                else:
                    img_array = np.array(img)

                img_rgb = img_array.astype(np.float32) / 255.0
                tensor = torch.from_numpy(np.transpose(img_rgb, (2, 0, 1)))
                images.append(tensor)
            except Exception as e:
                print(f"  Warning: Failed to load {dataset_name}[{sample_idx}]: {e}")
                attempts += 1
                continue

        attempts += 1

    print(f"Loaded {len(images)} validation samples (attempted {attempts} times)")
    return images


def apply_patch_ocr_mode(image, patch, center_ratio=0.6):
    """Apply adversarial patch to image (center region preserved).

    Args:
        image: [3, H, W] or [1, 3, H, W] tensor in [0, 1]
        patch: [3, patch_h, patch_w] tensor in [0, 1]
        center_ratio: Fraction of image to preserve in center (default: 0.6)

    Returns:
        result: [B, 3, H, W] patched image
    """
    # Handle single image
    if image.dim() == 3:
        image = image.unsqueeze(0)

    batch_size = image.shape[0]
    image_height, image_width = image.shape[2], image.shape[3]

    # Resize patch to match image dimensions
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
                             dtype=torch.float32)
    center_mask[:, :, pad_h:pad_h + center_h, pad_w:pad_w + center_w] = 1.0
    center_mask = center_mask.expand(-1, 3, -1, -1)  # [B, 3, H, W]

    # Blend: keep original image in center, use patch on borders
    result_image = image * center_mask + patch_batch * (1 - center_mask)
    result_image = torch.clamp(result_image, 0, 1)

    return result_image


def apply_neutral_border_ocr_mode(image, center_ratio=0.6, border_color=0.5):
    """Apply neutral grey border to image (center region preserved).

    Args:
        image: [3, H, W] or [1, 3, H, W] tensor in [0, 1]
        center_ratio: Fraction of image to preserve in center (default: 0.6)
        border_color: Value for neutral border (default: 0.5 = gray)

    Returns:
        result: [B, 3, H, W] image with grey border
    """
    # Handle single image
    if image.dim() == 3:
        image = image.unsqueeze(0)

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
                             dtype=torch.float32)
    center_mask[:, :, pad_h:pad_h + center_h, pad_w:pad_w + center_w] = 1.0
    center_mask = center_mask.expand(-1, 3, -1, -1)  # [B, 3, H, W]

    # Create neutral border
    neutral_border = torch.full_like(image, border_color)

    # Blend: keep original image in center, use neutral border on borders
    result_image = image * center_mask + neutral_border * (1 - center_mask)
    result_image = torch.clamp(result_image, 0, 1)

    return result_image


def find_latest_patches(run_dir):
    """Find the latest epoch/batch patches in the run directory.

    Returns:
        List of patch file paths
    """
    example_samples_dir = Path(run_dir) / "example_samples"
    if not example_samples_dir.exists():
        raise FileNotFoundError(f"example_samples directory not found in {run_dir}")

    # Find all batch directories
    batch_dirs = sorted([d for d in example_samples_dir.iterdir() if d.is_dir()])
    if not batch_dirs:
        raise FileNotFoundError(f"No batch directories found in {example_samples_dir}")

    latest_batch = batch_dirs[-1]
    print(f"Using latest batch: {latest_batch.name}")

    # Find all patch files in the latest batch
    patch_files = sorted(latest_batch.glob("patch_epoch_*_sample_*.png"))
    if not patch_files:
        raise FileNotFoundError(f"No patch files found in {latest_batch}")

    print(f"Found {len(patch_files)} patches")
    return patch_files


def main():
    parser = argparse.ArgumentParser(
        description='Composite adversarial patches with validation data.'
    )
    parser.add_argument('run_dir', help='Path to run directory (e.g., checkpoints/20260211_174724)')
    parser.add_argument('-o', '--outdir', default='composite_output',
                        help='Output directory for composite images (default: composite_output)')

    args = parser.parse_args()

    try:
        run_dir = Path(args.run_dir)
        if not run_dir.exists():
            print(f"Error: Run directory not found: {run_dir}", file=sys.stderr)
            sys.exit(1)

        # Create output directory
        output_dir = Path(args.outdir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Find latest patches
        print(f"Loading patches from {run_dir}...")
        patch_files = find_latest_patches(run_dir)
        num_patches = len(patch_files)

        # Load patches
        patches = []
        for patch_file in patch_files:
            patch = load_patch_from_png(patch_file)
            patches.append(patch)
            print(f"  Loaded {patch_file.name}")

        # Find CSV file
        csv_files = list(run_dir.glob("**/*.csv"))
        data_split_csv = None
        for csv_file in csv_files:
            if 'train_val_split' in csv_file.name or 'split' in csv_file.name:
                data_split_csv = csv_file
                break

        if data_split_csv is None:
            # Try current directory as fallback
            cwd_csv = list(Path('.').glob("train_val_split_*.csv"))
            if cwd_csv:
                data_split_csv = cwd_csv[-1]

        if data_split_csv is None:
            print("Error: Could not find train_val_split CSV file", file=sys.stderr)
            sys.exit(1)

        print(f"Using data split: {data_split_csv}")

        # Load validation samples (one per patch)
        print(f"Loading {num_patches} validation samples (one per patch)...")
        val_images = load_validation_samples_from_csv(
            data_split_csv, num_patches
        )

        if not val_images:
            print("Error: No validation samples loaded", file=sys.stderr)
            sys.exit(1)

        # Composite patches with samples (1-to-1 pairing)
        print(f"\nCreating {len(val_images)} composite images with controls...")
        print(f"Pairing {len(patches)} patches with {len(val_images)} validation samples (1-to-1)...")
        saved_count = 0
        for img_idx, (val_image, patch) in enumerate(zip(val_images, patches)):
            try:
                # Apply patch
                composite = apply_patch_ocr_mode(val_image, patch, center_ratio=0.6)
                composite = composite.squeeze(0)  # Remove batch dim

                # Convert to numpy and save
                composite_np = (composite.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                composite_bgr = cv2.cvtColor(composite_np, cv2.COLOR_RGB2BGR)

                output_path = output_dir / f"composite_{img_idx:02d}.jpg"
                cv2.imwrite(str(output_path), composite_bgr)

                # Apply grey control border
                control = apply_neutral_border_ocr_mode(val_image, center_ratio=0.6, border_color=0.5)
                control = control.squeeze(0)  # Remove batch dim

                # Convert to numpy and save
                control_np = (control.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                control_bgr = cv2.cvtColor(control_np, cv2.COLOR_RGB2BGR)

                control_path = output_dir / f"control_{img_idx:02d}.jpg"
                cv2.imwrite(str(control_path), control_bgr)

                print(f"  Saved composite_{img_idx:02d}.jpg and control_{img_idx:02d}.jpg")
                saved_count += 1
            except Exception as e:
                print(f"  Error processing image {img_idx}: {e}", file=sys.stderr)

        print(f"\nSuccessfully created {saved_count} composite and {saved_count} control image pairs")
        print(f"Saved to {output_dir}")

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
