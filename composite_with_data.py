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
from tqdm import tqdm


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
    """Load validation samples using combined dataset (matching training setup).

    Args:
        csv_path: Path to train_val_split CSV
        num_samples: Number of samples to load

    Returns:
        Tuple of (list of images as tensors [3, H, W] in [0, 1], list of (width, height) tuples)
    """
    # Import OCRDataset and ConcatDataset
    script_dir = Path(__file__).parent
    from torch.utils.data import ConcatDataset

    # Import OCRDataset from progressive_patch
    sys.path.insert(0, str(script_dir))
    try:
        from progressive_patch import OCRDataset
    except ImportError:
        raise ImportError("Could not import OCRDataset from progressive_patch.py")

    # Read CSV to get dataset names and validation indices
    val_indices = []
    dataset_names_in_csv = set()

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['split'].lower() == 'val':
                val_indices.append(int(row['index']))
                dataset_names_in_csv.add(row['dataset'])

    if not val_indices:
        raise ValueError(f"No validation samples found in {csv_path}")

    print(f"Found {len(val_indices)} validation samples in CSV")
    print(f"Datasets in CSV: {', '.join(sorted(dataset_names_in_csv))}")

    # Load and combine datasets in order
    datasets_to_combine = []
    for dataset_name in sorted(dataset_names_in_csv):
        print(f"  Loading {dataset_name}...")
        try:
            dataset = OCRDataset(
                dataset_name=dataset_name,
                split='train',
                transform=None,
                max_samples=None
            )
            datasets_to_combine.append(dataset)
            print(f"    Loaded {len(dataset)} samples from {dataset_name}")
        except Exception as e:
            print(f"  Error loading {dataset_name}: {e}", file=sys.stderr)
            raise

    # Combine datasets (matching how progressive_patch.py does it)
    if len(datasets_to_combine) > 1:
        combined_dataset = ConcatDataset(datasets_to_combine)
        print(f"Combined {len(datasets_to_combine)} datasets: {len(combined_dataset)} total samples")
    else:
        combined_dataset = datasets_to_combine[0]

    # Load validation samples using combined dataset indices
    images = []
    dimensions = []  # Track (width, height) for each image
    failed_samples = []

    # Randomly select validation indices to load
    selected_indices = random.sample(val_indices, min(num_samples, len(val_indices)))

    print(f"\nLoading {len(selected_indices)} validation samples from combined dataset...")
    for combined_idx in selected_indices:
        try:
            item = combined_dataset[combined_idx]
            img_tensor = item['prep_image']
            images.append(img_tensor)
            # Track dimensions: tensor is [3, H, W], so width=W, height=H
            height, width = img_tensor.shape[1], img_tensor.shape[2]
            dimensions.append((width, height))
        except Exception as e:
            error_msg = f"{type(e).__name__}: {e}"
            print(f"  Warning: Failed to load sample {combined_idx}: {error_msg}", file=sys.stderr)
            failed_samples.append((combined_idx, error_msg))

    print(f"Loaded {len(images)} validation samples")
    if failed_samples:
        print(f"Failed to load {len(failed_samples)} samples:", file=sys.stderr)
        for idx, error in failed_samples:
            print(f"  Index {idx}: {error}", file=sys.stderr)

    return images, dimensions


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
    parser.add_argument('-n', '--n-samples', type=int, default=1,
                        help='Number of validation samples per patch (default: 1). Total composites = num_patches * n_samples')

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

        # Load validation samples (n_samples per patch)
        num_val_samples = num_patches * args.n_samples
        print(f"Loading {num_val_samples} validation samples ({args.n_samples} per patch)...")
        val_images, dimensions = load_validation_samples_from_csv(
            data_split_csv, num_val_samples
        )

        if not val_images:
            print("Error: No validation samples loaded", file=sys.stderr)
            sys.exit(1)

        # Calculate and print image dimension statistics
        if dimensions:
            widths = [w for w, h in dimensions]
            heights = [h for w, h in dimensions]
            avg_width = np.mean(widths)
            avg_height = np.mean(heights)
            min_width = np.min(widths)
            max_width = np.max(widths)
            min_height = np.min(heights)
            max_height = np.max(heights)

            print(f"\nImage Dimension Statistics:")
            print(f"  Average size: {avg_width:.1f} × {avg_height:.1f} (W × H)")
            print(f"  Width range: {min_width} - {max_width}")
            print(f"  Height range: {min_height} - {max_height}")

        # Composite patches with samples (cycling through patches)
        print(f"\nCreating composite images with controls...")
        print(f"Pairing {len(patches)} patches with {len(val_images)} validation samples (cycling)...")

        pbar = tqdm(total=len(val_images), desc="Compositing")
        saved_count = 0
        for img_idx, val_image in enumerate(val_images):
            # Cycle through patches
            patch_idx = img_idx % len(patches)
            patch = patches[patch_idx]

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

                saved_count += 1
            except Exception as e:
                print(f"  Error processing image {img_idx}: {e}", file=sys.stderr)

            pbar.update(1)

        pbar.close()
        print(f"Saved to {output_dir}")

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
