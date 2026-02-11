#!/usr/bin/env python3
"""
Compute and display average within-layer image differences.
Usage: python image_diff.py /path/to/images -o output.png
"""

import argparse
import sys
from pathlib import Path
from glob import glob

import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.ndimage import gaussian_filter
from kornia.metrics import structural_similarity as kornia_ssim


def load_image(path):
    """Load an image and convert to RGB."""
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Could not load image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def load_all_images_from_dir(directory):
    """Load all PNG/JPG images from a directory."""
    images = []
    for ext in ['*.png', '*.jpg', '*.jpeg']:
        for filepath in sorted(glob(str(Path(directory) / ext))):
            try:
                img = load_image(filepath)
                images.append(img)
            except Exception as e:
                print(f"Warning: Could not load {filepath}: {e}", file=sys.stderr)
    return images


def create_center_mask(height, width):
    """Create a mask that zeros out the middle 60% of the image."""
    mask = np.ones((height, width), dtype=np.float32)
    center_h, center_w = int(height * 0.6), int(width * 0.6)
    start_h = (height - center_h) // 2
    start_w = (width - center_w) // 2
    mask[start_h:start_h + center_h, start_w:start_w + center_w] = 0
    return mask


def compute_ssim_kornia(img1, img2, window_size=11):
    """Compute SSIM dissimilarity using Kornia GPU-accelerated implementation.

    Args:
        img1, img2: Images as numpy arrays (H x W x C)
        window_size: Size of the SSIM computation window

    Returns:
        ssim_map: Per-pixel SSIM dissimilarity (1 - SSIM)
    """
    # Convert to grayscale and normalize to [0, 1]
    if len(img1.shape) == 3:
        img1_gray = cv2.cvtColor(img1.astype(np.uint8), cv2.COLOR_RGB2GRAY) / 255.0
        img2_gray = cv2.cvtColor(img2.astype(np.uint8), cv2.COLOR_RGB2GRAY) / 255.0
    else:
        img1_gray = img1.astype(np.float32) / 255.0
        img2_gray = img2.astype(np.float32) / 255.0

    # Convert to torch tensors and add batch/channel dimensions (1, 1, H, W)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tensor1 = torch.from_numpy(img1_gray[np.newaxis, np.newaxis, :, :]).float().to(device)
    tensor2 = torch.from_numpy(img2_gray[np.newaxis, np.newaxis, :, :]).float().to(device)

    # Compute SSIM map using Kornia
    ssim_map = kornia_ssim(tensor1, tensor2, window_size=window_size)

    # Convert back to numpy and return dissimilarity (1 - SSIM)
    ssim_map = ssim_map.squeeze().cpu().numpy()
    return 1 - ssim_map


def compute_average_diff_zones(images_list, gaussian_sigma=2, mask=None):
    """Compute average SSIM dissimilarity heatmap across images with log scaling.

    Args:
        images_list: List of images
        gaussian_sigma: Sigma for Gaussian smoothing
        mask: Optional mask to apply

    Returns:
        tuple: (log_scaled_heatmap, original_heatmap)
    """
    if not images_list or len(images_list) < 2:
        return None, None

    ssim_maps = []

    # Compute SSIM for all pairs
    print(f"  Computing SSIM for {len(images_list) * (len(images_list) - 1) // 2} image pairs...")
    for i in range(len(images_list)):
        for j in range(i + 1, len(images_list)):
            ssim_map = compute_ssim_kornia(images_list[i], images_list[j])
            ssim_maps.append(ssim_map)

    zone_heatmap = np.mean(ssim_maps, axis=0)
    if mask is not None:
        zone_heatmap = zone_heatmap * mask
    zone_heatmap = gaussian_filter(zone_heatmap, sigma=gaussian_sigma)
    original_heatmap = zone_heatmap.copy()
    # Log scale to prevent extreme values from dominating
    zone_heatmap = np.log1p(zone_heatmap)
    return zone_heatmap, original_heatmap


def display_average_diff(images, zone_data, directory, outfile=None):
    """Display average difference zones with sample images and greyscale overlay.

    Args:
        zone_data: tuple of (log_scaled_heatmap, original_heatmap)
    """
    zone_heatmap_log, zone_heatmap_orig = zone_data
    h, w = zone_heatmap_log.shape
    center_mask = create_center_mask(h, w)

    # Select two random samples
    idx1, idx2 = np.random.choice(len(images), 2, replace=False)
    sample1, sample2 = images[idx1], images[idx2]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Average Difference Zones: {Path(directory).name}',
                 fontsize=10, fontweight='normal')

    # Top row: two random samples with mask applied
    sample1_masked = sample1.astype(np.float32) * (center_mask[:, :, np.newaxis] * 0.5 + 0.5)
    sample2_masked = sample2.astype(np.float32) * (center_mask[:, :, np.newaxis] * 0.5 + 0.5)

    axes[0, 0].imshow(sample1_masked.astype(np.uint8))
    axes[0, 0].set_title('Sample 1')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(sample2_masked.astype(np.uint8))
    axes[0, 1].set_title('Sample 2')
    axes[0, 1].axis('off')

    # Bottom left: Heatmap (log-scaled display, original scale colorbar)
    im = axes[1, 0].imshow(zone_heatmap_log, cmap='hot')
    axes[1, 0].set_title('Average Difference Zones')
    axes[1, 0].axis('off')
    cbar = plt.colorbar(im, ax=axes[1, 0], fraction=0.046, pad=0.04)

    # Create custom colorbar with original values
    cbar_min, cbar_max = zone_heatmap_orig.min(), zone_heatmap_orig.max()
    log_min, log_max = zone_heatmap_log.min(), zone_heatmap_log.max()
    log_range = log_max - log_min if log_max > log_min else 1

    # Map colorbar positions to original values
    def format_cbar(x, pos):
        # x is in range [log_min, log_max], map back to original range
        if log_range > 0:
            ratio = (x - log_min) / log_range
            orig_val = cbar_min + ratio * (cbar_max - cbar_min)
        else:
            orig_val = cbar_min
        return f'{orig_val:.1f}'

    cbar.ax.yaxis.set_major_formatter(ticker.FuncFormatter(format_cbar))

    # Bottom right: Greyscale with red highlighting for all diffs
    grey = cv2.cvtColor(sample1, cv2.COLOR_RGB2GRAY)
    grey_rgb = np.stack([grey, grey, grey], axis=2).astype(np.float32)
    grey_rgb = grey_rgb * (center_mask[:, :, np.newaxis] * 0.5 + 0.5)

    zone_norm = np.clip(zone_heatmap_orig / (np.max(zone_heatmap_orig) + 1e-8), 0, 1)
    grey_rgb[:, :, 0] = np.maximum(grey_rgb[:, :, 0], zone_norm * 255)

    axes[1, 1].imshow(grey_rgb.astype(np.uint8))
    axes[1, 1].set_title('Highlighted (All Diffs)')
    axes[1, 1].axis('off')

    plt.tight_layout()
    if outfile:
        plt.savefig(outfile, dpi=100, bbox_inches='tight')
        print(f"Figure saved to {outfile}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description='Compute and display average within-layer image differences.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='Example:\n  python image_diff.py /path/to/images -o output.png'
    )
    parser.add_argument('directory', help='Path to directory of images')
    parser.add_argument('-o', '--outfile', default=None, help='Path to save output figure')

    args = parser.parse_args()

    try:
        path = Path(args.directory)
        if not path.is_dir():
            print(f"Error: {args.directory} is not a directory", file=sys.stderr)
            sys.exit(1)

        print(f"Loading images from {args.directory}...")
        images = load_all_images_from_dir(args.directory)

        if not images:
            print("Error: No images found", file=sys.stderr)
            sys.exit(1)

        print(f"Loaded {len(images)} images")
        print("Computing average SSIM-based differences...")

        h, w = images[0].shape[:2]
        center_mask = create_center_mask(h, w)
        zone_data = compute_average_diff_zones(images, mask=center_mask)

        if zone_data[0] is None:
            print("Error: Could not compute differences", file=sys.stderr)
            sys.exit(1)

        display_average_diff(images, zone_data, args.directory, args.outfile)

    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
