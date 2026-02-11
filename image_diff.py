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
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.ndimage import gaussian_filter


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


def compute_average_diff_across_images(images_list):
    """Compute average difference map across multiple image pairs."""
    if not images_list or len(images_list) < 2:
        return None

    # Ensure all images have same shape
    target_shape = images_list[0].shape
    resized = []
    for img in images_list:
        if img.shape != target_shape:
            h, w = target_shape[:2]
            img = cv2.resize(img, (w, h))
        resized.append(img.astype(np.float32))

    # Compute pairwise differences and average
    all_diffs = []
    for i in range(len(resized)):
        for j in range(i + 1, len(resized)):
            diff = cv2.absdiff(resized[i], resized[j])
            all_diffs.append(diff)

    return np.mean(all_diffs, axis=0) if all_diffs else None


def compute_average_diff_zones(diff_map, gaussian_sigma=2, mask=None):
    """Compute per-pixel difference heatmap."""
    zone_heatmap = np.mean(diff_map, axis=2)
    if mask is not None:
        zone_heatmap = zone_heatmap * mask
    return gaussian_filter(zone_heatmap, sigma=gaussian_sigma)


def display_average_diff(images, zone_heatmap, directory, outfile=None):
    """Display average difference zones with sample images and greyscale overlay."""
    h, w = zone_heatmap.shape
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

    # Bottom left: Heatmap
    im = axes[1, 0].imshow(zone_heatmap, cmap='hot')
    axes[1, 0].set_title('Average Difference Zones')
    axes[1, 0].axis('off')
    cbar = plt.colorbar(im, ax=axes[1, 0], fraction=0.046, pad=0.04)
    cbar.ax.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:.1f}'))

    # Bottom right: Greyscale with red highlighting for all diffs
    grey = cv2.cvtColor(sample1, cv2.COLOR_RGB2GRAY)
    grey_rgb = np.stack([grey, grey, grey], axis=2).astype(np.float32)
    grey_rgb = grey_rgb * (center_mask[:, :, np.newaxis] * 0.5 + 0.5)

    zone_norm = np.clip(zone_heatmap / (np.max(zone_heatmap) + 1e-8), 0, 1)
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
        print("Computing average within-layer differences...")

        avg_diff = compute_average_diff_across_images(images)
        if avg_diff is None:
            print("Error: Could not compute differences", file=sys.stderr)
            sys.exit(1)

        h, w = avg_diff.shape[:2]
        center_mask = create_center_mask(h, w)
        zone_heatmap = compute_average_diff_zones(avg_diff, mask=center_mask)

        display_average_diff(images, zone_heatmap, args.directory, args.outfile)

    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
