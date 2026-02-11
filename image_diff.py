#!/usr/bin/env python3
"""
Compare images and display their differences with zone analysis.
Usage: python image_diff.py image1.jpg image2.jpg
"""

import argparse
import sys
from pathlib import Path
from glob import glob

import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter


def load_image(path):
    """Load an image and convert to RGB."""
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Could not load image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def compute_diff(img1, img2):
    """Compute difference between two images, resizing if necessary."""
    if img1.shape != img2.shape:
        height = min(img1.shape[0], img2.shape[0])
        width = min(img1.shape[1], img2.shape[1])
        img1 = img1[:height, :width]
        img2 = img2[:height, :width]

    # Compute pixel-wise difference
    diff = cv2.absdiff(img1.astype(np.float32), img2.astype(np.float32))

    # Compute MSE and SSIM for statistics
    mse = np.mean(diff ** 2)

    return diff, mse


def create_center_mask(height, width):
    """Create a mask that zeros out the middle 60% of the image."""
    mask = np.ones((height, width), dtype=np.float32)

    # Calculate dimensions of the center 60% region
    center_h = int(height * 0.6)
    center_w = int(width * 0.6)

    # Calculate start coordinates for centered rectangle
    start_h = (height - center_h) // 2
    start_w = (width - center_w) // 2

    # Zero out the center region
    mask[start_h:start_h + center_h, start_w:start_w + center_w] = 0

    return mask


def compute_average_diff_zones(diff_map, gaussian_sigma=2, mask=None):
    """
    Compute per-pixel difference heatmap at original image resolution.

    Args:
        diff_map: HxWx3 difference map
        gaussian_sigma: sigma for smoothing the heatmap
        mask: Optional HxW mask to exclude regions from computation

    Returns:
        zone_heatmap: HxW heatmap of differences at full image resolution
    """
    # Compute mean magnitude of difference per pixel
    zone_heatmap = np.mean(diff_map, axis=2)

    # Apply mask if provided
    if mask is not None:
        zone_heatmap = zone_heatmap * mask

    # Smooth the heatmap for better visualization
    zone_heatmap = gaussian_filter(zone_heatmap, sigma=gaussian_sigma)

    return zone_heatmap


def display_comparison(img1, img2, diff, mse, path1, path2, outfile=None):
    """Display images side by side with zone-based difference visualization."""
    # Create center mask to exclude middle 60%
    h, w = diff.shape[:2]
    center_mask = create_center_mask(h, w)

    # Compute zone heatmap with mask
    zone_heatmap = compute_average_diff_zones(diff, mask=center_mask)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Image Comparison: {Path(path1).name} vs {Path(path2).name}\nMSE: {mse:.2f}',
                 fontsize=10, fontweight='normal')

    # Original images
    axes[0, 0].imshow(img1)
    axes[0, 0].set_title('Image 1')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(img2)
    axes[0, 1].set_title('Image 2')
    axes[0, 1].axis('off')

    # Zone-based difference heatmap
    im = axes[1, 0].imshow(zone_heatmap, cmap='hot')
    axes[1, 0].set_title('Average Difference Zones')
    axes[1, 0].axis('off')
    plt.colorbar(im, ax=axes[1, 0], fraction=0.046, pad=0.04)

    # Greyscale with red highlighting for top 25% threshold
    grey = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    grey_rgb = np.stack([grey, grey, grey], axis=2).astype(np.float32)

    # Calculate 75th percentile threshold (top 25%)
    threshold = np.percentile(zone_heatmap, 75)

    # Normalize heatmap for red intensity
    zone_norm = np.clip(zone_heatmap / (np.max(zone_heatmap) + 1e-8), 0, 1)

    # Apply mask darkening to greyscale
    grey_rgb = grey_rgb * (center_mask[:, :, np.newaxis] * 0.5 + 0.5)

    # Set red channel for high difference areas
    high_diff_mask = zone_heatmap > threshold
    grey_rgb[high_diff_mask, 0] = zone_norm[high_diff_mask] * 255

    overlay = grey_rgb.astype(np.uint8)
    axes[1, 1].imshow(overlay)
    axes[1, 1].set_title('Difference Zones Highlighted')
    axes[1, 1].axis('off')

    plt.tight_layout()
    if outfile:
        plt.savefig(outfile, dpi=100, bbox_inches='tight')
        print(f"Figure saved to {outfile}")
    else:
        plt.show()


def load_all_images_from_dir(directory):
    """Load all PNG/JPG images from a directory."""
    images = []
    paths = []
    for ext in ['*.png', '*.jpg', '*.jpeg']:
        for filepath in sorted(glob(str(Path(directory) / ext))):
            try:
                img = load_image(filepath)
                images.append(img)
                paths.append(filepath)
            except Exception as e:
                print(f"Warning: Could not load {filepath}: {e}", file=sys.stderr)
    return images, paths


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

    if not all_diffs:
        return None

    avg_diff = np.mean(all_diffs, axis=0)
    return avg_diff


def display_comparison_with_zones(img1, img2, diff, mse, path1, path2, outfile=None):
    """Display comparison with zone-based visualization."""
    # Create center mask to exclude middle 60%
    h, w = diff.shape[:2]
    center_mask = create_center_mask(h, w)

    zone_heatmap = compute_average_diff_zones(diff, mask=center_mask)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Image Comparison: {Path(path1).name} vs {Path(path2).name}\nMSE: {mse:.2f}',
                 fontsize=10, fontweight='normal')

    # Original images
    axes[0, 0].imshow(img1)
    axes[0, 0].set_title('Image 1')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(img2)
    axes[0, 1].set_title('Image 2')
    axes[0, 1].axis('off')

    # Zone-based difference heatmap
    im = axes[1, 0].imshow(zone_heatmap, cmap='hot')
    axes[1, 0].set_title('Average Difference Zones')
    axes[1, 0].axis('off')
    plt.colorbar(im, ax=axes[1, 0], fraction=0.046, pad=0.04)

    # Greyscale with red highlighting for top 25% threshold
    grey = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    grey_rgb = np.stack([grey, grey, grey], axis=2).astype(np.float32)

    # Calculate 75th percentile threshold (top 25%)
    threshold = np.percentile(zone_heatmap, 75)

    # Normalize heatmap for red intensity
    zone_norm = np.clip(zone_heatmap / (np.max(zone_heatmap) + 1e-8), 0, 1)

    # Apply mask darkening to greyscale
    grey_rgb = grey_rgb * (center_mask[:, :, np.newaxis] * 0.5 + 0.5)

    # Set red channel for high difference areas
    high_diff_mask = zone_heatmap > threshold
    grey_rgb[high_diff_mask, 0] = zone_norm[high_diff_mask] * 255

    overlay = grey_rgb.astype(np.uint8)
    axes[1, 1].imshow(overlay)
    axes[1, 1].set_title('Difference Zones Highlighted')
    axes[1, 1].axis('off')

    plt.tight_layout()
    if outfile:
        plt.savefig(outfile, dpi=100, bbox_inches='tight')
        print(f"Figure saved to {outfile}")
    else:
        plt.show()


def display_diff_of_diffs(within_layer_zone, across_layer_zone, outfile=None):
    """Display diff of differences between within-layer and across-layer average zones."""
    # Compute the difference between heatmaps
    diff_of_diffs = across_layer_zone - within_layer_zone

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle('Diff of Diffs: Within-Layer vs Across-Layer Average Difference Zones',
                 fontsize=10, fontweight='normal')

    # Within-layer average zones
    im1 = axes[0].imshow(within_layer_zone, cmap='hot')
    axes[0].set_title('Within-Layer Average Zones')
    axes[0].axis('off')
    plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)

    # Across-layer average zones
    im2 = axes[1].imshow(across_layer_zone, cmap='hot')
    axes[1].set_title('Across-Layer Average Zones')
    axes[1].axis('off')
    plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)

    # Difference (using diverging colormap for clarity)
    im3 = axes[2].imshow(diff_of_diffs, cmap='RdBu_r')
    axes[2].set_title('Difference (Red=More Across-Layer Diff)')
    axes[2].axis('off')
    plt.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04)

    plt.tight_layout()
    if outfile:
        plt.savefig(outfile, dpi=100, bbox_inches='tight')
        print(f"Figure saved to {outfile}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description='Compare images and display their differences with zone analysis.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python image_diff.py image1.jpg image2.jpg
  python image_diff.py /path/to/original.png /path/to/modified.png -o comparison.png
  python image_diff.py /path/to/layer0 /path/to/layer10 -d -o diff_of_diffs.png
        '''
    )
    parser.add_argument('image1', help='Path to first image or directory')
    parser.add_argument('image2', help='Path to second image or directory')
    parser.add_argument('-o', '--outfile', default=None,
                        help='Path to save the output figure')
    parser.add_argument('-d', '--diff-of-diffs', action='store_true',
                        help='Compute diff of diffs for directory comparisons')

    args = parser.parse_args()

    try:
        # Check if inputs are directories
        path1_is_dir = Path(args.image1).is_dir()
        path2_is_dir = Path(args.image2).is_dir()

        if path1_is_dir and path2_is_dir and args.diff_of_diffs:
            # Load all images from directories
            print(f"Loading images from {args.image1}...")
            images1, paths1 = load_all_images_from_dir(args.image1)
            print(f"Loaded {len(images1)} images")

            print(f"Loading images from {args.image2}...")
            images2, paths2 = load_all_images_from_dir(args.image2)
            print(f"Loaded {len(images2)} images")

            if not images1 or not images2:
                print("Error: No images found in one or both directories", file=sys.stderr)
                sys.exit(1)

            # Compute within-layer average diffs
            print("Computing within-layer average differences...")
            within_layer_diff = compute_average_diff_across_images(images1)
            if within_layer_diff is None:
                within_layer_diff = compute_average_diff_across_images(images2)

            # Compute across-layer average diffs
            print("Computing across-layer average differences...")
            across_layer_diff = compute_average_diff_across_images(images1 + images2)

            # Compute zone heatmaps with center mask
            h, w = within_layer_diff.shape[:2] if within_layer_diff is not None else across_layer_diff.shape[:2]
            center_mask = create_center_mask(h, w)
            within_zone = compute_average_diff_zones(within_layer_diff, mask=center_mask) if within_layer_diff is not None else None
            across_zone = compute_average_diff_zones(across_layer_diff, mask=center_mask) if across_layer_diff is not None else None

            if within_zone is not None and across_zone is not None:
                display_diff_of_diffs(within_zone, across_zone, args.outfile)
            else:
                print("Error: Could not compute zone differences", file=sys.stderr)
                sys.exit(1)

        else:
            # Standard single image comparison
            print(f"Loading images...")
            img1 = load_image(args.image1)
            img2 = load_image(args.image2)

            print(f"Image 1 shape: {img1.shape}")
            print(f"Image 2 shape: {img2.shape}")

            print(f"Computing differences...")
            diff, mse = compute_diff(img1, img2)

            print(f"Mean Squared Error: {mse:.2f}")
            print(f"Max pixel difference: {np.max(diff):.2f}")
            print(f"Mean pixel difference: {np.mean(diff):.2f}")

            display_comparison_with_zones(img1, img2, diff, mse, args.image1, args.image2, args.outfile)

    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
