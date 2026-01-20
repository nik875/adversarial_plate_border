#!/usr/bin/env python3
"""
Compute patch similarity across a directory of patch images.
Normalizes brightness, displays all patches, and shows aggregated difference heatmap.
"""
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import argparse
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Tuple
import math


def normalize_brightness(patches: List[torch.Tensor]) -> List[torch.Tensor]:
    """
    Normalize all patches in a batch to have the same average brightness.

    Args:
        patches: List of [3, H, W] patch tensors

    Returns:
        List of brightness-normalized patches
    """
    if len(patches) == 0:
        return patches

    # Stack patches for batch processing
    patches_stacked = torch.stack(patches, dim=0)  # [batch_size, 3, H, W]

    # Compute per-patch mean brightness (average across all pixels and channels)
    per_patch_brightness = patches_stacked.mean(dim=(1, 2, 3), keepdim=True)  # [batch_size, 1, 1, 1]

    # Compute global mean brightness across all patches
    global_mean_brightness = per_patch_brightness.mean()  # scalar

    # Normalize each patch: scale to match global mean brightness
    normalized_patches = patches_stacked * (global_mean_brightness / (per_patch_brightness + 1e-8))

    # Clamp to valid range [0, 1]
    normalized_patches = torch.clamp(normalized_patches, 0.0, 1.0)

    # Return as list to match input format
    return [normalized_patches[i] for i in range(len(patches))]


def normalize_patch(patch_tensor):
    """L2 normalize a patch tensor (3, H, W) -> (1, d)"""
    # Flatten
    flat = patch_tensor.reshape(1, -1)  # [1, channels*H*W]
    # L2 normalize
    normalized = F.normalize(flat, p=2, dim=1)  # [1, d]
    return normalized


def compute_pairwise_similarity(patches: List[torch.Tensor]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute pairwise cosine similarities and differences between all patches.

    Args:
        patches: List of [3, H, W] patch tensors

    Returns:
        similarity_matrix: [n_patches, n_patches] cosine similarities
        diff_heatmap: [H, W] aggregated absolute differences across all pairs
    """
    n = len(patches)
    similarity_matrix = np.zeros((n, n))

    # Aggregate differences across all pairs
    diff_heatmap = None
    pair_count = 0

    for i in range(n):
        for j in range(i + 1, n):
            # Normalize for cosine similarity
            p1_norm = normalize_patch(patches[i])  # [1, d]
            p2_norm = normalize_patch(patches[j])  # [1, d]

            # Compute cosine similarity (dot product of normalized vectors)
            similarity = torch.mm(p1_norm, p2_norm.t()).item()  # scalar
            similarity_matrix[i, j] = similarity
            similarity_matrix[j, i] = similarity

            # Compute absolute difference (mean across channels for grayscale)
            diff = torch.abs(patches[i] - patches[j]).mean(dim=0)  # [H, W]

            if diff_heatmap is None:
                diff_heatmap = diff
            else:
                diff_heatmap = diff_heatmap + diff

            pair_count += 1

    # Diagonal = 1 (similarity with self)
    np.fill_diagonal(similarity_matrix, 1.0)

    # Average aggregated heatmap
    if pair_count > 0:
        diff_heatmap = diff_heatmap / pair_count

    return similarity_matrix, diff_heatmap


def load_patch(image_path):
    """Load an image and convert to torch tensor [3, H, W]"""
    img = Image.open(image_path).convert('RGB')
    tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
    return tensor


def main():
    parser = argparse.ArgumentParser(description='Compute patch similarity from directory')
    parser.add_argument('directory', type=str, help='Directory containing patch PNG images')
    parser.add_argument('--output', type=str, default='patch_similarity.png',
                        help='Output figure path')
    parser.add_argument('--max-patches', type=int, default=None,
                        help='Maximum number of patches to display (default: all)')
    args = parser.parse_args()

    # Find all PNG files in directory
    patch_dir = Path(args.directory)
    if not patch_dir.exists():
        print(f"Error: Directory {args.directory} not found")
        return

    png_files = sorted(patch_dir.glob('*.png'))
    if not png_files:
        print(f"Error: No PNG files found in {args.directory}")
        return

    # Limit number of patches if specified
    if args.max_patches is not None:
        png_files = png_files[:args.max_patches]

    print(f"Loading {len(png_files)} patches from {args.directory}")

    # Load patches
    patches = []
    patch_names = []
    for png_file in png_files:
        try:
            patch = load_patch(str(png_file))
            patches.append(patch)
            patch_names.append(png_file.stem)
            print(f"  Loaded {png_file.name}: {patch.shape}")
        except Exception as e:
            print(f"  Warning: Failed to load {png_file.name}: {e}")

    if len(patches) == 0:
        print("Error: No patches loaded")
        return

    # Normalize brightness
    print(f"\nNormalizing brightness across {len(patches)} patches...")
    patches = normalize_brightness(patches)

    # Compute pairwise similarity
    print("Computing pairwise similarities...")
    similarity_matrix, diff_heatmap = compute_pairwise_similarity(patches)

    # Print statistics
    n_patches = len(patches)
    print(f"\n{'='*60}")
    print(f"Patch Similarity Statistics ({n_patches} patches)")
    print(f"{'='*60}")
    print(f"Mean Cosine Similarity (off-diagonal): {similarity_matrix[~np.eye(n_patches, dtype=bool)].mean():.4f}")
    print(f"Min Cosine Similarity: {similarity_matrix[~np.eye(n_patches, dtype=bool)].min():.4f}")
    print(f"Max Cosine Similarity: {similarity_matrix[~np.eye(n_patches, dtype=bool)].max():.4f}")
    print(f"Mean Absolute Difference (aggregated): {diff_heatmap.mean():.4f}")
    print(f"{'='*60}\n")

    # Convert patches to numpy for visualization
    patches_np = [p.permute(1, 2, 0).numpy() for p in patches]
    diff_np = diff_heatmap.numpy()

    # Determine grid layout for patches
    n_patches = len(patches)
    n_cols = min(n_patches, 5)  # Max 5 columns
    n_rows_patches = math.ceil(n_patches / n_cols)

    # Create figure with patch grid + similarity heatmap + aggregated diff
    fig = plt.figure(figsize=(4*n_cols, 4*(n_rows_patches + 2)))

    # Top section: all patches in grid
    for i, (patch_np, name) in enumerate(zip(patches_np, patch_names)):
        ax = plt.subplot(n_rows_patches + 2, n_cols, i + 1)
        ax.imshow(patch_np)
        ax.set_title(f'{name}\n{patch_np.shape[1]}x{patch_np.shape[0]}', fontsize=10)
        ax.axis('off')

    # Hide extra subplots if grid is not full
    total_patch_plots = n_rows_patches * n_cols
    for i in range(n_patches, total_patch_plots):
        ax = plt.subplot(n_rows_patches + 2, n_cols, i + 1)
        ax.axis('off')

    # Middle section: similarity matrix heatmap
    ax_sim = plt.subplot(n_rows_patches + 2, n_cols, total_patch_plots + 1)
    im_sim = ax_sim.imshow(similarity_matrix, cmap='coolwarm', vmin=-1, vmax=1)
    ax_sim.set_title('Cosine Similarity Matrix', fontsize=12, fontweight='bold')
    ax_sim.set_xlabel('Patch Index')
    ax_sim.set_ylabel('Patch Index')
    ax_sim.set_xticks(range(n_patches))
    ax_sim.set_yticks(range(n_patches))

    # Add colorbar for similarity
    cbar_sim = plt.colorbar(im_sim, ax=ax_sim)
    cbar_sim.set_label('Cosine Similarity')

    # Bottom section: aggregated difference heatmap
    ax_diff = plt.subplot(n_rows_patches + 2, n_cols, total_patch_plots + 2)
    diff_max = diff_np.max()
    im_diff = ax_diff.imshow(diff_np, cmap='hot', vmin=0, vmax=diff_max)
    ax_diff.set_title(f'Aggregated Difference\nMean: {diff_heatmap.mean():.4f}',
                     fontsize=12, fontweight='bold')
    ax_diff.axis('off')

    # Add colorbar for difference
    cbar_diff = plt.colorbar(im_diff, ax=ax_diff)
    cbar_diff.set_label('Mean Absolute Difference')

    # Hide remaining subplots in bottom sections
    for i in range(2, n_cols):
        ax = plt.subplot(n_rows_patches + 2, n_cols, total_patch_plots + i + 1)
        ax.axis('off')

    # Add overall title
    fig.suptitle(f'Patch Similarity Analysis ({len(patches)} patches)\n' +
                f'Mean Cosine Similarity: {similarity_matrix[~np.eye(n_patches, dtype=bool)].mean():.4f}',
                fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(args.output, dpi=150, bbox_inches='tight')
    print(f"Saved figure to {args.output}")
    plt.show()


if __name__ == '__main__':
    main()
