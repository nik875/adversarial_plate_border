#!/usr/bin/env python3
"""
Compute patch similarity using the same method as in optimize_basis.py/neural_basis.py
"""
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import argparse
import matplotlib.pyplot as plt


def normalize_patch(patch_tensor):
    """L2 normalize a patch tensor (3, H, W) -> (1, d)"""
    # Flatten
    flat = patch_tensor.reshape(1, -1)  # [1, channels*H*W]
    # L2 normalize
    normalized = F.normalize(flat, p=2, dim=1)  # [1, d]
    return normalized


def compute_similarity(patch1_tensor, patch2_tensor):
    """
    Compute cosine similarity between two patches

    Args:
        patch1_tensor: torch tensor [3, H, W]
        patch2_tensor: torch tensor [3, H, W]

    Returns:
        similarity: scalar cosine similarity [-1, 1]
        diff: absolute difference between patches [H, W]
    """
    # Normalize for cosine similarity
    p1_norm = normalize_patch(patch1_tensor)  # [1, d]
    p2_norm = normalize_patch(patch2_tensor)  # [1, d]

    # Compute cosine similarity (dot product of normalized vectors)
    similarity = torch.mm(p1_norm, p2_norm.t()).item()  # scalar

    # Compute absolute difference (mean across channels for grayscale)
    diff = torch.abs(patch1_tensor - patch2_tensor).mean(dim=0)  # [H, W]

    return similarity, diff


def load_patch(image_path):
    """Load an image and convert to torch tensor [3, H, W]"""
    img = Image.open(image_path).convert('RGB')
    tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
    return tensor


def main():
    parser = argparse.ArgumentParser(description='Compute patch similarity')
    parser.add_argument('patch1', type=str, help='Path to first patch image')
    parser.add_argument('patch2', type=str, help='Path to second patch image')
    parser.add_argument('--output', type=str, default='patch_similarity.png',
                        help='Output figure path')
    args = parser.parse_args()
    
    # Load patches
    print(f"Loading patches from {args.patch1} and {args.patch2}")
    p1 = load_patch(args.patch1)
    p2 = load_patch(args.patch2)
    
    print(f"Patch 1 shape: {p1.shape}")
    print(f"Patch 2 shape: {p2.shape}")
    
    # Compute similarity
    similarity, diff = compute_similarity(p1, p2)

    print(f"\n{'='*60}")
    print(f"Cosine Similarity: {similarity:.4f}")
    print(f"Angle: {np.degrees(np.arccos(np.clip(similarity, -1, 1))):.2f}°")
    print(f"Mean Absolute Difference: {diff.mean():.4f}")
    print(f"{'='*60}\n")

    # Convert tensors to numpy for visualization
    p1_np = p1.permute(1, 2, 0).numpy()
    p2_np = p2.permute(1, 2, 0).numpy()
    diff_np = diff.numpy()

    # Create simple figure with 2 rows: patches on top, diff on bottom
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))

    # Top row: the two patches
    axes[0, 0].imshow(p1_np)
    axes[0, 0].set_title(f'Patch 1\n{p1.shape[1]}x{p1.shape[2]}', fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(p2_np)
    axes[0, 1].set_title(f'Patch 2\n{p2.shape[1]}x{p2.shape[2]}', fontsize=12, fontweight='bold')
    axes[0, 1].axis('off')

    # Bottom row: difference
    diff_max = diff_np.max()
    axes[1, 0].imshow(diff_np, cmap='hot', vmin=0, vmax=diff_max)
    axes[1, 0].set_title(f'Absolute Difference\nMean: {diff.mean():.4f}', fontsize=12, fontweight='bold')
    axes[1, 0].axis('off')

    # Bottom right: hide
    axes[1, 1].axis('off')

    # Add similarity info
    fig.suptitle(f'Patch Comparison\nCosine Similarity: {similarity:.4f} | Angle: {np.degrees(np.arccos(np.clip(similarity, -1, 1))):.2f}°',
                 fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(args.output, dpi=150, bbox_inches='tight')
    print(f"Saved figure to {args.output}")
    plt.show()


if __name__ == '__main__':
    main()
