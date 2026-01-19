#!/usr/bin/env python3
"""
Compute patch similarity using the same method as in optimize_basis.py/neural_basis.py
"""
import torch
import torch.nn.functional as F
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
import numpy as np
import argparse
import kornia


def normalize_patch(patch_tensor):
    """L2 normalize a patch tensor (3, H, W) -> (1, d)"""
    # Flatten
    flat = patch_tensor.reshape(1, -1)  # [1, channels*H*W]
    # L2 normalize
    normalized = F.normalize(flat, p=2, dim=1)  # [1, d]
    return normalized


def compute_similarity(patch1_tensor, patch2_tensor):
    """
    Compute cosine similarity between two patches following the same method as the code

    Args:
        patch1_tensor: torch tensor [3, H, W] or [3, 512, 256]
        patch2_tensor: torch tensor [3, H, W] or [3, 512, 256]

    Returns:
        similarity: scalar cosine similarity [-1, 1]
    """
    # Store originals for diff computation
    p1_original = patch1_tensor.clone()
    p2_original = patch2_tensor.clone()

    # Add batch dimension
    p1 = patch1_tensor.unsqueeze(0)  # [1, 3, H, W]
    p2 = patch2_tensor.unsqueeze(0)  # [1, 3, H, W]

    # Apply small random pixel jitter (±2 pixels) FIRST
    jitter_x1 = np.random.randint(-2, 3)
    jitter_y1 = np.random.randint(-2, 3)
    jitter_x2 = np.random.randint(-2, 3)
    jitter_y2 = np.random.randint(-2, 3)

    # Use affine transform for subpixel jitter
    p1_jitter = kornia.geometry.transform.translate(p1, torch.tensor([[jitter_x1, jitter_y1]], dtype=torch.float32))
    p2_jitter = kornia.geometry.transform.translate(p2, torch.tensor([[jitter_x2, jitter_y2]], dtype=torch.float32))

    # THEN apply Gaussian blur (sigma=4px)
    kernel_size = int(4 * 4) + 1  # ~4 sigma
    if kernel_size % 2 == 0:
        kernel_size += 1
    p1_blur = kornia.filters.gaussian_blur2d(p1_jitter, (kernel_size, kernel_size), (4.0, 4.0))
    p2_blur = kornia.filters.gaussian_blur2d(p2_jitter, (kernel_size, kernel_size), (4.0, 4.0))

    # Downsample to 32x64
    p1_down = F.interpolate(
        p1_blur,
        size=(32, 64),
        mode='bilinear',
        align_corners=True
    ).squeeze(0)  # Remove batch dim: [3, 32, 64]

    p2_down = F.interpolate(
        p2_blur,
        size=(32, 64),
        mode='bilinear',
        align_corners=True
    ).squeeze(0)  # [3, 32, 64]

    # Compute differences between patches
    # Original resolution: p1_original - p2_original
    diff_original = torch.abs(p1_original - p2_original).mean(dim=0)  # [512, 256], grayscale

    # Downsampled: upsample back to original size for visualization
    p1_down_upsampled = F.interpolate(
        p1_down.unsqueeze(0),
        size=(p1_original.shape[1], p1_original.shape[2]),
        mode='bilinear',
        align_corners=True
    ).squeeze(0)  # [3, 512, 256]

    p2_down_upsampled = F.interpolate(
        p2_down.unsqueeze(0),
        size=(p2_original.shape[1], p2_original.shape[2]),
        mode='bilinear',
        align_corners=True
    ).squeeze(0)  # [3, 512, 256]

    # Downsampled resolution difference
    diff_downsampled = torch.abs(p1_down_upsampled - p2_down_upsampled).mean(dim=0)  # [512, 256], grayscale

    # Normalize
    p1_norm = normalize_patch(p1_down)  # [1, 6144]
    p2_norm = normalize_patch(p2_down)  # [1, 6144]

    # Compute cosine similarity (dot product of normalized vectors)
    similarity = torch.mm(p1_norm, p2_norm.t()).item()  # scalar

    return similarity, p1_down, p2_down, p1_jitter.squeeze(0), p2_jitter.squeeze(0), p1_blur.squeeze(0), p2_blur.squeeze(0), diff_original, diff_downsampled


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
    similarity, p1_down, p2_down, p1_jitter, p2_jitter, p1_blur, p2_blur, diff_original, diff_downsampled = compute_similarity(p1, p2)

    print(f"\n{'='*60}")
    print(f"Cosine Similarity: {similarity:.4f}")
    print(f"Angle: {np.degrees(np.arccos(np.clip(similarity, -1, 1))):.2f}°")
    print(f"{'='*60}\n")

    # Convert tensors to numpy for visualization
    p1_np = p1.permute(1, 2, 0).numpy()
    p2_np = p2.permute(1, 2, 0).numpy()
    p1_jitter_np = p1_jitter.permute(1, 2, 0).numpy()
    p2_jitter_np = p2_jitter.permute(1, 2, 0).numpy()
    p1_blur_np = p1_blur.permute(1, 2, 0).numpy()
    p2_blur_np = p2_blur.permute(1, 2, 0).numpy()
    p1_down_np = p1_down.permute(1, 2, 0).numpy()
    p2_down_np = p2_down.permute(1, 2, 0).numpy()
    diff_original_np = diff_original.numpy()
    diff_downsampled_np = diff_downsampled.numpy()

    # Create figure with 5 rows
    fig, axes = plt.subplots(5, 2, figsize=(12, 20))
    
    axes[0, 0].imshow(p1_np)
    axes[0, 0].set_title(f'Patch 1 (Original)\n{p1.shape[1]}x{p1.shape[2]}', fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(p2_np)
    axes[0, 1].set_title(f'Patch 2 (Original)\n{p2.shape[1]}x{p2.shape[2]}', fontsize=12, fontweight='bold')
    axes[0, 1].axis('off')
    
    axes[1, 0].imshow(p1_jitter_np)
    axes[1, 0].set_title(f'Patch 1 (Jittered)', fontsize=12, fontweight='bold')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(p2_jitter_np)
    axes[1, 1].set_title(f'Patch 2 (Jittered)', fontsize=12, fontweight='bold')
    axes[1, 1].axis('off')
    
    axes[2, 0].imshow(p1_blur_np)
    axes[2, 0].set_title(f'Patch 1 (Jittered + Blurred σ=4px)', fontsize=12, fontweight='bold')
    axes[2, 0].axis('off')
    
    axes[2, 1].imshow(p2_blur_np)
    axes[2, 1].set_title(f'Patch 2 (Jittered + Blurred σ=4px)', fontsize=12, fontweight='bold')
    axes[2, 1].axis('off')
    
    axes[3, 0].imshow(p1_down_np)
    axes[3, 0].set_title(f'Patch 1 (Final Downsampled)\n32x64', fontsize=12, fontweight='bold')
    axes[3, 0].axis('off')

    axes[3, 1].imshow(p2_down_np)
    axes[3, 1].set_title(f'Patch 2 (Final Downsampled)\n32x64', fontsize=12, fontweight='bold')
    axes[3, 1].axis('off')

    # Difference images (grayscale, white = high magnitude)
    # Left: patch1 - patch2 at original resolution
    axes[4, 0].imshow(diff_original_np, cmap='hot', vmin=0, vmax=max(diff_original_np.max(), diff_downsampled_np.max()))
    axes[4, 0].set_title(f'Patch1 - Patch2\n(Original Resolution)', fontsize=12, fontweight='bold')
    axes[4, 0].axis('off')

    # Right: patch1 - patch2 at downsampled resolution (upsampled for display)
    axes[4, 1].imshow(diff_downsampled_np, cmap='hot', vmin=0, vmax=max(diff_original_np.max(), diff_downsampled_np.max()))
    axes[4, 1].set_title(f'Patch1 - Patch2\n(Downsampled Resolution)', fontsize=12, fontweight='bold')
    axes[4, 1].axis('off')

    # Add similarity info
    fig.suptitle(f'Patch Similarity Analysis (Jitter → Blur → Downsample)\nCosine Similarity: {similarity:.4f} | Angle: {np.degrees(np.arccos(np.clip(similarity, -1, 1))):.2f}°',
                 fontsize=14, fontweight='bold', y=0.99)
    
    plt.tight_layout()
    plt.savefig(args.output, dpi=150, bbox_inches='tight')
    print(f"Saved figure to {args.output}")
    plt.show()


if __name__ == '__main__':
    main()
