#!/usr/bin/env python3
"""
Debug script to check if generator outputs differ for different latent codes.
"""
import torch
from progressive_patch import FoundationPatchGenerator, SimplePatchGenerator
import numpy as np

def debug_generator_outputs(
    checkpoint_path: str = "generator_export/final_layer_checkpoint_epoch_0100/generator_epoch_0104.pt",
    num_samples: int = 5,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
):
    """
    Check if generator produces different outputs for different latent codes.
    """
    print(f"Loading checkpoint from: {checkpoint_path}")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    basis_dim = checkpoint['basis_dim']
    patch_size = checkpoint['patch_size']
    patch_height, patch_width = patch_size

    print(f"Checkpoint info:")
    print(f"  - Basis dim: {basis_dim}")
    print(f"  - Patch size: {patch_height} x {patch_width}")

    # Load generator
    try:
        generator = FoundationPatchGenerator(
            latent_dim=basis_dim,
            patch_height=patch_height,
            patch_width=patch_width
        ).to(device)
        print("Using FoundationPatchGenerator")
    except Exception as e:
        print(f"FoundationPatchGenerator failed: {e}")
        generator = SimplePatchGenerator(
            latent_dim=basis_dim,
            patch_height=patch_height,
            patch_width=patch_width
        ).to(device)
        print("Using SimplePatchGenerator")

    # Load weights
    generator.load_state_dict(checkpoint['generator_state_dict'])
    generator.eval()

    print(f"\nGenerating {num_samples} patches with different latent codes...")

    with torch.no_grad():
        # Sample different latent codes
        z_samples = torch.rand(num_samples, basis_dim, device=device)

        # Generate patches
        patches = generator(z_samples)  # [num_samples, 3, H, W]

        print(f"\nRaw patch statistics:")
        print(f"  - Shape: {patches.shape}")
        print(f"  - Min value: {patches.min().item():.6f}")
        print(f"  - Max value: {patches.max().item():.6f}")
        print(f"  - Mean value: {patches.mean().item():.6f}")

        # Compare patches pairwise
        print(f"\nPairwise comparisons:")
        print(f"{'Pair':<10} {'L2 Distance':<15} {'Cosine Dist':<15} {'Pixel Max Diff':<15}")
        print("-" * 55)

        for i in range(num_samples):
            for j in range(i + 1, num_samples):
                patch_i_flat = patches[i].reshape(-1)
                patch_j_flat = patches[j].reshape(-1)

                # L2 distance
                l2_dist = torch.norm(patch_i_flat - patch_j_flat).item()

                # Cosine distance
                cosine_dist = 1 - torch.nn.functional.cosine_similarity(
                    patch_i_flat.unsqueeze(0),
                    patch_j_flat.unsqueeze(0)
                ).item()

                # Max pixel difference
                pixel_max_diff = (patches[i] - patches[j]).abs().max().item()

                print(f"{i}-{j}      {l2_dist:<15.6f} {cosine_dist:<15.6f} {pixel_max_diff:<15.6f}")

        # Check per-sample statistics
        print(f"\nPer-sample statistics:")
        print(f"{'Sample':<8} {'Min':<12} {'Max':<12} {'Mean':<12} {'Std':<12}")
        print("-" * 56)
        for i in range(num_samples):
            patch = patches[i]
            print(f"{i:<8} {patch.min().item():<12.6f} {patch.max().item():<12.6f} {patch.mean().item():<12.6f} {patch.std().item():<12.6f}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Debug generator outputs")
    parser.add_argument("--checkpoint", type=str, default="generator_export/final_layer_checkpoint_epoch_0100/generator_epoch_0104.pt",
                        help="Path to generator checkpoint")
    parser.add_argument("--num-samples", type=int, default=5,
                        help="Number of samples to generate")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use (cuda or cpu)")

    args = parser.parse_args()

    debug_generator_outputs(
        checkpoint_path=args.checkpoint,
        num_samples=args.num_samples,
        device=args.device
    )
