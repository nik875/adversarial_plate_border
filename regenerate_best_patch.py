#!/usr/bin/env python3
"""
Regenerate best_patch.png from best_z.npy with proper clamping.

This is useful after discovering that the generator outputs values outside [0, 1].
Regenerates the patch with clamping to match what actually gets composited.

Usage:
  python regenerate_best_patch.py cmaes_output
  python regenerate_best_patch.py cmaes_output --checkpoint checkpoints/20260211_220815
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import cv2
import torch

# Import model loading utilities
from load_exported_model import load_exported_generator


def load_generator_from_checkpoint(checkpoint_path):
    """Load generator from checkpoint directory."""
    try:
        from progressive_patch_v1 import FoundationPatchGenerator
        print("Using progressive_patch_v1.py (old architecture)")
    except ImportError:
        from progressive_patch import FoundationPatchGenerator
        print("WARNING: progressive_patch_v1.py not found, using current progressive_patch.py")

    checkpoint_dir = Path(checkpoint_path)

    # Find latest checkpoint (same logic as optimize_patch_cmaes.py)
    latest_checkpoint = None

    # Priority 1: Final training checkpoint
    final_dir = checkpoint_dir / "training_complete_final_model"
    if final_dir.exists() and final_dir.is_dir():
        checkpoint_files = sorted(final_dir.glob("generator_epoch_*.pt"))
        if checkpoint_files:
            latest_checkpoint = checkpoint_files[-1]

    # Priority 2: Best model checkpoint
    if latest_checkpoint is None:
        best_dir = checkpoint_dir / "best_progressive_patch"
        if best_dir.exists() and best_dir.is_dir():
            checkpoint_files = sorted(best_dir.glob("generator_epoch_*.pt"))
            if checkpoint_files:
                latest_checkpoint = checkpoint_files[-1]

    # Priority 3: Latest periodic checkpoint
    if latest_checkpoint is None:
        checkpoint_dirs = sorted([d for d in checkpoint_dir.iterdir()
                                 if d.is_dir() and d.name.startswith("checkpoint_epoch_")])
        if checkpoint_dirs:
            latest_checkpoint_dir = checkpoint_dirs[-1]
            checkpoint_files = sorted(latest_checkpoint_dir.glob("generator_epoch_*.pt"))
            if checkpoint_files:
                latest_checkpoint = checkpoint_files[-1]

    if latest_checkpoint is None:
        raise FileNotFoundError(f"No generator checkpoint found in {checkpoint_dir}")

    print(f"Loading checkpoint: {latest_checkpoint}")

    # Load checkpoint
    checkpoint = torch.load(latest_checkpoint, map_location='cpu')

    # Extract model parameters
    latent_dim = checkpoint['basis_dim']
    patch_height, patch_width = checkpoint['patch_size']
    use_vae_lora = checkpoint.get('use_vae_lora', True)
    lora_rank = checkpoint.get('lora_rank', 8)
    lora_alpha = checkpoint.get('lora_alpha', 16)

    print(f"  Latent dim: {latent_dim}")
    print(f"  Patch size: {patch_height}x{patch_width}")

    # Create generator
    generator = FoundationPatchGenerator(
        latent_dim=latent_dim,
        patch_height=patch_height,
        patch_width=patch_width,
        use_vae_lora=use_vae_lora,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
    )

    # Load state dict
    generator.load_state_dict(checkpoint['generator_state_dict'])
    generator.eval()

    print(f"Generator loaded successfully")

    return generator, latent_dim


def main():
    parser = argparse.ArgumentParser(
        description='Regenerate best_patch.png from best_z.npy with clamping'
    )
    parser.add_argument('output_dir', help='Path to output directory (e.g., cmaes_output)')
    parser.add_argument('--checkpoint', default=None,
                       help='Path to checkpoint directory (auto-detect from run_dir if not specified)')
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda'],
                       help='Device to use (default: cpu)')

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if not output_dir.exists():
        print(f"Error: Output directory not found: {output_dir}", file=sys.stderr)
        sys.exit(1)

    # Find best_z.npy
    z_path = output_dir / "best_z.npy"
    if not z_path.exists():
        print(f"Error: best_z.npy not found in {output_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading latent code from: {z_path}")
    best_z = np.load(z_path)
    print(f"  Shape: {best_z.shape}")

    # Find checkpoint directory if not specified
    if args.checkpoint is None:
        # Look for optimization_results.txt to get run_dir
        metadata_path = output_dir / "optimization_results.txt"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                for line in f:
                    if line.startswith("Run directory:"):
                        checkpoint_dir = Path(line.split(":", 1)[1].strip())
                        break
                else:
                    print("Error: Could not find run directory in optimization_results.txt", file=sys.stderr)
                    sys.exit(1)
        else:
            print("Error: --checkpoint not specified and optimization_results.txt not found", file=sys.stderr)
            sys.exit(1)
    else:
        checkpoint_dir = Path(args.checkpoint)

    print(f"\nLoading generator from: {checkpoint_dir}")
    device = torch.device(args.device)
    generator, latent_dim = load_generator_from_checkpoint(checkpoint_dir)
    generator = generator.to(device)

    # Generate patch
    print(f"\nGenerating patch from latent code...")
    with torch.no_grad():
        z_tensor = torch.from_numpy(best_z).float().unsqueeze(0).to(device)
        patch = generator(z_tensor)
        patch = patch.squeeze(0).cpu()  # [3, H, W]

    print(f"  Patch shape: {patch.shape}")
    print(f"  Before clamp - Min: {patch.min():.6f}, Max: {patch.max():.6f}, Mean: {patch.mean():.6f}")

    # Save original (unclamped) version
    patch_orig_np = (patch.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    patch_orig_bgr = cv2.cvtColor(patch_orig_np, cv2.COLOR_RGB2BGR)
    orig_output_path = output_dir / "best_patch_orig.png"
    cv2.imwrite(str(orig_output_path), patch_orig_bgr)
    print(f"\n✓ Saved original (unclamped) patch to: {orig_output_path}")

    # Clamp to [0, 1]
    patch_clamped = torch.clamp(patch, 0, 1)

    print(f"  After clamp  - Min: {patch_clamped.min():.6f}, Max: {patch_clamped.max():.6f}, Mean: {patch_clamped.mean():.6f}")

    # Save clamped version
    patch_np = (patch_clamped.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    patch_bgr = cv2.cvtColor(patch_np, cv2.COLOR_RGB2BGR)

    output_path = output_dir / "best_patch.png"
    cv2.imwrite(str(output_path), patch_bgr)

    print(f"✓ Saved clamped patch to: {output_path}")


if __name__ == '__main__':
    main()
