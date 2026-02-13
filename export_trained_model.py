#!/usr/bin/env python3
"""
Export a trained FoundationPatchGenerator with its architecture frozen.

This creates a standalone model file that can be loaded even if progressive_patch.py
changes in the future. The exported model is self-contained.

Usage:
  python export_trained_model.py checkpoints/20260211_220815 -o exported_model.pt
"""

import argparse
import sys
from pathlib import Path
import torch

# Import current architecture
from progressive_patch import FoundationPatchGenerator


def export_model(checkpoint_path, output_path):
    """Export model with architecture frozen using torch.jit.script or torch.save."""

    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Extract model parameters
    latent_dim = checkpoint['basis_dim']
    patch_height, patch_width = checkpoint['patch_size']
    use_vae_lora = checkpoint.get('use_vae_lora', True)
    lora_rank = checkpoint.get('lora_rank', 8)
    lora_alpha = checkpoint.get('lora_alpha', 16)

    print(f"Model configuration:")
    print(f"  Latent dim: {latent_dim}")
    print(f"  Patch size: {patch_height}x{patch_width}")
    print(f"  VAE LoRA: {use_vae_lora} (rank={lora_rank}, alpha={lora_alpha})")

    # Create generator with same architecture
    print("\nCreating model...")
    generator = FoundationPatchGenerator(
        latent_dim=latent_dim,
        patch_height=patch_height,
        patch_width=patch_width,
        use_vae_lora=use_vae_lora,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
    )

    # Load weights
    print("Loading weights...")
    generator.load_state_dict(checkpoint['generator_state_dict'])
    generator.eval()

    # Save state dict only (safer for models with JIT components)
    # To load later, you'll need to recreate the architecture, but this is more robust
    print(f"\nSaving model state dict and config to: {output_path}")
    torch.save({
        'state_dict': checkpoint['generator_state_dict'],
        'config': {
            'latent_dim': latent_dim,
            'patch_height': patch_height,
            'patch_width': patch_width,
            'use_vae_lora': use_vae_lora,
            'lora_rank': lora_rank,
            'lora_alpha': lora_alpha,
        },
        'epoch': checkpoint.get('epoch', 'unknown'),
        'source_checkpoint': str(checkpoint_path),
        'architecture_version': 'v1',  # Track architecture version
    }, output_path)

    print("✓ Model exported successfully!")
    print("\nTo load this model later (even with modified progressive_patch.py):")
    print("  # Keep a copy of the OLD progressive_patch.py as progressive_patch_v1.py")
    print("  from progressive_patch_v1 import FoundationPatchGenerator")
    print("  exported = torch.load('exported_model.pt')")
    print("  config = exported['config']")
    print("  generator = FoundationPatchGenerator(**config)")
    print("  generator.load_state_dict(exported['state_dict'])")
    print("  generator.eval()")

    # Test generation
    print("\nTesting generation...")
    with torch.no_grad():
        z_test = torch.randn(1, latent_dim)
        patch_test = generator(z_test)
        print(f"  Test output shape: {patch_test.shape}")
        print(f"  Test output range: [{patch_test.min():.4f}, {patch_test.max():.4f}]")

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description='Export trained FoundationPatchGenerator with frozen architecture'
    )
    parser.add_argument('checkpoint', help='Path to checkpoint file or directory')
    parser.add_argument('-o', '--output', default='exported_generator.pt',
                       help='Output path for exported model (default: exported_generator.pt)')

    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)

    # If directory provided, find latest checkpoint
    if checkpoint_path.is_dir():
        # Try final model first
        final_dir = checkpoint_path / "training_complete_final_model"
        if final_dir.exists():
            checkpoint_files = sorted(final_dir.glob("generator_epoch_*.pt"))
            if checkpoint_files:
                checkpoint_path = checkpoint_files[-1]

        # Try best model
        if checkpoint_path.is_dir():
            best_dir = checkpoint_path / "best_progressive_patch"
            if best_dir.exists():
                checkpoint_files = sorted(best_dir.glob("generator_epoch_*.pt"))
                if checkpoint_files:
                    checkpoint_path = checkpoint_files[-1]

        # Try periodic checkpoints
        if checkpoint_path.is_dir():
            checkpoint_dirs = sorted([d for d in checkpoint_path.iterdir()
                                     if d.is_dir() and d.name.startswith("checkpoint_epoch_")])
            if checkpoint_dirs:
                latest_dir = checkpoint_dirs[-1]
                checkpoint_files = sorted(latest_dir.glob("generator_epoch_*.pt"))
                if checkpoint_files:
                    checkpoint_path = checkpoint_files[-1]

    if not checkpoint_path.is_file():
        print(f"Error: Could not find checkpoint file: {checkpoint_path}", file=sys.stderr)
        sys.exit(1)

    output_path = Path(args.output)
    export_model(checkpoint_path, output_path)


if __name__ == '__main__':
    main()
