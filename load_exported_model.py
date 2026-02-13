#!/usr/bin/env python3
"""
Helper to load exported models, even if progressive_patch.py has changed.

Usage:
  from load_exported_model import load_exported_generator

  generator = load_exported_generator('exported_model.pt')
  z = torch.randn(1, 16)
  patch = generator(z)
"""

import torch
from pathlib import Path


def load_exported_generator(checkpoint_path, device='cpu'):
    """
    Load an exported generator model.

    Args:
        checkpoint_path: Path to exported .pt file
        device: Device to load model on ('cpu' or 'cuda')

    Returns:
        generator: Loaded FoundationPatchGenerator ready for inference
    """
    # Try importing current architecture first
    try:
        from progressive_patch import FoundationPatchGenerator
        print("Using current progressive_patch.py architecture")
    except ImportError:
        # Fall back to v1 if available
        try:
            from progressive_patch_v1 import FoundationPatchGenerator
            print("WARNING: Using progressive_patch_v1.py (old architecture)")
        except ImportError:
            raise ImportError(
                "Cannot find FoundationPatchGenerator. Please ensure either:\n"
                "  1. progressive_patch.py exists (current version), or\n"
                "  2. progressive_patch_v1.py exists (backed up old version)"
            )

    # Load checkpoint
    print(f"Loading model from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Extract config
    config = checkpoint['config']
    print(f"Model config: latent_dim={config['latent_dim']}, "
          f"patch_size={config['patch_height']}x{config['patch_width']}, "
          f"LoRA={config['use_vae_lora']}")

    # Create model
    generator = FoundationPatchGenerator(**config)

    # Load weights
    generator.load_state_dict(checkpoint['state_dict'])
    generator.to(device)
    generator.eval()

    print("✓ Model loaded successfully")

    return generator


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Test loading an exported model')
    parser.add_argument('checkpoint', help='Path to exported .pt file')
    parser.add_argument('--device', default='cpu', help='Device (cpu/cuda)')

    args = parser.parse_args()

    # Load model
    generator = load_exported_generator(args.checkpoint, args.device)

    # Test generation
    print("\nTesting generation...")
    with torch.no_grad():
        z_test = torch.randn(2, generator.latent_dim).to(args.device)
        patches = generator(z_test)
        print(f"  Input shape: {z_test.shape}")
        print(f"  Output shape: {patches.shape}")
        print(f"  Output range: [{patches.min():.4f}, {patches.max():.4f}]")
        print("✓ Generation successful!")
