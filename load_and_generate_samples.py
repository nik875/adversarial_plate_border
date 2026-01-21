#!/usr/bin/env python3
"""
Load a trained generator checkpoint and generate sample patches.
"""
import torch
from pathlib import Path
from progressive_patch import FoundationPatchGenerator, SimplePatchGenerator
import torchvision.transforms as T
import cv2
import numpy as np

def load_generator_and_generate_samples(
    checkpoint_path: str = "generator_export/training_complete_final_model/generator_epoch_0254.pt",
    num_samples: int = 9,
    output_dir: str = "generated_samples",
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
):
    """
    Load a generator checkpoint and generate sample patches.

    Args:
        checkpoint_path: Path to the generator checkpoint
        num_samples: Number of samples to generate
        output_dir: Directory to save sample images
        device: Device to use (cuda or cpu)
    """
    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)

    print(f"Loading checkpoint from: {checkpoint_path}")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    basis_dim = checkpoint['basis_dim']
    patch_size = checkpoint['patch_size']
    patch_height, patch_width = patch_size

    print(f"Checkpoint info:")
    print(f"  - Basis dim: {basis_dim}")
    print(f"  - Patch size: {patch_height} x {patch_width}")
    print(f"  - Epoch: {checkpoint['epoch']}")

    # Determine which generator type was used
    # Try FoundationPatchGenerator first (more common)
    try:
        generator = FoundationPatchGenerator(
            latent_dim=basis_dim,
            patch_height=patch_height,
            patch_width=patch_width
        ).to(device)
        print("Using FoundationPatchGenerator")
    except Exception as e:
        print(f"FoundationPatchGenerator failed: {e}")
        print("Trying SimplePatchGenerator...")
        generator = SimplePatchGenerator(
            latent_dim=basis_dim,
            patch_height=patch_height,
            patch_width=patch_width
        ).to(device)
        print("Using SimplePatchGenerator")

    # Load weights
    generator.load_state_dict(checkpoint['generator_state_dict'])
    generator.eval()

    print(f"\nGenerating {num_samples} sample patches...")

    # Generate samples
    with torch.no_grad():
        # Sample latent codes uniformly from [0, 1]
        z_samples = torch.rand(num_samples, basis_dim, device=device)

        # Generate patches
        patches = generator(z_samples)  # [num_samples, 3, H, W]

        # Save each patch as an image
        for i, patch in enumerate(patches):
            # Convert to PIL image and save
            patch_cpu = patch.detach().cpu()

            # CLAHE (Contrast Limited Adaptive Histogram Equalization) for better local structure preservation
            patch_uint8 = (patch_cpu * 255).byte()
            clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4, 4))

            # Apply CLAHE to each channel separately
            patch_enhanced = torch.stack([
                torch.from_numpy(clahe.apply(patch_uint8[c].numpy())).float() / 255.0
                for c in range(3)
            ])

            patch_pil = T.ToPILImage()(patch_enhanced)

            output_path = f"{output_dir}/sample_{i:02d}.png"
            patch_pil.save(output_path)
            print(f"Saved: {output_path}")

    print(f"\nSample patches saved to: {output_dir}/")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate sample patches from a trained generator")
    parser.add_argument("--checkpoint", type=str, default="generator_export/training_complete_final_model/generator_epoch_0254.pt",
                        help="Path to generator checkpoint")
    parser.add_argument("--num-samples", type=int, default=9,
                        help="Number of samples to generate")
    parser.add_argument("--output-dir", type=str, default="generated_samples",
                        help="Output directory for samples")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use (cuda or cpu)")

    args = parser.parse_args()

    load_generator_and_generate_samples(
        checkpoint_path=args.checkpoint,
        num_samples=args.num_samples,
        output_dir=args.output_dir,
        device=args.device
    )
