#!/usr/bin/env python3
"""
Apply all attack strategies to validation images using a trained patch generator.

For each attack strategy (border, sticker, perturbation):
- Iterate through validation images
- Generate one patch per image
- Apply that attack
- Save result

Output structure:
  attack_outputs/
    border/
      image_000_patch.png
      image_000_attacked.png
      ...
    sticker/
      image_000_patch.png
      image_000_attacked.png
      ...
    perturbation/
      image_000_patch.png
      image_000_attacked.png
      ...
  attack_outputs.tar.gz  (archived at project root)

Usage:
    python apply_attacks_to_images.py --manifest /path/to/manifest.csv --checkpoint /path/to/generator.pt
    python apply_attacks_to_images.py --image-dir /path/to/images --checkpoint /path/to/generator.pt
"""

import argparse
import csv
import tarfile
import torch
import torchvision.transforms as T
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from typing import Optional, List, Tuple

# Framework imports
from framework.generator import FoundationPatchGenerator
from framework.base.attack_strategy import (
    BorderStrategy,
    StickerStrategy,
    PerturbationStrategy,
)


def load_image_from_path(image_path: str) -> Tuple[torch.Tensor, str]:
    """
    Load an image and return as [3, H, W] tensor in [0, 1] range.

    Args:
        image_path: Path to image file

    Returns:
        Tuple of (tensor, filename)
    """
    try:
        img = Image.open(image_path).convert('RGB')
        img_tensor = T.ToTensor()(img)  # [3, H, W] in [0, 1]
        filename = Path(image_path).name
        return img_tensor, filename
    except Exception as e:
        print(f"Warning: could not load {image_path}: {e}")
        return None, None


def load_generator_checkpoint(checkpoint_path: str, device: str = 'cuda') -> FoundationPatchGenerator:
    """Load a trained generator from checkpoint."""
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    basis_dim = checkpoint['basis_dim']
    patch_size = checkpoint.get('patch_size', [224, 224])
    patch_height, patch_width = patch_size
    num_taesd = checkpoint.get('num_taesd', 1)

    # Get transformer params
    transformer_d_model = checkpoint.get('transformer_d_model', 256)
    transformer_nhead = checkpoint.get('transformer_nhead', 4)
    transformer_d_ff = checkpoint.get('transformer_d_ff', 1024)
    transformer_enc_layers = checkpoint.get('transformer_enc_layers', 2)
    transformer_dec_layers = checkpoint.get('transformer_dec_layers', 2)

    print(f"  Basis dim: {basis_dim}")
    print(f"  Patch size: {patch_height} × {patch_width}")
    print(f"  Num TAESD: {num_taesd}")

    generator = FoundationPatchGenerator(
        latent_dim=basis_dim,
        patch_height=patch_height,
        patch_width=patch_width,
        num_taesd=num_taesd,
        transformer_d_model=transformer_d_model,
        transformer_nhead=transformer_nhead,
        transformer_d_ff=transformer_d_ff,
        transformer_enc_layers=transformer_enc_layers,
        transformer_dec_layers=transformer_dec_layers,
    ).to(device)

    generator.load_state_dict(checkpoint['generator_state_dict'])
    generator.eval()
    print(f"✓ Generator loaded on {device}")

    return generator, patch_height, patch_width


def load_validation_images(
    manifest_path: Optional[str] = None,
    image_dir: Optional[str] = None,
    device: str = 'cuda',
) -> List[Tuple[torch.Tensor, str]]:
    """
    Load all validation images from either a manifest CSV or a directory.

    Args:
        manifest_path: Path to CSV manifest with 'path' column
        image_dir: Directory containing images
        device: Device to load images to

    Returns:
        List of (image_tensor, filename) tuples
    """
    images = []

    if manifest_path:
        print(f"Loading images from manifest: {manifest_path}")
        with open(manifest_path, 'r') as f:
            reader = csv.DictReader(f)
            paths = [row['path'] for row in reader]
    elif image_dir:
        print(f"Loading images from directory: {image_dir}")
        image_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp'}
        paths = sorted([
            str(p) for p in Path(image_dir).rglob('*')
            if p.suffix.lower() in image_exts
        ])
    else:
        raise ValueError("Must provide either --manifest or --image-dir")

    print(f"Found {len(paths)} images")

    for img_path in tqdm(paths, desc="Loading images"):
        img_tensor, filename = load_image_from_path(img_path)
        if img_tensor is not None:
            images.append((img_tensor.to(device), filename))

    print(f"✓ Loaded {len(images)} images")
    return images


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """Convert [3, H, W] tensor in [0, 1] to PIL Image."""
    # Clamp to [0, 1] to be safe
    tensor = torch.clamp(tensor, 0, 1)
    pil = T.ToPILImage()(tensor.cpu())
    return pil


def apply_attacks(
    generator: FoundationPatchGenerator,
    images: List[Tuple[torch.Tensor, str]],
    patch_height: int,
    patch_width: int,
    device: str = 'cuda',
) -> Path:
    """
    Apply all attack strategies. For each strategy, generate one patch per image.

    Args:
        generator: Trained patch generator
        images: List of (image_tensor, filename) tuples
        patch_height: Height of generated patches
        patch_width: Width of generated patches
        device: Torch device

    Returns:
        Path to output directory
    """
    output_dir = Path("attack_outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize strategies
    border_strategy = BorderStrategy(center_ratio=0.91)
    sticker_strategy = StickerStrategy(area_fraction=0.2)
    perturbation_strategy = PerturbationStrategy(budget=0.1, norm='linf')

    strategies = {
        'border': border_strategy,
        'sticker': sticker_strategy,
        'perturbation': perturbation_strategy,
    }

    print(f"\nApplying {len(strategies)} attack strategies to {len(images)} images...")
    print(f"  Border: center_ratio=0.91")
    print(f"  Sticker: area_fraction=0.2")
    print(f"  Perturbation: budget=0.1, norm=linf")

    with torch.no_grad():
        for strategy_name, strategy in strategies.items():
            strategy_dir = output_dir / strategy_name
            strategy_dir.mkdir(parents=True, exist_ok=True)

            for img_idx, (image, filename) in enumerate(tqdm(
                images, desc=f"Processing {strategy_name}", leave=False
            )):
                # Generate one patch for this image
                z = torch.rand(1, generator.latent_dim, device=device)
                patch = generator(z)[0]  # [3, H, W]

                # Prepare image as batch [1, 3, H, W]
                image_batch = image.unsqueeze(0)

                # Apply strategy
                if strategy_name == 'border':
                    # Resize image to patch size for border strategy
                    image_resized = torch.nn.functional.interpolate(
                        image_batch,
                        size=(patch_height, patch_width),
                        mode='bilinear',
                        align_corners=False
                    )
                    composited, _ = strategy.apply(image_resized, patch)
                else:
                    # Sticker and perturbation use original image size
                    composited, _ = strategy.apply(image_batch, patch)

                composited = composited.squeeze(0)

                # Save patch
                base_name = Path(filename).stem
                tensor_to_pil(patch).save(
                    strategy_dir / f"{img_idx:04d}_{base_name}_patch.png"
                )

                # Save attacked image
                tensor_to_pil(composited).save(
                    strategy_dir / f"{img_idx:04d}_{base_name}_attacked.png"
                )

    print(f"✓ Attack outputs saved to: {output_dir}")
    return output_dir


def create_tar_archive(output_dir: Path, tar_path: Path) -> None:
    """Create a tarred archive of the output directory."""
    print(f"\nCreating tar archive: {tar_path}")
    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(output_dir, arcname=output_dir.name)
    print(f"✓ Archive created: {tar_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Apply all attack strategies to validation images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python apply_attacks_to_images.py --manifest data/manifest.csv --checkpoint checkpoint.pt
  python apply_attacks_to_images.py --image-dir /path/to/images --checkpoint checkpoint.pt
        """
    )

    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        '--manifest',
        type=str,
        help='Path to manifest CSV with image paths (path column)'
    )
    input_group.add_argument(
        '--image-dir',
        type=str,
        help='Directory containing images'
    )

    # Checkpoint
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to trained generator checkpoint'
    )

    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device (cuda/cpu, default: cuda if available)'
    )

    args = parser.parse_args()

    # Validate checkpoint exists
    if not Path(args.checkpoint).exists():
        print(f"Error: checkpoint not found: {args.checkpoint}")
        return 1

    # Load generator
    try:
        generator, patch_h, patch_w = load_generator_checkpoint(args.checkpoint, args.device)
    except Exception as e:
        print(f"Error loading generator: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Load images
    try:
        images = load_validation_images(
            manifest_path=args.manifest,
            image_dir=args.image_dir,
            device=args.device,
        )
    except Exception as e:
        print(f"Error loading images: {e}")
        import traceback
        traceback.print_exc()
        return 1

    if not images:
        print("Error: no images loaded")
        return 1

    # Apply attacks
    try:
        output_dir = apply_attacks(
            generator,
            images,
            patch_h,
            patch_w,
            device=args.device,
        )
    except Exception as e:
        print(f"Error applying attacks: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Create tar archive at project root
    try:
        tar_path = Path("attack_outputs.tar.gz")
        create_tar_archive(output_dir, tar_path)
    except Exception as e:
        print(f"Error creating tar archive: {e}")
        import traceback
        traceback.print_exc()
        return 1

    print(f"\n✓ Done! Results:")
    print(f"  Directory: {output_dir}")
    print(f"  Archive: {tar_path}")

    return 0


if __name__ == '__main__':
    exit(main())
