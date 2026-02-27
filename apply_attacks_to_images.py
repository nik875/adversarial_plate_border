#!/usr/bin/env python3
"""
Apply all attack strategies to validation images using example patches from checkpoint.

For each attack strategy (border, sticker, perturbation):
- Load example patches from checkpoint outputs
- Iterate through validation images
- Apply patches using that strategy
- Save results

Output structure:
  attack_outputs/
    border/
      image_000_patch_0_attacked.png
      image_000_patch_1_attacked.png
      ...
    sticker/
      image_000_patch_0_attacked.png
      image_000_patch_1_attacked.png
      ...
    perturbation/
      image_000_patch_0_attacked.png
      image_000_patch_1_attacked.png
      ...
  attack_outputs.tar.gz  (archived at project root)

Usage:
    python apply_attacks_to_images.py --manifest /path/to/manifest.csv --patch-dir /path/to/patches
    python apply_attacks_to_images.py --image-dir /path/to/images --patch-dir /path/to/patches
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


def find_latest_checkpoint_patches(run_dir: str) -> Path:
    """
    Find the latest checkpoint's example_samples in a run directory.

    Looks for checkpoint_epoch_XXXX directories and returns example_samples
    from the one with the highest epoch number.

    Args:
        run_dir: Path to run directory (e.g., runs/broad_ensemble/run_20260227_051206/)

    Returns:
        Path to example_samples directory of latest checkpoint
    """
    run_dir = Path(run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    checkpoint_dirs = sorted([
        d for d in run_dir.rglob('checkpoint_epoch_*')
        if d.is_dir()
    ])

    if not checkpoint_dirs:
        raise FileNotFoundError(f"No checkpoint directories found in {run_dir}")

    latest_ckpt = checkpoint_dirs[-1]
    example_samples = latest_ckpt / 'example_samples'
    if not example_samples.exists():
        raise FileNotFoundError(f"example_samples not found in {latest_ckpt}")

    print(f"Using latest checkpoint: {latest_ckpt.name}")
    return example_samples


def load_patches(patch_dir: str, device: str = 'cuda') -> Tuple[List[torch.Tensor], List[str]]:
    """
    Load all patch images from a directory.

    Args:
        patch_dir: Directory containing patch PNG files
        device: Device to load patches to

    Returns:
        Tuple of (list of patch tensors, list of patch filenames)
    """
    print(f"Loading patches from: {patch_dir}")
    patch_dir = Path(patch_dir)

    patches = []
    filenames = []

    patch_files = sorted(patch_dir.glob('*.png'))
    print(f"Found {len(patch_files)} patch images")

    for patch_path in tqdm(patch_files, desc="Loading patches"):
        img, filename = load_image_from_path(str(patch_path))
        if img is not None:
            patches.append(img.to(device))
            filenames.append(filename)

    print(f"✓ Loaded {len(patches)} patches")
    return patches, filenames


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
    patches: List[torch.Tensor],
    patch_filenames: List[str],
    images: List[Tuple[torch.Tensor, str]],
    device: str = 'cuda',
) -> Path:
    """
    Apply all attack strategies using example patches.

    For each strategy, apply all patches to all images.

    Args:
        patches: List of patch tensors [3, H, W]
        patch_filenames: List of patch filenames (for naming)
        images: List of (image_tensor, filename) tuples
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

    print(f"\nApplying {len(strategies)} attack strategies...")
    print(f"  {len(patches)} patches × {len(images)} images = {len(patches) * len(images)} attacks per strategy")
    print(f"  Border: center_ratio=0.91")
    print(f"  Sticker: area_fraction=0.2")
    print(f"  Perturbation: budget=0.1, norm=linf")

    with torch.no_grad():
        for strategy_name, strategy in strategies.items():
            strategy_dir = output_dir / strategy_name
            strategy_dir.mkdir(parents=True, exist_ok=True)

            output_idx = 0
            for img_idx, (image, img_filename) in enumerate(tqdm(
                images, desc=f"Processing {strategy_name}", leave=False
            )):
                img_base = Path(img_filename).stem

                for patch_idx, patch in enumerate(patches):
                    # Prepare image as batch [1, 3, H, W]
                    image_batch = image.unsqueeze(0)

                    # Apply strategy
                    if strategy_name == 'border':
                        # Resize image to patch size for border strategy
                        patch_h, patch_w = patch.shape[1], patch.shape[2]
                        image_resized = torch.nn.functional.interpolate(
                            image_batch,
                            size=(patch_h, patch_w),
                            mode='bilinear',
                            align_corners=False
                        )
                        composited, _ = strategy.apply(image_resized, patch)
                    else:
                        # Sticker and perturbation use original image size
                        composited, _ = strategy.apply(image_batch, patch)

                    composited = composited.squeeze(0)

                    # Save attacked image
                    tensor_to_pil(composited).save(
                        strategy_dir / f"{output_idx:06d}_{img_base}_patch{patch_idx}_attacked.png"
                    )
                    output_idx += 1

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
        description="Apply all attack strategies to validation images using example patches",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python apply_attacks_to_images.py --manifest ~/.cache/adversarial_plate_manifest.csv --patch-dir runs/broad_ensemble/run_20260227_051206/
  python apply_attacks_to_images.py --image-dir /path/to/images --patch-dir runs/my_run/
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

    # Patches
    parser.add_argument(
        '--patch-dir',
        type=str,
        required=True,
        help='Run directory (e.g., runs/broad_ensemble/run_20260227_051206/). Script finds latest checkpoint within it'
    )

    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device (cuda/cpu, default: cuda if available)'
    )

    args = parser.parse_args()

    # Find latest checkpoint's example_samples in the run directory
    try:
        patch_dir = find_latest_checkpoint_patches(args.patch_dir)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return 1

    # Load patches
    try:
        patches, patch_filenames = load_patches(str(patch_dir), args.device)
    except Exception as e:
        print(f"Error loading patches: {e}")
        import traceback
        traceback.print_exc()
        return 1

    if not patches:
        print("Error: no patches loaded")
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
            patches,
            patch_filenames,
            images,
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
