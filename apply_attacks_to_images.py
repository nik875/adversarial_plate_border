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
from typing import Optional, List, Tuple, Dict

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


def find_latest_patches(run_dir: str) -> Path:
    """
    Find the latest sample patches in a run directory.

    Looks for step_XXXXXXX.tar files and extracts patches from the latest one.
    Tar structure is step_XXXXXXX/{strategy}/{model}/{i}.png

    Args:
        run_dir: Path to run directory (e.g., runs/broad_ensemble/run_20260227_051206/)

    Returns:
        Path to directory containing all patch PNG files (flattened)
    """
    import tempfile

    run_dir = Path(run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    samples_dir = run_dir / 'samples'
    if not samples_dir.exists():
        raise FileNotFoundError(f"samples directory not found in {run_dir}")

    # Find latest step tar file
    step_tars = sorted(samples_dir.glob('step_*.tar'))
    if not step_tars:
        raise FileNotFoundError(f"No step_*.tar files found in {samples_dir}")

    latest_tar = step_tars[-1]
    print(f"Using latest samples: {latest_tar.name}")

    # Extract tar to temp directory
    extract_dir = Path(tempfile.mkdtemp(prefix='patches_'))
    with tarfile.open(latest_tar, 'r') as tar:
        tar.extractall(extract_dir)

    # Find all patch PNG files (they're nested in step_XXXXX/strategy/model/i.png)
    patch_files = list(extract_dir.rglob('*.png'))
    if not patch_files:
        raise FileNotFoundError(f"No patch PNG files found in extracted tar")

    print(f"Extracted {len(patch_files)} patches from {latest_tar.name}")
    return extract_dir


def load_patches_by_strategy(patch_dir: str, device: str = 'cuda') -> Dict[str, List[Tuple[torch.Tensor, str, str]]]:
    """
    Load patch images organized by strategy, preserving model hierarchy.

    Tar structure: step_XXXXX/{strategy}/{model}/{i}.png

    Args:
        patch_dir: Directory containing patch PNG files (extracted tar root)
        device: Device to load patches to

    Returns:
        Dict mapping strategy name → list of (patch_tensor, model_name, patch_id)
    """
    print(f"Loading patches from: {patch_dir}")
    patch_dir = Path(patch_dir)

    # Find strategy directories (border, sticker, perturbation)
    strategy_dirs = {
        d.name: d for d in patch_dir.rglob('*')
        if d.is_dir() and d.name in {'border', 'sticker', 'perturbation'}
    }

    patches_by_strategy = {}

    for strategy_name in ['border', 'sticker', 'perturbation']:
        if strategy_name not in strategy_dirs:
            print(f"Warning: no {strategy_name} patches found")
            patches_by_strategy[strategy_name] = []
            continue

        strat_dir = strategy_dirs[strategy_name]
        patches_list = []

        # Iterate through model subdirectories
        for model_dir in sorted(strat_dir.iterdir()):
            if not model_dir.is_dir():
                continue

            model_name = model_dir.name
            patch_files = sorted(model_dir.glob('*.png'))

            for patch_path in patch_files:
                img, filename = load_image_from_path(str(patch_path))
                if img is not None:
                    patch_id = patch_path.stem
                    patches_list.append((img.to(device), model_name, patch_id))

        print(f"Found {len(patches_list)} {strategy_name} patches")
        print(f"  ✓ Loaded {len(patches_list)} {strategy_name} patches")
        patches_by_strategy[strategy_name] = patches_list

    return patches_by_strategy


def load_validation_images(
    manifest_path: Optional[str] = None,
    image_dir: Optional[str] = None,
    num_samples: int = 50,
    device: str = 'cuda',
) -> List[Tuple[torch.Tensor, str]]:
    """
    Randomly sample and load validation images from manifest or directory.

    If manifest is used, filters for split='val' (ImageNet val, COCO val).

    Args:
        manifest_path: Path to CSV manifest with 'path' and 'split' columns
        image_dir: Directory containing images
        num_samples: Number of images to randomly sample (default: 50)
        device: Device to load images to

    Returns:
        List of (image_tensor, filename) tuples
    """
    import random

    if manifest_path:
        print(f"Loading validation images from manifest: {manifest_path}")
        with open(manifest_path, 'r') as f:
            reader = csv.DictReader(f)
            # Filter for validation split only
            all_paths = [row['path'] for row in reader if row.get('split') == 'val']
        print(f"Found {len(all_paths)} validation images")
    elif image_dir:
        print(f"Loading images from directory: {image_dir}")
        image_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp'}
        all_paths = sorted([
            str(p) for p in Path(image_dir).rglob('*')
            if p.suffix.lower() in image_exts
        ])
    else:
        raise ValueError("Must provide either --manifest or --image-dir")

    print(f"Found {len(all_paths)} images total")

    # Randomly sample
    paths = random.sample(all_paths, min(num_samples, len(all_paths)))
    print(f"Sampling {len(paths)} images")

    images = []
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
    patches_by_strategy: Dict[str, List[Tuple[torch.Tensor, str, str]]],
    images: List[Tuple[torch.Tensor, str]],
    device: str = 'cuda',
) -> Path:
    """
    Apply attack strategies using strategy-specific patches.

    Output structure mirrors input:
      attack_outputs/
        border/
          SmolVLM/
            0_attacked.png
            1_attacked.png
            ...
          CLIP/
            0_attacked.png
            ...
        sticker/
          SmolVLM/
            ...
        perturbation/
          ...

    Args:
        patches_by_strategy: Dict mapping strategy name → list of (patch_tensor, model_name, patch_id)
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

    print(f"\nApplying attack strategies with their respective patches...")
    print(f"  Border: center_ratio=0.91")
    print(f"  Sticker: area_fraction=0.2")
    print(f"  Perturbation: budget=0.1, norm=linf")

    with torch.no_grad():
        for strategy_name, strategy in strategies.items():
            patches_list = patches_by_strategy[strategy_name]

            if not patches_list:
                print(f"Skipping {strategy_name}: no patches loaded")
                continue

            print(f"\n{strategy_name}: {len(patches_list)} patches (1 image per patch)")

            # Count patches per model for progress tracking
            models = set(model for _, model, _ in patches_list)
            for model_name in sorted(models):
                model_patches = [(p, pid) for p, m, pid in patches_list if m == model_name]
                model_dir = output_dir / strategy_name / model_name
                model_dir.mkdir(parents=True, exist_ok=True)

                for patch_idx, (patch, patch_id) in enumerate(tqdm(
                    model_patches, desc=f"{strategy_name}/{model_name}", leave=False
                )):
                    # Pick one random image for this patch
                    image, img_filename = images[patch_idx % len(images)]

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
                        kwargs = strategy.sample_kwargs(image_resized, patch.shape[1], patch.shape[2])
                        composited, _ = strategy.apply(image_resized, patch, **kwargs)
                    else:
                        # Sticker and perturbation: sample kwargs (gives random bbox for sticker)
                        kwargs = strategy.sample_kwargs(image_batch, patch.shape[1], patch.shape[2])
                        composited, _ = strategy.apply(image_batch, patch, **kwargs)

                    composited = composited.squeeze(0)

                    # Save attacked image: {patch_id}_attacked.png
                    tensor_to_pil(composited).save(
                        model_dir / f"{patch_id}_attacked.png"
                    )

    print(f"\n✓ Attack outputs saved to: {output_dir}")
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

    # Find and extract latest patches from run directory
    try:
        patch_dir = find_latest_patches(args.patch_dir)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return 1

    # Load patches organized by strategy
    try:
        patches_by_strategy = load_patches_by_strategy(str(patch_dir), args.device)
    except Exception as e:
        print(f"Error loading patches: {e}")
        import traceback
        traceback.print_exc()
        return 1

    total_patches = sum(len(p) for p in patches_by_strategy.values())
    if total_patches == 0:
        print("Error: no patches loaded")
        return 1

    # Load exactly enough images to match total patches
    num_images_needed = total_patches
    print(f"Total patches: {total_patches}, sampling {num_images_needed} images")

    try:
        images = load_validation_images(
            manifest_path=args.manifest,
            image_dir=args.image_dir,
            num_samples=num_images_needed,
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
            patches_by_strategy,
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
