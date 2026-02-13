#!/usr/bin/env python3
"""
CMA-ES optimization of adversarial patches to maximize misreads on validation set.

Uses CMA-ES to optimize latent codes z that generate patches via the VAE decoder.
The objective is to maximize the number of misreads (changed OCR predictions) when
patches are composited onto validation images.

Usage:
  python optimize_patch_cmaes.py run_dir --popsize 20 --maxiter 100 --sigma0 0.5
"""

import argparse
import sys
from pathlib import Path
import csv
import random
from datetime import datetime

import cv2
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import Levenshtein

try:
    from fast_alpr import ALPR
except ImportError:
    ALPR = None

try:
    import cma
except ImportError:
    cma = None


def load_validation_samples_from_csv(csv_path, num_samples):
    """Load validation samples using combined dataset (matching training setup).

    Args:
        csv_path: Path to train_val_split CSV
        num_samples: Number of samples to load

    Returns:
        Tuple of (list of images as tensors [3, H, W] in [0, 1], list of (width, height) tuples)
    """
    # Import OCRDataset and ConcatDataset
    script_dir = Path(__file__).parent
    from torch.utils.data import ConcatDataset

    # Import OCRDataset from progressive_patch
    sys.path.insert(0, str(script_dir))
    try:
        from progressive_patch import OCRDataset
    except ImportError:
        raise ImportError("Could not import OCRDataset from progressive_patch.py")

    # Read CSV to get dataset names and validation indices
    val_indices = []
    dataset_names_in_csv = set()

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['split'].lower() == 'val':
                val_indices.append(int(row['index']))
                dataset_names_in_csv.add(row['dataset'])

    if not val_indices:
        raise ValueError(f"No validation samples found in {csv_path}")

    print(f"Found {len(val_indices)} validation samples in CSV")
    print(f"Datasets in CSV: {', '.join(sorted(dataset_names_in_csv))}")

    # Load and combine datasets in order
    datasets_to_combine = []
    for dataset_name in sorted(dataset_names_in_csv):
        print(f"  Loading {dataset_name}...")
        try:
            dataset = OCRDataset(
                dataset_name=dataset_name,
                split='train',
                transform=None,
                max_samples=None
            )
            datasets_to_combine.append(dataset)
            print(f"    Loaded {len(dataset)} samples from {dataset_name}")
        except Exception as e:
            print(f"  Error loading {dataset_name}: {e}", file=sys.stderr)
            raise

    # Combine datasets (matching how progressive_patch.py does it)
    if len(datasets_to_combine) > 1:
        combined_dataset = ConcatDataset(datasets_to_combine)
        print(f"Combined {len(datasets_to_combine)} datasets: {len(combined_dataset)} total samples")
    else:
        combined_dataset = datasets_to_combine[0]

    # Load all validation samples upfront
    images = []
    dimensions = []
    failed_samples = []

    print(f"\nLoading all {len(val_indices)} validation samples from combined dataset...")
    for combined_idx in val_indices:
        try:
            item = combined_dataset[combined_idx]
            img_tensor = item['prep_image']
            # Track dimensions: tensor is [3, H, W], so width=W, height=H
            height, width = img_tensor.shape[1], img_tensor.shape[2]

            images.append(img_tensor)
            dimensions.append((width, height))
        except Exception as e:
            error_msg = f"{type(e).__name__}: {e}"
            failed_samples.append((combined_idx, error_msg))

    # Randomly select down to requested number
    if len(images) > num_samples:
        indices_to_keep = random.sample(range(len(images)), num_samples)
        images = [images[i] for i in sorted(indices_to_keep)]
        dimensions = [dimensions[i] for i in sorted(indices_to_keep)]

    print(f"Loaded {len(images)} validation samples")
    if failed_samples:
        print(f"Failed to load {len(failed_samples)} samples", file=sys.stderr)

    return images, dimensions


def apply_patch_ocr_mode(image, patch, center_ratio=0.6):
    """Apply adversarial patch to image (center region preserved).

    Args:
        image: [3, H, W] or [1, 3, H, W] tensor in [0, 1]
        patch: [3, patch_h, patch_w] tensor in [0, 1]
        center_ratio: Fraction of image to preserve in center (default: 0.6)

    Returns:
        result: [B, 3, H, W] patched image
    """
    # Handle single image
    if image.dim() == 3:
        image = image.unsqueeze(0)

    batch_size = image.shape[0]
    image_height, image_width = image.shape[2], image.shape[3]

    # Resize patch to match image dimensions
    patch_resized = F.interpolate(
        patch.unsqueeze(0),  # [1, 3, patch_h, patch_w]
        size=(image_height, image_width),
        mode='bilinear',
        align_corners=False
    )  # [1, 3, H, W]

    # Expand to batch size
    patch_batch = patch_resized.repeat(batch_size, 1, 1, 1)  # [B, 3, H, W]

    # Create center mask (1 in center, 0 on borders)
    center_h = int(image_height * center_ratio)
    center_w = int(image_width * center_ratio)

    # Calculate padding to center the mask
    pad_h = (image_height - center_h) // 2
    pad_w = (image_width - center_w) // 2

    # Create mask: 1 in center region, 0 elsewhere
    center_mask = torch.zeros(batch_size, 1, image_height, image_width,
                             dtype=torch.float32)
    center_mask[:, :, pad_h:pad_h + center_h, pad_w:pad_w + center_w] = 1.0
    center_mask = center_mask.expand(-1, 3, -1, -1)  # [B, 3, H, W]

    # Blend: keep original image in center, use patch on borders
    result_image = image * center_mask + patch_batch * (1 - center_mask)
    result_image = torch.clamp(result_image, 0, 1)

    return result_image


def apply_neutral_border_ocr_mode(image, center_ratio=0.6, border_color=0.5):
    """Apply neutral grey border to image (center region preserved).

    Args:
        image: [3, H, W] or [1, 3, H, W] tensor in [0, 1]
        center_ratio: Fraction of image to preserve in center (default: 0.6)
        border_color: Value for neutral border (default: 0.5 = gray)

    Returns:
        result: [B, 3, H, W] image with grey border
    """
    # Handle single image
    if image.dim() == 3:
        image = image.unsqueeze(0)

    batch_size = image.shape[0]
    image_height, image_width = image.shape[2], image.shape[3]

    # Create center mask (1 in center, 0 on borders)
    center_h = int(image_height * center_ratio)
    center_w = int(image_width * center_ratio)

    # Calculate padding to center the mask
    pad_h = (image_height - center_h) // 2
    pad_w = (image_width - center_w) // 2

    # Create mask: 1 in center region, 0 elsewhere
    center_mask = torch.zeros(batch_size, 1, image_height, image_width,
                             dtype=torch.float32)
    center_mask[:, :, pad_h:pad_h + center_h, pad_w:pad_w + center_w] = 1.0
    center_mask = center_mask.expand(-1, 3, -1, -1)  # [B, 3, H, W]

    # Create neutral border
    neutral_border = torch.full_like(image, border_color)

    # Blend: keep original image in center, use neutral border on borders
    result_image = image * center_mask + neutral_border * (1 - center_mask)
    result_image = torch.clamp(result_image, 0, 1)

    return result_image


def load_generator(run_dir):
    """Load the FoundationPatchGenerator from the run directory.

    Args:
        run_dir: Path to run directory

    Returns:
        Tuple of (generator model, latent_dim, device)
    """
    import sys
    from pathlib import Path

    # Import from progressive_patch.py
    sys.path.insert(0, str(Path(__file__).parent))
    try:
        from progressive_patch import FoundationPatchGenerator
    except ImportError:
        raise ImportError("Could not import FoundationPatchGenerator from progressive_patch.py")

    # Search for checkpoints in this priority order:
    # 1. training_complete_final_model (final checkpoint at end of training)
    # 2. best_progressive_patch (best model during training)
    # 3. checkpoint_epoch_XXXX (periodic checkpoints)
    # 4. Any other directories with generator_epoch_*.pt files

    run_path = Path(run_dir)
    latest_checkpoint = None
    checkpoint_source = None

    # Priority 1: Final training checkpoint
    final_dir = run_path / "training_complete_final_model"
    if final_dir.exists() and final_dir.is_dir():
        checkpoint_files = sorted(final_dir.glob("generator_epoch_*.pt"))
        if checkpoint_files:
            latest_checkpoint = checkpoint_files[-1]
            checkpoint_source = "final training checkpoint"

    # Priority 2: Best model checkpoint
    if latest_checkpoint is None:
        best_dir = run_path / "best_progressive_patch"
        if best_dir.exists() and best_dir.is_dir():
            checkpoint_files = sorted(best_dir.glob("generator_epoch_*.pt"))
            if checkpoint_files:
                latest_checkpoint = checkpoint_files[-1]
                checkpoint_source = "best model checkpoint"

    # Priority 3: Latest periodic checkpoint
    if latest_checkpoint is None:
        checkpoint_dirs = sorted([d for d in run_path.iterdir()
                                 if d.is_dir() and d.name.startswith("checkpoint_epoch_")])
        if checkpoint_dirs:
            latest_checkpoint_dir = checkpoint_dirs[-1]
            checkpoint_files = sorted(latest_checkpoint_dir.glob("generator_epoch_*.pt"))
            if checkpoint_files:
                latest_checkpoint = checkpoint_files[-1]
                checkpoint_source = f"periodic checkpoint ({latest_checkpoint_dir.name})"

    # Priority 4: Any other checkpoint directory
    if latest_checkpoint is None:
        # Search all subdirectories for generator checkpoints
        all_checkpoints = sorted(run_path.glob("**/generator_epoch_*.pt"))
        if all_checkpoints:
            latest_checkpoint = all_checkpoints[-1]
            checkpoint_source = f"found in {latest_checkpoint.parent.name}"

    if latest_checkpoint is None:
        raise FileNotFoundError(
            f"No generator checkpoint files found in {run_dir}\n"
            f"Searched for:\n"
            f"  - training_complete_final_model/generator_epoch_*.pt\n"
            f"  - best_progressive_patch/generator_epoch_*.pt\n"
            f"  - checkpoint_epoch_*/generator_epoch_*.pt\n"
            f"  - **/generator_epoch_*.pt"
        )

    print(f"Loading checkpoint: {latest_checkpoint}")
    print(f"  Source: {checkpoint_source}")

    # Load checkpoint
    checkpoint = torch.load(latest_checkpoint, map_location='cpu')

    # Extract model parameters from checkpoint
    latent_dim = checkpoint['basis_dim']
    patch_height, patch_width = checkpoint['patch_size']
    use_vae_lora = checkpoint.get('use_vae_lora', True)
    lora_rank = checkpoint.get('lora_rank', 8)
    lora_alpha = checkpoint.get('lora_alpha', 16)

    print(f"  Latent dim: {latent_dim}")
    print(f"  Patch size: {patch_height}x{patch_width}")
    print(f"  VAE LoRA: {use_vae_lora} (rank={lora_rank}, alpha={lora_alpha})")

    # Create generator with same architecture
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

    # Use GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    generator = generator.to(device)
    generator.eval()

    print(f"Loaded generator on device={device}")

    return generator, latent_dim, device


def generate_patch_from_z(generator, z, device):
    """Generate a patch from latent code z using the generator.

    Args:
        generator: FoundationPatchGenerator model
        z: Latent code as numpy array [latent_dim]
        device: torch device

    Returns:
        patch: [3, H, W] tensor in [0, 1]
    """
    with torch.no_grad():
        z_tensor = torch.from_numpy(z).float().unsqueeze(0).to(device)  # [1, latent_dim]
        patch = generator(z_tensor)  # [1, 3, H, W]
        # Generator output is already in [0, 1] range (uses tanh scaled to [0, 1])
        patch = patch.squeeze(0).cpu()  # [3, H, W]

    return patch


def evaluate_patch(patch, val_images, alpr, control_texts, center_ratio=0.6):
    """Evaluate a patch by computing edit distance between control and composite OCR.

    Args:
        patch: [3, H, W] tensor in [0, 1]
        val_images: List of validation image tensors [3, H, W] in [0, 1]
        alpr: fast-alpr instance for OCR
        control_texts: List of precomputed control OCR texts
        center_ratio: Center ratio for compositing

    Returns:
        Tuple of (total_edit_distance, num_misreads, avg_edit_distance)
    """
    total_edit_distance = 0
    misreads = 0
    num_evaluated = 0

    for val_image, control_text in zip(val_images, control_texts):
        try:
            # Create composite (with patch)
            composite = apply_patch_ocr_mode(val_image, patch, center_ratio=center_ratio)
            composite = composite.squeeze(0)  # Remove batch dim

            # Convert to numpy for OCR
            composite_np = (composite.permute(1, 2, 0).numpy() * 255).astype(np.uint8)

            # Run OCR on composite only (control already precomputed)
            composite_result = alpr.ocr.predict(composite_np)
            composite_text = composite_result.text if composite_result is not None else ""

            # Calculate Levenshtein edit distance
            edit_dist = Levenshtein.distance(control_text, composite_text)
            total_edit_distance += edit_dist

            # Count misread if texts differ
            if composite_text != control_text:
                misreads += 1

            num_evaluated += 1

        except Exception as e:
            # Treat errors as 0 edit distance (conservative)
            num_evaluated += 1
            pass

    avg_edit_distance = total_edit_distance / num_evaluated if num_evaluated > 0 else 0
    return total_edit_distance, misreads, avg_edit_distance


def evaluate_patch_with_debug(patch, val_images, alpr, control_texts, center_ratio=0.6, debug_dir=None, candidate_idx=0):
    """Evaluate a patch and save debug images for all validation samples.

    Args:
        patch: [3, H, W] tensor in [0, 1]
        val_images: List of validation image tensors [3, H, W] in [0, 1]
        alpr: fast-alpr instance for OCR
        control_texts: List of precomputed control OCR texts
        center_ratio: Center ratio for compositing
        debug_dir: Directory to save debug images
        candidate_idx: Index of the candidate being evaluated

    Returns:
        Tuple of (total_edit_distance, num_misreads, avg_edit_distance)
    """
    total_edit_distance = 0
    misreads = 0
    num_evaluated = 0
    debug_results = []

    for img_idx, (val_image, control_text) in enumerate(zip(val_images, control_texts)):
        try:
            # Create composite (with patch)
            composite = apply_patch_ocr_mode(val_image, patch, center_ratio=center_ratio)
            composite = composite.squeeze(0)  # Remove batch dim

            # Convert to numpy for OCR
            composite_np = (composite.permute(1, 2, 0).numpy() * 255).astype(np.uint8)

            # Run OCR on composite only (control already precomputed)
            composite_result = alpr.ocr.predict(composite_np)
            composite_text = composite_result.text if composite_result is not None else ""

            # Calculate Levenshtein edit distance
            edit_dist = Levenshtein.distance(control_text, composite_text)
            total_edit_distance += edit_dist

            # Count misread if texts differ
            is_misread = (composite_text != control_text)
            if is_misread:
                misreads += 1

            num_evaluated += 1

            # Save debug images
            if debug_dir is not None:
                # Save composite
                composite_bgr = cv2.cvtColor(composite_np, cv2.COLOR_RGB2BGR)
                comp_path = debug_dir / f"iter0_candidate{candidate_idx:02d}_img{img_idx:02d}_composite.jpg"
                cv2.imwrite(str(comp_path), composite_bgr)

                # Save control (regenerate for debug visualization)
                control = apply_neutral_border_ocr_mode(val_image, center_ratio=center_ratio, border_color=0.5)
                control = control.squeeze(0)
                control_np = (control.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                control_bgr = cv2.cvtColor(control_np, cv2.COLOR_RGB2BGR)
                ctrl_path = debug_dir / f"iter0_candidate{candidate_idx:02d}_img{img_idx:02d}_control.jpg"
                cv2.imwrite(str(ctrl_path), control_bgr)

                # Track results
                debug_results.append({
                    'img_idx': img_idx,
                    'control_text': control_text,
                    'composite_text': composite_text,
                    'edit_distance': edit_dist,
                    'is_misread': is_misread,
                })

        except Exception as e:
            # Treat errors as 0 edit distance (conservative)
            num_evaluated += 1
            pass

    # Save debug summary
    avg_edit_distance = total_edit_distance / num_evaluated if num_evaluated > 0 else 0
    if debug_dir is not None and debug_results:
        summary_path = debug_dir / f"iter0_candidate{candidate_idx:02d}_summary.txt"
        with open(summary_path, 'w') as f:
            f.write(f"Candidate {candidate_idx} Debug Summary\n")
            f.write(f"{'=' * 80}\n")
            f.write(f"Total edit distance: {total_edit_distance}\n")
            f.write(f"Average edit distance: {avg_edit_distance:.2f}\n")
            f.write(f"Total misreads: {misreads}/{len(val_images)} ({misreads/len(val_images)*100:.1f}%)\n\n")
            f.write(f"Per-image results:\n")
            for result in debug_results:
                status = "MISREAD" if result['is_misread'] else "MATCH"
                f.write(f"  Image {result['img_idx']:2d}: {status:7s} | EditDist: {result['edit_distance']:2d} | "
                       f"Control: '{result['control_text']:15s}' → "
                       f"Composite: '{result['composite_text']:15s}'\n")

    return total_edit_distance, misreads, avg_edit_distance


def main():
    parser = argparse.ArgumentParser(
        description='CMA-ES optimization of adversarial patches to maximize misreads.'
    )
    parser.add_argument('run_dir', help='Path to run directory with trained VAE')
    parser.add_argument('--csv', default=None,
                        help='Path to train_val_split CSV (default: auto-detect from run_dir)')
    parser.add_argument('--n-eval-samples', type=int, default=50,
                        help='Number of validation samples to evaluate on (default: 50)')
    parser.add_argument('--center-ratio', type=float, default=0.6,
                        help='Center ratio for compositing (default: 0.6)')

    # CMA-ES parameters
    parser.add_argument('--popsize', type=int, default=20,
                        help='CMA-ES population size (default: 20)')
    parser.add_argument('--maxiter', type=int, default=100,
                        help='CMA-ES maximum iterations (default: 100)')
    parser.add_argument('--sigma0', type=float, default=0.5,
                        help='CMA-ES initial standard deviation (default: 0.5)')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility (default: None)')

    # OCR model
    parser.add_argument('--white-box', action='store_true',
                        help='Use smaller xs model instead of s model')

    # Output
    parser.add_argument('--outdir', default='cmaes_output',
                        help='Output directory for results (default: cmaes_output)')

    # Mode selection
    parser.add_argument('--composite-only', action='store_true',
                        help='Composite mode: just composite patches from run_dir with n validation samples and save, then exit (no optimization)')

    args = parser.parse_args()

    # Check dependencies (CMA-ES only needed for optimization mode)
    if not args.composite_only:
        if ALPR is None:
            print("Error: fast-alpr not installed. Install with: pip install fast-alpr",
                  file=sys.stderr)
            sys.exit(1)

        if cma is None:
            print("Error: cma not installed. Install with: pip install cma",
                  file=sys.stderr)
            sys.exit(1)

    # Set random seed
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        print(f"Set random seed: {args.seed}")

    # Create output directory
    output_dir = Path(args.outdir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load run directory
    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        print(f"Error: Run directory not found: {run_dir}", file=sys.stderr)
        sys.exit(1)

    # Find CSV file
    if args.csv:
        csv_path = Path(args.csv)
    else:
        csv_files = list(run_dir.glob("**/*.csv"))
        csv_path = None
        for csv_file in csv_files:
            if 'train_val_split' in csv_file.name or 'split' in csv_file.name:
                csv_path = csv_file
                break

        if csv_path is None:
            # Try current directory as fallback
            cwd_csv = list(Path('.').glob("train_val_split_*.csv"))
            if cwd_csv:
                csv_path = cwd_csv[-1]

        if csv_path is None:
            print("Error: Could not find train_val_split CSV file", file=sys.stderr)
            sys.exit(1)

    print(f"Using data split: {csv_path}")

    # Load validation samples
    print(f"\nLoading {args.n_eval_samples} validation samples...")
    val_images, dimensions = load_validation_samples_from_csv(csv_path, args.n_eval_samples)

    if not val_images:
        print("Error: No validation samples loaded", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {len(val_images)} validation samples")

    # Load generator
    print(f"\nLoading generator from {run_dir}...")
    generator, latent_dim, device = load_generator(run_dir)

    # Composite-only mode: just generate and save composites, then exit
    if args.composite_only:
        print("\n" + "="*80)
        print("COMPOSITE-ONLY MODE")
        print("="*80)

        # Generate random latent codes for patches
        num_patches = args.popsize if args.popsize else 10
        print(f"\nGenerating {num_patches} random patches...")

        patches = []
        for i in range(num_patches):
            z = np.random.randn(latent_dim) * args.sigma0
            patch = generate_patch_from_z(generator, z, device)
            patches.append(patch)

        print(f"Generated {len(patches)} patches")

        # Create debug directory for quantization comparison
        debug_dir = output_dir / "quantization_debug"
        debug_dir.mkdir(parents=True, exist_ok=True)

        # Debug: save downscaling comparison for first patch only
        print(f"\nSaving quantization debug outputs for first patch...")
        if len(patches) > 0 and len(val_images) > 0:
            first_patch = patches[0]
            first_val_image = val_images[0]

            # Get image dimensions
            img_h, img_w = first_val_image.shape[1], first_val_image.shape[2]

            # Method 1: Direct downscale from float32 generator output
            patch_downscaled_float = F.interpolate(
                first_patch.unsqueeze(0),
                size=(img_h, img_w),
                mode='bilinear',
                align_corners=False
            ).squeeze(0)  # [3, H, W] float32

            # Save float-downscaled version
            float_np = (patch_downscaled_float.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            float_bgr = cv2.cvtColor(float_np, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(debug_dir / "patch_downscaled_float32.png"), float_bgr)

            # Method 2: Quantize to uint8 first, THEN downscale
            # Convert patch to uint8 PNG format
            patch_np = (first_patch.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            patch_bgr = cv2.cvtColor(patch_np, cv2.COLOR_RGB2BGR)

            # Save original patch
            cv2.imwrite(str(debug_dir / "patch_original.png"), patch_bgr)

            # Convert back to tensor (simulating PNG save/load)
            patch_rgb = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            patch_quantized = torch.from_numpy(np.transpose(patch_rgb, (2, 0, 1)))  # [3, H, W]

            # Now downscale the quantized version
            patch_downscaled_quantized = F.interpolate(
                patch_quantized.unsqueeze(0),
                size=(img_h, img_w),
                mode='bilinear',
                align_corners=False
            ).squeeze(0)  # [3, H, W]

            # Save quantized-downscaled version
            quant_np = (patch_downscaled_quantized.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            quant_bgr = cv2.cvtColor(quant_np, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(debug_dir / "patch_downscaled_quantized.png"), quant_bgr)

            # Debug: Composite a grey rectangle onto the downscaled patch
            # Create a grey rectangle (64x128 in center)
            grey_rect_h, grey_rect_w = 64, 128
            grey_rect = torch.full((3, grey_rect_h, grey_rect_w), 0.5, dtype=torch.float32)  # Mid-grey

            # Upscale grey rectangle to image size
            grey_rect_scaled = F.interpolate(
                grey_rect.unsqueeze(0),
                size=(img_h, img_w),
                mode='bilinear',
                align_corners=False
            ).squeeze(0)

            # Create mask for center region (60% like in apply_patch_ocr_mode)
            center_ratio = 0.6
            center_h = int(img_h * center_ratio)
            center_w = int(img_w * center_ratio)
            pad_h = (img_h - center_h) // 2
            pad_w = (img_w - center_w) // 2

            center_mask = torch.zeros(1, img_h, img_w, dtype=torch.float32)
            center_mask[:, pad_h:pad_h + center_h, pad_w:pad_w + center_w] = 1.0
            center_mask = center_mask.expand(3, -1, -1)  # [3, H, W]

            # Composite: keep original in center, grey rectangle on borders
            composite_grey = patch_downscaled_float * center_mask + grey_rect_scaled * (1 - center_mask)
            composite_grey = torch.clamp(composite_grey, 0, 1)

            # Save composited version
            comp_grey_np = (composite_grey.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            comp_grey_bgr = cv2.cvtColor(comp_grey_np, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(debug_dir / "patch_composited_grey_rect.png"), comp_grey_bgr)

            print(f"  Saved to {debug_dir}/")
            print(f"    - patch_original.png (original generator output)")
            print(f"    - patch_downscaled_float32.png (float → downscale)")
            print(f"    - patch_downscaled_quantized.png (float → uint8 → float → downscale)")
            print(f"    - patch_composited_grey_rect.png (composited with grey 64x128 rectangle on border)")

        # Composite patches with validation samples (ONLY ONE IMAGE PER PATCH)
        print(f"\nCompositing {len(patches)} patches with 1 validation sample each...")
        print(f"Total outputs: {len(patches)} composite + control pairs")

        pbar = tqdm(total=len(patches), desc="Compositing")

        for patch_idx, patch in enumerate(patches):
            # Use only first validation image
            val_image = val_images[0]

            try:
                # Apply patch
                composite = apply_patch_ocr_mode(val_image, patch, center_ratio=args.center_ratio)
                composite = composite.squeeze(0)  # Remove batch dim

                # Convert to numpy and save
                composite_np = (composite.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                composite_bgr = cv2.cvtColor(composite_np, cv2.COLOR_RGB2BGR)

                output_path = output_dir / f"composite_{patch_idx:04d}.jpg"
                cv2.imwrite(str(output_path), composite_bgr)

                # Apply grey control border (only save once)
                if patch_idx == 0:
                    control = apply_neutral_border_ocr_mode(val_image, center_ratio=args.center_ratio, border_color=0.5)
                    control = control.squeeze(0)  # Remove batch dim

                    # Convert to numpy and save
                    control_np = (control.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                    control_bgr = cv2.cvtColor(control_np, cv2.COLOR_RGB2BGR)

                    control_path = output_dir / f"control_0000.jpg"
                    cv2.imwrite(str(control_path), control_bgr)

            except Exception as e:
                print(f"  Error processing patch {patch_idx}: {e}", file=sys.stderr)

            pbar.update(1)

        pbar.close()
        print(f"\nSaved {len(patches)} composite images + 1 control to {output_dir}")
        print("Done!")
        return

    # Initialize ALPR
    print("\nInitializing fast-alpr (OCR-only mode)...")
    ocr_model = "cct-xs-v1-global-model" if args.white_box else "cct-s-v1-global-model"
    print(f"Using OCR model: {ocr_model}")
    alpr = ALPR(
        detector=None,
        ocr_model=ocr_model,
    )
    print("fast-alpr loaded")

    # Precompute control OCR outputs once (with grey border)
    print("\nPrecomputing control OCR outputs...")
    control_texts = []
    for val_image in tqdm(val_images, desc="Control OCR"):
        try:
            # Create control (with grey border)
            control = apply_neutral_border_ocr_mode(val_image, center_ratio=args.center_ratio, border_color=0.5)
            control = control.squeeze(0)  # Remove batch dim

            # Convert to numpy for OCR
            control_np = (control.permute(1, 2, 0).numpy() * 255).astype(np.uint8)

            # Run OCR
            control_result = alpr.ocr.predict(control_np)
            control_text = control_result.text if control_result is not None else ""
            control_texts.append(control_text)

        except Exception as e:
            # Treat errors as empty string
            control_texts.append("")

    print(f"Precomputed {len(control_texts)} control OCR outputs")

    # Create debug output directory
    debug_dir = output_dir / "debug_output"
    debug_dir.mkdir(parents=True, exist_ok=True)

    # Define objective function for CMA-ES (minimize negative edit distance)
    eval_count = [0]  # Track number of evaluations
    best_edit_distance = [0]
    best_avg_edit_distance = [0]
    best_misread_pct = [0]
    best_z = [None]
    current_iteration = [0]  # Track current iteration
    all_edit_distances = []  # Track all edit distances for progress bar

    def objective(z, candidate_idx=None):
        """Objective function: returns negative edit distance (CMA-ES minimizes)."""
        eval_count[0] += 1

        # Generate patch from z
        patch = generate_patch_from_z(generator, z, device)

        # For first iteration only, save debug output
        save_debug = (current_iteration[0] == 0 and candidate_idx is not None)

        if save_debug:
            # Save the patch itself
            patch_np = (patch.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            patch_bgr = cv2.cvtColor(patch_np, cv2.COLOR_RGB2BGR)
            patch_path = debug_dir / f"iter0_candidate{candidate_idx:02d}_patch.png"
            cv2.imwrite(str(patch_path), patch_bgr)

        # Evaluate on validation set with optional debug output
        if save_debug:
            total_edit_dist, misreads, avg_edit_dist = evaluate_patch_with_debug(
                patch, val_images, alpr, control_texts,
                center_ratio=args.center_ratio,
                debug_dir=debug_dir,
                candidate_idx=candidate_idx
            )
        else:
            total_edit_dist, misreads, avg_edit_dist = evaluate_patch(
                patch, val_images, alpr, control_texts,
                center_ratio=args.center_ratio
            )

        # Track for progress bar
        all_edit_distances.append(avg_edit_dist)

        # Track best
        if total_edit_dist > best_edit_distance[0]:
            best_edit_distance[0] = total_edit_dist
            best_avg_edit_distance[0] = avg_edit_dist
            best_misread_pct[0] = (misreads / len(val_images) * 100) if len(val_images) > 0 else 0
            best_z[0] = z.copy()

        # Return negative edit distance (CMA-ES minimizes, we want to maximize)
        return -total_edit_dist

    # Initialize CMA-ES
    print(f"\nInitializing CMA-ES:")
    print(f"  Latent dimension: {latent_dim}")
    print(f"  Population size: {args.popsize}")
    print(f"  Max iterations: {args.maxiter}")
    print(f"  Initial sigma: {args.sigma0}")

    x0 = np.zeros(latent_dim)  # Start from zero (neutral latent code)

    es = cma.CMAEvolutionStrategy(
        x0,
        args.sigma0,
        {
            'popsize': args.popsize,
            'maxiter': args.maxiter,
            'verb_disp': 1,
            'verb_log': 0,
        }
    )

    # Run optimization
    print(f"\nStarting CMA-ES optimization...")
    print(f"Evaluating on {len(val_images)} validation samples per iteration")
    print(f"Objective: Maximize Levenshtein edit distance between control and composite OCR")
    print("=" * 80)

    iteration = 0

    # Create progress bar for iterations
    pbar = tqdm(total=args.maxiter, desc="CMA-ES", unit="iter", position=0)

    while not es.stop():
        solutions = es.ask()
        fitness_values = []

        # Save debug output for first iteration only
        if iteration == 0:
            tqdm.write(f"[First iteration: saving debug output to {debug_dir}]")

        current_iteration[0] = iteration  # Update iteration counter for objective function

        # Clear edit distances for this iteration
        all_edit_distances.clear()

        # Evaluate all candidates with a nested progress bar
        for i, z in enumerate(solutions):
            # Pass candidate index only for first iteration
            fitness = objective(z, candidate_idx=i if iteration == 0 else None)
            fitness_values.append(fitness)

        es.tell(solutions, fitness_values)

        # Calculate iteration statistics
        avg_edit_dist = np.mean(all_edit_distances) if all_edit_distances else 0

        # Update progress bar with current metrics
        pbar.set_postfix({
            'best_edit': f'{best_avg_edit_distance[0]:.2f}',
            'avg_edit': f'{avg_edit_dist:.2f}',
            'misread%': f'{best_misread_pct[0]:.1f}%'
        })
        pbar.update(1)

        iteration += 1

    pbar.close()

    # Print final results
    print("\n" + "=" * 80)
    print("OPTIMIZATION COMPLETE")
    print("=" * 80)
    print(f"Total evaluations: {eval_count[0]}")
    print(f"Best total edit distance: {best_edit_distance[0]}")
    print(f"Best average edit distance: {best_avg_edit_distance[0]:.2f}")
    print(f"Best misread percentage: {best_misread_pct[0]:.1f}%")

    # Save best patch
    if best_z[0] is not None:
        best_patch = generate_patch_from_z(generator, best_z[0], device)

        # Save as PNG
        patch_np = (best_patch.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        patch_bgr = cv2.cvtColor(patch_np, cv2.COLOR_RGB2BGR)
        patch_path = output_dir / "best_patch.png"
        cv2.imwrite(str(patch_path), patch_bgr)
        print(f"\nSaved best patch to: {patch_path}")

        # Save latent code
        z_path = output_dir / "best_z.npy"
        np.save(z_path, best_z[0])
        print(f"Saved latent code to: {z_path}")

        # Save metadata
        metadata_path = output_dir / "optimization_results.txt"
        with open(metadata_path, 'w') as f:
            f.write(f"CMA-ES Optimization Results\n")
            f.write(f"{'=' * 80}\n")
            f.write(f"Timestamp: {datetime.now().isoformat()}\n\n")
            f.write(f"Run directory: {run_dir}\n")
            f.write(f"CSV file: {csv_path}\n\n")
            f.write(f"CMA-ES Parameters:\n")
            f.write(f"  Population size: {args.popsize}\n")
            f.write(f"  Max iterations: {args.maxiter}\n")
            f.write(f"  Initial sigma: {args.sigma0}\n")
            f.write(f"  Latent dimension: {latent_dim}\n\n")
            f.write(f"Evaluation:\n")
            f.write(f"  Validation samples: {len(val_images)}\n")
            f.write(f"  Center ratio: {args.center_ratio}\n")
            f.write(f"  OCR model: {ocr_model}\n")
            f.write(f"  Objective: Maximize Levenshtein edit distance\n\n")
            f.write(f"Results:\n")
            f.write(f"  Total evaluations: {eval_count[0]}\n")
            f.write(f"  Best total edit distance: {best_edit_distance[0]}\n")
            f.write(f"  Best average edit distance: {best_avg_edit_distance[0]:.2f}\n")
            f.write(f"  Best misread percentage: {best_misread_pct[0]:.1f}%\n")
            f.write(f"  Best z shape: {best_z[0].shape}\n")

        print(f"Saved metadata to: {metadata_path}")

    print(f"\nAll results saved to: {output_dir}")


if __name__ == '__main__':
    main()
