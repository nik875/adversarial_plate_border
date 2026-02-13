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

    # Find the latest checkpoint
    checkpoint_dir = Path(run_dir)
    checkpoint_files = sorted(checkpoint_dir.glob("generator_epoch_*.pt"))

    if not checkpoint_files:
        raise FileNotFoundError(f"No generator checkpoint files found in {run_dir}")

    latest_checkpoint = checkpoint_files[-1]
    print(f"Loading checkpoint: {latest_checkpoint}")

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


def evaluate_patch(patch, val_images, alpr, center_ratio=0.6):
    """Evaluate a patch by counting misreads on validation images.

    Args:
        patch: [3, H, W] tensor in [0, 1]
        val_images: List of validation image tensors [3, H, W] in [0, 1]
        alpr: fast-alpr instance for OCR
        center_ratio: Center ratio for compositing

    Returns:
        num_misreads: Number of images where composite OCR differs from control OCR
    """
    misreads = 0

    for val_image in val_images:
        try:
            # Create composite (with patch)
            composite = apply_patch_ocr_mode(val_image, patch, center_ratio=center_ratio)
            composite = composite.squeeze(0)  # Remove batch dim

            # Create control (with grey border)
            control = apply_neutral_border_ocr_mode(val_image, center_ratio=center_ratio, border_color=0.5)
            control = control.squeeze(0)  # Remove batch dim

            # Convert to numpy for OCR
            composite_np = (composite.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            control_np = (control.permute(1, 2, 0).numpy() * 255).astype(np.uint8)

            # Run OCR
            composite_result = alpr.ocr.predict(composite_np)
            control_result = alpr.ocr.predict(control_np)

            composite_text = composite_result.text if composite_result is not None else ""
            control_text = control_result.text if control_result is not None else ""

            # Count misread if texts differ
            if composite_text != control_text:
                misreads += 1

        except Exception as e:
            # Treat errors as non-misreads (conservative)
            pass

    return misreads


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

    args = parser.parse_args()

    # Check dependencies
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

    # Initialize ALPR
    print("\nInitializing fast-alpr (OCR-only mode)...")
    ocr_model = "cct-xs-v1-global-model" if args.white_box else "cct-s-v1-global-model"
    print(f"Using OCR model: {ocr_model}")
    alpr = ALPR(
        detector=None,
        ocr_model=ocr_model,
    )
    print("fast-alpr loaded")

    # Define objective function for CMA-ES (minimize negative misreads)
    eval_count = [0]  # Track number of evaluations
    best_score = [0]
    best_z = [None]

    def objective(z):
        """Objective function: returns negative misreads (CMA-ES minimizes)."""
        eval_count[0] += 1

        # Generate patch from z
        patch = generate_patch_from_z(generator, z, device)

        # Evaluate on validation set
        misreads = evaluate_patch(patch, val_images, alpr, center_ratio=args.center_ratio)

        # Track best
        if misreads > best_score[0]:
            best_score[0] = misreads
            best_z[0] = z.copy()
            print(f"  New best: {misreads}/{len(val_images)} misreads (eval {eval_count[0]})")

        # Return negative (CMA-ES minimizes)
        return -misreads

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
    print("=" * 80)

    iteration = 0
    while not es.stop():
        solutions = es.ask()
        fitness_values = []

        print(f"\nIteration {iteration + 1}/{args.maxiter}")
        print(f"  Evaluating {len(solutions)} candidates...")

        for i, z in enumerate(solutions):
            fitness = objective(z)
            fitness_values.append(fitness)

        es.tell(solutions, fitness_values)

        # Print iteration summary
        avg_fitness = np.mean(fitness_values)
        min_fitness = np.min(fitness_values)
        print(f"  Avg misreads: {-avg_fitness:.1f}/{len(val_images)}")
        print(f"  Best this iter: {-min_fitness:.0f}/{len(val_images)}")
        print(f"  Best overall: {best_score[0]}/{len(val_images)}")

        iteration += 1

    # Print final results
    print("\n" + "=" * 80)
    print("OPTIMIZATION COMPLETE")
    print("=" * 80)
    print(f"Total evaluations: {eval_count[0]}")
    print(f"Best score: {best_score[0]}/{len(val_images)} misreads ({best_score[0]/len(val_images)*100:.1f}%)")

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
            f.write(f"  OCR model: {ocr_model}\n\n")
            f.write(f"Results:\n")
            f.write(f"  Total evaluations: {eval_count[0]}\n")
            f.write(f"  Best score: {best_score[0]}/{len(val_images)} misreads ({best_score[0]/len(val_images)*100:.1f}%)\n")
            f.write(f"  Best z shape: {best_z[0].shape}\n")

        print(f"Saved metadata to: {metadata_path}")

    print(f"\nAll results saved to: {output_dir}")


if __name__ == '__main__':
    main()
