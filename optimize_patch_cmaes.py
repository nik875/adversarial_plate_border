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

    # Load all validation samples upfront (no downsampling - we'll sample per iteration)
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

    print(f"Loaded {len(images)} validation samples (will sample {min(num_samples, len(images))} per iteration)")
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


def load_generator(run_dir, device=None):
    """Load the FoundationPatchGenerator from the run directory.

    Args:
        run_dir: Path to run directory
        device: torch device (if None, auto-detect)

    Returns:
        Tuple of (generator model, latent_dim, device)
    """
    import sys
    from pathlib import Path

    # Import from progressive_patch.py (original architecture)
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

    # Use provided device or auto-detect
    if device is None:
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


def create_ocr_model(ocr_model_type, white_box=False):
    """Create OCR model based on type selection.

    Args:
        ocr_model_type: 'fast-alpr' or 'opencv-crnn'
        white_box: If True and using fast-alpr, use smaller xs model

    Returns:
        OCR model object with a predict(image) method
    """
    if ocr_model_type == 'fast-alpr':
        from fast_alpr import ALPR
        ocr_model_name = "cct-xs-v1-global-model" if white_box else "cct-s-v1-global-model"
        print(f"Initializing fast-alpr with model: {ocr_model_name}")
        alpr = ALPR(
            detector=None,
            ocr_model=ocr_model_name,
        )
        return alpr.ocr  # Return the OCR component

    elif ocr_model_type == 'opencv-crnn':
        # CRNN ONNX model converted to PyTorch
        print("Initializing CRNN OCR model (PyTorch from ONNX)...")
        try:
            import onnx
            from onnx2torch import convert

            crnn_model_path = Path("CRNN_VGG_BiLSTM_CTC.onnx")
            crnn_dict_path = Path("alphabet_36.txt")

            if not crnn_model_path.exists():
                raise FileNotFoundError(
                    f"CRNN model not found: {crnn_model_path}\n"
                    f"Please ensure the model is saved in the current directory."
                )

            if not crnn_dict_path.exists():
                raise FileNotFoundError(
                    f"Alphabet dictionary not found: {crnn_dict_path}\n"
                    f"Please ensure the alphabet file is in the current directory."
                )

            print(f"Loading and converting ONNX model to PyTorch...")
            # Load ONNX model
            onnx_model = onnx.load(str(crnn_model_path))

            # Convert to PyTorch
            print(f"  Converting model architecture...")
            pytorch_model = convert(onnx_model)
            pytorch_model.eval()

            with open(crnn_dict_path, 'r') as f:
                alphabet = f.read().strip()

            class CRNNWrapper:
                def __init__(self, model, alphabet):
                    self.model = model
                    self.alphabet = alphabet

                def predict(self, image):
                    """Predict text from image using CRNN."""
                    import cv2

                    # Convert to grayscale if needed
                    if len(image.shape) == 3:
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

                    # CRNN expects 32x128 input, normalize to [0, 1]
                    resized = cv2.resize(image, (128, 32))
                    normalized = resized.astype(np.float32) / 255.0

                    # Add batch and channel dimensions: [1, 1, 32, 128]
                    input_tensor = torch.from_numpy(normalized).unsqueeze(0).unsqueeze(0)

                    # Run inference
                    with torch.no_grad():
                        output = self.model(input_tensor)

                    # Handle output - could be tensor or tuple
                    if isinstance(output, tuple):
                        output = output[0]

                    # Decode output to text
                    output = output.squeeze(0)  # Remove batch dimension
                    if len(output.shape) > 1:
                        # [seq_len, num_classes]
                        indices = torch.argmax(output, dim=1).cpu().numpy()
                    else:
                        indices = np.array([torch.argmax(output).cpu().numpy()])

                    # Decode to text
                    text = ''
                    prev_idx = -1
                    for idx in indices:
                        if idx > 0 and idx != prev_idx:  # Skip blank (0)
                            if idx - 1 < len(self.alphabet):
                                text += self.alphabet[idx - 1]
                        prev_idx = idx

                    class Result:
                        def __init__(self, text):
                            self.text = text

                    return Result(text)

                def get_logits(self, image):
                    """Get raw logits from model for an image.

                    Args:
                        image: Input image in BGR format (as uint8 numpy array)

                    Returns:
                        logits: [seq_len, num_classes] numpy array of raw logits
                    """
                    import cv2

                    # Convert to grayscale if needed
                    if len(image.shape) == 3:
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

                    # CRNN expects 32x128 input, normalize to [0, 1]
                    resized = cv2.resize(image, (128, 32))
                    normalized = resized.astype(np.float32) / 255.0

                    # Add batch and channel dimensions: [1, 1, 32, 128]
                    input_tensor = torch.from_numpy(normalized).unsqueeze(0).unsqueeze(0)

                    # Run inference
                    with torch.no_grad():
                        output = self.model(input_tensor)

                    # Handle output - could be tensor or tuple
                    if isinstance(output, tuple):
                        output = output[0]

                    # Return raw logits as numpy array [seq_len, num_classes]
                    output = output.squeeze(0)  # Remove batch dimension
                    return output.cpu().numpy()

            crnn = CRNNWrapper(pytorch_model, alphabet)
            return crnn

        except ImportError as e:
            print(f"Error: Required package not installed: {e}", file=sys.stderr)
            print("Install with: pip install onnx onnx2torch", file=sys.stderr)
            sys.exit(1)
        except Exception as e:
            print(f"Error: Could not initialize CRNN: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
            sys.exit(1)

    else:
        raise ValueError(f"Unknown OCR model type: {ocr_model_type}")


def evaluate_patch(patch, val_images, ocr, control_texts, center_ratio=0.6):
    """Evaluate a patch by computing edit distance between control and composite OCR.

    Args:
        patch: [3, H, W] tensor in [0, 1]
        val_images: List of validation image tensors [3, H, W] in [0, 1]
        ocr: OCR model instance with predict(image) method
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
            composite_result = ocr.predict(composite_np)
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


def evaluate_patch_with_debug(patch, val_images, ocr, control_texts, center_ratio=0.6, debug_dir=None, candidate_idx=0):
    """Evaluate a patch and save debug images for all validation samples.

    Args:
        patch: [3, H, W] tensor in [0, 1]
        val_images: List of validation image tensors [3, H, W] in [0, 1]
        ocr: OCR model instance with predict(image) method
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
            composite_result = ocr.predict(composite_np)
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


def evaluate_patch_logit_delta(patch, val_images, ocr, control_logits_list, center_ratio=0.6):
    """Evaluate patch by measuring logit differences between control and composite (MSE).

    Args:
        patch: [3, H, W] tensor in [0, 1]
        val_images: List of validation image tensors [3, H, W] in [0, 1]
        ocr: OCR model instance with get_logits(image) method
        control_logits_list: List of precomputed control logits (numpy arrays)
        center_ratio: Center ratio for compositing

    Returns:
        Total MSE across all samples
    """
    total_mse = 0.0
    num_evaluated = 0

    for val_image, control_logits in zip(val_images, control_logits_list):
        try:
            # Create composite (with patch)
            composite = apply_patch_ocr_mode(val_image, patch, center_ratio=center_ratio)
            composite = composite.squeeze(0)  # Remove batch dim

            # Convert to numpy for OCR
            composite_np = (composite.permute(1, 2, 0).numpy() * 255).astype(np.uint8)

            # Get logits for composite
            composite_logits = ocr.get_logits(composite_np)

            # Compute MSE between control and composite logits
            # Handle cases where logits have different shapes (shouldn't happen, but be safe)
            min_len = min(control_logits.shape[0], composite_logits.shape[0])
            logit_diff = control_logits[:min_len] - composite_logits[:min_len]
            mse = np.mean(logit_diff ** 2)
            total_mse += mse

            num_evaluated += 1

        except Exception as e:
            # Treat errors as 0 MSE (conservative)
            num_evaluated += 1
            pass

    return total_mse if num_evaluated > 0 else 0.0


def evaluate_patch_logit_delta_with_debug(patch, val_images, ocr, control_logits_list, center_ratio=0.6, debug_dir=None, candidate_idx=0):
    """Evaluate patch by measuring logit MSE and save debug images.

    Args:
        patch: [3, H, W] tensor in [0, 1]
        val_images: List of validation image tensors [3, H, W] in [0, 1]
        ocr: OCR model instance with get_logits(image) method
        control_logits_list: List of precomputed control logits (numpy arrays)
        center_ratio: Center ratio for compositing
        debug_dir: Directory to save debug images
        candidate_idx: Index of the candidate being evaluated

    Returns:
        Total MSE across all samples
    """
    total_mse = 0.0
    num_evaluated = 0
    debug_results = []

    for img_idx, (val_image, control_logits) in enumerate(zip(val_images, control_logits_list)):
        try:
            # Create composite (with patch)
            composite = apply_patch_ocr_mode(val_image, patch, center_ratio=center_ratio)
            composite = composite.squeeze(0)  # Remove batch dim

            # Convert to numpy for OCR
            composite_np = (composite.permute(1, 2, 0).numpy() * 255).astype(np.uint8)

            # Get logits for composite
            composite_logits = ocr.get_logits(composite_np)

            # Compute MSE
            min_len = min(control_logits.shape[0], composite_logits.shape[0])
            logit_diff = control_logits[:min_len] - composite_logits[:min_len]
            mse = np.mean(logit_diff ** 2)
            total_mse += mse

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
                    'mse': mse,
                    'control_logit_shape': control_logits.shape,
                    'composite_logit_shape': composite_logits.shape,
                })

        except Exception as e:
            num_evaluated += 1
            pass

    # Save debug summary
    if debug_dir is not None and debug_results:
        summary_path = debug_dir / f"iter0_candidate{candidate_idx:02d}_summary.txt"
        with open(summary_path, 'w') as f:
            f.write(f"Candidate {candidate_idx} Debug Summary (Logit MSE Mode)\n")
            f.write(f"{'=' * 80}\n")
            f.write(f"Total MSE: {total_mse:.2f}\n\n")
            f.write(f"Per-image results:\n")
            for result in debug_results:
                f.write(f"  Image {result['img_idx']:2d}: MSE: {result['mse']:10.2f} | "
                       f"Shapes: {result['control_logit_shape']} vs {result['composite_logit_shape']}\n")

    return total_mse


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
    parser.add_argument('--ocr-model', choices=['fast-alpr', 'opencv-crnn'], default='fast-alpr',
                        help='OCR model to use (default: fast-alpr)')
    parser.add_argument('--white-box', action='store_true',
                        help='Use smaller xs model instead of s model (only for fast-alpr)')

    # Device
    parser.add_argument('--device', default=None,
                        help='Device to use (default: auto-detect cuda/cpu). Examples: cpu, cuda, cuda:0, cuda:1')

    # Output
    parser.add_argument('--outdir', default='cmaes_output',
                        help='Output directory for results (default: cmaes_output)')

    # Mode selection
    parser.add_argument('--composite-only', action='store_true',
                        help='Composite mode: just composite patches from run_dir with n validation samples and save, then exit (no optimization)')

    args = parser.parse_args()

    # Parse device
    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = None  # Will auto-detect in load_generator

    # Check dependencies (CMA-ES only needed for optimization mode)
    if not args.composite_only:
        # Check for OCR model dependencies
        if args.ocr_model == 'fast-alpr':
            if ALPR is None:
                print("Error: fast-alpr not installed. Install with: pip install fast-alpr",
                      file=sys.stderr)
                sys.exit(1)
        elif args.ocr_model == 'opencv-crnn':
            import onnx
            import onnx2torch

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
    generator, latent_dim, device = load_generator(run_dir, device=device)

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

        # Print generator output statistics
        if len(patches) > 0:
            first_patch = patches[0]
            print(f"\nGenerator output statistics (before any clamping):")
            print(f"  Min value: {first_patch.min().item():.6f}")
            print(f"  Max value: {first_patch.max().item():.6f}")
            print(f"  Mean value: {first_patch.mean().item():.6f}")
            print(f"  Values < 0: {(first_patch < 0).sum().item()} pixels")
            print(f"  Values > 1: {(first_patch > 1).sum().item()} pixels")

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

            # Debug: Composite the patch around a grey rectangle (like apply_patch_ocr_mode does)
            # Create a grey rectangle base image
            grey_base = torch.full((3, img_h, img_w), 0.5, dtype=torch.float32)  # Mid-grey

            # Create mask for center region (60% like in apply_patch_ocr_mode)
            center_ratio = 0.6
            center_h = int(img_h * center_ratio)
            center_w = int(img_w * center_ratio)
            pad_h = (img_h - center_h) // 2
            pad_w = (img_w - center_w) // 2

            center_mask = torch.zeros(1, img_h, img_w, dtype=torch.float32)
            center_mask[:, pad_h:pad_h + center_h, pad_w:pad_w + center_w] = 1.0
            center_mask = center_mask.expand(3, -1, -1)  # [3, H, W]

            # Composite: grey rectangle in center, patch on borders
            composite_grey = grey_base * center_mask + patch_downscaled_float * (1 - center_mask)
            composite_grey = torch.clamp(composite_grey, 0, 1)

            # Save composited version
            comp_grey_np = (composite_grey.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            comp_grey_bgr = cv2.cvtColor(comp_grey_np, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(debug_dir / "patch_composited_grey_rect.png"), comp_grey_bgr)

            # Debug 3: Composite grey rectangle on ORIGINAL patch (no downscaling)
            # Get original patch dimensions
            patch_h, patch_w = first_patch.shape[1], first_patch.shape[2]

            # Create grey base at original patch size
            grey_base_orig = torch.full((3, patch_h, patch_w), 0.5, dtype=torch.float32)

            # Create mask for center region at original size
            center_h_orig = int(patch_h * center_ratio)
            center_w_orig = int(patch_w * center_ratio)
            pad_h_orig = (patch_h - center_h_orig) // 2
            pad_w_orig = (patch_w - center_w_orig) // 2

            center_mask_orig = torch.zeros(1, patch_h, patch_w, dtype=torch.float32)
            center_mask_orig[:, pad_h_orig:pad_h_orig + center_h_orig, pad_w_orig:pad_w_orig + center_w_orig] = 1.0
            center_mask_orig = center_mask_orig.expand(3, -1, -1)

            # Composite at original resolution
            composite_orig = grey_base_orig * center_mask_orig + first_patch * (1 - center_mask_orig)
            composite_orig = torch.clamp(composite_orig, 0, 1)

            # Save
            comp_orig_np = (composite_orig.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            comp_orig_bgr = cv2.cvtColor(comp_orig_np, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(debug_dir / "patch_composited_original_res.png"), comp_orig_bgr)

            # Debug 4: Downscale patch to 64x128, then composite small grey square off-center
            patch_small = F.interpolate(
                first_patch.unsqueeze(0),
                size=(64, 128),
                mode='bilinear',
                align_corners=False
            ).squeeze(0)  # [3, 64, 128]

            # Create small grey square (32x32) at position (10, 20) - top-left offset
            grey_square = torch.full((3, 64, 128), 0.0, dtype=torch.float32)  # Start with zeros
            square_size = 32
            offset_y, offset_x = 10, 20
            grey_square[:, offset_y:offset_y+square_size, offset_x:offset_x+square_size] = 0.5

            # Create mask for the square
            square_mask = torch.zeros(1, 64, 128, dtype=torch.float32)
            square_mask[:, offset_y:offset_y+square_size, offset_x:offset_x+square_size] = 1.0
            square_mask = square_mask.expand(3, -1, -1)

            # Composite: grey square overlaid on patch
            composite_small = patch_small * (1 - square_mask) + grey_square * square_mask
            composite_small = torch.clamp(composite_small, 0, 1)

            # Save
            comp_small_np = (composite_small.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            comp_small_bgr = cv2.cvtColor(comp_small_np, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(debug_dir / "patch_64x128_with_grey_square.png"), comp_small_bgr)

            # Debug 5: Composite a single white pixel at top-left corner on downscaled patch
            patch_with_pixel = patch_downscaled_float.clone()  # Clone to avoid modifying original

            # Set top-left pixel to white (1.0 in all channels)
            patch_with_pixel[:, 0, 0] = 1.0

            # Save
            pixel_np = (patch_with_pixel.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            pixel_bgr = cv2.cvtColor(pixel_np, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(debug_dir / "patch_downscaled_with_white_pixel.png"), pixel_bgr)

            # Debug 6: Composite 1x1 white rectangle at top-left using EXACT compositing logic
            # Create white 1x1 "image"
            white_pixel = torch.ones((3, img_h, img_w), dtype=torch.float32)

            # Create mask for single pixel at [0, 0]
            pixel_mask = torch.zeros(1, img_h, img_w, dtype=torch.float32)
            pixel_mask[:, 0, 0] = 1.0
            pixel_mask = pixel_mask.expand(3, -1, -1)  # [3, H, W]

            # Composite: white pixel at [0,0], patch everywhere else
            composite_1x1 = patch_downscaled_float * (1 - pixel_mask) + white_pixel * pixel_mask
            composite_1x1 = torch.clamp(composite_1x1, 0, 1)

            # Save
            comp_1x1_np = (composite_1x1.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            comp_1x1_bgr = cv2.cvtColor(comp_1x1_np, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(debug_dir / "patch_composited_1x1_white.png"), comp_1x1_bgr)

            # Debug 7: Identity operation - composite patch with all-zeros using all-ones mask
            # This should return the patch unchanged, but let's see if mask ops introduce artifacts
            zeros_image = torch.zeros((3, img_h, img_w), dtype=torch.float32)
            ones_mask = torch.ones((3, img_h, img_w), dtype=torch.float32)

            # Composite: patch everywhere (mask=1), zeros nowhere (mask=0 would show zeros)
            # Result should be identical to patch_downscaled_float
            identity_composite = zeros_image * (1 - ones_mask) + patch_downscaled_float * ones_mask
            identity_composite = torch.clamp(identity_composite, 0, 1)

            # Save
            identity_np = (identity_composite.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            identity_bgr = cv2.cvtColor(identity_np, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(debug_dir / "patch_identity_composite.png"), identity_bgr)

            # Debug 8: Just clamp the downscaled patch, no compositing at all
            patch_clamped = torch.clamp(patch_downscaled_float, 0, 1)

            clamped_np = (patch_clamped.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            clamped_bgr = cv2.cvtColor(clamped_np, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(debug_dir / "patch_downscaled_clamped.png"), clamped_bgr)

            print(f"  Saved to {debug_dir}/")
            print(f"    - patch_original.png (original generator output)")
            print(f"    - patch_downscaled_float32.png (float → downscale)")
            print(f"    - patch_downscaled_quantized.png (float → uint8 → float → downscale)")
            print(f"    - patch_composited_grey_rect.png (patch on border, grey in center, downscaled)")
            print(f"    - patch_composited_original_res.png (patch on border, grey in center, NO downscale)")
            print(f"    - patch_64x128_with_grey_square.png (patch @ 64x128 with 32x32 grey square at offset)")
            print(f"    - patch_downscaled_with_white_pixel.png (downscaled patch + 1 white pixel at [0,0])")
            print(f"    - patch_composited_1x1_white.png (composited 1x1 white rect at [0,0] using mask logic)")
            print(f"    - patch_identity_composite.png (zeros * 0 + patch * 1 - should be identical to patch)")

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

    # Initialize OCR model
    print(f"\nInitializing OCR model: {args.ocr_model}")
    ocr = create_ocr_model(args.ocr_model, white_box=args.white_box)
    print("OCR model loaded")

    # Precompute control OCR outputs once (with grey border)
    # For opencv-crnn, precompute logits; for others, precompute text
    use_logit_mse = (args.ocr_model == 'opencv-crnn')

    if use_logit_mse:
        print("\nPrecomputing control logits (logit MSE mode)...")
        control_logits_list = []
        for val_image in tqdm(val_images, desc="Control logits"):
            try:
                # Create control (with grey border)
                control = apply_neutral_border_ocr_mode(val_image, center_ratio=args.center_ratio, border_color=0.5)
                control = control.squeeze(0)  # Remove batch dim

                # Convert to numpy for OCR
                control_np = (control.permute(1, 2, 0).numpy() * 255).astype(np.uint8)

                # Get logits
                control_logits = ocr.get_logits(control_np)
                control_logits_list.append(control_logits)

            except Exception as e:
                # Treat errors as zeros with default shape
                control_logits_list.append(np.zeros((26, 37)))  # Default CRNN output shape

        print(f"Precomputed {len(control_logits_list)} control logits")
        control_data = control_logits_list
    else:
        print("\nPrecomputing control OCR outputs (text mode)...")
        control_texts = []
        for val_image in tqdm(val_images, desc="Control OCR"):
            try:
                # Create control (with grey border)
                control = apply_neutral_border_ocr_mode(val_image, center_ratio=args.center_ratio, border_color=0.5)
                control = control.squeeze(0)  # Remove batch dim

                # Convert to numpy for OCR
                control_np = (control.permute(1, 2, 0).numpy() * 255).astype(np.uint8)

                # Run OCR
                control_result = ocr.predict(control_np)
                control_text = control_result.text if control_result is not None else ""
                control_texts.append(control_text)

            except Exception as e:
                # Treat errors as empty string
                control_texts.append("")

        print(f"Precomputed {len(control_texts)} control OCR outputs")
        control_data = control_texts

    # Create debug output directory
    debug_dir = output_dir / "debug_output"
    debug_dir.mkdir(parents=True, exist_ok=True)

    # Define objective function for CMA-ES
    eval_count = [0]  # Track number of evaluations
    best_metric = [0]  # Best edit distance or logit delta (depending on mode)
    best_avg_metric = [0]  # Average metric
    best_misread_pct = [0]  # Only used in text mode
    best_z = [None]
    current_iteration = [0]  # Track current iteration
    all_metrics = []  # Track all metrics for progress bar
    sampled_indices = []  # Will hold random sample indices for current iteration
    sampled_val_images = []  # Will hold sampled validation images
    sampled_control_data = []  # Will hold sampled control texts or logits

    def objective(z, candidate_idx=None):
        """Objective function: returns negative metric (CMA-ES minimizes).

        Metric is either edit distance (text mode) or logit delta (logit delta mode).
        """
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

        # Evaluate on sampled validation set with optional debug output
        if use_logit_mse:
            # Logit delta mode (opencv-crnn only)
            if save_debug:
                total_metric = evaluate_patch_logit_delta_with_debug(
                    patch, sampled_val_images, ocr, sampled_control_data,
                    center_ratio=args.center_ratio,
                    debug_dir=debug_dir,
                    candidate_idx=candidate_idx
                )
            else:
                total_metric = evaluate_patch_logit_delta(
                    patch, sampled_val_images, ocr, sampled_control_data,
                    center_ratio=args.center_ratio
                )
            avg_metric = total_metric / len(sampled_val_images) if len(sampled_val_images) > 0 else 0
            misreads = 0  # Not applicable in logit delta mode
        else:
            # Text edit distance mode (default)
            if save_debug:
                total_metric, misreads, avg_metric = evaluate_patch_with_debug(
                    patch, sampled_val_images, ocr, sampled_control_data,
                    center_ratio=args.center_ratio,
                    debug_dir=debug_dir,
                    candidate_idx=candidate_idx
                )
            else:
                total_metric, misreads, avg_metric = evaluate_patch(
                    patch, sampled_val_images, ocr, sampled_control_data,
                    center_ratio=args.center_ratio
                )

        # Track for progress bar
        all_metrics.append(avg_metric)

        # Track best
        if total_metric > best_metric[0]:
            best_metric[0] = total_metric
            best_avg_metric[0] = avg_metric
            if not use_logit_mse:
                best_misread_pct[0] = (misreads / len(sampled_val_images) * 100) if len(sampled_val_images) > 0 else 0
            best_z[0] = z.copy()

        # Return negative metric (CMA-ES minimizes, we want to maximize)
        return -total_metric

    # Initialize CMA-ES
    print(f"\nInitializing CMA-ES:")
    print(f"  Latent dimension: {latent_dim}")
    print(f"  Population size: {args.popsize}")
    print(f"  Max iterations: {args.maxiter}")
    print(f"  Initial sigma: {args.sigma0}")
    if use_logit_mse:
        print(f"  Mode: Logit MSE optimization (opencv-crnn)")
    else:
        print(f"  Mode: Text edit distance optimization")

    x0 = np.zeros(latent_dim)  # Start from zero (neutral latent code)

    es = cma.CMAEvolutionStrategy(
        x0,
        args.sigma0,
        {
            'popsize': args.popsize,
            'maxiter': args.maxiter,
            'verb_disp': 1,
            'verb_log': 0,
            'tolstagnation': np.inf,      # Disable stagnation (plateau) early stopping
            'tolfun': -np.inf,             # Disable function value convergence (set very low)
            'tolflatfitness': np.inf,     # Disable flat fitness early stopping
            'tolxstagnation': np.inf,     # Disable x-space stagnation
            'tolx': -np.inf,               # Disable x convergence
        }
    )

    # Run optimization
    print(f"\nStarting CMA-ES optimization...")
    num_samples_to_use = min(args.n_eval_samples, len(val_images))
    print(f"Full validation set: {len(val_images)} samples")
    print(f"Sampling {num_samples_to_use} samples randomly each iteration (different subset per iteration)")
    if use_logit_mse:
        print(f"Objective: Maximize logit MSE (mean squared error of logits)")
    else:
        print(f"Objective: Maximize Levenshtein edit distance between control and composite OCR")
    print("=" * 80)

    iteration = 0

    # Create progress bar for iterations
    pbar = tqdm(total=args.maxiter, desc="CMA-ES", unit="iter", position=0)

    # Run for exactly maxiter iterations (disable all early stopping criteria)
    while iteration < args.maxiter:
        solutions = es.ask()
        fitness_values = []

        # Save debug output for first iteration only
        if iteration == 0:
            tqdm.write(f"[First iteration: saving debug output to {debug_dir}]")

        current_iteration[0] = iteration  # Update iteration counter for objective function

        # Randomly sample validation subset for this iteration
        num_samples_to_use = min(args.n_eval_samples, len(val_images))
        sampled_indices = random.sample(range(len(val_images)), num_samples_to_use)
        sampled_val_images = [val_images[i] for i in sampled_indices]
        sampled_control_data = [control_data[i] for i in sampled_indices]

        # Clear metrics for this iteration
        all_metrics.clear()

        # Evaluate all candidates with a nested progress bar
        for i, z in enumerate(solutions):
            # Pass candidate index only for first iteration
            fitness = objective(z, candidate_idx=i if iteration == 0 else None)
            fitness_values.append(fitness)

        es.tell(solutions, fitness_values)

        # Calculate iteration statistics
        avg_metric = np.mean(all_metrics) if all_metrics else 0

        # Update progress bar with current metrics
        if use_logit_mse:
            pbar.set_postfix({
                'best_mse': f'{best_avg_metric[0]:.2f}',
                'avg_mse': f'{avg_metric:.2f}',
            })
        else:
            pbar.set_postfix({
                'best_edit': f'{best_avg_metric[0]:.2f}',
                'avg_edit': f'{avg_metric:.2f}',
                'misread%': f'{best_misread_pct[0]:.1f}%'
            })
        pbar.update(1)

        iteration += 1

    pbar.close()

    # Print final results
    print("\n" + "=" * 80)
    print("OPTIMIZATION COMPLETE")
    print("=" * 80)
    print(f"Completed {iteration} iterations (maxiter={args.maxiter})")
    print(f"Total evaluations: {eval_count[0]}")
    if use_logit_mse:
        print(f"Best total logit MSE: {best_metric[0]:.2f}")
        print(f"Best average logit MSE: {best_avg_metric[0]:.2f}")
    else:
        print(f"Best total edit distance: {best_metric[0]:.2f}")
        print(f"Best average edit distance: {best_avg_metric[0]:.2f}")
        print(f"Best misread percentage: {best_misread_pct[0]:.1f}%")

    # Save best patch
    if best_z[0] is not None:
        best_patch = generate_patch_from_z(generator, best_z[0], device)

        # Clamp to [0, 1] to match compositing behavior
        best_patch_clamped = torch.clamp(best_patch, 0, 1)

        # Save as PNG
        patch_np = (best_patch_clamped.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
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
            f.write(f"  OCR model type: {args.ocr_model}\n")
            f.write(f"  Device: {device}\n")
            if use_logit_mse:
                f.write(f"  Objective: Maximize logit MSE (mean squared error of logits)\n\n")
            else:
                f.write(f"  Objective: Maximize Levenshtein edit distance\n\n")
            f.write(f"Results:\n")
            f.write(f"  Total evaluations: {eval_count[0]}\n")
            if use_logit_mse:
                f.write(f"  Best total logit MSE: {best_metric[0]:.2f}\n")
                f.write(f"  Best average logit MSE: {best_avg_metric[0]:.2f}\n")
            else:
                f.write(f"  Best total edit distance: {best_metric[0]:.2f}\n")
                f.write(f"  Best average edit distance: {best_avg_metric[0]:.2f}\n")
                f.write(f"  Best misread percentage: {best_misread_pct[0]:.1f}%\n")
            f.write(f"  Best z shape: {best_z[0].shape}\n")

        print(f"Saved metadata to: {metadata_path}")

    print(f"\nAll results saved to: {output_dir}")


if __name__ == '__main__':
    main()
