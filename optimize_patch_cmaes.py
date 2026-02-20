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
import shutil
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

try:
    import openai
except ImportError:
    openai = None


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
    for combined_idx in tqdm(val_indices, desc="Loading samples"):
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


def load_validation_samples_from_preproc_csv(csv_path, num_samples):
    """Load validation samples from a preproc_labels CSV using AdversarialPatchDataset.

    Args:
        csv_path: Path to preproc_labels CSV (dataset.py format)
        num_samples: Number of samples to load (randomly sampled if dataset is larger)

    Returns:
        Tuple of (list of images as tensors [3, H, W] in [0, 1] RGB, list of (width, height) tuples)
    """
    import pandas as pd
    import torchvision.transforms as T

    script_dir = Path(__file__).parent
    sys.path.insert(0, str(script_dir))
    try:
        from dataset import AdversarialPatchDataset
    except ImportError:
        raise ImportError("Could not import AdversarialPatchDataset from dataset.py")

    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows from {csv_path}")

    # dataset.py loads images as BGR numpy arrays via cv2; convert to RGB for OCR models
    transform = T.Compose([
        T.Lambda(lambda x: cv2.cvtColor(x, cv2.COLOR_BGR2RGB)),
        T.ToTensor(),
    ])

    dataset = AdversarialPatchDataset(df, transform=transform)

    images = []
    dimensions = []
    print(f"Loading all {len(dataset)} samples from preproc dataset...")
    for idx in tqdm(range(len(dataset)), desc="Loading samples"):
        item = dataset[idx]
        img_tensor = item['prep_image']  # [3, H, W] float in [0, 1]
        height, width = img_tensor.shape[1], img_tensor.shape[2]
        images.append(img_tensor)
        dimensions.append((width, height))

    print(f"Loaded {len(images)} samples (will sample {min(num_samples, len(images))} per iteration)")
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
    use_omniglot = checkpoint.get('use_omniglot', False)

    print(f"  Latent dim: {latent_dim}")
    print(f"  Patch size: {patch_height}x{patch_width}")
    print(f"  VAE LoRA: {use_vae_lora} (rank={lora_rank}, alpha={lora_alpha})")
    print(f"  Omniglot conditioning: {use_omniglot}")

    # Create generator with same architecture
    generator = FoundationPatchGenerator(
        latent_dim=latent_dim,
        patch_height=patch_height,
        patch_width=patch_width,
        use_vae_lora=use_vae_lora,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        use_omniglot=use_omniglot,
    )

    # Load state dict, falling back to use_omniglot=False if there's a mismatch
    try:
        generator.load_state_dict(checkpoint['generator_state_dict'])
    except RuntimeError as e:
        if use_omniglot:
            print(f"  Warning: state dict mismatch with use_omniglot=True, retrying with use_omniglot=False")
            print(f"  ({e})")
            use_omniglot = False
            generator = FoundationPatchGenerator(
                latent_dim=latent_dim,
                patch_height=patch_height,
                patch_width=patch_width,
                use_vae_lora=use_vae_lora,
                lora_rank=lora_rank,
                lora_alpha=lora_alpha,
                use_omniglot=False,
            )
            generator.load_state_dict(checkpoint['generator_state_dict'])
        else:
            raise

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


def create_ocr_model(ocr_model_type, white_box=False, device=None, api_key=None):
    """Create OCR model based on type selection.

    Args:
        ocr_model_type: 'fast-alpr', 'opencv-crnn', 'vitstr', or 'trocr'
        white_box: If True and using fast-alpr, use smaller xs model
        device: torch device for vitstr/trocr (if None, auto-detect)

    Returns:
        OCR model object with a predict(image) method
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
            import onnxruntime

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

            print(f"Loading ONNX model with ONNX Runtime...")
            # Create ONNX Runtime session
            session = onnxruntime.InferenceSession(str(crnn_model_path), providers=['CPUExecutionProvider'])

            # Print model input/output info for debugging
            inputs = session.get_inputs()
            outputs = session.get_outputs()
            print(f"  Model inputs:")
            for inp in inputs:
                print(f"    {inp.name}: shape={inp.shape}, type={inp.type}")
            print(f"  Model outputs:")
            for out in outputs:
                print(f"    {out.name}: shape={out.shape}, type={out.type}")

            input_name = inputs[0].name

            with open(crnn_dict_path, 'r') as f:
                alphabet = f.read().strip()

            class CRNNWrapper:
                def __init__(self, session, alphabet, input_name, input_shape):
                    self.session = session
                    self.alphabet = alphabet
                    self.input_name = input_name
                    # input_shape is [batch, channels, height, width]
                    self.input_height = input_shape[2]
                    self.input_width = input_shape[3]

                def _preprocess(self, image):
                    """Preprocess image for CRNN inference."""
                    import cv2

                    # Convert to grayscale if needed
                    if len(image.shape) == 3:
                        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

                    # Resize to match model's expected input (height x width from ONNX shape)
                    resized = cv2.resize(image, (self.input_width, self.input_height))
                    normalized = resized.astype(np.float32) / 255.0

                    # Add batch and channel dimensions: [1, 1, H, W]
                    return normalized[np.newaxis, np.newaxis, :, :]

                def _run(self, input_data):
                    """Run inference and return raw logits."""
                    outputs = self.session.run(None, {self.input_name: input_data})
                    return outputs[0].squeeze(1)  # [24, 1, 37] -> [24, 37] = [seq_len, num_classes]

                def _decode(self, logits):
                    """Decode logits to text using CTC decoding."""
                    indices = np.argmax(logits, axis=1)
                    text = ''
                    prev_idx = -1
                    for idx in indices:
                        if idx > 0 and idx != prev_idx:  # Skip blank (0)
                            if idx - 1 < len(self.alphabet):
                                text += self.alphabet[idx - 1]
                        prev_idx = idx
                    return text

                def predict(self, image):
                    """Predict text from image using CRNN."""
                    input_data = self._preprocess(image)
                    logits = self._run(input_data)
                    text = self._decode(logits)

                    class Result:
                        def __init__(self, text):
                            self.text = text

                    return Result(text)

                def get_logits(self, image):
                    """Get raw logits from model for an image.

                    Args:
                        image: Input image in RGB format (as uint8 numpy array)

                    Returns:
                        logits: [seq_len, num_classes] numpy array of raw logits
                    """
                    input_data = self._preprocess(image)
                    return self._run(input_data)

            input_shape = inputs[0].shape
            crnn = CRNNWrapper(session, alphabet, input_name, input_shape)

            # Smoke test: run on a dummy image and print logit stats
            dummy = np.zeros((input_shape[2], input_shape[3]), dtype=np.uint8)
            dummy_logits = crnn.get_logits(dummy)
            print(f"  Smoke test - logit shape: {dummy_logits.shape}, "
                  f"min: {dummy_logits.min():.4f}, max: {dummy_logits.max():.4f}, "
                  f"mean: {dummy_logits.mean():.4f}")
            return crnn

        except ImportError as e:
            print(f"Error: Required package not installed: {e}", file=sys.stderr)
            print("Install with: pip install onnxruntime", file=sys.stderr)
            sys.exit(1)
        except Exception as e:
            print(f"Error: Could not initialize CRNN: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
            sys.exit(1)

    elif ocr_model_type == 'vitstr':
        print("Initializing ViTSTR model (doctr)...")
        from doctr.models import vitstr_small

        model = vitstr_small(pretrained=True)
        model.eval()
        model = model.to(device)
        print(f"  Loaded vitstr_small on {device}")

        class ViTSTRWrapper:
            def __init__(self, model, device):
                self.model = model
                self.device = device

            def _preprocess(self, image):
                """Preprocess RGB uint8 image for ViTSTR."""
                resized = cv2.resize(image, (128, 32))  # (W, H)
                tensor = torch.from_numpy(resized).float() / 255.0
                tensor = tensor.permute(2, 0, 1)  # [H, W, 3] -> [3, H, W]
                return tensor.unsqueeze(0).to(self.device)  # [1, 3, 32, 128]

            def get_logits(self, image):
                """Return raw model output as numpy array.

                Uses return_model_output=True in eval mode to get 'out_map'
                (raw logits) without needing labels or switching to train mode.
                """
                input_tensor = self._preprocess(image)
                with torch.no_grad():
                    out = self.model(input_tensor, return_model_output=True)
                if isinstance(out, dict):
                    # 'out_map' contains raw logits before postprocessor
                    if "out_map" in out:
                        logits = out["out_map"]
                    elif "logits" in out:
                        logits = out["logits"]
                    else:
                        logits = list(out.values())[0]
                elif isinstance(out, torch.Tensor):
                    logits = out
                else:
                    raise RuntimeError(f"Unexpected ViTSTR output type: {type(out)}")
                return logits.squeeze(0).cpu().numpy()

            def predict(self, image):
                """Predict text using the model in eval mode."""
                input_tensor = self._preprocess(image)
                with torch.no_grad():
                    out = self.model(input_tensor)
                # In eval mode, doctr returns a dict with 'preds': [(text, conf), ...]
                if isinstance(out, dict):
                    preds = out.get("preds", [])
                    text = preds[0][0] if preds and preds[0] else ""
                elif isinstance(out, list):
                    text = out[0][0] if out and out[0] else ""
                else:
                    text = ""

                class Result:
                    def __init__(self, text):
                        self.text = text
                return Result(text)

        return ViTSTRWrapper(model, device)

    elif ocr_model_type == 'trocr':
        print("Initializing TrOCR model (microsoft/trocr-small-printed)...")
        from transformers import VisionEncoderDecoderModel, TrOCRProcessor
        from PIL import Image as PILImage

        processor = TrOCRProcessor.from_pretrained("microsoft/trocr-small-printed")
        full_model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-small-printed")
        full_model.eval()
        full_model = full_model.to(device)
        print(f"  Loaded trocr-small-printed on {device}")

        class TrOCRWrapper:
            def __init__(self, full_model, processor, device):
                self.full_model = full_model
                self.processor = processor
                self.device = device

            def _preprocess(self, image):
                """Preprocess RGB uint8 image for TrOCR."""
                pil_image = PILImage.fromarray(image)
                pixel_values = self.processor(images=pil_image, return_tensors="pt").pixel_values
                return pixel_values.to(self.device)

            def get_logits(self, image):
                """Return encoder last hidden state as numpy array [seq_len, hidden_size]."""
                pixel_values = self._preprocess(image)
                with torch.no_grad():
                    encoder_out = self.full_model.encoder(pixel_values=pixel_values)
                return encoder_out.last_hidden_state.squeeze(0).cpu().numpy()

            def predict(self, image):
                """Predict text using full encoder-decoder with generate()."""
                pixel_values = self._preprocess(image)
                with torch.no_grad():
                    generated_ids = self.full_model.generate(pixel_values)
                text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

                class Result:
                    def __init__(self, text):
                        self.text = text
                return Result(text)

        return TrOCRWrapper(full_model, processor, device)

    elif ocr_model_type == 'qwen3-vl':
        print("Initializing Qwen3-VL-7B-Instruct...")
        from transformers import AutoModelForImageTextToText, AutoProcessor
        from PIL import Image as PILImage

        model_id = "Qwen/Qwen3-VL-8B-Instruct"
        processor = AutoProcessor.from_pretrained(model_id)
        full_model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        full_model.eval()
        print(f"  Loaded {model_id}")

        class Qwen3VLWrapper:
            def __init__(self, full_model, processor):
                self.full_model = full_model
                self.processor = processor

            def predict(self, image):
                """Predict text from RGB uint8 image using Qwen3-VL."""
                import re
                pil_image = PILImage.fromarray(image)
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": pil_image},
                            {"type": "text", "text": 'Read the text in this image. Respond in this exact format:\nThe text is: [text here]'},
                        ],
                    }
                ]
                inputs = self.processor.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_dict=True,
                    return_tensors="pt",
                )
                inputs = inputs.to(self.full_model.device)
                with torch.no_grad():
                    generated_ids = self.full_model.generate(**inputs, max_new_tokens=32)
                raw = self.processor.decode(
                    generated_ids[0][inputs.input_ids.shape[-1]:],
                    skip_special_tokens=True,
                )
                match = re.search(r'The text is:\s*(.+)', raw)
                text = match.group(1).strip() if match else ""

                class Result:
                    def __init__(self, text):
                        self.text = text
                return Result(text)

        return Qwen3VLWrapper(full_model, processor)

    elif ocr_model_type == 'gpt-5-mini':
        import base64
        import re
        from io import BytesIO
        from PIL import Image as PILImage

        if openai is None:
            raise ImportError("openai not installed. Install with: pip install openai")
        if not api_key:
            raise ValueError("--openai-api-key is required for gpt-5-mini")

        client = openai.OpenAI(api_key=api_key)

        class GPT5MiniWrapper:
            def __init__(self, client):
                self.client = client

            def predict(self, image):
                """Predict text from RGB uint8 image using GPT-5 mini."""
                import time
                pil_image = PILImage.fromarray(image)
                buf = BytesIO()
                pil_image.save(buf, format="PNG")
                b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

                messages = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{b64}"},
                            },
                            {
                                "type": "text",
                                "text": "Read the text in this image. Respond in this exact format:\nThe text is: [text here]",
                            },
                        ],
                    }
                ]

                time.sleep(0.1)
                while True:
                    try:
                        response = self.client.chat.completions.create(
                            model="gpt-5-mini",
                            messages=messages,
                            max_tokens=32,
                        )
                        break
                    except openai.RateLimitError:
                        print("\n  [gpt-5-mini] Rate limited, retrying in 5s...")
                        time.sleep(5)

                raw = response.choices[0].message.content or ""
                match = re.search(r'The text is:\s*(.+)', raw)
                text = match.group(1).strip() if match else ""

                class Result:
                    def __init__(self, text):
                        self.text = text
                return Result(text)

        return GPT5MiniWrapper(client)

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


def evaluate_patch_logit_delta(patch, val_images, ocr, control_logits_list, control_texts, center_ratio=0.6):
    """Evaluate patch by measuring logit differences between control and composite (MSE).

    Args:
        patch: [3, H, W] tensor in [0, 1]
        val_images: List of validation image tensors [3, H, W] in [0, 1]
        ocr: OCR model instance with get_logits(image) method and predict(image) method
        control_logits_list: List of precomputed control logits (numpy arrays)
        control_texts: List of precomputed control OCR texts
        center_ratio: Center ratio for compositing

    Returns:
        Tuple of (total_mse, total_edit_distance, num_misreads)
    """
    total_mse = 0.0
    total_edit_distance = 0
    num_misreads = 0
    num_evaluated = 0

    for val_image, control_logits, control_text in zip(val_images, control_logits_list, control_texts):
        # Create composite (with patch)
        composite = apply_patch_ocr_mode(val_image, patch, center_ratio=center_ratio)
        composite = composite.squeeze(0)  # Remove batch dim

        # Convert to numpy for OCR
        composite_np = (composite.permute(1, 2, 0).numpy() * 255).astype(np.uint8)

        # Get logits for composite
        composite_logits = ocr.get_logits(composite_np)

        # Compute MSE between control and composite logits
        min_len = min(control_logits.shape[0], composite_logits.shape[0])
        logit_diff = control_logits[:min_len] - composite_logits[:min_len]
        mse = np.mean(logit_diff ** 2)
        total_mse += mse

        # Compute edit distance using precomputed control text
        composite_result = ocr.predict(composite_np)
        composite_text = composite_result.text if composite_result is not None else ""
        edit_dist = Levenshtein.distance(control_text, composite_text)
        total_edit_distance += edit_dist

        if composite_text != control_text:
            num_misreads += 1

        num_evaluated += 1

    return total_mse if num_evaluated > 0 else 0.0, total_edit_distance, num_misreads


def evaluate_patch_logit_delta_with_debug(patch, val_images, ocr, control_logits_list, control_texts, center_ratio=0.6, debug_dir=None, candidate_idx=0):
    """Evaluate patch by measuring logit MSE and save debug images.

    Args:
        patch: [3, H, W] tensor in [0, 1]
        val_images: List of validation image tensors [3, H, W] in [0, 1]
        ocr: OCR model instance with get_logits(image) method and predict(image) method
        control_logits_list: List of precomputed control logits (numpy arrays)
        control_texts: List of precomputed control OCR texts
        center_ratio: Center ratio for compositing
        debug_dir: Directory to save debug images
        candidate_idx: Index of the candidate being evaluated

    Returns:
        Tuple of (total_mse, total_edit_distance, num_misreads)
    """
    total_mse = 0.0
    total_edit_distance = 0
    num_misreads = 0
    num_evaluated = 0
    debug_results = []

    for img_idx, (val_image, control_logits, control_text) in enumerate(zip(val_images, control_logits_list, control_texts)):
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

        # Compute edit distance using precomputed control text
        composite_result = ocr.predict(composite_np)
        composite_text = composite_result.text if composite_result is not None else ""
        edit_dist = Levenshtein.distance(control_text, composite_text)
        total_edit_distance += edit_dist

        if composite_text != control_text:
            num_misreads += 1

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
                'edit_distance': edit_dist,
                'control_text': control_text,
                'composite_text': composite_text,
                'control_logit_shape': control_logits.shape,
                'composite_logit_shape': composite_logits.shape,
            })

    # Save debug summary
    if debug_dir is not None and debug_results:
        summary_path = debug_dir / f"iter0_candidate{candidate_idx:02d}_summary.txt"
        with open(summary_path, 'w') as f:
            f.write(f"Candidate {candidate_idx} Debug Summary (Logit MSE Mode)\n")
            f.write(f"{'=' * 80}\n")
            f.write(f"Total MSE: {total_mse:.2f}\n")
            f.write(f"Total edit distance: {total_edit_distance}\n")
            f.write(f"Misreads: {num_misreads}/{len(val_images)}\n\n")
            f.write(f"Per-image results:\n")
            for result in debug_results:
                f.write(f"  Image {result['img_idx']:2d}: MSE: {result['mse']:10.2f} | EditDist: {result['edit_distance']:2d} | "
                       f"Control: '{result['control_text']:15s}' → Composite: '{result['composite_text']:15s}'\n")

    return total_mse, total_edit_distance, num_misreads


def main():
    parser = argparse.ArgumentParser(
        description='CMA-ES optimization of adversarial patches to maximize misreads.'
    )
    parser.add_argument('run_dir', help='Path to run directory with trained VAE')
    parser.add_argument('--csv', default=None,
                        help='Path to train_val_split CSV (default: auto-detect from run_dir)')
    parser.add_argument('--preproc-csv', default=None,
                        help='Path to preproc_labels CSV (dataset.py format). If provided, uses AdversarialPatchDataset instead of OCRDataset')
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
    parser.add_argument('--ocr-model', choices=['fast-alpr', 'opencv-crnn', 'vitstr', 'trocr', 'qwen3-vl', 'gpt-5-mini'], default='fast-alpr',
                        help='OCR model to use (default: fast-alpr)')
    parser.add_argument('--white-box', action='store_true',
                        help='Use smaller xs model instead of s model (only for fast-alpr)')
    parser.add_argument('--openai-api-key', default=None,
                        help='OpenAI API key for gpt-5-mini (can also be set via OPENAI_API_KEY env var)')
    parser.add_argument('--correct-text', default=None,
                        help='Known correct license plate text for all images. Skips control OCR computation.')

    # Device
    parser.add_argument('--device', default=None,
                        help='Device to use (default: auto-detect cuda/cpu). Examples: cpu, cuda, cuda:0, cuda:1')

    # Output
    parser.add_argument('--outdir', default='cmaes_output',
                        help='Output directory for results (default: cmaes_output)')

    # Optimization mode
    parser.add_argument('--use-logit-mse', action='store_true',
                        help='Optimize for logit MSE instead of text edit distance (only for opencv-crnn, default: text edit distance)')

    # Mode selection
    parser.add_argument('--composite-only', action='store_true',
                        help='Composite mode: just composite patches from run_dir with n validation samples and save, then exit (no optimization)')

    args = parser.parse_args()

    # Resolve OpenAI API key: CLI arg takes precedence over env var
    if args.openai_api_key is None:
        import os
        args.openai_api_key = os.environ.get('OPENAI_API_KEY', None)

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
            import onnxruntime
        elif args.ocr_model == 'vitstr':
            from doctr.models import vitstr_small
        elif args.ocr_model == 'trocr':
            from transformers import VisionEncoderDecoderModel, TrOCRProcessor
        elif args.ocr_model == 'gpt-5-mini':
            if openai is None:
                print("Error: openai not installed. Install with: pip install openai",
                      file=sys.stderr)
                sys.exit(1)
            if not args.openai_api_key:
                print("Error: --openai-api-key or OPENAI_API_KEY env var is required for gpt-5-mini",
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

    # Find CSV file (only needed if not using --preproc-csv)
    csv_path = None
    if not args.preproc_csv:
        if args.csv:
            csv_path = Path(args.csv)
        else:
            csv_files = list(run_dir.glob("**/*.csv"))
            for csv_file in csv_files:
                if 'train_val_split' in csv_file.name or 'split' in csv_file.name:
                    csv_path = csv_file
                    break

            if csv_path is None:
                cwd_csv = list(Path('.').glob("train_val_split_*.csv"))
                if cwd_csv:
                    csv_path = cwd_csv[-1]

            if csv_path is None:
                print("Error: Could not find train_val_split CSV file. "
                      "Pass --csv or use --preproc-csv for a preproc_labels CSV.", file=sys.stderr)
                sys.exit(1)

    # Load validation samples
    if args.preproc_csv:
        preproc_csv_path = Path(args.preproc_csv)
        if not preproc_csv_path.exists():
            print(f"Error: preproc CSV not found: {preproc_csv_path}", file=sys.stderr)
            sys.exit(1)
        print(f"Using preproc dataset: {preproc_csv_path}")
        print(f"\nLoading {args.n_eval_samples} samples from preproc CSV...")
        val_images, dimensions = load_validation_samples_from_preproc_csv(preproc_csv_path, args.n_eval_samples)
    else:
        print(f"Using data split: {csv_path}")
        print(f"\nLoading {args.n_eval_samples} validation samples...")
        val_images, dimensions = load_validation_samples_from_csv(csv_path, args.n_eval_samples)

    if not val_images:
        print("Error: No validation samples loaded", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {len(val_images)} validation samples")

    # Load generator
    print(f"\nLoading generator from {run_dir}...")
    generator, latent_dim, device = load_generator(run_dir, device=device)

    # Composite-only mode: sample 10 images, apply 10 patches (1-to-1), run OCR, save CSV
    if args.composite_only:
        print("\n" + "="*80)
        print("COMPOSITE-ONLY MODE")
        print("="*80)

        # Delete and recreate output directory
        if output_dir.exists():
            shutil.rmtree(output_dir)
            print(f"Deleted existing output directory: {output_dir}")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize OCR model
        print(f"\nInitializing OCR model: {args.ocr_model}")
        ocr = create_ocr_model(args.ocr_model, white_box=args.white_box, device=device, api_key=args.openai_api_key)

        n = 10
        images_to_use = val_images[:n]
        print(f"\nGenerating {n} patches...")
        patches = []
        for i in range(n):
            z = np.random.randn(latent_dim) * args.sigma0
            patch = generate_patch_from_z(generator, z, device)
            patches.append(patch)

        print(f"Processing {n} image-patch pairs...")
        rows = []
        for idx, (val_image, patch) in enumerate(zip(images_to_use, patches)):
            # Control: grey border
            control = apply_neutral_border_ocr_mode(val_image, center_ratio=args.center_ratio, border_color=0.5)
            control_np = (control.squeeze(0).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            cv2.imwrite(str(output_dir / f"{idx:02d}_control.jpg"),
                        cv2.cvtColor(control_np, cv2.COLOR_RGB2BGR))

            # Composite: with patch
            composite = apply_patch_ocr_mode(val_image, patch, center_ratio=args.center_ratio)
            composite_np = (composite.squeeze(0).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            cv2.imwrite(str(output_dir / f"{idx:02d}_composite.jpg"),
                        cv2.cvtColor(composite_np, cv2.COLOR_RGB2BGR))

            # OCR
            control_result = ocr.predict(control_np)
            control_text = control_result.text if control_result is not None else ""
            composite_result = ocr.predict(composite_np)
            composite_text = composite_result.text if composite_result is not None else ""

            rows.append({
                'image_idx': idx,
                'control_text': control_text,
                'composite_text': composite_text,
                'changed': control_text != composite_text,
            })
            print(f"  [{idx:02d}] control='{control_text}' | composite='{composite_text}'"
                  + (" *" if control_text != composite_text else ""))

        # Save CSV
        csv_out = output_dir / "predictions.csv"
        with open(csv_out, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['image_idx', 'control_text', 'composite_text', 'changed'])
            writer.writeheader()
            writer.writerows(rows)

        changed = sum(r['changed'] for r in rows)
        print(f"\nSaved {n} image pairs to {output_dir}")
        print(f"CSV: {csv_out}")
        print(f"Changed: {changed}/{n}")
        return

    # Initialize OCR model
    print(f"\nInitializing OCR model: {args.ocr_model}")
    ocr = create_ocr_model(args.ocr_model, white_box=args.white_box, device=device)
    print("OCR model loaded")

    # Determine optimization mode
    use_logit_mse = args.use_logit_mse
    logit_capable_models = {'opencv-crnn', 'vitstr', 'trocr'}
    if use_logit_mse and args.ocr_model not in logit_capable_models:
        print(f"Error: --use-logit-mse is not supported for '{args.ocr_model}'. "
              f"Supported models: {', '.join(logit_capable_models)}", file=sys.stderr)
        sys.exit(1)

    # Always precompute control texts (used in both text and logit MSE mode)
    if args.correct_text is not None:
        print(f"\nUsing provided correct text for all images: '{args.correct_text}' (skipping control OCR)")
        control_texts = [args.correct_text] * len(val_images)
    else:
        print("\nPrecomputing control OCR texts...")
        control_texts = []
        for val_image in tqdm(val_images, desc="Control OCR"):
            try:
                control = apply_neutral_border_ocr_mode(val_image, center_ratio=args.center_ratio, border_color=0.5)
                control = control.squeeze(0)
                control_np = (control.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                control_result = ocr.predict(control_np)
                control_texts.append(control_result.text if control_result is not None else "")
            except Exception as e:
                control_texts.append("")
        print(f"Precomputed {len(control_texts)} control texts")

    # In logit MSE mode, also precompute control logits
    if use_logit_mse:
        print("\nPrecomputing control logits (logit MSE mode)...")
        control_logits_list = []
        for val_image in tqdm(val_images, desc="Control logits"):
            try:
                control = apply_neutral_border_ocr_mode(val_image, center_ratio=args.center_ratio, border_color=0.5)
                control = control.squeeze(0)
                control_np = (control.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                control_logits_list.append(ocr.get_logits(control_np))
            except Exception as e:
                control_logits_list.append(None)
        print(f"Precomputed {len(control_logits_list)} control logits")
        control_data = control_logits_list
    else:
        control_data = control_texts

    # Create debug output directory
    debug_dir = output_dir / "debug_output"
    debug_dir.mkdir(parents=True, exist_ok=True)

    # Define objective function for CMA-ES
    eval_count = [0]  # Track number of evaluations
    best_metric = [0]  # Best edit distance or logit MSE (depending on mode)
    best_avg_metric = [0]  # Average metric
    best_misread_pct = [0]  # Misread percentage (text mode) or avg edit distance (logit MSE mode)
    best_secondary_metric = [0]  # For tracking secondary metric in logit MSE mode
    best_z = [None]
    current_iteration = [0]  # Track current iteration
    all_metrics = []  # Track all metrics for progress bar
    all_secondary_metrics = []  # Track edit distance when in logit MSE mode
    sampled_indices = []  # Will hold random sample indices for current iteration
    sampled_val_images = []  # Will hold sampled validation images
    sampled_control_data = []  # Will hold sampled control texts or logits
    sampled_control_texts = []  # Will hold sampled control texts (always, for edit distance)

    best_patches_dir = output_dir / "best_patches"
    best_patches_dir.mkdir(parents=True, exist_ok=True)

    def save_best_patch(z, iteration, avg_metric):
        """Save the patch and 10 composites whenever a new best is found."""
        patch = generate_patch_from_z(generator, z, device)
        patch_clamped = torch.clamp(patch, 0, 1)
        patch_np = (patch_clamped.permute(1, 2, 0).numpy() * 255).astype(np.uint8)

        tag = f"iter{iteration:04d}"
        save_dir = best_patches_dir / tag
        save_dir.mkdir(parents=True, exist_ok=True)

        # Save the patch itself
        patch_bgr = cv2.cvtColor(patch_np, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(save_dir / "patch.png"), patch_bgr)

        # Save 10 composites sampled from the full validation set
        preview_indices = random.sample(range(len(val_images)), min(10, len(val_images)))
        for j, idx in enumerate(preview_indices):
            composite = apply_patch_ocr_mode(val_images[idx], patch_clamped, center_ratio=args.center_ratio)
            composite = composite.squeeze(0)
            composite_np = (composite.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            composite_bgr = cv2.cvtColor(composite_np, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(save_dir / f"composite_{j:02d}.png"), composite_bgr)

        tqdm.write(f"  [new best] iter={iteration} avg_metric={avg_metric:.4f} -> saved to {save_dir}")

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
            # Logit MSE mode (opencv-crnn only) - optimize MSE but track edit distance
            if save_debug:
                total_mse, total_edit_distance, misreads = evaluate_patch_logit_delta_with_debug(
                    patch, sampled_val_images, ocr, sampled_control_data, sampled_control_texts,
                    center_ratio=args.center_ratio,
                    debug_dir=debug_dir,
                    candidate_idx=candidate_idx
                )
            else:
                total_mse, total_edit_distance, misreads = evaluate_patch_logit_delta(
                    patch, sampled_val_images, ocr, sampled_control_data, sampled_control_texts,
                    center_ratio=args.center_ratio
                )
            # Primary metric is MSE for optimization
            total_metric = total_mse
            avg_metric = total_mse / len(sampled_val_images) if len(sampled_val_images) > 0 else 0
            # Secondary metric is edit distance for reporting
            avg_edit_distance = total_edit_distance / len(sampled_val_images) if len(sampled_val_images) > 0 else 0
            all_secondary_metrics.append(avg_edit_distance)
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
            best_misread_pct[0] = (misreads / len(sampled_val_images) * 100) if len(sampled_val_images) > 0 else 0
            if use_logit_mse:
                best_secondary_metric[0] = avg_edit_distance
            best_z[0] = z.copy()
            save_best_patch(z, current_iteration[0], avg_metric)

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
        sampled_control_texts = [control_texts[i] for i in sampled_indices]

        # Clear metrics for this iteration
        all_metrics.clear()
        if use_logit_mse:
            all_secondary_metrics.clear()

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
            avg_secondary = np.mean(all_secondary_metrics) if all_secondary_metrics else 0
            pbar.set_postfix({
                'best_mse': f'{best_avg_metric[0]:.2e}',
                'avg_mse': f'{avg_metric:.2e}',
                'best_edit': f'{best_secondary_metric[0]:.1f}',
                'avg_edit': f'{avg_secondary:.1f}',
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
        print(f"Best average logit MSE: {best_avg_metric[0]:.2e}")
        print(f"Best average edit distance: {best_secondary_metric[0]:.1f}")
        print(f"Best misread percentage: {best_misread_pct[0]:.1f}%")
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
                f.write(f"  Best average logit MSE: {best_avg_metric[0]:.2e}\n")
                f.write(f"  Best average edit distance: {best_secondary_metric[0]:.1f}\n")
                f.write(f"  Best misread percentage: {best_misread_pct[0]:.1f}%\n")
            else:
                f.write(f"  Best total edit distance: {best_metric[0]:.2f}\n")
                f.write(f"  Best average edit distance: {best_avg_metric[0]:.2f}\n")
                f.write(f"  Best misread percentage: {best_misread_pct[0]:.1f}%\n")
            f.write(f"  Best z shape: {best_z[0].shape}\n")

        print(f"Saved metadata to: {metadata_path}")

    print(f"\nAll results saved to: {output_dir}")


if __name__ == '__main__':
    main()
