#!/usr/bin/env python3
"""
Profile OCR Models with RDM

Profiles two OCR models using the RDM (Representational Dissimilarity Matrix) profiler:
1. CCT-XS-V1 Global (from progressive_patch.py)
2. Microsoft TrOCR Small Printed (vision encoder)

Profiles every layer of each model over the entire training dataset from progressive_patch.py.
Saves layer profiles to an output directory with clear labels.
"""

import os
import argparse
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import onnx
import onnx2torch
from transformers import VisionEncoderDecoderModel, TrOCRProcessor
import warnings
from rdm_profiler import ModelRDMProfiler
from dataset import create_dataloaders
import torchvision.transforms as T
import kornia.geometry as K

warnings.filterwarnings("ignore")


class OCRImageDataset(Dataset):
    """
    Wrapper to extract cropped license plate regions from AdversarialPatchDataset.

    This extracts the license plate region using corners (just like progressive_patch.py)
    and prepares them for OCR model profiling.

    Note: Uses manual iteration (no batching) because progressive_patch.py dataloader
    returns variable-sized images that can't be stacked.
    """
    def __init__(self, dataloader, target_size=(64, 128), preprocessor=None,
                 device='cuda', channels_last=False, scale_to_255=False):
        """
        Args:
            dataloader: DataLoader from progressive_patch.py dataset (batch_size must be 1)
            target_size: (height, width) for OCR input (used if preprocessor is None)
            preprocessor: Optional preprocessing function/processor (e.g., TrOCRProcessor)
                         If provided, applies after cropping
            device: Device for kornia operations
            channels_last: If True, output [H, W, C] instead of [C, H, W] (for CCT model)
            scale_to_255: If True, scale values from [0,1] to [0,255] (for CCT model)
        """
        self.images = []
        self.target_size = target_size
        self.preprocessor = preprocessor
        self.device = device
        self.channels_last = channels_last
        self.scale_to_255 = scale_to_255

        if preprocessor is not None:
            print(f"Extracting cropped plate regions with custom preprocessor...")
        else:
            print(f"Extracting cropped plate regions (target size: {target_size})...")
            if channels_last:
                print(f"  Format: [H, W, C] (channels last)")
            if scale_to_255:
                print(f"  Scaling: [0, 1] -> [0, 255]")
        print("(No batching - iterating individual images due to variable sizes)")

        # Manually iterate without batching
        for i, batch in enumerate(dataloader):
            # Each batch is a dict with single image (batch_size=1)
            prep_image = batch['prep_image']  # Shape: [1, 3, H, W]
            new_corners = batch['new_corners']  # Shape: [1, 4, 2] - corners in preprocessed image

            # Move to device for kornia operations
            prep_image = prep_image.to(self.device)
            new_corners = new_corners.to(self.device)

            # Use plate corners directly (no border scaling)
            plate_corners = new_corners  # [1, 4, 2]

            # Crop and resize plate region using kornia
            cropped_plate = K.crop_and_resize(
                prep_image,
                plate_corners,
                target_size  # (H, W)
            )

            # Extract single image and move to CPU
            img = cropped_plate[0].cpu()  # Shape: [3, H, W]

            # Preprocess image if processor provided
            if preprocessor is not None:
                # Convert tensor to PIL Image for processor
                from PIL import Image
                import numpy as np

                # Convert [C, H, W] tensor to PIL Image
                img_np = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                pil_img = Image.fromarray(img_np)

                # Process with TrOCRProcessor (or other processor)
                processed = preprocessor(images=pil_img, return_tensors="pt")
                img_processed = processed.pixel_values.squeeze(0)  # Remove batch dim
                self.images.append(img_processed)
            else:
                # Apply format transformations for specific models
                if channels_last:
                    # CCT model expects [H, W, C] format
                    img = img.permute(1, 2, 0)  # [C, H, W] -> [H, W, C]

                if scale_to_255:
                    # CCT model expects [0, 255] range
                    img = img * 255

                self.images.append(img)

            if (i + 1) % 100 == 0:
                print(f"  Processed {i + 1} images...")

        print(f"Loaded {len(self.images)} cropped plates total")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx]


def load_cct_model(device='cuda'):
    """
    Load CCT-XS-V1 Global model from progressive_patch.py.

    Returns:
        model: PyTorch model
        model_name: String identifier
    """
    print("\n" + "="*80)
    print("Loading CCT-XS-V1 Global Model")
    print("="*80)

    ocr_path = Path.home() / ".cache/fast-plate-ocr/cct-xs-v1-global-model/cct_xs_v1_global.onnx"

    if not ocr_path.exists():
        raise FileNotFoundError(
            f"CCT model not found at: {ocr_path}\n"
            f"Please run progressive_patch.py first to download the model."
        )

    print(f"Loading from: {ocr_path}")
    ocr_model = onnx.load(str(ocr_path))
    model = onnx2torch.convert(ocr_model).to(device)
    model.eval()

    print(f"Model loaded successfully")
    return model, "cct_xs_v1_global"


def load_trocr_model(device='cuda'):
    """
    Load microsoft/trocr-small-printed from HuggingFace.

    This is a Vision Encoder-Decoder model for OCR. We profile the vision encoder
    which is the part that processes images into representations.

    Returns:
        model: PyTorch model (vision encoder only)
        model_name: String identifier
        processor: TrOCRProcessor for image preprocessing
    """
    print("\n" + "="*80)
    print("Loading Microsoft TrOCR Small Printed Model")
    print("="*80)

    try:
        # Load processor for image preprocessing
        processor = TrOCRProcessor.from_pretrained("microsoft/trocr-small-printed")

        # Load full model
        full_model = VisionEncoderDecoderModel.from_pretrained(
            "microsoft/trocr-small-printed"
        ).to(device)
        full_model.eval()

        # Extract vision encoder (the part that processes images)
        # The decoder is for text generation, not relevant for RDM profiling
        model = full_model.encoder

        print("Model loaded successfully (vision encoder)")
        print(f"Encoder type: {type(model).__name__}")

        # Get expected input size (try both old and new attribute names)
        if hasattr(processor, 'image_processor'):
            print(f"Expected input size: {processor.image_processor.size}")
        elif hasattr(processor, 'feature_extractor'):
            print(f"Expected input size: {processor.feature_extractor.size}")

        return model, "trocr_small_printed_encoder", processor
    except Exception as e:
        print(f"Error loading TrOCR model: {e}")
        raise RuntimeError(
            f"Failed to load TrOCR model from HuggingFace. Error: {e}\n"
            "Please ensure transformers is installed: pip install transformers"
        )


def profile_model(model, model_name, dataset, output_dir, device='cuda', batch_size=16):
    """
    Profile a single model and save RDMs for all layers.

    Args:
        model: PyTorch model to profile
        model_name: Name identifier for the model
        dataset: Image dataset to profile over
        output_dir: Directory to save RDM profiles
        device: Device to run on
        batch_size: Batch size for profiling
    """
    print("\n" + "="*80)
    print(f"Profiling {model_name}")
    print("="*80)

    # Create output directory for this model
    model_output_dir = Path(output_dir) / model_name
    model_output_dir.mkdir(parents=True, exist_ok=True)

    # Create profiler for ALL layers
    profiler = ModelRDMProfiler(
        model=model,
        model_name=model_name,
        device=device,
        layer_names=None,  # Profile all layers
        layer_indices=None
    )

    # Profile and save
    save_path = model_output_dir / f"{model_name}_rdms.h5"
    rdms = profiler.profile(
        image_dataset=dataset,
        batch_size=batch_size,
        save_path=save_path
    )

    print(f"\n{model_name} profiling complete!")
    print(f"Profiled {len(rdms)} layers")
    print(f"Results saved to: {save_path}")
    print(f"\nLayer names:")
    for i, layer_name in enumerate(rdms.keys(), 1):
        print(f"  {i:3d}. {layer_name}")

    return rdms


def main():
    parser = argparse.ArgumentParser(description='Profile OCR Models with RDM')
    parser.add_argument('--csv-path', type=str, default='preproc_labels.csv',
                        help='Path to CSV file with training data (default: preproc_labels.csv)')
    parser.add_argument('--output-dir', type=str, default='rdm_profiles',
                        help='Output directory for RDM profiles (default: rdm_profiles)')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (cuda/mps/cpu). Auto-detects if not specified.')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Batch size for profiling (default: 16)')
    parser.add_argument('--models', type=str, default='cct,trocr',
                        help='Comma-separated list of models to profile: cct,trocr (default: all)')
    parser.add_argument('--limit-images', type=int, default=0,
                        help='Limit number of images to profile (0=all, default: 0)')
    args = parser.parse_args()

    # Auto-detect device if not specified
    if args.device is None:
        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
    else:
        device = args.device

    print("="*80)
    print("OCR Model RDM Profiler")
    print("="*80)
    print(f"Device: {device}")
    print(f"Output directory: {args.output_dir}")
    print(f"Batch size: {args.batch_size}")
    print()

    # Parse which models to profile
    models_to_profile = [m.strip().lower() for m in args.models.split(',')]

    # Load training dataset from progressive_patch.py
    # Note: We'll create model-specific datasets as needed since different models
    # require different preprocessing
    print("Loading training dataset...")
    transform = T.Compose([
        T.ToPILImage(),
        T.ToTensor()
    ])

    # Must use batch_size=1 because images have variable sizes and can't be stacked
    train_loader, val_loader = create_dataloaders(
        args.csv_path,
        transform=transform,
        preload=True,
        batch_size=1,  # MUST be 1 - images have different sizes, can't batch
        n_jobs=0,
        use_all_for_train=True
    )

    print(f"Loaded dataloader with {len(train_loader)} images\n")

    # Profile each requested model
    results = {}

    if 'cct' in models_to_profile:
        try:
            model, model_name = load_cct_model(device)

            # Create CCT-specific dataset (64x128 for license plate OCR)
            # CCT model expects [H, W, C] format and [0, 255] range (matches progressive_patch.py)
            print(f"\nCreating dataset for {model_name}...")
            cct_dataset = OCRImageDataset(
                train_loader,
                target_size=(64, 128),  # Matches progressive_patch.py ocr_input_shape
                device=device,
                channels_last=True,  # CCT expects [H, W, C] not [C, H, W]
                scale_to_255=True    # CCT expects [0, 255] not [0, 1]
            )
            if args.limit_images > 0:
                print(f"Limiting to {args.limit_images} images")
                cct_dataset.images = cct_dataset.images[:args.limit_images]

            rdms = profile_model(model, model_name, cct_dataset, args.output_dir, device, args.batch_size)
            results[model_name] = rdms
            del model, cct_dataset  # Free memory
            torch.cuda.empty_cache() if device == 'cuda' else None
        except Exception as e:
            print(f"\nERROR profiling CCT model: {e}")
            import traceback
            traceback.print_exc()

    if 'trocr' in models_to_profile:
        try:
            model, model_name, processor = load_trocr_model(device)

            # Create TrOCR-specific dataset
            # First crop to plate region, then use processor for final preprocessing
            print(f"\nCreating dataset for {model_name}...")
            trocr_dataset = OCRImageDataset(
                train_loader,
                target_size=(64, 128),  # Initial crop size before processor
                preprocessor=processor,  # Processor will resize to model's expected size
                device=device
            )
            if args.limit_images > 0:
                print(f"Limiting to {args.limit_images} images")
                trocr_dataset.images = trocr_dataset.images[:args.limit_images]

            rdms = profile_model(model, model_name, trocr_dataset, args.output_dir, device, args.batch_size)
            results[model_name] = rdms
            del model, trocr_dataset, processor  # Free memory
            torch.cuda.empty_cache() if device == 'cuda' else None
        except Exception as e:
            print(f"\nERROR profiling TrOCR model: {e}")
            import traceback
            traceback.print_exc()

    # Summary
    print("\n" + "="*80)
    print("PROFILING COMPLETE")
    print("="*80)
    print(f"Successfully profiled {len(results)} model(s)")
    for model_name, rdms in results.items():
        print(f"  - {model_name}: {len(rdms)} layers")
    print(f"\nResults saved to: {args.output_dir}/")
    print("\nTo inspect results, use:")
    print("  from rdm_profiler import RDMStorage")
    print(f"  storage = RDMStorage('{args.output_dir}/<model_name>/<model_name>_rdms.h5')")
    print("  rdms, metadata = storage.load('<model_name>')")
    print("="*80)


if __name__ == "__main__":
    main()
