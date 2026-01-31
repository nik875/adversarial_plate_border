#!/usr/bin/env python3
"""
Profile OCR Models with RDM

Profiles three OCR models using the RDM (Representational Dissimilarity Matrix) profiler:
1. CCT-XS-V1 Global (from progressive_patch.py)
2. PaddlePaddle/en_PP-OCRv5_mobile_rec
3. opencv/text_recognition_crnn

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
from transformers import AutoModel
import warnings
from rdm_profiler import ModelRDMProfiler
from dataset import create_dataloaders
import torchvision.transforms as T

warnings.filterwarnings("ignore")


class OCRImageDataset(Dataset):
    """
    Wrapper to convert AdversarialPatchDataset to simple image dataset for RDM profiling.
    Extracts preprocessed images and resizes them for OCR models.
    """
    def __init__(self, dataloader, target_size=(64, 256)):
        """
        Args:
            dataloader: DataLoader from progressive_patch.py dataset
            target_size: (height, width) for OCR input
        """
        self.images = []
        self.target_size = target_size

        print(f"Extracting images from dataloader (target size: {target_size})...")
        for batch in dataloader:
            # Extract preprocessed images from batch
            prep_images = batch['prep_image']  # Shape: [batch, 3, H, W]

            # Process each image in batch
            for img in prep_images:
                # Resize to OCR input size
                img_resized = T.Resize(target_size)(img)
                self.images.append(img_resized)

        print(f"Loaded {len(self.images)} images")

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


def load_paddleocr_model(device='cuda'):
    """
    Load PaddlePaddle/en_PP-OCRv5_mobile_rec from HuggingFace.

    Returns:
        model: PyTorch model
        model_name: String identifier
    """
    print("\n" + "="*80)
    print("Loading PaddlePaddle en_PP-OCRv5_mobile_rec Model")
    print("="*80)

    try:
        model = AutoModel.from_pretrained(
            "PaddlePaddle/en_PP-OCRv5_mobile_rec",
            trust_remote_code=True
        ).to(device)
        model.eval()
        print("Model loaded successfully")
        return model, "paddle_ppocr_v5_mobile"
    except Exception as e:
        print(f"Error loading PaddleOCR model: {e}")
        print("Attempting alternative loading method...")

        # Alternative: Try loading with different config
        try:
            from paddleocr import PaddleOCR
            # Note: This may require paddlepaddle installation
            raise NotImplementedError(
                "PaddleOCR model loading requires custom implementation. "
                "Please install paddlepaddle and implement custom loader."
            )
        except ImportError:
            raise RuntimeError(
                "Failed to load PaddleOCR model. Please install required dependencies:\n"
                "pip install paddlepaddle paddleocr"
            )


def load_opencv_crnn_model(device='cuda'):
    """
    Load opencv/text_recognition_crnn from HuggingFace.

    Returns:
        model: PyTorch model
        model_name: String identifier
    """
    print("\n" + "="*80)
    print("Loading OpenCV Text Recognition CRNN Model")
    print("="*80)

    try:
        model = AutoModel.from_pretrained(
            "opencv/text_recognition_crnn",
            trust_remote_code=True
        ).to(device)
        model.eval()
        print("Model loaded successfully")
        return model, "opencv_crnn"
    except Exception as e:
        print(f"Error loading OpenCV CRNN model: {e}")
        raise RuntimeError(
            f"Failed to load OpenCV CRNN model from HuggingFace. Error: {e}\n"
            "The model may not be available or may require custom loading."
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
    parser.add_argument('--models', type=str, default='cct,paddle,opencv',
                        help='Comma-separated list of models to profile: cct,paddle,opencv (default: all)')
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
    print("Loading training dataset...")
    transform = T.Compose([
        T.ToPILImage(),
        T.ToTensor()
    ])

    train_loader, val_loader = create_dataloaders(
        args.csv_path,
        transform=transform,
        preload=True,
        batch_size=32,  # Use larger batch for faster loading
        n_jobs=0,
        use_all_for_train=True
    )

    # Convert to OCR dataset (extract and resize images)
    # Note: Different models may need different input sizes
    ocr_dataset = OCRImageDataset(train_loader, target_size=(64, 256))

    # Limit images if specified
    if args.limit_images > 0:
        print(f"Limiting to {args.limit_images} images")
        ocr_dataset.images = ocr_dataset.images[:args.limit_images]

    print(f"Dataset ready: {len(ocr_dataset)} images\n")

    # Profile each requested model
    results = {}

    if 'cct' in models_to_profile:
        try:
            model, model_name = load_cct_model(device)
            rdms = profile_model(model, model_name, ocr_dataset, args.output_dir, device, args.batch_size)
            results[model_name] = rdms
            del model  # Free memory
            torch.cuda.empty_cache() if device == 'cuda' else None
        except Exception as e:
            print(f"\nERROR profiling CCT model: {e}")
            import traceback
            traceback.print_exc()

    if 'paddle' in models_to_profile:
        try:
            model, model_name = load_paddleocr_model(device)
            rdms = profile_model(model, model_name, ocr_dataset, args.output_dir, device, args.batch_size)
            results[model_name] = rdms
            del model  # Free memory
            torch.cuda.empty_cache() if device == 'cuda' else None
        except Exception as e:
            print(f"\nERROR profiling PaddleOCR model: {e}")
            import traceback
            traceback.print_exc()

    if 'opencv' in models_to_profile:
        try:
            model, model_name = load_opencv_crnn_model(device)
            rdms = profile_model(model, model_name, ocr_dataset, args.output_dir, device, args.batch_size)
            results[model_name] = rdms
            del model  # Free memory
            torch.cuda.empty_cache() if device == 'cuda' else None
        except Exception as e:
            print(f"\nERROR profiling OpenCV CRNN model: {e}")
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
