#!/usr/bin/env python3
"""
Profile OCR Models with RDM

Profiles three OCR models using the RDM (Representational Dissimilarity Matrix) profiler:
1. ViTSTR Small (from doctr - Vision Transformer STR)
2. CCT-XS-V1 Global (from progressive_patch.py)
3. Microsoft TrOCR Small Printed (vision encoder)

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
import urllib.request
import warnings
import numpy as np
from datetime import datetime
from tqdm import tqdm
from rdm_profiler import ModelRDMProfiler
from dataset import create_dataloaders
import torchvision.transforms as T
import kornia.geometry as K
import h5py

warnings.filterwarnings("ignore")


class CroppedPlateDataset(Dataset):
    """
    Loads and crops license plate regions once from the dataset.
    Stores raw cropped plates that can be preprocessed differently per model.
    """
    def __init__(self, dataloader, target_size=(64, 128), device='cuda'):
        """
        Args:
            dataloader: DataLoader from progressive_patch.py dataset (batch_size must be 1)
            target_size: (height, width) for cropping
            device: Device for kornia operations
        """
        self.images = []
        self.target_size = target_size
        self.device = device

        print(f"Extracting cropped plate regions (target size: {target_size})...")
        print("(No batching - iterating individual images due to variable sizes)")

        # Manually iterate without batching
        for i, batch in enumerate(dataloader):
            prep_image = batch['prep_image']  # [1, 3, H, W]
            new_corners = batch['new_corners']  # [1, 4, 2]

            prep_image = prep_image.to(self.device)
            new_corners = new_corners.to(self.device)

            # Crop and resize plate region using kornia
            cropped_plate = K.crop_and_resize(
                prep_image,
                new_corners,
                target_size
            )

            # Move to CPU and store: [3, H, W] in [0, 1] range
            img = cropped_plate[0].cpu()  # Shape: [3, H, W]
            self.images.append(img)

            if (i + 1) % 100 == 0:
                print(f"  Processed {i + 1} images...")

        print(f"Loaded {len(self.images)} cropped plates total")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx]


class OCRImageDataset(Dataset):
    """
    Applies model-specific preprocessing to raw cropped plates.
    Wraps a CroppedPlateDataset with different format/scale options.
    """
    def __init__(self, cropped_dataset, preprocessor=None,
                 channels_last=False, scale_to_255=False):
        """
        Args:
            cropped_dataset: CroppedPlateDataset with raw plates
            preprocessor: Optional TrOCRProcessor
            channels_last: If True, output [H, W, C] instead of [C, H, W]
            scale_to_255: If True, scale values from [0,1] to [0,255]
        """
        self.cropped_dataset = cropped_dataset
        self.preprocessor = preprocessor
        self.channels_last = channels_last
        self.scale_to_255 = scale_to_255

    def __len__(self):
        return len(self.cropped_dataset)

    def __getitem__(self, idx):
        img = self.cropped_dataset[idx].clone()  # [3, H, W] in [0, 1]

        # Apply processor if provided (TrOCR)
        if self.preprocessor is not None:
            from PIL import Image
            import numpy as np

            # Convert to PIL Image
            img_np = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            pil_img = Image.fromarray(img_np)

            # Process with TrOCRProcessor
            processed = self.preprocessor(images=pil_img, return_tensors="pt")
            return processed.pixel_values.squeeze(0)

        # Apply format transformations for other models
        if self.channels_last:
            img = img.permute(1, 2, 0)  # [C, H, W] -> [H, W, C]

        if self.scale_to_255:
            img = img * 255  # [0, 1] -> [0, 255]

        return img


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


def load_vitstr_model(device='cuda'):
    """
    Load ViTSTR Small model from doctr.

    Returns:
        model: PyTorch model
        model_name: String identifier
    """
    print("\n" + "="*80)
    print("Loading ViTSTR Small Model (doctr)")
    print("="*80)

    try:
        from doctr.models import vitstr_small

        print("Loading pretrained ViTSTR model...")
        model = vitstr_small(pretrained=True)
        model.eval()  # Set to eval mode for profiling
        model.to(device)

        print("Model loaded successfully")
        print(f"Model type: {type(model).__name__}")
        return model, "vitstr_small"
    except ImportError:
        raise RuntimeError(
            "doctr library not found. Install it with:\n"
            "pip install python-doctr[torch]"
        )
    except Exception as e:
        print(f"Error loading ViTSTR model: {e}")
        raise RuntimeError(
            f"Failed to load ViTSTR model. Error: {e}"
        )


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


def compute_activation_statistics(all_activations, layer_names):
    """
    Compute mean and standard deviation for each neuron's activations across dataset.

    Args:
        all_activations: Dict mapping layer names to lists of activation arrays
                        Each array has shape [batch_size, n_features]
        layer_names: List of layer names to compute statistics for

    Returns:
        Dict mapping layer names to dicts with 'mean' and 'std' arrays
        Each array has shape [n_features]
    """
    activation_stats = {}

    for layer_name in layer_names:
        if layer_name not in all_activations or not all_activations[layer_name]:
            continue

        # Concatenate all activations for this layer across all batches
        layer_acts = np.concatenate(all_activations[layer_name], axis=0)  # [n_images, n_features]

        # Compute mean and std per neuron (across images)
        mean = layer_acts.mean(axis=0)  # [n_features]
        std = layer_acts.std(axis=0)    # [n_features]

        activation_stats[layer_name] = {
            'mean': mean.astype(np.float32),
            'std': std.astype(np.float32),
            'n_images': layer_acts.shape[0],
            'n_features': layer_acts.shape[1]
        }

    return activation_stats


def save_activation_statistics(activation_stats, save_path):
    """
    Save activation statistics to HDF5 file.

    Args:
        activation_stats: Dict mapping layer names to stats dicts
        save_path: Path to save HDF5 file
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(save_path, 'w') as f:
        stats_group = f.create_group('activation_statistics')

        for layer_name, stats in activation_stats.items():
            layer_group = stats_group.create_group(layer_name)

            # Save mean and std
            layer_group.create_dataset(
                'mean',
                data=stats['mean'],
                compression='gzip',
                compression_opts=4,
                dtype=np.float32
            )
            layer_group.create_dataset(
                'std',
                data=stats['std'],
                compression='gzip',
                compression_opts=4,
                dtype=np.float32
            )

            # Save metadata
            layer_group.attrs['n_images'] = stats['n_images']
            layer_group.attrs['n_features'] = stats['n_features']


def profile_model(model, model_name, dataset, output_dir, device='cuda', batch_size=16):
    """
    Profile a single model and save RDMs and activation statistics for all layers.

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

    # Need to capture activations during profiling for statistics
    # We'll do this by extending the profiler workflow
    from torch.utils.data import DataLoader

    profiler.extractor.register_hooks()
    hooked_layers = profiler.extractor.get_hooked_layers()

    if not hooked_layers:
        raise RuntimeError("No layers matched filter criteria. Check layer names/indices.")

    print(f"Profiling {len(hooked_layers)} layers on {device}")
    print(f"Layers: {hooked_layers}")

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    # Collect activations from all images
    all_activations = {layer_name: [] for layer_name in hooked_layers}
    layer_info = {layer_name: {} for layer_name in hooked_layers}

    expected_batch_size = None

    print("\nRunning inference and collecting activations...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Processing batches")):
            # Handle different batch formats
            if isinstance(batch, (tuple, list)):
                images = batch[0]
            else:
                images = batch

            # Move to device
            if isinstance(images, torch.Tensor):
                images = images.to(device)

            # Track expected batch size from first batch
            if expected_batch_size is None:
                expected_batch_size = images.shape[0]

            # Forward pass
            _ = model(images)

            # Collect activations from this batch
            for layer_name in hooked_layers:
                if layer_name in profiler.extractor.activations:
                    activation = profiler.extractor.activations[layer_name]

                    # Store shape info from first batch
                    if batch_idx == 0:
                        layer_info[layer_name]['activation_shape'] = tuple(activation.shape)

                    # Move to CPU and flatten
                    activation_cpu = activation.cpu().numpy()
                    from rdm_profiler import ActivationProcessor
                    flattened = ActivationProcessor.flatten_activation(
                        torch.from_numpy(activation_cpu)
                    ).numpy()

                    # Check for sequence-collapsed activations
                    if flattened.shape[0] != expected_batch_size:
                        if flattened.shape[0] % expected_batch_size == 0:
                            seq_len = flattened.shape[0] // expected_batch_size
                            reshaped = flattened.reshape(expected_batch_size, seq_len, -1)
                            flattened = reshaped.mean(axis=1)

                            if batch_idx == 0:
                                warnings.warn(
                                    f"Layer {layer_name}: Detected sequence-collapsed activation "
                                    f"[{flattened.shape[0]}x{seq_len}, {flattened.shape[1]}]. "
                                    f"Aggregating across sequence dimension."
                                )
                        else:
                            warnings.warn(
                                f"Layer {layer_name}: Unexpected activation shape "
                                f"{flattened.shape} (expected batch_size={expected_batch_size})"
                            )

                    all_activations[layer_name].append(flattened)

            # Clear activations for next batch
            profiler.extractor.clear_activations()

    # Remove hooks
    profiler.extractor.remove_hooks()

    # Compute RDMs and activation statistics
    print("\nComputing RDMs and activation statistics...")
    rdms = {}

    for layer_name in tqdm(hooked_layers, desc="Computing RDMs"):
        if not all_activations[layer_name]:
            warnings.warn(f"No activations collected for layer {layer_name}")
            continue

        layer_activations = np.concatenate(all_activations[layer_name], axis=0)
        n_images = layer_activations.shape[0]

        # Compute RDM
        rdm = profiler.rdm_computer.compute_rdm(layer_activations)
        rdms[layer_name] = rdm

        # Get layer type
        layer_module = None
        for name, module in model.named_modules():
            if name == layer_name:
                layer_module = module
                break

        layer_info[layer_name]['layer_type'] = (
            layer_module.__class__.__name__ if layer_module else 'Unknown'
        )

    # Compute activation statistics
    print("\nComputing activation statistics (mean/std per neuron)...")
    activation_stats = compute_activation_statistics(all_activations, hooked_layers)

    # Save RDMs
    save_path = model_output_dir / f"{model_name}_rdms.h5"
    from rdm_profiler import RDMStorage
    metadata = {
        'n_images': n_images,
        'metric': 'correlation',
        'timestamp': datetime.now().isoformat(),
        'device': device,
        'batch_size': batch_size
    }
    storage = RDMStorage(save_path)
    storage.save(model_name, rdms, layer_info, metadata)
    print(f"Saved RDMs to {save_path}")

    # Save activation statistics
    stats_path = model_output_dir / f"{model_name}_activation_statistics.h5"
    save_activation_statistics(activation_stats, stats_path)
    print(f"Saved activation statistics to {stats_path}")

    print(f"\n{model_name} profiling complete!")
    print(f"Profiled {len(rdms)} layers")
    print(f"\nLayer activation statistics summary:")
    for i, (layer_name, stats) in enumerate(activation_stats.items(), 1):
        print(f"  {i:3d}. {layer_name}")
        print(f"       Images: {stats['n_images']}, Features: {stats['n_features']}")
        print(f"       Mean range: [{stats['mean'].min():.4f}, {stats['mean'].max():.4f}]")
        print(f"       Std range: [{stats['std'].min():.4f}, {stats['std'].max():.4f}]")

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
    parser.add_argument('--models', type=str, default='vitstr,cct,trocr',
                        help='Comma-separated list of models to profile: vitstr,cct,trocr (default: all)')
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
    # preload=False to reduce initial overhead (images loaded on-demand)
    train_loader, val_loader = create_dataloaders(
        args.csv_path,
        transform=transform,
        preload=False,  # Don't preload - reduces initial overhead
        batch_size=1,  # MUST be 1 - images have different sizes, can't batch
        n_jobs=0,
        use_all_for_train=True
    )

    print(f"Loaded dataloader with {len(train_loader)} images\n")

    # Load and crop plates once (done at 64x128 which both models use)
    print("Loading and cropping license plate regions (done once for all models)...")
    cropped_plates = CroppedPlateDataset(
        train_loader,
        target_size=(64, 128),  # Standard size for both CCT and TrOCR
        device=device
    )

    if args.limit_images > 0:
        print(f"Limiting to {args.limit_images} images")
        cropped_plates.images = cropped_plates.images[:args.limit_images]

    # Profile each requested model
    # ViTSTR is profiled FIRST to catch errors early
    results = {}

    if 'vitstr' in models_to_profile:
        try:
            model, model_name = load_vitstr_model(device)

            # Create ViTSTR dataset wrapper with model-specific preprocessing
            # ViTSTR needs 32x128, so we'll resize on-the-fly
            print(f"\nCreating preprocessing wrapper for {model_name}...")
            vitstr_dataset = OCRImageDataset(
                cropped_plates,
                preprocessor=None,
                channels_last=False,  # Use [C, H, W] format
                scale_to_255=False    # Keep [0, 1] range
            )
            # Note: ViTSTR resizing is done in profile_model's extract_activations

            rdms = profile_model(model, model_name, vitstr_dataset, args.output_dir, device, args.batch_size)
            results[model_name] = rdms
            del model, vitstr_dataset  # Free memory
            torch.cuda.empty_cache() if device == 'cuda' else None
        except Exception as e:
            print(f"\nERROR profiling ViTSTR model: {e}")
            import traceback
            traceback.print_exc()

    if 'cct' in models_to_profile:
        try:
            model, model_name = load_cct_model(device)

            # Create CCT-specific preprocessing wrapper
            print(f"\nCreating preprocessing wrapper for {model_name}...")
            cct_dataset = OCRImageDataset(
                cropped_plates,
                preprocessor=None,
                channels_last=True,  # CCT expects [H, W, C]
                scale_to_255=True    # CCT expects [0, 255]
            )

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

            # Create TrOCR-specific preprocessing wrapper
            print(f"\nCreating preprocessing wrapper for {model_name}...")
            trocr_dataset = OCRImageDataset(
                cropped_plates,
                preprocessor=processor,
                channels_last=False,
                scale_to_255=False
            )

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
