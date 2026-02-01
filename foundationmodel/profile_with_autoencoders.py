#!/usr/bin/env python3
"""
Profile OCR Models with Autoencoder Surrogates

For each layer of each model, trains an autoencoder surrogate that replicates its behavior.

Process:
1. Load 1024 random cropped images from load_lp_crops.py
2. For each layer Li:
   - Pass images through model and collect Li-1 outputs (inputs to Li) and Li outputs
   - Apply PCA to compress/expand to fixed autoencoder size
   - Train 4-layer autoencoder (128-128-128-128) to map compressed inputs to compressed outputs
   - Save autoencoder + PCA operators as layer profile

Flow: Li-1 output -> Li input PCA -> autoencoder_i -> Li output inverse PCA -> Li+1 input
"""

import os
import argparse
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import onnx
import onnx2torch
from transformers import VisionEncoderDecoderModel, TrOCRProcessor
import numpy as np
from datetime import datetime
from tqdm import tqdm
import warnings
import pickle
from sklearn.decomposition import PCA
from PIL import Image
import torchvision.transforms as T

# Import from foundationmodel
import sys
sys.path.insert(0, str(Path(__file__).parent))
from load_lp_crops import build_anchor_pool

warnings.filterwarnings("ignore")


# ============================================================================
# Autoencoder Architecture
# ============================================================================

class LayerAutoencoder(nn.Module):
    """
    4-layer autoencoder with 128-unit hidden layers.
    Maps compressed layer input to compressed layer output.
    """
    def __init__(self, input_dim, output_dim, hidden_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


# ============================================================================
# Dataset for License Plate Crops
# ============================================================================

class LPCropsDataset(Dataset):
    """
    Loads random cropped license plates from load_lp_crops.py.
    Stores raw PIL images and applies preprocessing per model.
    """
    def __init__(self, num_samples=1024, target_size=(64, 128), seed=42):
        """
        Args:
            num_samples: Number of random samples to load
            target_size: (height, width) for resizing
            seed: Random seed for reproducibility
        """
        print(f"Loading {num_samples} random license plate crops...")

        # Set random seed for reproducibility
        np.random.seed(seed)
        torch.manual_seed(seed)

        # Load samples from build_anchor_pool
        samples = []
        for img, text, meta in build_anchor_pool(max_per_dataset=10000):
            samples.append(img)
            if len(samples) >= num_samples:
                break

        # Shuffle and take num_samples
        if len(samples) > num_samples:
            indices = np.random.permutation(len(samples))[:num_samples]
            samples = [samples[i] for i in indices]

        print(f"Loaded {len(samples)} images")

        # Resize all images to target size
        self.images = []
        transform = T.Compose([
            T.Resize(target_size),
            T.ToTensor()
        ])

        for img in tqdm(samples, desc="Resizing images"):
            tensor = transform(img)  # [3, H, W] in [0, 1]
            self.images.append(tensor)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx]


class ModelSpecificDataset(Dataset):
    """
    Applies model-specific preprocessing to raw cropped plates.
    """
    def __init__(self, base_dataset, preprocessor=None, channels_last=False, scale_to_255=False, target_size=None):
        self.base_dataset = base_dataset
        self.preprocessor = preprocessor
        self.channels_last = channels_last
        self.scale_to_255 = scale_to_255
        self.target_size = target_size

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        img = self.base_dataset[idx].clone()  # [3, H, W] in [0, 1]

        # Resize if needed
        if self.target_size is not None:
            import torch.nn.functional as F
            img = F.interpolate(
                img.unsqueeze(0),
                size=self.target_size,
                mode='bilinear',
                align_corners=False
            ).squeeze(0)

        # Apply TrOCR processor if provided
        if self.preprocessor is not None:
            img_np = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            pil_img = Image.fromarray(img_np)
            processed = self.preprocessor(images=pil_img, return_tensors="pt")
            return processed.pixel_values.squeeze(0)

        # Apply format transformations
        if self.channels_last:
            img = img.permute(1, 2, 0)  # [C, H, W] -> [H, W, C]

        if self.scale_to_255:
            img = img * 255  # [0, 1] -> [0, 255]

        return img


# ============================================================================
# Model Loaders
# ============================================================================

def load_cct_model(device='cuda'):
    """Load CCT-XS-V1 Global model."""
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

    print("Model loaded successfully")
    return model, "cct_xs_v1_global"


def load_vitstr_model(device='cuda'):
    """Load ViTSTR Small model from doctr."""
    print("\n" + "="*80)
    print("Loading ViTSTR Small Model (doctr)")
    print("="*80)

    try:
        from doctr.models import vitstr_small

        print("Loading pretrained ViTSTR model...")
        model = vitstr_small(pretrained=True)
        model.eval()
        model.to(device)

        print("Model loaded successfully")
        return model, "vitstr_small"
    except ImportError:
        raise RuntimeError(
            "doctr library not found. Install it with:\n"
            "pip install python-doctr[torch]"
        )


def load_trocr_model(device='cuda'):
    """Load microsoft/trocr-small-printed vision encoder."""
    print("\n" + "="*80)
    print("Loading Microsoft TrOCR Small Printed Model")
    print("="*80)

    try:
        processor = TrOCRProcessor.from_pretrained("microsoft/trocr-small-printed")
        full_model = VisionEncoderDecoderModel.from_pretrained(
            "microsoft/trocr-small-printed"
        ).to(device)
        full_model.eval()

        model = full_model.encoder

        print("Model loaded successfully (vision encoder)")
        return model, "trocr_small_printed_encoder", processor
    except Exception as e:
        raise RuntimeError(
            f"Failed to load TrOCR model from HuggingFace. Error: {e}\n"
            "Please ensure transformers is installed: pip install transformers"
        )


# ============================================================================
# Layer Activation Extraction
# ============================================================================

def get_all_layers(model):
    """
    Get all leaf modules (layers with parameters) from a model.

    Returns:
        List of (layer_name, layer_module) tuples
    """
    layers = []
    for name, module in model.named_modules():
        # Skip container modules
        if isinstance(module, (nn.Sequential, nn.ModuleList, nn.ModuleDict)):
            continue

        # Include modules with parameters or buffers
        has_params = len(list(module.parameters())) > 0
        has_buffers = len(list(module.buffers())) > 0

        if has_params or has_buffers:
            layers.append((name, module))

    return layers


class LayerActivationCapture:
    """
    Captures activations before and after a specific layer.
    """
    def __init__(self, model):
        self.model = model
        self.activations = {}
        self.hooks = []

    def register_layer_hooks(self, layer_name, layer_module):
        """
        Register hooks to capture input and output of a specific layer.
        """
        self.activations = {}

        def hook_fn(module, input, output):
            # Store input and output
            if isinstance(input, tuple):
                self.activations['input'] = input[0].detach()
            else:
                self.activations['input'] = input.detach()

            if isinstance(output, torch.Tensor):
                self.activations['output'] = output.detach()
            elif isinstance(output, tuple):
                self.activations['output'] = output[0].detach()

        hook = layer_module.register_forward_hook(hook_fn)
        self.hooks.append(hook)

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

    def clear_activations(self):
        self.activations.clear()


def collect_layer_activations(model, dataset, layer_name, layer_module, device='cuda', batch_size=32):
    """
    Collect input and output activations for a specific layer across the dataset.

    Returns:
        Tuple of (inputs, outputs) as numpy arrays
        - inputs: [n_samples, input_features]
        - outputs: [n_samples, output_features]
    """
    capture = LayerActivationCapture(model)
    capture.register_layer_hooks(layer_name, layer_module)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    all_inputs = []
    all_outputs = []

    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)

            # Forward pass
            _ = model(batch)

            if 'input' in capture.activations and 'output' in capture.activations:
                # Flatten activations to [batch, features]
                input_act = capture.activations['input']
                output_act = capture.activations['output']

                # Flatten all dimensions except batch
                input_flat = input_act.reshape(input_act.shape[0], -1).cpu().numpy()
                output_flat = output_act.reshape(output_act.shape[0], -1).cpu().numpy()

                all_inputs.append(input_flat)
                all_outputs.append(output_flat)

            capture.clear_activations()

    capture.remove_hooks()

    if not all_inputs:
        return None, None

    inputs = np.concatenate(all_inputs, axis=0)
    outputs = np.concatenate(all_outputs, axis=0)

    return inputs, outputs


# ============================================================================
# PCA Compression/Expansion
# ============================================================================

def fit_pca_transform(data, target_dim, name="data"):
    """
    Fit PCA to data and return transformer that outputs target_dim dimensions.

    If data has more features than target_dim: compress using PCA
    If data has fewer features than target_dim: expand by padding with zeros

    Returns:
        pca_obj: Fitted PCA or None if no compression needed
        transform_fn: Function that transforms data to target_dim
        inverse_fn: Function that transforms from target_dim back to original
    """
    n_samples, n_features = data.shape

    print(f"  {name}: {n_features} features -> {target_dim} dimensions")

    if n_features > target_dim:
        # Compress using PCA
        pca = PCA(n_components=target_dim, random_state=42)
        pca.fit(data)

        explained_var = pca.explained_variance_ratio_.sum()
        print(f"    PCA compression: {explained_var:.4f} variance explained")

        def transform_fn(x):
            return pca.transform(x)

        def inverse_fn(x):
            return pca.inverse_transform(x)

        return pca, transform_fn, inverse_fn

    elif n_features < target_dim:
        # Expand by padding with zeros
        print(f"    Expanding with zero-padding")

        def transform_fn(x):
            padding = np.zeros((x.shape[0], target_dim - n_features))
            return np.concatenate([x, padding], axis=1)

        def inverse_fn(x):
            return x[:, :n_features]

        return None, transform_fn, inverse_fn

    else:
        # No transformation needed
        print(f"    No transformation needed")

        def identity_fn(x):
            return x

        return None, identity_fn, identity_fn


# ============================================================================
# Autoencoder Training
# ============================================================================

def train_autoencoder(inputs, outputs, latent_dim=128, epochs=50, batch_size=32, lr=0.001, device='cuda'):
    """
    Train autoencoder to map inputs to outputs.

    Args:
        inputs: Input data [n_samples, input_dim]
        outputs: Output data [n_samples, output_dim]
        latent_dim: Hidden layer dimension
        epochs: Number of training epochs
        batch_size: Batch size for training
        lr: Learning rate
        device: Device to train on

    Returns:
        Trained autoencoder model
    """
    input_dim = inputs.shape[1]
    output_dim = outputs.shape[1]

    # Create model
    model = LayerAutoencoder(input_dim, output_dim, hidden_dim=latent_dim).to(device)

    # Create dataset
    inputs_tensor = torch.from_numpy(inputs).float()
    outputs_tensor = torch.from_numpy(outputs).float()
    dataset = torch.utils.data.TensorDataset(inputs_tensor, outputs_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # Training loop
    model.train()
    pbar = tqdm(range(epochs), desc="    Training autoencoder", leave=False)

    for epoch in pbar:
        total_loss = 0
        for batch_inputs, batch_outputs in dataloader:
            batch_inputs = batch_inputs.to(device)
            batch_outputs = batch_outputs.to(device)

            optimizer.zero_grad()
            predictions = model(batch_inputs)
            loss = criterion(predictions, batch_outputs)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        pbar.set_postfix({'loss': f'{avg_loss:.6f}'})

    model.eval()
    return model


# ============================================================================
# Main Profiling Function
# ============================================================================

def profile_model_with_autoencoders(model, model_name, dataset, output_dir,
                                    device='cuda', batch_size=32,
                                    pca_dim=128, ae_epochs=50):
    """
    Profile a model by training autoencoder surrogates for each layer.

    Args:
        model: PyTorch model to profile
        model_name: Name identifier for the model
        dataset: Image dataset to profile over
        output_dir: Directory to save profiles
        device: Device to run on
        batch_size: Batch size for activation collection
        pca_dim: Target dimension for PCA compression/expansion
        ae_epochs: Number of epochs to train each autoencoder
    """
    print("\n" + "="*80)
    print(f"Profiling {model_name} with Autoencoder Surrogates")
    print("="*80)

    # Create output directory
    model_output_dir = Path(output_dir) / model_name
    model_output_dir.mkdir(parents=True, exist_ok=True)

    # Get all layers
    layers = get_all_layers(model)
    print(f"Found {len(layers)} layers to profile")

    # Profile each layer
    layer_profiles = {}

    for i, (layer_name, layer_module) in enumerate(tqdm(layers, desc=f"Profiling layers")):
        print(f"\nLayer {i+1}/{len(layers)}: {layer_name}")
        print(f"  Type: {layer_module.__class__.__name__}")

        try:
            # Collect activations
            print("  Collecting activations...")
            inputs, outputs = collect_layer_activations(
                model, dataset, layer_name, layer_module, device, batch_size
            )

            if inputs is None or outputs is None:
                print("  ⚠️  Failed to collect activations, skipping layer")
                continue

            print(f"  Input shape: {inputs.shape}, Output shape: {outputs.shape}")

            # Fit PCA transforms
            print("  Fitting PCA transforms...")
            input_pca, input_transform, input_inverse = fit_pca_transform(
                inputs, pca_dim, name="Input"
            )
            output_pca, output_transform, output_inverse = fit_pca_transform(
                outputs, pca_dim, name="Output"
            )

            # Transform data
            inputs_compressed = input_transform(inputs)
            outputs_compressed = output_transform(outputs)

            # Train autoencoder
            print(f"  Training autoencoder ({pca_dim}→{pca_dim}, {ae_epochs} epochs)...")
            autoencoder = train_autoencoder(
                inputs_compressed,
                outputs_compressed,
                latent_dim=128,
                epochs=ae_epochs,
                batch_size=batch_size,
                lr=0.001,
                device=device
            )

            # Evaluate reconstruction error
            with torch.no_grad():
                test_inputs = torch.from_numpy(inputs_compressed).float().to(device)
                predictions = autoencoder(test_inputs).cpu().numpy()
                mse = np.mean((predictions - outputs_compressed) ** 2)
                print(f"  Reconstruction MSE: {mse:.6f}")

            # Save profile (only save picklable objects - not local functions)
            profile = {
                'layer_name': layer_name,
                'layer_type': layer_module.__class__.__name__,
                'input_shape': inputs.shape,
                'output_shape': outputs.shape,
                'pca_dim': pca_dim,
                'input_pca': input_pca,
                'output_pca': output_pca,
                'autoencoder': autoencoder.cpu().state_dict(),
                'reconstruction_mse': float(mse),
            }

            layer_profiles[layer_name] = profile

            # Save individual layer profile
            layer_file = model_output_dir / f"layer_{i:03d}_{layer_name.replace('.', '_')}.pkl"
            with open(layer_file, 'wb') as f:
                pickle.dump(profile, f)

            print(f"  ✓ Saved to {layer_file.name}")

        except Exception as e:
            print(f"  ⚠️  Error profiling layer: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Save summary
    summary = {
        'model_name': model_name,
        'n_layers': len(layer_profiles),
        'pca_dim': pca_dim,
        'ae_epochs': ae_epochs,
        'timestamp': datetime.now().isoformat(),
        'layer_names': list(layer_profiles.keys()),
    }

    summary_file = model_output_dir / "profile_summary.pkl"
    with open(summary_file, 'wb') as f:
        pickle.dump(summary, f)

    print(f"\n✓ Profiling complete for {model_name}")
    print(f"  Profiled {len(layer_profiles)} layers")
    print(f"  Saved to {model_output_dir}")

    return layer_profiles


# ============================================================================
# Main Script
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Profile OCR Models with Autoencoder Surrogates')
    parser.add_argument('--output-dir', type=str, default='autoencoder_profiles',
                        help='Output directory for layer profiles (default: autoencoder_profiles)')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (cuda/mps/cpu). Auto-detects if not specified.')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for profiling (default: 32)')
    parser.add_argument('--num-samples', type=int, default=1024,
                        help='Number of random images to use (default: 1024)')
    parser.add_argument('--pca-dim', type=int, default=128,
                        help='Target dimension for PCA compression (default: 128)')
    parser.add_argument('--ae-epochs', type=int, default=50,
                        help='Number of epochs to train each autoencoder (default: 50)')
    parser.add_argument('--models', type=str, default='vitstr,cct,trocr',
                        help='Comma-separated list of models to profile: vitstr,cct,trocr (default: all)')
    args = parser.parse_args()

    # Auto-detect device
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
    print("OCR Model Autoencoder Profiler")
    print("="*80)
    print(f"Device: {device}")
    print(f"Output directory: {args.output_dir}")
    print(f"Number of samples: {args.num_samples}")
    print(f"PCA target dimension: {args.pca_dim}")
    print(f"Autoencoder epochs: {args.ae_epochs}")
    print(f"Batch size: {args.batch_size}")
    print()

    # Parse which models to profile
    models_to_profile = [m.strip().lower() for m in args.models.split(',')]

    # Load base dataset (shared across all models)
    print("Loading license plate crops dataset...")
    base_dataset = LPCropsDataset(
        num_samples=args.num_samples,
        target_size=(64, 128),
        seed=42
    )

    # Profile each requested model
    results = {}

    if 'vitstr' in models_to_profile:
        try:
            model, model_name = load_vitstr_model(device)

            # ViTSTR-specific preprocessing
            vitstr_dataset = ModelSpecificDataset(
                base_dataset,
                preprocessor=None,
                channels_last=False,
                scale_to_255=False,
                target_size=(32, 128)  # ViTSTR expects 32x128 input
            )

            profiles = profile_model_with_autoencoders(
                model, model_name, vitstr_dataset, args.output_dir,
                device, args.batch_size, args.pca_dim, args.ae_epochs
            )
            results[model_name] = profiles

            del model, vitstr_dataset
            torch.cuda.empty_cache() if device == 'cuda' else None
        except Exception as e:
            print(f"\n⚠️  ERROR profiling ViTSTR model: {e}")
            import traceback
            traceback.print_exc()

    if 'cct' in models_to_profile:
        try:
            model, model_name = load_cct_model(device)

            # CCT-specific preprocessing
            cct_dataset = ModelSpecificDataset(
                base_dataset,
                preprocessor=None,
                channels_last=True,
                scale_to_255=True
            )

            profiles = profile_model_with_autoencoders(
                model, model_name, cct_dataset, args.output_dir,
                device, args.batch_size, args.pca_dim, args.ae_epochs
            )
            results[model_name] = profiles

            del model, cct_dataset
            torch.cuda.empty_cache() if device == 'cuda' else None
        except Exception as e:
            print(f"\n⚠️  ERROR profiling CCT model: {e}")
            import traceback
            traceback.print_exc()

    if 'trocr' in models_to_profile:
        try:
            model, model_name, processor = load_trocr_model(device)

            # TrOCR-specific preprocessing
            trocr_dataset = ModelSpecificDataset(
                base_dataset,
                preprocessor=processor,
                channels_last=False,
                scale_to_255=False
            )

            profiles = profile_model_with_autoencoders(
                model, model_name, trocr_dataset, args.output_dir,
                device, args.batch_size, args.pca_dim, args.ae_epochs
            )
            results[model_name] = profiles

            del model, trocr_dataset, processor
            torch.cuda.empty_cache() if device == 'cuda' else None
        except Exception as e:
            print(f"\n⚠️  ERROR profiling TrOCR model: {e}")
            import traceback
            traceback.print_exc()

    # Summary
    print("\n" + "="*80)
    print("PROFILING COMPLETE")
    print("="*80)
    print(f"Successfully profiled {len(results)} model(s)")
    for model_name, profiles in results.items():
        print(f"  - {model_name}: {len(profiles)} layers")
    print(f"\nResults saved to: {args.output_dir}/")
    print("="*80)


if __name__ == "__main__":
    main()
