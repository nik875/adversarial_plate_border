#!/usr/bin/env python3
"""
Profile OCR Models with Autoencoder Surrogates

For each layer of each model, trains an autoencoder surrogate that replicates its behavior.

Process:
1. Load all available cropped images from roboflow_lpr, kaggle_lp, mercosur datasets
2. For each layer Li:
   - Pass images through model and collect Li-1 outputs (inputs to Li) and Li outputs
   - Apply PCA to compress/expand to fixed dimension (default 256)
   - Train deep autoencoder (3-layer encoder/decoder with 256-unit hidden layers + 10% dropout)
   - With validation split (20%), cosine annealing LR (5e-3 → 1e-6), and early stopping (patience 5)
   - Save autoencoder + PCA operators as layer profile
   - Save detailed training history and metrics as JSON for analysis

Output per layer:
- layer_XXX_name.pkl: Full profile with autoencoder weights, PCA objects
- layer_XXX_name_metrics.json: Training history and performance metrics

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
import json
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
    Deep autoencoder with 256-unit hidden layers and dropout.
    Maps compressed layer input to compressed layer output.
    """
    def __init__(self, input_dim, output_dim, hidden_dim=256, dropout_rate=0.1):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
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
    def __init__(self, num_samples=1024, target_size=(64, 128), seed=42, datasets=None):
        """
        Args:
            num_samples: Number of random samples to load
            target_size: (height, width) for resizing
            seed: Random seed for reproducibility
            datasets: List of dataset names to load. If None, loads from all datasets.
        """
        if num_samples <= 0:
            print(f"Loading all available license plate crops...")
        else:
            print(f"Loading up to {num_samples} random license plate crops...")
        if datasets:
            print(f"  Using datasets: {', '.join(datasets)}")

        # Set random seed for reproducibility
        np.random.seed(seed)
        torch.manual_seed(seed)

        # Load samples from specified datasets
        samples = []
        total_loaded = 0

        if datasets is None:
            # Load from all datasets
            from load_lp_crops import build_anchor_pool
            iterator = build_anchor_pool(max_per_dataset=10000)
        else:
            # Load from specific datasets only
            from load_lp_crops import iter_dataset
            def custom_pool():
                for dataset_name in datasets:
                    for split in ['train', 'test', 'valid', 'val']:
                        try:
                            yield from iter_dataset(dataset_name, split, max_samples=10000)
                        except (ValueError, StopIteration):
                            pass
            iterator = custom_pool()

        for img, text, meta in iterator:
            samples.append(img)
            total_loaded += 1

        # Shuffle all samples
        indices = np.random.permutation(len(samples))
        samples = [samples[i] for i in indices]

        # If num_samples > 0 and less than total, sample; otherwise use all
        if num_samples > 0 and len(samples) > num_samples:
            samples = samples[:num_samples]
            print(f"Loaded {len(samples)} license plate images (sampled from {total_loaded} total available)")
        else:
            print(f"Loaded {total_loaded} license plate images (all available)")

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


def collect_layer_activations(model, dataset, layer_name, layer_module, device='cuda',
                             batch_size=32, max_memory_gb=2.0):
    """
    Collect input and output activations for a specific layer across the dataset.

    Strategy: Collect activations up to memory limit, then return them.
    The caller will fit PCA on this sample and apply to full dataset iteratively.

    Args:
        model: Model to profile
        dataset: Dataset to collect from
        layer_name: Name of layer
        layer_module: Layer module object
        device: Device to use
        batch_size: Batch size for collection
        max_memory_gb: Max memory to use for collecting activations (default 2GB)

    Returns:
        Tuple of (inputs, outputs) as numpy arrays (sample for PCA fitting)
        - inputs: [n_samples, input_features] (limited by memory)
        - outputs: [n_samples, output_features] (limited by memory)
    """
    capture = LayerActivationCapture(model)
    capture.register_layer_hooks(layer_name, layer_module)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    all_inputs = []
    all_outputs = []
    max_memory_bytes = max_memory_gb * 1024 * 1024 * 1024

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

                # Check memory usage
                current_memory = sum(x.nbytes for x in all_inputs) + sum(x.nbytes for x in all_outputs)
                if current_memory > max_memory_bytes:
                    break

            capture.clear_activations()

    capture.remove_hooks()

    if not all_inputs:
        # No activations captured - likely hook not triggered
        return None, None

    inputs = np.concatenate(all_inputs, axis=0)
    outputs = np.concatenate(all_outputs, axis=0)

    return inputs, outputs


def apply_pca_to_dataset(model, dataset, layer_name, layer_module, input_pca, output_pca,
                         input_transform, output_transform, device='cuda', batch_size=32):
    """
    Iterate through entire dataset, applying PCA compression to activations on-the-fly.
    Returns compressed activations without storing raw activations in memory.

    Returns:
        Tuple of (compressed_inputs, compressed_outputs) as numpy arrays
    """
    capture = LayerActivationCapture(model)
    capture.register_layer_hooks(layer_name, layer_module)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    all_inputs_compressed = []
    all_outputs_compressed = []

    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)

            # Forward pass
            _ = model(batch)

            if 'input' in capture.activations and 'output' in capture.activations:
                # Flatten activations to [batch, features]
                input_act = capture.activations['input']
                output_act = capture.activations['output']

                input_flat = input_act.reshape(input_act.shape[0], -1).cpu().numpy()
                output_flat = output_act.reshape(output_act.shape[0], -1).cpu().numpy()

                # Apply PCA immediately (no storage of raw activations)
                input_compressed = input_transform(input_flat)
                output_compressed = output_transform(output_flat)

                all_inputs_compressed.append(input_compressed)
                all_outputs_compressed.append(output_compressed)

            capture.clear_activations()

    capture.remove_hooks()

    if not all_inputs_compressed:
        return None, None

    inputs_compressed = np.concatenate(all_inputs_compressed, axis=0)
    outputs_compressed = np.concatenate(all_outputs_compressed, axis=0)

    return inputs_compressed, outputs_compressed


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

def train_autoencoder(inputs, outputs, latent_dim=256, epochs=100, batch_size=32,
                     max_lr=5e-3, min_lr=1e-6, val_split=0.2, early_stop_patience=5,
                     device='cuda', dropout_rate=0.2):
    """
    Train autoencoder to map inputs to outputs with validation and early stopping.

    Args:
        inputs: Input data [n_samples, input_dim]
        outputs: Output data [n_samples, output_dim]
        latent_dim: Hidden layer dimension
        epochs: Max number of training epochs
        batch_size: Batch size for training
        max_lr: Maximum learning rate for cosine annealing
        min_lr: Minimum learning rate for cosine annealing
        val_split: Fraction of data to use for validation (0.2 = 20%)
        early_stop_patience: Stop if validation loss doesn't improve for N epochs
        device: Device to train on
        dropout_rate: Dropout rate for each layer (default 0.2 = 20%)

    Returns:
        Tuple of (trained_model, best_val_loss, training_history)
    """
    input_dim = inputs.shape[1]
    output_dim = outputs.shape[1]

    # Create model
    model = LayerAutoencoder(input_dim, output_dim, hidden_dim=latent_dim,
                            dropout_rate=dropout_rate).to(device)

    # Create dataset and split into train/val
    inputs_tensor = torch.from_numpy(inputs).float()
    outputs_tensor = torch.from_numpy(outputs).float()
    dataset = torch.utils.data.TensorDataset(inputs_tensor, outputs_tensor)

    n_samples = len(dataset)
    n_val = int(n_samples * val_split)
    n_train = n_samples - n_val

    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [n_train, n_val]
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=max_lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=min_lr
    )
    criterion = nn.MSELoss()

    # Training loop with early stopping and history tracking
    best_val_loss = float('inf')
    patience_counter = 0
    history = {
        'train_loss': [],
        'val_loss': [],
        'learning_rate': [],
        'epoch': []
    }

    pbar = tqdm(range(epochs), desc="    Training autoencoder", leave=False)

    for epoch in pbar:
        # Training phase
        model.train()
        train_loss = 0
        for batch_inputs, batch_outputs in train_loader:
            batch_inputs = batch_inputs.to(device)
            batch_outputs = batch_outputs.to(device)

            optimizer.zero_grad()
            predictions = model(batch_inputs)
            loss = criterion(predictions, batch_outputs)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        # Validation phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_inputs, batch_outputs in val_loader:
                batch_inputs = batch_inputs.to(device)
                batch_outputs = batch_outputs.to(device)

                predictions = model(batch_inputs)
                loss = criterion(predictions, batch_outputs)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)

        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']

        # Learning rate scheduling
        scheduler.step()

        # Record history
        history['epoch'].append(epoch)
        history['train_loss'].append(float(avg_train_loss))
        history['val_loss'].append(float(avg_val_loss))
        history['learning_rate'].append(float(current_lr))

        # Early stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
        else:
            patience_counter += 1

        # Update progress bar
        pbar.set_postfix({
            'train': f'{avg_train_loss:.6f}',
            'val': f'{avg_val_loss:.6f}',
            'patience': f'{patience_counter}/{early_stop_patience}'
        })

        # Early stopping
        if patience_counter >= early_stop_patience:
            pbar.close()
            break

    model.eval()
    return model, best_val_loss, history


# ============================================================================
# Main Profiling Function
# ============================================================================

def profile_model_with_autoencoders(model, model_name, dataset, output_dir,
                                    device='cuda', batch_size=32,
                                    pca_dim=256, ae_epochs=100, max_lr=5e-3, min_lr=1e-6,
                                    val_split=0.2, early_stop_patience=5, max_pca_memory_gb=2.0):
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
            # PASS 1: Collect sample activations (up to memory limit) for PCA fitting
            print("  Collecting activations sample (up to {:.1f}GB)...".format(max_pca_memory_gb))
            inputs_sample, outputs_sample = collect_layer_activations(
                model, dataset, layer_name, layer_module, device, batch_size, max_pca_memory_gb
            )

            if inputs_sample is None or outputs_sample is None:
                reason = []
                if inputs_sample is None:
                    reason.append("input hook not triggered")
                if outputs_sample is None:
                    reason.append("output hook not triggered")
                print(f"  ⚠️  Failed to collect activations ({', '.join(reason)}) - skipping layer")
                continue

            print(f"  Sample shape: inputs {inputs_sample.shape}, outputs {outputs_sample.shape}")

            # Fit PCA transforms on sample
            print("  Fitting PCA transforms...")
            input_pca, input_transform, input_inverse = fit_pca_transform(
                inputs_sample, pca_dim, name="Input"
            )
            output_pca, output_transform, output_inverse = fit_pca_transform(
                outputs_sample, pca_dim, name="Output"
            )

            # Calculate explained variance metrics
            input_explained_var = float(input_pca.explained_variance_ratio_.sum()) if input_pca is not None else 1.0
            output_explained_var = float(output_pca.explained_variance_ratio_.sum()) if output_pca is not None else 1.0

            # PASS 2: Apply PCA to entire dataset iteratively (memory efficient)
            print("  Applying PCA to full dataset...")
            inputs_compressed, outputs_compressed = apply_pca_to_dataset(
                model, dataset, layer_name, layer_module, input_pca, output_pca,
                input_transform, output_transform, device, batch_size
            )

            if inputs_compressed is None or outputs_compressed is None:
                reason = []
                if inputs_compressed is None:
                    reason.append("input PCA application failed")
                if outputs_compressed is None:
                    reason.append("output PCA application failed")
                print(f"  ⚠️  Failed to process full dataset ({', '.join(reason)}) - skipping layer")
                continue

            print(f"  Full dataset compressed: {inputs_compressed.shape}")
            print(f"  PCA quality: Input {input_explained_var:.4f} | Output {output_explained_var:.4f} variance explained")

            # Train autoencoder
            print(f"  Training autoencoder ({pca_dim}→{pca_dim}, {ae_epochs} epochs, val_split={val_split*100:.0f}%)...")
            autoencoder, best_val_loss, train_history = train_autoencoder(
                inputs_compressed,
                outputs_compressed,
                latent_dim=pca_dim,
                epochs=ae_epochs,
                batch_size=batch_size,
                max_lr=max_lr,
                min_lr=min_lr,
                val_split=val_split,
                early_stop_patience=early_stop_patience,
                device=device
            )

            # Evaluate reconstruction error on training set
            with torch.no_grad():
                test_inputs = torch.from_numpy(inputs_compressed).float().to(device)
                predictions = autoencoder(test_inputs).cpu().numpy()
                mse = np.mean((predictions - outputs_compressed) ** 2)
                print(f"  Final train MSE: {mse:.6f}, best val MSE: {best_val_loss:.6f}")

            # Save profile (only save picklable objects - not local functions)
            profile = {
                'layer_name': layer_name,
                'layer_type': layer_module.__class__.__name__,
                'input_shape': inputs_compressed.shape,  # Shape after full dataset processing
                'output_shape': outputs_compressed.shape,  # Shape after full dataset processing
                'pca_dim': pca_dim,
                'input_pca': input_pca,
                'output_pca': output_pca,
                'input_pca_explained_variance': input_explained_var,
                'output_pca_explained_variance': output_explained_var,
                'autoencoder': autoencoder.cpu().state_dict(),
                'train_mse': float(mse),
                'val_mse': float(best_val_loss),
                'training_history': train_history,
                'epochs_trained': len(train_history['epoch']),
            }

            layer_profiles[layer_name] = profile

            # Save individual layer profile
            layer_file = model_output_dir / f"layer_{i:03d}_{layer_name.replace('.', '_')}.pkl"
            with open(layer_file, 'wb') as f:
                pickle.dump(profile, f)

            # Save training metrics as JSON for easy analysis
            metrics_file = model_output_dir / f"layer_{i:03d}_{layer_name.replace('.', '_')}_metrics.json"
            metrics = {
                'layer_name': layer_name,
                'layer_type': layer_module.__class__.__name__,
                'input_shape_full': list(inputs_compressed.shape),  # Full dataset
                'output_shape_full': list(outputs_compressed.shape),  # Full dataset
                'input_shape_sample': list(inputs_sample.shape),  # Used for PCA fitting
                'output_shape_sample': list(outputs_sample.shape),  # Used for PCA fitting
                'pca_dim': pca_dim,
                'input_pca_explained_variance': round(input_explained_var, 6),  # Proportion of variance explained
                'output_pca_explained_variance': round(output_explained_var, 6),  # Proportion of variance explained
                'train_mse': float(mse),
                'val_mse': float(best_val_loss),
                'epochs_trained': len(train_history['epoch']),
                'training_history': {
                    'epoch': train_history['epoch'],
                    'train_loss': train_history['train_loss'],
                    'val_loss': train_history['val_loss'],
                    'learning_rate': train_history['learning_rate']
                }
            }
            with open(metrics_file, 'w') as f:
                json.dump(metrics, f, indent=2)

            print(f"  ✓ Saved profile to {layer_file.name}")
            print(f"    Metrics: {metrics_file.name}")

        except Exception as e:
            error_type = type(e).__name__
            error_msg = str(e)
            print(f"  ⚠️  Error profiling layer: {error_type}")
            print(f"     Details: {error_msg}")
            if 'CUDA out of memory' in error_msg or 'RuntimeError' in error_type:
                print(f"     Suggestion: Reduce --batch-size or --pca-dim")
            elif 'NaN' in error_msg or 'inf' in error_msg:
                print(f"     Suggestion: Check for exploding/vanishing gradients in activations")
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
                        help='Base output directory for layer profiles (default: autoencoder_profiles)')
    parser.add_argument('--unique-run', action='store_true', default=True,
                        help='Append timestamp to output directory (default: True)')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (cuda/mps/cpu). Auto-detects if not specified.')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for profiling (default: 32)')
    parser.add_argument('--num-samples', type=int, default=0,
                        help='Max number of random images to use (0=all available, default: 0)')
    parser.add_argument('--pca-dim', type=int, default=256,
                        help='Target dimension for PCA compression (default: 256)')
    parser.add_argument('--ae-epochs', type=int, default=100,
                        help='Max number of epochs to train each autoencoder (default: 100)')
    parser.add_argument('--max-lr', type=float, default=5e-3,
                        help='Maximum learning rate for cosine annealing (default: 5e-3)')
    parser.add_argument('--min-lr', type=float, default=1e-6,
                        help='Minimum learning rate for cosine annealing (default: 1e-6)')
    parser.add_argument('--val-split', type=float, default=0.2,
                        help='Fraction of data to use for validation (default: 0.2)')
    parser.add_argument('--early-stop-patience', type=int, default=5,
                        help='Stop training if validation loss doesn\'t improve for N epochs (default: 5)')
    parser.add_argument('--max-pca-memory', type=float, default=2.0,
                        help='Max memory in GB for collecting activations before PCA fitting (default: 2.0)')
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
    # Append timestamp to output directory for unique runs
    if args.unique_run:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"{args.output_dir}_{timestamp}"
    else:
        output_dir = args.output_dir

    print(f"Device: {device}")
    print(f"Output directory: {output_dir}")
    print(f"Number of samples: {args.num_samples}")
    print(f"PCA memory limit: {args.max_pca_memory:.1f}GB (collects sample, then processes full dataset)")
    print(f"PCA target dimension: {args.pca_dim}")
    print(f"Autoencoder epochs: {args.ae_epochs} (early stopping patience: {args.early_stop_patience})")
    print(f"Learning rate: {args.max_lr} → {args.min_lr} (cosine annealing)")
    print(f"Validation split: {args.val_split*100:.0f}%")
    print(f"Batch size: {args.batch_size}")
    print()

    # Parse which models to profile
    models_to_profile = [m.strip().lower() for m in args.models.split(',')]

    # Load base dataset (shared across all models)
    print("Loading license plate crops dataset...")
    base_dataset = LPCropsDataset(
        num_samples=args.num_samples,
        target_size=(64, 128),
        seed=42,
        datasets=['roboflow_lpr', 'kaggle_lp', 'mercosur']
    )

    # Profile each requested model (in order: cct, vitstr, trocr)
    results = {}

    if 'cct' in models_to_profile:
        try:
            model, model_name = load_cct_model(device)

            # CCT-specific preprocessing
            print(f"\nProfiling CCT...")
            cct_dataset = ModelSpecificDataset(
                base_dataset,
                preprocessor=None,
                channels_last=True,  # CCT expects [H, W, C]
                scale_to_255=True    # CCT expects [0, 255]
            )

            profiles = profile_model_with_autoencoders(
                model, model_name, cct_dataset, output_dir,
                device, args.batch_size, args.pca_dim, args.ae_epochs,
                args.max_lr, args.min_lr, args.val_split, args.early_stop_patience,
                args.max_pca_memory
            )
            results[model_name] = profiles

            del model, cct_dataset
            torch.cuda.empty_cache() if device == 'cuda' else None
        except Exception as e:
            print(f"\n✗ ERROR profiling CCT: {e}")
            import traceback
            traceback.print_exc()

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
                model, model_name, vitstr_dataset, output_dir,
                device, args.batch_size, args.pca_dim, args.ae_epochs,
                args.max_lr, args.min_lr, args.val_split, args.early_stop_patience,
                args.max_pca_memory
            )
            results[model_name] = profiles

            del model, vitstr_dataset
            torch.cuda.empty_cache() if device == 'cuda' else None
        except Exception as e:
            print(f"\n⚠️  ERROR profiling ViTSTR model: {e}")
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
                device, args.batch_size, args.pca_dim, args.ae_epochs,
                args.max_lr, args.min_lr, args.val_split, args.early_stop_patience,
                args.max_pca_memory
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
