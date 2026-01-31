"""
Model RDM (Representational Dissimilarity Matrix) Profiler

A generalizable PyTorch model profiler that computes Representational Dissimilarity
Matrices (RDMs) for all layers when given a model and sample images.

This module provides tools to:
1. Hook into any PyTorch model to capture layer activations
2. Compute RDMs using correlation distance metric
3. Store RDMs with full metadata in HDF5 format
4. Load and inspect RDMs across multiple models

==============================================================================
USAGE EXAMPLES
==============================================================================

Example 1: Profile all layers in a model
-------------------------------------------
from rdm_profiler import ModelRDMProfiler
from torch.utils.data import DataLoader, Dataset
import torchvision.models as models

# Prepare your dataset
class SimpleImageDataset(Dataset):
    def __init__(self, image_list):
        self.images = image_list

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx]

# Load model and images
model = models.resnet18(pretrained=True)
dataset = SimpleImageDataset(image_tensors)

# Create profiler for all layers
profiler = ModelRDMProfiler(
    model=model,
    model_name='resnet18',
    device='cuda'
)

# Profile with batching
rdms = profiler.profile(
    image_dataset=dataset,
    batch_size=16,
    save_path='rdms/resnet18_rdms.h5'
)

# Access RDMs
print(rdms.keys())  # Layer names
layer_rdm = rdms['layer1.0.conv1']  # [n_images, n_images]


Example 2: Profile specific layers by name
-------------------------------------------
profiler = ModelRDMProfiler(
    model=model,
    model_name='resnet18',
    device='cuda',
    layer_names=['layer1.0.conv1', 'layer2.0.conv1', 'layer4']
)

rdms = profiler.profile(dataset, batch_size=16, save_path='rdms/resnet18_selective.h5')


Example 3: Profile specific layers by index
-------------------------------------------
# Profile 1st, 6th, 11th, and 16th layers
profiler = ModelRDMProfiler(
    model=model,
    model_name='resnet18',
    device='cuda',
    layer_indices=[0, 5, 10, 15]
)

rdms = profiler.profile(dataset, batch_size=16)


Example 4: Load and analyze saved RDMs
-------------------------------------------
from rdm_profiler import RDMStorage

storage = RDMStorage('rdms/resnet18_rdms.h5')
rdms, metadata = storage.load('resnet18')

print(f"Profiled {metadata['n_images']} images")
print(f"Metric: {metadata['metric']}")
print(f"Layers: {list(rdms.keys())}")


Example 5: Use with custom model zoo
-------------------------------------------
# If you have a custom function to load models
def load_custom_model(model_name, device):
    # Your custom loading logic
    return model.to(device)

model = load_custom_model('my_model', 'cuda')
profiler = ModelRDMProfiler(model, model_name='my_model', device='cuda')
rdms = profiler.profile(dataset, batch_size=8, save_path='rdms/my_model.h5')

==============================================================================
"""

import torch
import torch.nn as nn
import numpy as np
import h5py
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Union
from collections import OrderedDict
from datetime import datetime
from tqdm import tqdm
import warnings


def get_device(device: Optional[str] = None) -> str:
    """
    Auto-detect compute device if not specified.

    Args:
        device: Target device ('cuda', 'mps', 'cpu', or None for auto-detect)

    Returns:
        str: Device identifier ('cuda', 'mps', or 'cpu')
    """
    if device is None:
        if torch.cuda.is_available():
            return 'cuda'
        elif torch.backends.mps.is_available():
            return 'mps'
        else:
            return 'cpu'
    return device


class LayerActivationExtractor:
    """
    Hooks model layers and captures activations during forward pass.

    This class automatically discovers leaf modules (Conv, Linear, etc.) in a model
    and registers forward hooks to capture their outputs. Supports filtering by:
    - Layer names (exact match or regex patterns)
    - Layer indices (position in model)

    Attributes:
        model: PyTorch model to hook
        layer_names: Optional list of exact layer names to hook
        layer_indices: Optional list of layer indices to hook
        activations: Dict mapping layer names to their captured activations
    """

    def __init__(
        self,
        model: nn.Module,
        layer_names: Optional[List[str]] = None,
        layer_indices: Optional[List[int]] = None
    ):
        """
        Initialize the activation extractor.

        Args:
            model: PyTorch model to extract activations from
            layer_names: List of exact layer names to hook (e.g., ['layer1.0.conv1', 'layer3'])
                        If None, all leaf layers are hooked
            layer_indices: List of layer indices to hook (e.g., [0, 5, 10])
                          If None, all leaf layers are hooked
                          Mutually exclusive with layer_names

        Raises:
            ValueError: If both layer_names and layer_indices are specified
        """
        if layer_names is not None and layer_indices is not None:
            raise ValueError("Cannot specify both layer_names and layer_indices")

        self.model = model
        self.layer_names = layer_names
        self.layer_indices = layer_indices
        self.activations = {}
        self.hooks = []

    def _is_leaf_module(self, module: nn.Module) -> bool:
        """
        Check if module is a leaf module (not a container).

        Leaf modules are ones that have parameters or buffers (e.g., Conv, Linear, BatchNorm).
        Container modules like Sequential, ModuleList are skipped.

        Args:
            module: Module to check

        Returns:
            bool: True if module is a leaf module
        """
        # Container modules to skip
        container_types = (
            nn.Sequential, nn.ModuleList, nn.ModuleDict,
            nn.ParameterList, nn.ParameterDict
        )

        if isinstance(module, container_types):
            return False

        # A leaf module should have parameters or buffers
        has_params = len(list(module.parameters())) > 0
        has_buffers = len(list(module.buffers())) > 0

        return has_params or has_buffers

    def _should_hook_layer(self, name: str, idx: int) -> bool:
        """
        Check if a layer matches the filter criteria.

        Args:
            name: Layer name from named_modules
            idx: Layer index in enumeration order

        Returns:
            bool: True if layer should be hooked
        """
        # No filtering - hook all layers
        if self.layer_names is None and self.layer_indices is None:
            return True

        # Filter by exact layer names
        if self.layer_names is not None:
            return name in self.layer_names

        # Filter by layer indices
        if self.layer_indices is not None:
            return idx in self.layer_indices

        return False

    def _make_hook_fn(self, layer_name: str):
        """
        Create a hook function closure for capturing activations.

        Args:
            layer_name: Name of the layer for storage key

        Returns:
            callable: Hook function that captures activations
        """
        def hook_fn(module, input, output):
            # Store output - don't detach, let caller decide based on use_grad
            # Move to CPU if on GPU to save memory during collection
            if isinstance(output, torch.Tensor):
                self.activations[layer_name] = output.detach()
            elif isinstance(output, tuple):
                # Some layers return tuples - take first element
                self.activations[layer_name] = output[0].detach()
            else:
                warnings.warn(f"Layer {layer_name} returned unexpected type {type(output)}")

        return hook_fn

    def register_hooks(self):
        """
        Register forward hooks on all matching layers.

        Discovers all leaf modules in the model and registers hooks on those
        matching the filter criteria (if any).
        """
        self.clear_activations()

        # Enumerate all modules to get indices
        leaf_idx = 0
        for name, module in self.model.named_modules():
            if not self._is_leaf_module(module):
                continue

            # Check if this layer should be hooked
            if not self._should_hook_layer(name, leaf_idx):
                leaf_idx += 1
                continue

            # Register hook
            hook_fn = self._make_hook_fn(name)
            hook_handle = module.register_forward_hook(hook_fn)
            self.hooks.append(hook_handle)

            leaf_idx += 1

    def remove_hooks(self):
        """Remove all registered hooks to cleanup."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

    def clear_activations(self):
        """Clear stored activations to free memory."""
        self.activations.clear()

    def get_hooked_layers(self) -> List[str]:
        """
        Get list of all hooked layers.

        Returns:
            list: Layer names that are currently hooked
        """
        hooked_layers = []
        leaf_idx = 0

        for name, module in self.model.named_modules():
            if not self._is_leaf_module(module):
                continue

            if self._should_hook_layer(name, leaf_idx):
                hooked_layers.append(name)

            leaf_idx += 1

        return hooked_layers


class ActivationProcessor:
    """
    Processes and normalizes activations for RDM computation.

    Handles different activation shapes from various layer types:
    - Conv layers: [batch, C, H, W] → [batch, features]
    - Linear layers: [batch, features] → [batch, features]
    - General: Flatten all dimensions except batch
    """

    @staticmethod
    def flatten_activation(activation: torch.Tensor) -> torch.Tensor:
        """
        Flatten activation to 2D tensor [batch, features].

        Args:
            activation: Activation tensor of shape [batch, ...] where ... can be any dimensions

        Returns:
            torch.Tensor: 2D tensor [batch_size, flattened_features]
        """
        if len(activation.shape) == 2:
            # Already 2D - assume [batch, features]
            return activation

        elif len(activation.shape) == 4:
            # Conv layer: [batch, channels, height, width] → [batch, channels*height*width]
            # Use reshape instead of view to handle non-contiguous tensors
            batch_size = activation.shape[0]
            flattened = activation.reshape(batch_size, -1)
            return flattened

        else:
            # General case: flatten everything except first dimension
            # Use reshape instead of view to handle non-contiguous tensors
            batch_size = activation.shape[0]
            flattened = activation.reshape(batch_size, -1)
            return flattened


class RDMComputer:
    """
    Computes Representational Dissimilarity Matrices (RDMs).

    Uses correlation distance metric (1 - Pearson correlation coefficient)
    to compute pairwise dissimilarities between images based on layer activations.

    The RDM is a symmetric matrix where RDM[i,j] represents the dissimilarity
    between image i and image j based on a layer's activations.
    """

    def __init__(self, metric: str = 'correlation'):
        """
        Initialize RDM computer.

        Args:
            metric: Distance metric to use. Currently supports 'correlation'.
                   (correlation distance = 1 - Pearson correlation coefficient)
        """
        if metric != 'correlation':
            raise ValueError(f"Metric '{metric}' not supported. Use 'correlation'.")
        self.metric = metric

    def compute_rdm(self, activations: np.ndarray) -> np.ndarray:
        """
        Compute RDM from activations.

        Args:
            activations: Activation matrix [n_images, n_features] as numpy array

        Returns:
            np.ndarray: RDM matrix [n_images, n_images] with values in [0, 2]
                       - 0 = identical activations
                       - 2 = perfect negative correlation
        """
        if self.metric == 'correlation':
            return self._correlation_distance_matrix(activations)
        else:
            raise ValueError(f"Unknown metric: {self.metric}")

    @staticmethod
    def _correlation_distance_matrix(X: np.ndarray) -> np.ndarray:
        """
        Compute correlation distance matrix for all image pairs.

        Distance = 1 - Pearson correlation coefficient

        Args:
            X: Activation matrix [n_images, n_features]

        Returns:
            np.ndarray: Distance matrix [n_images, n_images]
        """
        # Normalize activations (z-score normalization)
        X_normalized = (X - X.mean(axis=1, keepdims=True)) / (X.std(axis=1, keepdims=True) + 1e-8)

        # Compute correlation matrix: X_norm @ X_norm.T
        # Each element [i,j] = sum(X_norm[i] * X_norm[j]) / n_features
        # This is the Pearson correlation coefficient
        n_features = X.shape[1]
        correlation_matrix = (X_normalized @ X_normalized.T) / n_features

        # Convert correlation to distance: distance = 1 - correlation
        rdm = 1.0 - correlation_matrix

        # Ensure RDM is symmetric (due to numerical precision)
        rdm = (rdm + rdm.T) / 2.0

        # Ensure diagonal is zero (distance to self)
        np.fill_diagonal(rdm, 0.0)

        return rdm.astype(np.float32)


class RDMStorage:
    """
    Saves and loads RDMs with metadata in HDF5 format.

    File structure:
    /model_name/
      /layer_0/
        rdm: [n_images, n_images] dataset
        layer_name: str attribute
        layer_type: str attribute
        activation_shape: tuple attribute
      /layer_1/
        ...
      /metadata/
        n_images, metric, timestamp, etc.
    """

    def __init__(self, filepath: Union[str, Path]):
        """
        Initialize RDM storage manager.

        Args:
            filepath: Path to HDF5 file for saving/loading RDMs
        """
        self.filepath = Path(filepath)
        self.filepath.parent.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        model_name: str,
        rdms: Dict[str, np.ndarray],
        layer_info: Dict[str, Dict],
        metadata: Dict
    ):
        """
        Save RDMs and metadata to HDF5 file.

        Args:
            model_name: Name of the model
            rdms: Dict mapping layer names to RDM matrices [n_images, n_images]
            layer_info: Dict mapping layer names to info dicts with keys:
                       - 'layer_type': str
                       - 'activation_shape': tuple
            metadata: Dict with keys like 'n_images', 'metric', 'timestamp', etc.
        """
        with h5py.File(self.filepath, 'a') as f:
            # Create model group
            if model_name in f:
                del f[model_name]

            model_group = f.create_group(model_name)

            # Save RDMs for each layer
            for layer_name, rdm in rdms.items():
                layer_group = model_group.create_group(f"layer_{len(model_group)-1}")

                # Save RDM matrix with compression
                layer_group.create_dataset(
                    'rdm',
                    data=rdm,
                    compression='gzip',
                    compression_opts=4,
                    dtype=np.float32
                )

                # Save layer metadata as attributes
                layer_group.attrs['layer_name'] = layer_name
                layer_group.attrs['layer_type'] = layer_info[layer_name]['layer_type']
                layer_group.attrs['activation_shape'] = str(layer_info[layer_name]['activation_shape'])

            # Save global metadata
            metadata_group = model_group.create_group('metadata')
            for key, value in metadata.items():
                if isinstance(value, (str, int, float)):
                    metadata_group.attrs[key] = value
                else:
                    metadata_group.attrs[key] = str(value)

    def load(self, model_name: str) -> Tuple[Dict[str, np.ndarray], Dict]:
        """
        Load RDMs and metadata from HDF5 file.

        Args:
            model_name: Name of the model to load

        Returns:
            Tuple of:
            - rdms: Dict mapping layer names to RDM matrices
            - metadata: Dict with profiling metadata

        Raises:
            KeyError: If model_name not found in file
            FileNotFoundError: If HDF5 file doesn't exist
        """
        if not self.filepath.exists():
            raise FileNotFoundError(f"HDF5 file not found: {self.filepath}")

        rdms = OrderedDict()
        metadata = {}

        with h5py.File(self.filepath, 'r') as f:
            if model_name not in f:
                raise KeyError(f"Model '{model_name}' not found in {self.filepath}")

            model_group = f[model_name]

            # Load RDMs (skip metadata group)
            for key in sorted(model_group.keys()):
                if key == 'metadata':
                    continue

                layer_group = model_group[key]
                layer_name = layer_group.attrs['layer_name']
                rdms[layer_name] = layer_group['rdm'][:]

            # Load metadata
            if 'metadata' in model_group:
                metadata_group = model_group['metadata']
                for key in metadata_group.attrs.keys():
                    metadata[key] = metadata_group.attrs[key]

        return rdms, metadata


class ModelRDMProfiler:
    """
    High-level orchestrator for profiling models with RDMs.

    Main API that combines layer hooking, activation processing, RDM computation,
    and storage into a single workflow.

    Supports:
    - Automatic layer discovery with optional filtering
    - Batched inference for memory efficiency
    - Progress tracking with tqdm
    - Automatic GPU/device management
    - HDF5 storage with full metadata
    """

    def __init__(
        self,
        model: nn.Module,
        model_name: str,
        device: Optional[str] = None,
        layer_names: Optional[List[str]] = None,
        layer_indices: Optional[List[int]] = None
    ):
        """
        Initialize the model RDM profiler.

        Args:
            model: PyTorch model to profile
            model_name: Name for this model (used in storage)
            device: Device to run on ('cuda', 'mps', 'cpu', or None for auto-detect)
            layer_names: Optional list of exact layer names to profile
                        E.g., ['layer1.0.conv1', 'layer3.fc']
                        If None and layer_indices is None, profiles all leaf layers
            layer_indices: Optional list of layer indices to profile
                          E.g., [0, 5, 10] profiles 1st, 6th, 11th layers
                          Mutually exclusive with layer_names

        Raises:
            ValueError: If both layer_names and layer_indices are specified
        """
        self.model = model.eval()  # Set to evaluation mode
        self.model_name = model_name
        self.device = get_device(device)
        self.model.to(self.device)

        # Create activation extractor
        self.extractor = LayerActivationExtractor(
            model=self.model,
            layer_names=layer_names,
            layer_indices=layer_indices
        )

        # Create processors and computers
        self.processor = ActivationProcessor()
        self.rdm_computer = RDMComputer(metric='correlation')

    def profile(
        self,
        image_dataset,
        batch_size: int = 8,
        save_path: Optional[Union[str, Path]] = None
    ) -> Dict[str, np.ndarray]:
        """
        Profile model on dataset and compute RDMs for all hooked layers.

        Args:
            image_dataset: PyTorch Dataset providing batches of preprocessed images
                          Should yield tensors of shape [3, H, W] (or [C, H, W] for custom models)
            batch_size: Batch size for inference
            save_path: Optional path to save RDMs to HDF5 file

        Returns:
            Dict mapping layer names to RDM matrices [n_images, n_images]

        Raises:
            RuntimeError: If no layers are hooked
        """
        from torch.utils.data import DataLoader

        # Register hooks on matching layers
        self.extractor.register_hooks()
        hooked_layers = self.extractor.get_hooked_layers()

        if not hooked_layers:
            raise RuntimeError("No layers matched filter criteria. Check layer names/indices.")

        print(f"Profiling {len(hooked_layers)} layers on {self.device}")
        print(f"Layers: {hooked_layers}")

        # Create dataloader
        dataloader = DataLoader(
            image_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0  # Disable multiprocessing to avoid device issues
        )

        # Collect activations from all images
        all_activations = {layer_name: [] for layer_name in hooked_layers}
        layer_info = {layer_name: {} for layer_name in hooked_layers}

        # Track expected batch size for detecting sequence-collapsed activations
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
                    images = images.to(self.device)

                # Track expected batch size from first batch
                if expected_batch_size is None:
                    expected_batch_size = images.shape[0]

                # Forward pass
                _ = self.model(images)

                # Collect activations from this batch
                for layer_name in hooked_layers:
                    if layer_name in self.extractor.activations:
                        activation = self.extractor.activations[layer_name]

                        # Store shape info from first batch
                        if batch_idx == 0:
                            layer_info[layer_name]['activation_shape'] = tuple(activation.shape)

                        # Move to CPU and flatten
                        activation_cpu = activation.cpu().numpy()
                        flattened = self.processor.flatten_activation(
                            torch.from_numpy(activation_cpu)
                        ).numpy()

                        # Check for sequence-collapsed activations
                        # If first dim != batch_size but is a multiple, reshape and aggregate
                        if flattened.shape[0] != expected_batch_size:
                            if flattened.shape[0] % expected_batch_size == 0:
                                seq_len = flattened.shape[0] // expected_batch_size
                                # Reshape [batch*seq_len, features] -> [batch, seq_len, features]
                                reshaped = flattened.reshape(expected_batch_size, seq_len, -1)
                                # Aggregate across sequence dimension (mean pooling)
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
                self.extractor.clear_activations()

        # Remove hooks
        self.extractor.remove_hooks()

        # Concatenate activations and compute RDMs
        print("\nComputing RDMs...")
        rdms = {}

        for layer_name in tqdm(hooked_layers, desc="Computing RDMs"):
            # Concatenate all batch activations
            if not all_activations[layer_name]:
                warnings.warn(f"No activations collected for layer {layer_name}")
                continue

            layer_activations = np.concatenate(all_activations[layer_name], axis=0)
            n_images = layer_activations.shape[0]

            # Compute RDM
            rdm = self.rdm_computer.compute_rdm(layer_activations)
            rdms[layer_name] = rdm

            # Get layer type
            layer_module = None
            for name, module in self.model.named_modules():
                if name == layer_name:
                    layer_module = module
                    break

            layer_info[layer_name]['layer_type'] = (
                layer_module.__class__.__name__ if layer_module else 'Unknown'
            )

        # Save if path provided
        if save_path:
            metadata = {
                'n_images': n_images,
                'metric': 'correlation',
                'timestamp': datetime.now().isoformat(),
                'device': self.device,
                'batch_size': batch_size
            }

            storage = RDMStorage(save_path)
            storage.save(self.model_name, rdms, layer_info, metadata)
            print(f"\nSaved RDMs to {save_path}")

        print(f"\nProfiling complete. Computed RDMs for {len(rdms)} layers.")

        return rdms
