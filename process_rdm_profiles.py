#!/usr/bin/env python3
"""
Process RDM Profiles: Double-center and eigendecompose

For each RDM layer profile:
1. Load the RDM matrix [n_samples, n_samples]
2. Double-center the RDM
3. Eigendecompose to extract k=8 principal modes
4. Save the layer profile (8 features per sample)

Output: layer_profiles/ directory with one file per model, containing
all layer profiles stacked together.
"""

import numpy as np
import h5py
from pathlib import Path
from tqdm import tqdm
import json


def double_center_distance_matrix(D: np.ndarray) -> np.ndarray:
    """
    Double-center a distance matrix to convert to Gram-like matrix.

    Formula: G[i,j] = -0.5 * (D[i,j] - D[i,:].mean() - D[:,j].mean() + D.mean())

    Args:
        D: Distance matrix [n, n]

    Returns:
        Gram-like matrix [n, n] (symmetric)
    """
    row_mean = D.mean(axis=1, keepdims=True)
    col_mean = D.mean(axis=0, keepdims=True)
    grand_mean = D.mean()

    G = -0.5 * (D - row_mean - col_mean + grand_mean)

    return G


def eigendecompose_gram(G: np.ndarray, k: int = 8) -> tuple:
    """
    Eigendecompose a Gram matrix and extract top k features.

    Args:
        G: Gram matrix [n, n]
        k: Number of principal modes to extract (default: 8)

    Returns:
        eigenvectors: Top k eigenvectors [n, k]
        eigenvalues: Top k eigenvalues [k]
    """
    # Compute eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eigh(G)

    # Sort by eigenvalue (descending)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Take top k
    eigenvalues_k = eigenvalues[:k]
    eigenvectors_k = eigenvectors[:, :k]

    return eigenvectors_k, eigenvalues_k


def process_rdm_file(rdm_path: Path, output_dir: Path, k: int = 8):
    """
    Process a single RDM file: double-center and eigendecompose all layers.

    Args:
        rdm_path: Path to RDM HDF5 file
        output_dir: Output directory for layer profiles
        k: Number of principal modes (default: 8)
    """
    model_name = rdm_path.parent.name

    print(f"\nProcessing {model_name}...")
    print(f"Loading RDMs from {rdm_path}")

    # Load RDMs and metadata
    with h5py.File(rdm_path, "r") as f:
        model_group = f[model_name]
        n_images = model_group["metadata"].attrs["n_images"]

        # Get all layer names (skip metadata group)
        layer_keys = sorted([k for k in model_group.keys() if k != "metadata"])

        # Initialize arrays to store all layer profiles
        # Shape: [n_images, n_layers * k]
        layer_profiles = []
        layer_names_list = []

        print(f"Processing {len(layer_keys)} layers...")

        # Check RDM sizes first
        rdm_sizes = {}
        for layer_key in layer_keys:
            layer_group = model_group[layer_key]
            rdm_shape = layer_group["rdm"].shape
            layer_name = layer_group.attrs["layer_name"]
            rdm_sizes[layer_name] = rdm_shape[0]

        # Find expected size (most common)
        from collections import Counter

        size_counts = Counter(rdm_sizes.values())
        expected_size = size_counts.most_common(1)[0][0]

        print(f"  Expected n_images: {expected_size}")
        mismatched_layers = [
            (name, size)
            for name, size in rdm_sizes.items()
            if size != expected_size
        ]
        if mismatched_layers:
            print(
                f"  WARNING: {len(mismatched_layers)} layers have mismatched sizes:"
            )
            for name, size in mismatched_layers[:5]:  # Show first 5
                print(f"    - {name}: {size}")
            if len(mismatched_layers) > 5:
                print(f"    ... and {len(mismatched_layers) - 5} more")

        with tqdm(layer_keys, desc="Eigendecomposing layers") as pbar:
            for layer_key in pbar:
                layer_group = model_group[layer_key]
                layer_name = layer_group.attrs["layer_name"]

                # Load RDM
                rdm = layer_group["rdm"][:]  # [n_images, n_images]

                # Skip layers with mismatched size
                if rdm.shape[0] != expected_size:
                    pbar.write(
                        f"  Skipping {layer_name} (size {rdm.shape[0]} != {expected_size})"
                    )
                    continue

                # Double-center
                G = double_center_distance_matrix(rdm)

                # Eigendecompose
                eigenvectors, eigenvalues = eigendecompose_gram(G, k=k)

                # Store eigenvectors as features for this layer
                # Shape: [n_images, k]
                layer_profiles.append(eigenvectors)
                layer_names_list.append(layer_name)

                pbar.set_postfix({"layer": layer_name[:30]})

        if not layer_profiles:
            print(f"  ERROR: No valid layers found for {model_name}")
            return

        # Stack all layers: [n_images, n_layers * k]
        all_profiles = np.concatenate(layer_profiles, axis=1)

        # Save to output directory
        output_file = output_dir / f"{model_name}_layer_profiles.npz"
        np.savez_compressed(
            output_file,
            profiles=all_profiles.astype(np.float32),
            n_images=n_images,
            n_layers=len(layer_names_list),
            k=k,
        )

        # Save layer names for reference
        metadata = {
            "model_name": model_name,
            "n_images": int(n_images),
            "n_layers": len(layer_names_list),
            "k": k,
            "layer_names": layer_names_list,
        }
        metadata_file = output_dir / f"{model_name}_metadata.json"
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"Saved profiles: {output_file}")
        print(f"  Shape: {all_profiles.shape}")
        print(f"  Layers: {len(layer_names_list)}")
        print(f"  Features per layer: {k}")
        print(f"Saved metadata: {metadata_file}")

        return all_profiles


def main():
    # Setup paths
    rdm_dir = Path("rdm_profiles")
    output_dir = Path("layer_profiles")

    if not rdm_dir.exists():
        print(f"RDM directory not found: {rdm_dir}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Process RDM Profiles: Double-center and Eigendecompose")
    print("=" * 80)
    print(f"Input directory: {rdm_dir}")
    print(f"Output directory: {output_dir}")
    print()

    # Find all RDM files
    rdm_files = list(rdm_dir.glob("*/") )
    rdm_files = [d for d in rdm_files if d.is_dir()]

    if not rdm_files:
        print(f"No model directories found in {rdm_dir}")
        return

    print(f"Found {len(rdm_files)} models to process")
    print()

    # Process each model
    for model_dir in rdm_files:
        rdm_file = model_dir / f"{model_dir.name}_rdms.h5"

        if not rdm_file.exists():
            print(f"RDM file not found: {rdm_file}")
            continue

        try:
            process_rdm_file(rdm_file, output_dir, k=8)
        except Exception as e:
            print(f"ERROR processing {model_dir.name}: {e}")
            import traceback

            traceback.print_exc()

    # Summary
    print("\n" + "=" * 80)
    print("PROCESSING COMPLETE")
    print("=" * 80)
    print(f"Layer profiles saved to: {output_dir}")
    print()
    print("Output files:")
    print("  - *_layer_profiles.npz: Eigendecomposed features [n_images, n_layers*k]")
    print("  - *_metadata.json: Layer names and metadata")
    print()
    print("To load profiles:")
    print("  data = np.load('layer_profiles/model_layer_profiles.npz')")
    print("  profiles = data['profiles']  # [n_images, n_layers*k]")
    print("  metadata = json.load(open('layer_profiles/model_metadata.json'))")
    print("=" * 80)


if __name__ == "__main__":
    main()
