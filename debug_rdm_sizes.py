#!/usr/bin/env python3
"""
Debug RDM Sizes: Investigate why some layers have mismatched RDM dimensions
"""

import h5py
from pathlib import Path
from collections import Counter


def inspect_rdm_file(rdm_path: Path):
    """Inspect RDM file and report size mismatches."""
    model_name = rdm_path.parent.name

    print(f"\n{'='*80}")
    print(f"Model: {model_name}")
    print(f"File: {rdm_path}")
    print(f"{'='*80}")

    with h5py.File(rdm_path, "r") as f:
        model_group = f[model_name]
        n_images_expected = model_group["metadata"].attrs["n_images"]

        print(f"Expected n_images (from metadata): {n_images_expected}")
        print()

        # Collect all layer info
        layer_info = []
        layer_keys = sorted([k for k in model_group.keys() if k != "metadata"])

        for layer_key in layer_keys:
            layer_group = model_group[layer_key]
            layer_name = layer_group.attrs["layer_name"]
            layer_type = layer_group.attrs.get("layer_type", "Unknown")
            activation_shape = layer_group.attrs.get("activation_shape", "Unknown")
            rdm_shape = layer_group["rdm"].shape

            layer_info.append(
                {
                    "key": layer_key,
                    "name": layer_name,
                    "type": layer_type,
                    "activation_shape": activation_shape,
                    "rdm_shape": rdm_shape,
                    "n_samples": rdm_shape[0],
                }
            )

        # Group by size
        size_groups = {}
        for info in layer_info:
            n = info["n_samples"]
            if n not in size_groups:
                size_groups[n] = []
            size_groups[n].append(info)

        # Report
        print(f"Found {len(layer_keys)} layers with {len(size_groups)} different RDM sizes:")
        print()

        for size in sorted(size_groups.keys()):
            layers = size_groups[size]
            print(f"Size: {size} ({len(layers)} layers)")

            if size != n_images_expected:
                ratio = size / n_images_expected
                print(f"  ⚠️  Mismatch! {ratio:.1f}x expected size")

            # Show first few layers
            for info in layers[:3]:
                print(f"  - {info['name']}")
                print(f"    Type: {info['type']}")
                print(f"    Activation shape: {info['activation_shape']}")
                print(f"    RDM shape: {info['rdm_shape']}")

            if len(layers) > 3:
                print(f"  ... and {len(layers) - 3} more layers")

            print()

        # Analyze patterns
        print("\nPattern Analysis:")
        print("-" * 80)

        # Check if size ratios are integer multiples
        sizes = sorted(size_groups.keys())
        if len(sizes) > 1:
            base_size = sizes[0]
            for size in sizes[1:]:
                ratio = size / base_size
                if ratio == int(ratio):
                    print(
                        f"Size {size} is exactly {int(ratio)}x the base size {base_size}"
                    )
                    print(
                        f"  → Likely caused by sequence/patch dimension being flattened incorrectly"
                    )

                    # Show which layer types have this issue
                    types_with_issue = Counter(
                        [info["type"] for info in size_groups[size]]
                    )
                    print(f"  → Affected layer types: {dict(types_with_issue)}")
                    print()

        # Check activation shapes
        print("\nActivation Shape Patterns:")
        print("-" * 80)
        shape_to_rdm_size = {}
        for info in layer_info:
            shape_str = str(info["activation_shape"])
            n_samples = info["n_samples"]

            if shape_str not in shape_to_rdm_size:
                shape_to_rdm_size[shape_str] = []
            shape_to_rdm_size[shape_str].append((info["name"], n_samples))

        for shape_str, samples_list in sorted(shape_to_rdm_size.items()):
            rdm_sizes = Counter([s[1] for s in samples_list])
            if len(rdm_sizes) == 1:
                size = list(rdm_sizes.keys())[0]
                print(f"Activation shape {shape_str} → RDM size {size}")
                print(f"  ({len(samples_list)} layers)")
            else:
                print(f"Activation shape {shape_str} → Multiple RDM sizes: {dict(rdm_sizes)}")
                print(f"  ⚠️  Inconsistent! Same activation shape should give same RDM size")


def main():
    rdm_dir = Path("rdm_profiles")

    if not rdm_dir.exists():
        print(f"RDM directory not found: {rdm_dir}")
        return

    print("RDM Size Debugging")
    print("=" * 80)

    # Find all RDM files
    rdm_files = list(rdm_dir.glob("*/*_rdms.h5"))

    if not rdm_files:
        print(f"No RDM files found in {rdm_dir}")
        return

    print(f"Found {len(rdm_files)} RDM files to inspect")

    for rdm_file in rdm_files:
        try:
            inspect_rdm_file(rdm_file)
        except Exception as e:
            print(f"ERROR inspecting {rdm_file}: {e}")
            import traceback

            traceback.print_exc()

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
