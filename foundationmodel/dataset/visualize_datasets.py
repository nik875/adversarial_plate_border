#!/usr/bin/env python3
"""
Visualize example images from each dataset and count total images.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from load_datasets import iter_dataset, DATASETS

# Datasets to visualize
DATASET_NAMES = ["iiit5k", "mjsynth", "iam_line", "icdar2015"]

# Create figure with subplots
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes = axes.flatten()

# Count images per dataset
print("Counting images in each dataset...\n")
dataset_counts = {}
total_images = 0

for dataset_name in DATASET_NAMES:
    try:
        dataset_counts[dataset_name] = 0
        splits = DATASETS[dataset_name]["splits"]

        for split in splits:
            for img, text, meta in iter_dataset(dataset_name, split):
                dataset_counts[dataset_name] += 1

        total_images += dataset_counts[dataset_name]
        print(f"  {dataset_name:12} : {dataset_counts[dataset_name]:>6} images")
    except Exception as e:
        dataset_counts[dataset_name] = 0
        print(f"  {dataset_name:12} : Error - {str(e)[:50]}")

print(f"\n  {'TOTAL':12} : {total_images:>6} images\n")

# Visualize examples
for idx, dataset_name in enumerate(DATASET_NAMES):
    ax = axes[idx]

    try:
        # Get first split available
        cfg_splits = {
            "iiit5k": "test",
            "mjsynth": "train",
            "iam_line": "train",
            "icdar2015": "train",
        }
        split = cfg_splits[dataset_name]

        # Load one sample
        for img, text, meta in iter_dataset(dataset_name, split, max_samples=1):
            ax.imshow(img)
            ax.set_title(f"{dataset_name.upper()}\nLabel: {text}", fontsize=10, fontweight='bold')
            ax.axis('off')
            break
    except Exception as e:
        ax.text(0.5, 0.5, f"Error loading {dataset_name}:\n{str(e)}",
                ha='center', va='center', transform=ax.transAxes,
                fontsize=10, color='red')
        ax.set_title(dataset_name.upper(), fontsize=10, fontweight='bold')
        ax.axis('off')

plt.tight_layout()
plt.savefig('dataset_examples.png', dpi=150, bbox_inches='tight')
print("✓ Saved dataset examples to dataset_examples.png")
plt.show()
