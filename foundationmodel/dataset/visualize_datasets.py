#!/usr/bin/env python3
"""
Visualize example images from each dataset.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from load_datasets import iter_dataset

# Datasets to visualize
DATASETS = ["iiit5k", "mjsynth", "iam_line", "icdar2015"]

# Create figure with subplots
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes = axes.flatten()

for idx, dataset_name in enumerate(DATASETS):
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
