#!/usr/bin/env python3
"""
Visualize example images from each dataset and count total images.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import traceback
from load_datasets import iter_dataset, DATASETS

# Datasets to visualize (dataset, split)
DATASETS_TO_VIZ = [
    ("iiit5k", "test"),
    ("mjsynth", "train"),
    ("iam_line", "train"),
    ("icdar2013", "train"),
    ("icdar2013", "test"),
    ("icdar2015", "train"),
    ("icdar2015", "test"),
]

# Create figure with subplots (7 datasets, use 2x4 grid with one empty)
fig, axes = plt.subplots(2, 4, figsize=(18, 8))
axes = axes.flatten()

# Count images per dataset split
print("Counting images in each dataset...\n")
dataset_counts = {}
total_images = 0

for dataset_name, split in DATASETS_TO_VIZ:
    key = f"{dataset_name}/{split}"
    dataset_counts[key] = 0

    sample_idx = 0
    try:
        for img, text, meta in iter_dataset(dataset_name, split):
            sample_idx += 1
            dataset_counts[key] += 1
    except Exception as e:
        print(f"\n  ✗ {dataset_name.upper()} ({split}) ERROR (sample_idx={sample_idx}):")
        print(f"    Type: {type(e).__name__}")
        print(f"    Message: {str(e)}")
        print(f"    Full traceback:")
        for line in traceback.format_exc().split('\n'):
            if line.strip():
                print(f"      {line}")
        print()

    total_images += dataset_counts[key]
    print(f"  {dataset_name:12} ({split:5}) : {dataset_counts[key]:>6} images")

print(f"\n  {'TOTAL':12}          : {total_images:>6} images\n")

# Visualize examples
for idx, (dataset_name, split) in enumerate(DATASETS_TO_VIZ):
    ax = axes[idx]

    try:
        # Load one sample
        for img, text, meta in iter_dataset(dataset_name, split, max_samples=1):
            ax.imshow(img)
            ax.set_title(f"{dataset_name.upper()} ({split})\nLabel: {text}", fontsize=10, fontweight='bold')
            ax.axis('off')
            break
    except Exception as e:
        ax.text(0.5, 0.5, f"Error loading {dataset_name}:\n{str(e)}",
                ha='center', va='center', transform=ax.transAxes,
                fontsize=10, color='red')
        ax.set_title(f"{dataset_name.upper()} ({split})", fontsize=10, fontweight='bold')
        ax.axis('off')

# Hide the extra subplot
axes[-1].axis('off')

plt.tight_layout()
plt.savefig('dataset_examples.png', dpi=150, bbox_inches='tight')
print("✓ Saved dataset examples to dataset_examples.png")
plt.show()
