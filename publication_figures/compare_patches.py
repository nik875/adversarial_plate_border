#!/usr/bin/env python3
"""
Create a side-by-side comparison of Zhang et al's patch and our impersonation patch.
"""

import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

# Load images
zhang_patch = Image.open('zhang_patch.png')
our_patch = Image.open('final_patches/SHX8459/best_patches/patch_epoch_0095.png')

# Create figure with two subplots
fig, axes = plt.subplots(1, 2, figsize=(14, 7))

# Display Zhang et al's patch
axes[0].imshow(zhang_patch)
axes[0].set_title('Zhang et al. (2024)',
                  fontsize=14, fontweight='bold', pad=15)
axes[0].axis('off')

# Add label overlay for Zhang's patch
height_zhang, width_zhang = np.array(zhang_patch).shape[:2]
axes[0].text(width_zhang/2, height_zhang/2, 'License Plate\nAttaches Here',
            ha='center', va='center', fontsize=12, fontweight='bold',
            color='black',
            bbox=dict(boxstyle='round,pad=0.7', facecolor='white',
                     alpha=0.9, edgecolor='black', linewidth=2))

# Display our patch
axes[1].imshow(our_patch)
axes[1].set_title('SPAR Disruption Attack',
                  fontsize=14, fontweight='bold', pad=15)
axes[1].axis('off')

# Add label overlay for our patch
height_our, width_our = np.array(our_patch).shape[:2]
axes[1].text(width_our/2, height_our/2, 'License Plate\nAttaches Here',
            ha='center', va='center', fontsize=12, fontweight='bold',
            color='black',
            bbox=dict(boxstyle='round,pad=0.7', facecolor='white',
                     alpha=0.9, edgecolor='black', linewidth=2))

plt.suptitle('Adversarial Rim Patch Comparison',
             fontsize=16, fontweight='bold', y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('patch_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Patch comparison saved to patch_comparison.png")
