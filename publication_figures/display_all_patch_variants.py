#!/usr/bin/env python3
"""
Display all 8 patch variants (2 final + 6 ablations) in a grid.
No PII - uses generic labels.
"""

import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
import glob

def get_best_patch(variant_dir):
    """Find the best (last) patch image in best_patches directory"""
    best_patches_dir = variant_dir / 'best_patches'
    if not best_patches_dir.exists():
        return None

    # Find all .png files
    patch_files = sorted(best_patches_dir.glob('patch_epoch_*.png'))
    if not patch_files:
        return None

    # Return the last one (highest epoch number)
    return patch_files[-1]

def create_variant_label(variant_name):
    """Create clean label without PII"""
    if '_notv' in variant_name and '_nohomo' in variant_name:
        return 'No TV + No Homography'
    elif '_notv' in variant_name:
        return 'No TV Loss'
    elif '_nohomo' in variant_name:
        return 'No Homography'
    else:
        return 'Full (TV + Homography)'

# Define variant order (2 rows x 2 columns, disruption only)
variant_order = [
    ('SHX8459', 0, 0),
    ('SHX8459_nohomo', 0, 1),
    ('SHX8459_notv', 1, 0),
    ('SHX8459_notv_nohomo', 1, 1),
]

# Try loading from final_patches first for the full variant
final_patches_map = {
    'SHX8459': Path('final_patches/SHX8459/best_patches')
}

# Base directory for ablation variants
ablation_base = Path('patch_variants_20260101_001131')

# Create figure
fig, axes = plt.subplots(2, 2, figsize=(10, 6))

for variant_name, row, col in variant_order:
    ax = axes[row, col]

    # Try to find the patch
    patch_path = None

    # Check final_patches for full variants
    if variant_name in final_patches_map and final_patches_map[variant_name].exists():
        patch_files = sorted(final_patches_map[variant_name].glob('patch_epoch_*.png'))
        if patch_files:
            patch_path = patch_files[-1]

    # If not found, check ablation directory
    if patch_path is None:
        variant_dir = ablation_base / f'patches_{variant_name}'
        patch_path = get_best_patch(variant_dir)

    # Display patch or placeholder
    if patch_path and patch_path.exists():
        img = Image.open(patch_path)
        ax.imshow(img)

        # Add label overlay
        height, width = img.height, img.width
        ax.text(width/2, height/2, 'License Plate\nAttaches Here',
               ha='center', va='center', fontsize=9, fontweight='bold',
               color='black',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                        alpha=0.8, edgecolor='black', linewidth=1.5))

        epoch_num = patch_path.stem.split('_')[-1]
        print(f"✓ Loaded {variant_name} (epoch {epoch_num})")
    else:
        ax.text(0.5, 0.5, 'No patch\navailable', ha='center', va='center',
               transform=ax.transAxes, fontsize=10)
        print(f"✗ Could not find patch for {variant_name}")

    # Set title
    ax.set_title(create_variant_label(variant_name), fontsize=11, fontweight='bold', pad=8)
    ax.set_aspect('auto')
    ax.axis('off')

plt.suptitle('Ablation Cases (Disruption)',
             fontsize=15, fontweight='bold')
plt.tight_layout()
plt.subplots_adjust(hspace=0.15)
plt.savefig('all_patch_variants_display.png', dpi=300, bbox_inches='tight')
print("\n✅ Saved: all_patch_variants_display.png")
