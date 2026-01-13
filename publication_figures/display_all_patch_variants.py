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
    if variant_name.startswith('VJJ7744'):
        attack_type = 'Impersonation'
    else:
        attack_type = 'Disruption'

    if '_notv' in variant_name and '_nohomo' in variant_name:
        config = 'No TV + No Homography'
    elif '_notv' in variant_name:
        config = 'No TV Loss'
    elif '_nohomo' in variant_name:
        config = 'No Homography'
    else:
        config = 'Full (TV + Homography)'

    return f"{attack_type}\n{config}"

# Define variant order (4 rows x 2 columns)
variant_order = [
    ('VJJ7744', 0, 0),
    ('SHX8459', 0, 1),
    ('VJJ7744_nohomo', 1, 0),
    ('SHX8459_nohomo', 1, 1),
    ('VJJ7744_notv', 2, 0),
    ('SHX8459_notv', 2, 1),
    ('VJJ7744_notv_nohomo', 3, 0),
    ('SHX8459_notv_nohomo', 3, 1)
]

# Try loading from final_patches first for the full variants
final_patches_map = {
    'VJJ7744': Path('final_patches/VJJ7744/best_patches'),
    'SHX8459': Path('final_patches/SHX8459/best_patches')
}

# Base directory for ablation variants
ablation_base = Path('patch_variants_20260101_001131')

# Create figure
fig, axes = plt.subplots(4, 2, figsize=(10, 16))

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
    ax.axis('off')

# Add column labels at top
column_labels = ['Impersonation Attack', 'Disruption Attack']
for j, label in enumerate(column_labels):
    axes[0, j].text(0.5, 1.15, label.upper(), ha='center', va='bottom',
                   transform=axes[0, j].transAxes, fontsize=13, fontweight='bold')

plt.suptitle('Adversarial Rim Patches - Ablation Study',
             fontsize=15, fontweight='bold', y=0.995)
plt.tight_layout(rect=[0, 0, 1, 0.99])
plt.savefig('all_patch_variants_display.png', dpi=300, bbox_inches='tight')
print("\n✅ Saved: all_patch_variants_display.png")
