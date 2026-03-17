#!/usr/bin/env python3
"""
Display the final disruption and impersonation patches side by side.
"""

import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path

VARIANTS_DIR = Path('patch_variants_20260101_001131')

patches = {
    'Disruption':     VARIANTS_DIR / 'patches_SHX8459'  / 'best_patches',
    'Impersonation':  VARIANTS_DIR / 'patches_VJJ7744'  / 'best_patches',
}

fig, axes = plt.subplots(1, 2, figsize=(10, 5))

for ax, (label, best_dir) in zip(axes, patches.items()):
    patch_files = sorted(best_dir.glob('patch_epoch_*.png'))
    if not patch_files:
        ax.text(0.5, 0.5, 'No patch found', ha='center', va='center',
                transform=ax.transAxes, fontsize=11)
        ax.axis('off')
        ax.set_title(label, fontsize=13, fontweight='bold', pad=8)
        print(f"✗ No patch found for {label}")
        continue

    patch_path = patch_files[-1]
    img = Image.open(patch_path)
    ax.imshow(img)

    h, w = img.height, img.width
    ax.text(w / 2, h / 2, 'License Plate\nAttaches Here',
            ha='center', va='center', fontsize=10, fontweight='bold',
            color='black',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                      alpha=0.8, edgecolor='black', linewidth=1.5))

    ax.set_title(label, fontsize=13, fontweight='bold', pad=8)
    ax.axis('off')
    print(f"✓ Loaded {label} ({patch_path.stem})")

plt.tight_layout()
plt.savefig('final_patches_display.png', dpi=300, bbox_inches='tight')
print("\n✅ Saved: final_patches_display.png")
