#!/usr/bin/env python3
"""
Visualize how the adapter transforms a patch at a given epoch.

Loads a patch and the most recent adapter checkpoint, then shows:
- Original patch (what we optimize)
- Adapted patch (what the white-box model sees)
"""

import argparse
import glob
from pathlib import Path
import torch
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Import from optimize_patch to get PatchAdapter class
from optimize_patch import PatchAdapter, PATCH_HEIGHT, PATCH_WIDTH


def find_patch_file(epoch: int, patches_dir: str = "bb_patches") -> Path:
    """Find the patch file for a given epoch."""
    patch_path = Path(patches_dir) / f"patch_epoch_{epoch:04d}.pt"
    if not patch_path.exists():
        raise FileNotFoundError(f"Patch file not found: {patch_path}")
    return patch_path


def find_latest_adapter(epoch: int, patches_dir: str = "bb_patches") -> Path:
    """Find the most recent adapter checkpoint at or before the given epoch."""
    # Look for adapter files
    pattern = str(Path(patches_dir) / "models_adapter_epoch_*.pt")
    adapter_files = glob.glob(pattern)

    if not adapter_files:
        raise FileNotFoundError(f"No adapter checkpoints found in {patches_dir}")

    # Extract epoch numbers
    valid_adapters = []
    for path in adapter_files:
        try:
            # Extract epoch from filename: models_adapter_epoch_0012.pt -> 12
            filename = Path(path).stem
            adapter_epoch = int(filename.split('_')[-1])
            if adapter_epoch <= epoch:
                valid_adapters.append((adapter_epoch, path))
        except (ValueError, IndexError):
            continue

    if not valid_adapters:
        raise FileNotFoundError(f"No adapter checkpoint found at or before epoch {epoch}")

    # Get the most recent one
    valid_adapters.sort(reverse=True)
    latest_epoch, latest_path = valid_adapters[0]

    print(f"Using adapter from epoch {latest_epoch}")
    return Path(latest_path)


def load_patch(patch_path: Path, device: str = 'cpu') -> torch.Tensor:
    """Load patch tensor from .pt file."""
    state = torch.load(patch_path, map_location=device)
    patch = state['patch']
    print(f"Loaded patch from epoch {state['epoch']}")
    return patch


def load_adapter(adapter_path: Path, device: str = 'cpu') -> PatchAdapter:
    """Load adapter model from checkpoint."""
    # Create adapter
    adapter = PatchAdapter(patch_height=PATCH_HEIGHT, patch_width=PATCH_WIDTH)

    # Load state dict
    state = torch.load(adapter_path, map_location=device)
    if 'adapter' in state:
        adapter.load_state_dict(state['adapter'])
    else:
        raise ValueError(f"No adapter found in checkpoint: {adapter_path}")

    adapter.to(device)
    adapter.eval()

    return adapter


def visualize_patches(
    patch: torch.Tensor,
    adapted_patch: torch.Tensor,
    epoch: int,
    adapter_epoch: int,
    save_path: str = None
):
    """
    Visualize original and adapted patches side by side.

    Args:
        patch: Original patch tensor [3, H, W]
        adapted_patch: Adapted patch tensor [3, H, W]
        epoch: Patch epoch number
        adapter_epoch: Adapter checkpoint epoch
        save_path: Optional path to save figure
    """
    fig = plt.figure(figsize=(15, 6))
    gs = GridSpec(1, 3, figure=fig, width_ratios=[1, 1, 1])

    # Normalize patches to [0, 1] for display
    patch_img = torch.sigmoid(patch).permute(1, 2, 0).cpu().numpy()
    adapted_img = adapted_patch.permute(1, 2, 0).cpu().numpy()

    # Compute difference
    diff = adapted_img - patch_img
    diff_magnitude = ((diff ** 2).sum(axis=2) ** 0.5)

    # Original patch
    ax1 = fig.add_subplot(gs[0])
    ax1.imshow(patch_img)
    ax1.set_title(f"Original Patch\n(Epoch {epoch})", fontsize=12, fontweight='bold')
    ax1.axis('off')

    # Adapted patch
    ax2 = fig.add_subplot(gs[1])
    ax2.imshow(adapted_img)
    ax2.set_title(f"Adapted Patch\n(Adapter from Epoch {adapter_epoch})", fontsize=12, fontweight='bold')
    ax2.axis('off')

    # Difference heatmap
    ax3 = fig.add_subplot(gs[2])
    im = ax3.imshow(diff_magnitude, cmap='hot', vmin=0, vmax=0.5)
    ax3.set_title("Difference Magnitude\n(L2 per pixel)", fontsize=12, fontweight='bold')
    ax3.axis('off')
    plt.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)

    plt.suptitle(f"Adapter Transformation Analysis", fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")

    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Visualize adapter transformation of a patch at a given epoch"
    )
    parser.add_argument("epoch", type=int, help="Epoch number to visualize")
    parser.add_argument("--patches-dir", type=str, default="bb_patches",
                        help="Directory containing patches and adapter checkpoints")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to use (cuda/mps/cpu, default: auto)")
    parser.add_argument("--save", type=str, default=None,
                        help="Path to save visualization (optional)")

    args = parser.parse_args()

    # Device setup
    if args.device is None:
        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
    else:
        device = args.device

    print(f"Using device: {device}")
    print("=" * 60)

    # Find files
    print(f"Looking for patch at epoch {args.epoch}...")
    patch_path = find_patch_file(args.epoch, args.patches_dir)
    print(f"Found patch: {patch_path}")

    print(f"\nLooking for adapter checkpoint...")
    adapter_path = find_latest_adapter(args.epoch, args.patches_dir)
    print(f"Found adapter: {adapter_path}")

    # Extract adapter epoch from filename
    adapter_epoch = int(adapter_path.stem.split('_')[-1])

    # Load patch
    print("\nLoading patch...")
    patch = load_patch(patch_path, device)

    # Load adapter
    print("Loading adapter...")
    adapter = load_adapter(adapter_path, device)

    # Process patch through adapter
    print("\nProcessing patch through adapter...")
    with torch.no_grad():
        # Normalize patch to [0, 1]
        patch_normalized = torch.sigmoid(patch)

        # Apply adapter
        adapted_patch = adapter(patch_normalized.unsqueeze(0)).squeeze(0)

    print(f"Patch shape: {patch.shape}")
    print(f"Adapted patch shape: {adapted_patch.shape}")

    # Compute statistics
    patch_img = torch.sigmoid(patch)
    diff = (adapted_patch - patch_img).abs()
    print(f"\nDifference statistics:")
    print(f"  Mean absolute difference: {diff.mean():.6f}")
    print(f"  Max absolute difference: {diff.max():.6f}")
    print(f"  Min absolute difference: {diff.min():.6f}")

    # Visualize
    print("\nGenerating visualization...")
    visualize_patches(patch, adapted_patch, args.epoch, adapter_epoch, args.save)


if __name__ == "__main__":
    main()
