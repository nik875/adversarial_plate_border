"""
Generator loader — self-contained module for loading a trained FoundationPatchGenerator
and generating patches from latent codes.

Imports FoundationPatchGenerator from framework/generator.py (not progressive_patch.py).
Both optimize_patch_cmaes.py (via --domain framework) and framework/scripts/cmaes_domain.py
import from here.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from torch import Tensor

from framework.generator import FoundationPatchGenerator


def load_generator(
    run_dir: str | Path,
    device: Optional[torch.device] = None,
) -> Tuple[FoundationPatchGenerator, int, torch.device]:
    """
    Load a FoundationPatchGenerator checkpoint from a run directory.

    Searches for checkpoints in priority order:
      1. training_complete_final_model/generator_epoch_*.pt
      2. best_progressive_patch/generator_epoch_*.pt
      3. checkpoint_epoch_*/generator_epoch_*.pt  (latest)
      4. **/generator_epoch_*.pt  (any)

    Args:
        run_dir: Path to the training run directory.
        device: torch.device; if None, auto-detected (CUDA > CPU).

    Returns:
        (generator, latent_dim, device)
    """
    run_path = Path(run_dir)
    latest_checkpoint = None
    checkpoint_source = None

    # Priority 1: final checkpoint
    final_dir = run_path / "training_complete_final_model"
    if final_dir.exists():
        ckpts = sorted(final_dir.glob("generator_epoch_*.pt"))
        if ckpts:
            latest_checkpoint = ckpts[-1]
            checkpoint_source = "final training checkpoint"

    # Priority 2: best model
    if latest_checkpoint is None:
        best_dir = run_path / "best_progressive_patch"
        if best_dir.exists():
            ckpts = sorted(best_dir.glob("generator_epoch_*.pt"))
            if ckpts:
                latest_checkpoint = ckpts[-1]
                checkpoint_source = "best model checkpoint"

    # Priority 3: latest periodic checkpoint
    if latest_checkpoint is None:
        ckpt_dirs = sorted(
            [d for d in run_path.iterdir()
             if d.is_dir() and d.name.startswith("checkpoint_epoch_")]
        )
        if ckpt_dirs:
            ckpts = sorted(ckpt_dirs[-1].glob("generator_epoch_*.pt"))
            if ckpts:
                latest_checkpoint = ckpts[-1]
                checkpoint_source = f"periodic checkpoint ({ckpt_dirs[-1].name})"

    # Priority 4: any checkpoint in subtree
    if latest_checkpoint is None:
        all_ckpts = sorted(run_path.glob("**/generator_epoch_*.pt"))
        if all_ckpts:
            latest_checkpoint = all_ckpts[-1]
            checkpoint_source = f"found in {latest_checkpoint.parent.name}"

    if latest_checkpoint is None:
        raise FileNotFoundError(
            f"No generator checkpoint files found in {run_dir}\n"
            f"Searched for generator_epoch_*.pt in standard subdirectories."
        )

    print(f"Loading checkpoint: {latest_checkpoint}")
    print(f"  Source: {checkpoint_source}")

    ckpt = torch.load(latest_checkpoint, map_location='cpu')

    latent_dim = ckpt['basis_dim']
    patch_height, patch_width = ckpt['patch_size']
    use_vae_lora = ckpt.get('use_vae_lora', True)
    lora_rank = ckpt.get('lora_rank', 8)
    lora_alpha = ckpt.get('lora_alpha', 16)
    use_omniglot = ckpt.get('use_omniglot', False)

    print(f"  Latent dim: {latent_dim}")
    print(f"  Patch size: {patch_height}x{patch_width}")
    print(f"  VAE LoRA: {use_vae_lora} (rank={lora_rank}, alpha={lora_alpha})")
    print(f"  Omniglot conditioning: {use_omniglot}")

    def _build_and_load(omniglot_flag):
        gen = FoundationPatchGenerator(
            latent_dim=latent_dim,
            patch_height=patch_height,
            patch_width=patch_width,
            use_vae_lora=use_vae_lora,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            use_omniglot=omniglot_flag,
        )
        gen.load_state_dict(ckpt['generator_state_dict'])
        return gen

    try:
        generator = _build_and_load(use_omniglot)
    except RuntimeError as e:
        alt = not use_omniglot
        print(f"  Warning: state dict mismatch with use_omniglot={use_omniglot}, "
              f"retrying with use_omniglot={alt}")
        try:
            generator = _build_and_load(alt)
        except RuntimeError:
            raise e  # re-raise original

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    generator = generator.to(device)
    generator.eval()

    print(f"Generator loaded on device={device}")
    return generator, latent_dim, device


def generate_patch_from_z(
    generator: FoundationPatchGenerator,
    z: np.ndarray,
    device: torch.device,
) -> Tensor:
    """
    Generate a single patch tensor from a numpy latent code vector.

    Args:
        generator: loaded FoundationPatchGenerator (eval mode)
        z: 1-D numpy array of shape [latent_dim]
        device: torch.device

    Returns:
        patch: [3, H, W] float tensor in [0, 1] on CPU
    """
    with torch.no_grad():
        z_t = torch.from_numpy(z).float().unsqueeze(0).to(device)   # [1, D]
        patch = generator(z_t)                                        # [1, 3, H, W]
        patch = patch.squeeze(0).cpu()                                # [3, H, W]
    return patch
