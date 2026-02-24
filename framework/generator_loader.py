"""
Generator loader — self-contained module for loading a trained FoundationPatchGenerator
and generating patches from latent codes.

Imports FoundationPatchGenerator from framework/generator.py.
Both optimize_patch_cmaes.py (via --domain framework) and framework/scripts/cmaes_domain.py
import from here.

NOTE: Old SDXL-LoRA checkpoints are architecturally incompatible with the new multi-TAESD
architecture. A clear RuntimeError is raised rather than silently loading mismatched weights.
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

    # Reject old SDXL-LoRA checkpoints immediately
    if 'use_vae_lora' in ckpt:
        raise RuntimeError(
            "Checkpoint is from the old SDXL-LoRA architecture and is not compatible "
            "with the new multi-TAESD generator. Start a new training run."
        )

    latent_dim = ckpt['basis_dim']
    patch_height, patch_width = ckpt['patch_size']
    num_taesd             = ckpt.get('num_taesd', 6)
    transformer_d_model   = ckpt.get('transformer_d_model', 256)
    transformer_nhead     = ckpt.get('transformer_nhead', 4)
    transformer_d_ff      = ckpt.get('transformer_d_ff', 1024)
    transformer_enc_layers = ckpt.get('transformer_enc_layers', 2)
    transformer_dec_layers = ckpt.get('transformer_dec_layers', 2)

    print(f"  Latent dim:          {latent_dim}")
    print(f"  Patch size:          {patch_height}x{patch_width}")
    print(f"  Num TAESD decoders:  {num_taesd}")
    print(f"  Transformer d_model: {transformer_d_model}, nhead={transformer_nhead}, "
          f"d_ff={transformer_d_ff}")
    print(f"  Transformer layers:  enc={transformer_enc_layers}, dec={transformer_dec_layers}")

    generator = FoundationPatchGenerator(
        latent_dim=latent_dim,
        patch_height=patch_height,
        patch_width=patch_width,
        num_taesd=num_taesd,
        transformer_d_model=transformer_d_model,
        transformer_nhead=transformer_nhead,
        transformer_d_ff=transformer_d_ff,
        transformer_enc_layers=transformer_enc_layers,
        transformer_dec_layers=transformer_dec_layers,
    )
    generator.load_state_dict(ckpt['generator_state_dict'])

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
