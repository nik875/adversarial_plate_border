"""
Pure-math loss functions, self-contained (no domain coupling).

Reproduced from progressive_patch.py with one key generalisation:
visibility_mask is passed explicitly instead of being computed from a
hard-coded border scale, so these functions work for any AttackStrategy.

Functions:
  - total_variation_loss(patches, visibility_mask)
  - compute_spectrum_loss(patches, visibility_mask)
  - compute_activation_diversity(patch_acts, baseline_acts)
"""
from __future__ import annotations

from typing import List

import torch
import torch.nn.functional as F
from torch import Tensor
from kornia.metrics import ssim


def total_variation_loss(patches: Tensor, visibility_mask: Tensor) -> Tensor:
    """
    L2 total variation regularisation loss, restricted to the visible patch region.

    Args:
        patches: [B, 3, H, W] in [0, 1]
        visibility_mask: [1, 1, H, W] float (1 = visible, 0 = hidden)

    Returns:
        scalar TV loss (averaged over batch and normalised by visible pixels)
    """
    batch_size = patches.shape[0]
    total_tv = 0.0

    for i in range(batch_size):
        patch = patches[i:i+1]  # [1, 3, H, W]

        tv_h = torch.pow(patch[:, :, :, 1:] - patch[:, :, :, :-1], 2)
        tv_v = torch.pow(patch[:, :, 1:, :] - patch[:, :, :-1, :], 2)

        mask_h = visibility_mask[:, :, :, :-1]  # [1, 1, H, W-1]
        mask_v = visibility_mask[:, :, :-1, :]  # [1, 1, H-1, W]

        tv_h_masked = (tv_h * mask_h).sum()
        tv_v_masked = (tv_v * mask_v).sum()

        num_vis_h = mask_h.sum()
        num_vis_v = mask_v.sum()

        denom = num_vis_h + num_vis_v
        patch_tv = ((tv_h_masked + tv_v_masked) / denom * 2.5
                    if denom > 0
                    else torch.tensor(0.0, device=patches.device))

        total_tv = total_tv + patch_tv

    return total_tv / batch_size


def compute_spectrum_loss(patches: Tensor, visibility_mask: Tensor) -> Tensor:
    """
    SSIM-based structural similarity penalty encouraging patch diversity.

    Computes mean pairwise SSIM over the visible region.  Minimising this
    penalty maximises structural diversity across patches in the batch.

    Args:
        patches: [B, 3, H, W] in [0, 1]
        visibility_mask: [1, 1, H, W] float (1 = visible, 0 = hidden)

    Returns:
        scalar SSIM penalty (higher = more similar = less diverse)
    """
    batch_size = patches.shape[0]
    if batch_size < 2:
        return torch.tensor(0.0, device=patches.device, dtype=patches.dtype)

    ssim_sum = 0.0
    pair_count = 0

    for i in range(batch_size):
        for j in range(i + 1, batch_size):
            patch_i = patches[i:i+1]
            patch_j = patches[j:j+1]

            ssim_map = ssim(patch_i, patch_j, window_size=11)          # [1, 3, H, W]
            ssim_map_avg = ssim_map.mean(dim=1, keepdim=True)           # [1, 1, H, W]
            masked_ssim = ssim_map_avg * visibility_mask

            num_vis = visibility_mask.sum()
            ssim_val = masked_ssim.sum() / num_vis if num_vis > 0 else 0.0

            ssim_sum = ssim_sum + ssim_val
            pair_count += 1

    if pair_count > 0:
        return ssim_sum / pair_count
    return torch.tensor(0.0, device=patches.device, dtype=patches.dtype)


def compute_activation_diversity(
    patch_acts: List[Tensor],
    baseline_acts: List[Tensor],
    device: torch.device = None,
) -> Tensor:
    """
    Diversity as log-det of the Gram matrix of activation deltas.

    For each patch, compute activation delta = act_with_patch − baseline.
    Measure diversity as log-det of the Gram matrix of unit-normalised deltas.
    A higher log-det means the patches produce more orthogonal (diverse) effects.

    Args:
        patch_acts: list of [*] activation tensors, one per patch
        baseline_acts: list of [*] baseline activation tensors (same shapes)
        device: target device (inferred from patch_acts if None)

    Returns:
        log_det: scalar diversity score
    """
    if device is None and patch_acts:
        device = patch_acts[0].device

    batch_size = len(patch_acts)
    deltas = []
    for act, base in zip(patch_acts, baseline_acts):
        delta = (act - base).reshape(-1)
        deltas.append(delta)

    deltas_flat = torch.stack(deltas, dim=0)        # [B, D]
    normalized = F.normalize(deltas_flat, p=2, dim=1)

    gram = normalized @ normalized.t()              # [B, B]

    epsilon = max(1e-6, 1e-2 / batch_size)
    gram = gram + epsilon * torch.eye(batch_size, device=device)

    sign, log_det = torch.slogdet(gram)

    if torch.isnan(log_det) or sign <= 0:
        log_det = torch.tensor(-20.0, device=device, dtype=log_det.dtype)

    return log_det
