"""
TaskEncoder — enriches a latent code z with task metadata via an additive delta.

Architecture:
    context = cat([one_hot(model_idx), one_hot(strategy_idx), clip_embed])
    context → MLP → task_delta  (Tanh, bounded to [-1, 1])
    z_enriched = z + alpha * task_delta

Key design properties:
    - MLP operates on task context only (not z), so z_enriched's dynamic range
      equals z shifted by at most ±alpha per dimension.
    - Final Linear is zero-initialized → task_delta = 0 at init (safe warmup,
      no sudden distribution shift when starting ensemble training).
    - alpha = 0.5 by default (shifts by at most half a std-dev).
    - clip_embed is a [B, clip_embed_dim] CLIP ViT-L/14 image embedding,
      giving the encoder image-level context about what is being attacked.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class TaskEncoder(nn.Module):
    """
    Conditions a latent code z on task metadata (model, strategy) and the
    CLIP embedding of the input image being attacked.

    Usage::

        enc = TaskEncoder(num_models=8, num_strategies=1, latent_dim=16)
        z_enriched = enc(z, model_idx, strategy_idx, clip_embed)
    """

    def __init__(
        self,
        num_models: int,
        num_strategies: int,
        latent_dim: int,
        clip_embed_dim: int = 1024,
        hidden_dim: int = 128,
        num_layers: int = 3,
        alpha: float = 0.5,
    ):
        """
        Args:
            num_models:     number of distinct target models in the ensemble pool
            num_strategies: number of distinct attack strategies
            latent_dim:     dimensionality of z (matches generator's basis_dim)
            clip_embed_dim: dimensionality of the CLIP image embedding (default 1024
                            for openai/clip-vit-large-patch14 ViT-L)
            hidden_dim:     width of all hidden MLP layers
            num_layers:     total MLP depth (input layer + hidden layers + output)
            alpha:          additive blend strength; z_enriched = z + alpha * delta
        """
        super().__init__()

        self.num_models = num_models
        self.num_strategies = num_strategies
        self.clip_embed_dim = clip_embed_dim
        self.latent_dim = latent_dim
        self.alpha = alpha

        input_dim = num_models + num_strategies + clip_embed_dim

        layers: list[nn.Module] = []
        # Input layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.SiLU())
        # Hidden layers (num_layers - 1 additional hidden layers after the first)
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.SiLU())
        # Output layer: latent_dim with Tanh
        layers.append(nn.Linear(hidden_dim, latent_dim))
        layers.append(nn.Tanh())

        self.mlp = nn.Sequential(*layers)

        # Zero-initialize the last Linear so task_delta = 0 at init
        final_linear = self.mlp[-2]  # nn.Linear before Tanh
        nn.init.zeros_(final_linear.weight)
        nn.init.zeros_(final_linear.bias)

    def forward(
        self,
        z: Tensor,
        model_idx: Tensor,
        strategy_idx: Tensor,
        clip_embed: Tensor,
    ) -> Tensor:
        """
        Enrich z with task-specific conditioning.

        Args:
            z:            [B, latent_dim]  base latent codes
            model_idx:    [B]  int64 model indices into the ensemble pool
            strategy_idx: [B]  int64 strategy indices
            clip_embed:   [B, clip_embed_dim]  CLIP ViT-L/14 image embedding

        Returns:
            z_enriched: [B, latent_dim]  z shifted by at most ±alpha per dimension
        """
        device = z.device
        B = z.shape[0]

        # Build one-hot vectors for model and strategy
        m_oh = torch.zeros(B, self.num_models, device=device, dtype=z.dtype)
        m_oh.scatter_(1, model_idx.view(B, 1).long(), 1.0)

        s_oh = torch.zeros(B, self.num_strategies, device=device, dtype=z.dtype)
        s_oh.scatter_(1, strategy_idx.view(B, 1).long(), 1.0)

        # Concatenate into context vector: [B, num_m + num_s + clip_embed_dim]
        context = torch.cat([m_oh, s_oh, clip_embed.to(dtype=z.dtype)], dim=1)

        # MLP → task_delta in [-1, 1] per dim
        task_delta = self.mlp(context)   # [B, latent_dim]

        return z + self.alpha * task_delta
