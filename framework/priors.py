"""
PriorRegistry — pluggable registry of JIT character-decoder priors.

Each registered prior is a JIT-traced decoder:
    input:  [B, char_embed_dim]   (char embedding produced by a scale-MLP)
    output: [B, 1, h, w]         (character-like texture image)

The registry renders textures at 4 spatial scales (8, 16, 32, 64 px) for each
prior and concatenates them as extra input channels to BottleneckDenseRefiner's
spatial_layers.  This is a direct generalisation of the hardcoded omniglot path:

    - 1 prior  × 4 scales = 4  extra channels  (matches old use_omniglot=True)
    - 2 priors × 4 scales = 8  extra channels
    - 0 priors             = 0  extra channels  (matches old use_omniglot=False)

Device note:
    JIT (ScriptModule) objects are not tracked by nn.Module's parameter
    bookkeeping, so they are not automatically moved by .to(device).  This
    class overrides .to() to move all JIT decoders manually.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


_SCALES: List[int] = [8, 16, 32, 64]


class PriorRegistry(nn.Module):
    """
    Registry of N pluggable JIT character-decoder priors.

    Usage::

        reg = PriorRegistry(patch_height=256, patch_width=512, latent_dim=16)
        reg.add_prior('omniglot', 'omniglot_ae_export/decoder_traced.pt')
        reg.add_prior('mnist',    'mnist_ae_export/decoder_traced.pt')

        # In BottleneckDenseRefiner forward:
        prior_feats = reg(z_enriched)   # List of [B, 1, 256, 512]
        combined = torch.cat([patches, refined] + prior_feats, dim=1)
    """

    def __init__(
        self,
        patch_height: int,
        patch_width: int,
        latent_dim: int,
        char_embed_dim: int = 32,
    ):
        """
        Args:
            patch_height:   output patch height in pixels
            patch_width:    output patch width in pixels
            latent_dim:     dimensionality of z_enriched (drives the scale MLPs)
            char_embed_dim: dimensionality of the embedding passed to each JIT decoder
        """
        super().__init__()

        self.patch_height = patch_height
        self.patch_width = patch_width
        self.latent_dim = latent_dim
        self.char_embed_dim = char_embed_dim

        # JIT decoders: plain dict (NOT tracked by nn.Module; moved manually in .to())
        self._decoders: Dict[str, Any] = {}
        self._decoder_names: List[str] = []   # insertion-order list

        # Scale MLPs: registered as ModuleDict (parameters tracked by nn.Module)
        # Key format: "{prior_name}_{scale}"
        self.scale_mlps = nn.ModuleDict()

    # ------------------------------------------------------------------
    # Prior registration
    # ------------------------------------------------------------------

    def add_prior(
        self,
        name: str,
        decoder_path: str,
        decoder_latent_dim: int = 32,
    ) -> None:
        """
        Register a JIT-traced decoder as a named prior.

        The decoder must accept [B, char_embed_dim] and return [B, 1, h, w].
        It should have been traced with torch.enable_grad() to support
        gradient flow during training.

        Args:
            name:               unique identifier for this prior
            decoder_path:       path to the JIT .pt file
            decoder_latent_dim: latent dim expected by this specific decoder
                                (currently informational; char_embed_dim is used)
        """
        if name in self._decoders:
            raise ValueError(f"Prior '{name}' is already registered.")

        path = Path(decoder_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Decoder not found at '{decoder_path}'. "
                f"Trace and export with torch.jit.trace() first."
            )

        decoder = torch.jit.load(str(path), map_location='cpu')
        decoder.train()
        self._decoders[name] = decoder
        self._decoder_names.append(name)

        # Create one scale MLP per spatial scale for this prior
        for scale in _SCALES:
            num_h = self.patch_height // scale
            num_w = self.patch_width // scale
            output_size = self.char_embed_dim * num_h * num_w
            key = f"{name}_{scale}"
            mlp = nn.Sequential(
                nn.Linear(self.latent_dim, 64),
                nn.SiLU(inplace=True),
                nn.Linear(64, 64),
                nn.SiLU(inplace=True),
                nn.Linear(64, output_size),
            )
            self.scale_mlps[key] = mlp

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, z_enriched: Tensor) -> List[Tensor]:
        """
        Render character-texture feature maps at all scales for all priors.

        Args:
            z_enriched: [B, latent_dim]

        Returns:
            List of [B, 1, patch_height, patch_width] tensors,
            length = num_priors × 4   (outer loop: priors, inner: scales)
        """
        B = z_enriched.shape[0]
        device = z_enriched.device
        outputs: List[Tensor] = []

        for name in self._decoder_names:
            decoder = self._decoders[name]

            for scale in _SCALES:
                key = f"{name}_{scale}"
                mlp = self.scale_mlps[key]

                num_h = self.patch_height // scale
                num_w = self.patch_width // scale
                num_patches = num_h * num_w

                # MLP: [B, latent_dim] → [B, embed * num_h * num_w]
                mlp_out = mlp(z_enriched)
                # Reshape to [B*num_patches, char_embed_dim]
                char_emb = mlp_out.view(B, num_patches, self.char_embed_dim)
                char_emb_flat = char_emb.reshape(B * num_patches, self.char_embed_dim)

                # Decoder: [B*P, embed] → [B*P, 1, h, w]
                chars = decoder(char_emb_flat)

                # Resize each tile to (scale, scale) px
                chars_rs = F.interpolate(chars, size=(scale, scale),
                                         mode='bilinear', align_corners=True)

                # Arrange tiles into a full [B, 1, H, W] map
                # chars_rs: [B*num_patches, 1, scale, scale]
                chars_rs = chars_rs.view(B, num_h, num_w, 1, scale, scale)
                # permute to [B, 1, num_h, scale, num_w, scale]
                chars_rs = chars_rs.permute(0, 3, 1, 4, 2, 5).contiguous()
                char_scale = chars_rs.view(B, 1, self.patch_height, self.patch_width)
                outputs.append(char_scale)

        return outputs

    # ------------------------------------------------------------------
    # Property
    # ------------------------------------------------------------------

    @property
    def num_output_channels(self) -> int:
        """Total extra channels contributed to spatial concat: num_priors × 4."""
        return len(self._decoders) * len(_SCALES)

    # ------------------------------------------------------------------
    # Device management — override to move JIT decoders manually
    # ------------------------------------------------------------------

    def to(self, *args, **kwargs):
        """
        Move all registered parameters AND JIT decoders to the target device.

        nn.Module.to() handles registered parameters (scale_mlps) automatically,
        but JIT modules are plain Python objects and must be moved manually.
        """
        device = None
        if args:
            first = args[0]
            if isinstance(first, (str, torch.device)):
                device = torch.device(first)
            # If it's a dtype-only call, no device change is needed

        result = super().to(*args, **kwargs)

        if device is not None:
            for name in list(self._decoder_names):
                self._decoders[name] = self._decoders[name].to(device)

        return result

    def __repr__(self) -> str:
        names = list(self._decoder_names)
        return (
            f"PriorRegistry(priors={names}, "
            f"num_output_channels={self.num_output_channels}, "
            f"patch={self.patch_height}x{self.patch_width})"
        )
