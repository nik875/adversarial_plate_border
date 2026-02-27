"""
FoundationPatchGenerator — multi-stream TAESD architecture.

Architecture:
  - 6 TAESD decoder copies (AutoencoderTiny, encoder deleted, fully trainable)
  - 6 linear adapters  Linear(latent_dim, vae_latent_dim)
  - 6 LightPatchTransformers (spatial transformer encoder-decoder)
  - 1 ChannelMixer (spatially-varying attention blend, 18→3 channels)

Output: 512×512 adversarial patch in [0, 1].

Contains:
  - DilatedResidualSmoother  (used by ChannelMixer)
  - LightPatchTransformer
  - ChannelMixer
  - FoundationPatchGenerator
"""
from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from diffusers import AutoencoderTiny


# ---------------------------------------------------------------------------
# DilatedResidualSmoother
# ---------------------------------------------------------------------------

class DilatedResidualSmoother(nn.Module):
    """
    Coarse-to-fine dilated conv chain with residual connections.

    Each block adds its output to a skip of the block input:
    - Channel-changing blocks (3→16, 16→3) use a 1×1 projection skip.
    - Same-channel blocks (16→16) use an identity skip.
    """

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=32, dilation=32)
        self.skip1 = nn.Conv2d(3, 16, kernel_size=1)

        self.conv2 = nn.Conv2d(16, 16, kernel_size=3, padding=16, dilation=16)
        self.conv3 = nn.Conv2d(16, 16, kernel_size=3, padding=8,  dilation=8)
        self.conv4 = nn.Conv2d(16, 16, kernel_size=5, padding=8,  dilation=4)
        self.conv5 = nn.Conv2d(16, 16, kernel_size=5, padding=4,  dilation=2)

        self.conv6 = nn.Conv2d(16, 3, kernel_size=7, padding=3)
        self.skip6 = nn.Conv2d(16, 3, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        x = F.leaky_relu(self.conv1(x)) + self.skip1(x)
        x = F.leaky_relu(self.conv2(x)) + x
        x = F.leaky_relu(self.conv3(x)) + x
        x = F.leaky_relu(self.conv4(x)) + x
        x = F.leaky_relu(self.conv5(x)) + x
        x = self.conv6(x) + self.skip6(x)
        return x


# ---------------------------------------------------------------------------
# LightPatchTransformer
# ---------------------------------------------------------------------------

class LightPatchTransformer(nn.Module):
    """
    Lightweight refinement transformer for TAESD outputs.
    Keeps conceptual elegance of transformer refinement while minimizing compute.

    Architecture:
        - patch_embed: Conv2d(3, d_model, 16, 16) → [B, 256, d_model]
        - 1-layer transformer encoder
        - 1-layer transformer decoder
        - output_proj: Linear(d_model, 768) → reshape → [B, 3, 256, 256]
        - upsample: ConvTranspose2d(3, 3, 4, 2, 1) → [B, 3, 512, 512]

    Reduced from 2 layers to 1 per phase. d_model=128 instead of 256 for 4× smaller
    attention maps: [B, nhead, 256, 128] vs [B, nhead, 256, 256].
    """

    def __init__(
        self,
        d_model: int = 256,
        nhead: int = 4,
        d_ff: int = 1024,
        num_enc_layers: int = 2,
        num_dec_layers: int = 2,
        input_size: int = 256,
    ):
        super().__init__()

        self.input_size = input_size
        num_tokens = (input_size // 16) ** 2  # 256 for 256×256 input

        # Patch embedding: Conv2d produces [B, d_model, 16, 16] for 256×256 input
        self.patch_embed = nn.Conv2d(3, d_model, kernel_size=16, stride=16)

        # Learnable positional embeddings: 16×16 = 256 tokens
        self.encoder_pos_embed = nn.Parameter(torch.zeros(1, num_tokens, d_model))
        nn.init.trunc_normal_(self.encoder_pos_embed, std=0.02)

        # Encoder
        self.encoder_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=d_ff,
                dropout=0.0,
                batch_first=True,
                norm_first=True,
            )
            for _ in range(num_enc_layers)
        ])

        # Decoder queries and positional embeddings
        self.decoder_queries = nn.Parameter(torch.zeros(1, num_tokens, d_model))
        nn.init.trunc_normal_(self.decoder_queries, std=0.02)

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_tokens, d_model))
        nn.init.trunc_normal_(self.decoder_pos_embed, std=0.02)

        # Decoder
        self.decoder_layers = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=d_ff,
                dropout=0.0,
                batch_first=True,
                norm_first=True,
            )
            for _ in range(num_dec_layers)
        ])

        # Output projection: d_model → 16×16×3 = 768 values per patch token
        self.output_norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, 768)  # 768 = 16*16*3

        # 2× upsample: 256×256 → 512×512
        self.upsample = nn.ConvTranspose2d(3, 3, kernel_size=4, stride=2, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: [B, 3, 256, 256] TAESD output in [0, 1]

        Returns:
            [B, 3, 512, 512] upsampled refined patch
        """
        B = x.shape[0]
        x_orig = x  # Save original input for residual connection

        # Embed patches: [B, 3, 256, 256] → [B, d_model, 16, 16] → [B, 256, d_model]
        tokens = self.patch_embed(x).flatten(2).transpose(1, 2)
        tokens = tokens + self.encoder_pos_embed

        # Encoder
        for layer in self.encoder_layers:
            tokens = layer(tokens)
        encoder_out = tokens

        # Decoder
        queries = self.decoder_queries.expand(B, -1, -1) + self.decoder_pos_embed
        for layer in self.decoder_layers:
            queries = layer(queries, encoder_out)

        # Project to pixel space: [B, 256, d_model] → [B, 256, 768]
        out = self.output_proj(self.output_norm(queries))

        # Reshape to image: [B, 256, 768] → [B, 3, 256, 256]
        # 256 = 16×16 patch grid, 768 = 3×16×16 pixels per patch
        grid = self.input_size // 16
        out = out.view(B, grid, grid, 3, 16, 16)
        out = out.permute(0, 3, 1, 4, 2, 5).contiguous()
        out = out.view(B, 3, self.input_size, self.input_size)

        # Residual + sigmoid, then upsample to 512×512
        out = torch.sigmoid(out + x_orig)
        return self.upsample(out)


# ---------------------------------------------------------------------------
# ChannelMixer
# ---------------------------------------------------------------------------

class ChannelMixer(nn.Module):
    """
    Spatially-varying channel mixing attention mechanism.

    Takes the concatenated outputs of N=6 TAESD streams (18 channels) and
    the latent code z, and blends them via learned spatial attention into a
    single [B, 3, patch_height, patch_width] output.

    Architecture mirrors the spatial attention section of BottleneckDenseRefiner,
    adapted for 18-channel input and 512×512 output.
    """

    def __init__(
        self,
        patch_height: int = 512,
        patch_width: int = 512,
        latent_dim: int = 16,
        num_taesd: int = 1,
        num_modes: int = 6,
    ):
        super().__init__()

        self.patch_height = patch_height
        self.patch_width = patch_width
        self.num_modes = num_modes

        # Low-resolution attention grid
        self.attn_grid_h = patch_height // 32
        self.attn_grid_w = patch_width  // 32

        # Input channels: num_taesd streams × 3 RGB channels
        in_channels = num_taesd * 3

        # Spatial feature extraction from concatenated stream outputs
        self.spatial_layers = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
        )

        # Per-mode 3-channel projection heads
        self.proj_modes = nn.ModuleList([
            nn.Conv2d(32, 3, kernel_size=1)
            for _ in range(num_modes)
        ])

        # Attention weight generation from latent code
        self.attention_proj = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.LeakyReLU(inplace=True),
            nn.Linear(128, 256),
            nn.LeakyReLU(inplace=True),
            nn.Linear(256, num_modes * self.attn_grid_h * self.attn_grid_w),
        )

        # Upsample attention grid from (attn_grid_h, attn_grid_w) → (patch_height, patch_width)
        # For default 512×512: 16×16 → 64×64 → 256×256 → 512×512
        self.attention_upsample = nn.Sequential(
            nn.ConvTranspose2d(num_modes, 32, kernel_size=4, stride=4),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=4),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(16, num_modes, kernel_size=4, stride=2, padding=1),
        )

        self.post_smooth = DilatedResidualSmoother()
        self.final_activation = nn.Sigmoid()

        # Weight initialization
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        total_params = sum(p.numel() for p in self.parameters())
        print(f"ChannelMixer initialized: ~{total_params:,} parameters")

    def forward(self, combined: Tensor, z: Tensor) -> Tensor:
        """
        Args:
            combined: [B, 18, H, W]  concatenated stream outputs
            z:        [B, latent_dim] latent code for attention conditioning

        Returns:
            [B, 3, H, W] blended patch in [0, 1]
        """
        batch_size = z.shape[0]

        # Spatial feature extraction
        spatial_features = self.spatial_layers(combined)              # [B, 32, H, W]
        mode_outputs = torch.stack(
            [m(spatial_features) for m in self.proj_modes], dim=1
        )                                                             # [B, num_modes, 3, H, W]

        # Spatial attention weights from latent code
        attn_grid = self.attention_proj(z).view(
            batch_size, self.num_modes, self.attn_grid_h, self.attn_grid_w
        )                                                             # [B, num_modes, gh, gw]
        blend_weights = self.attention_upsample(attn_grid)            # [B, num_modes, H, W]
        blend_weights = torch.softmax(blend_weights, dim=1)
        blend_weights = blend_weights.unsqueeze(2)                    # [B, num_modes, 1, H, W]

        # Weighted sum across modes
        refined_patches = (mode_outputs * blend_weights).sum(dim=1)  # [B, 3, H, W]
        refined_patches = F.leaky_relu(refined_patches)
        refined_patches = self.post_smooth(refined_patches)
        return self.final_activation(refined_patches)


# ---------------------------------------------------------------------------
# FoundationPatchGenerator
# ---------------------------------------------------------------------------

class FoundationPatchGenerator(nn.Module):
    """
    Patch generator: latent code z → adversarial patch in [0, 1].

    Pipeline:
        z → 6 adapters → 6 TAESD decoders → 6 spatial transformers
        → concatenate [B, 18, H, W] → ChannelMixer → patch [B, 3, H, W]

    All TAESD decoders are fully trainable (no frozen weights).
    """

    def __init__(
        self,
        latent_dim: int = 16,
        patch_height: int = 512,
        patch_width: int = 512,
        num_taesd: int = 1,
        transformer_d_model: int = 256,
        transformer_nhead: int = 4,
        transformer_d_ff: int = 1024,
        transformer_enc_layers: int = 2,
        transformer_dec_layers: int = 2,
    ):
        super().__init__()

        self.latent_dim = latent_dim
        self.patch_height = patch_height
        self.patch_width = patch_width
        self.num_taesd = num_taesd

        # Store transformer hyperparams as attributes for checkpoint serialisation
        self.transformer_d_model = transformer_d_model
        self.transformer_nhead = transformer_nhead
        self.transformer_d_ff = transformer_d_ff
        self.transformer_enc_layers = transformer_enc_layers
        self.transformer_dec_layers = transformer_dec_layers

        # TAESD targets 256×256 internally; transformer upsamples 2× to patch_height×patch_width
        self.vae_latent_h = 256 // 8   # = 32
        self.vae_latent_w = 256 // 8   # = 32
        self.vae_latent_dim = 4 * self.vae_latent_h * self.vae_latent_w  # = 4096

        # 6 adapters: Linear(latent_dim → vae_latent_dim) each
        self.adapters = nn.ModuleList([
            nn.Linear(latent_dim, self.vae_latent_dim)
            for _ in range(num_taesd)
        ])

        # 6 TAESD decoder copies (encoder deleted, fully trainable)
        print(f"Loading {num_taesd} TAESD decoder copies from madebyollin/taesd ...")
        self.taesd_decoders = nn.ModuleList()
        for i in range(num_taesd):
            vae = AutoencoderTiny.from_pretrained(
                "madebyollin/taesd",
                torch_dtype=torch.float32,
            )
            del vae.encoder
            vae.encoder = None
            self.taesd_decoders.append(vae)
            print(f"  TAESD decoder {i + 1}/{num_taesd} loaded")

        # 6 spatial transformers (operate at 256×256, upsample to 512×512)
        self.transformers = nn.ModuleList([
            LightPatchTransformer(
                d_model=transformer_d_model,
                nhead=transformer_nhead,
                d_ff=transformer_d_ff,
                num_enc_layers=transformer_enc_layers,
                num_dec_layers=transformer_dec_layers,
                input_size=256,
            )
            for _ in range(num_taesd)
        ])
        print(f"LightPatchTransformer ×{num_taesd} initialized "
              f"(d_model={transformer_d_model}, nhead={transformer_nhead}, "
              f"d_ff={transformer_d_ff}, "
              f"enc={transformer_enc_layers}, dec={transformer_dec_layers})")

        # 1 channel mixer (knows how many input streams to expect)
        self.channel_mixer = ChannelMixer(patch_height, patch_width, latent_dim, num_taesd)

        total_params = sum(p.numel() for p in self.parameters())
        print(f"FoundationPatchGenerator total parameters: {total_params:,}")

    def forward(
        self,
        z: Tensor,
        z_enriched: Optional[Tensor] = None,
        active_streams: Optional[List[int]] = None,
    ) -> Tensor:
        """
        Generate an adversarial patch from latent codes.

        Args:
            z:              [B, latent_dim]  base latent codes
            z_enriched:     [B, latent_dim]  task-conditioned codes (from TaskEncoder).
                            If None, falls back to z.
            active_streams: optional list of stream indices to compute. Inactive
                            streams are replaced with zeros (no forward pass).
                            If None, all streams are active (normal training).

        Returns:
            patches: [B, 3, patch_height, patch_width] in [0, 1]
        """
        if z_enriched is None:
            z_enriched = z

        B = z_enriched.shape[0]

        def _run_stream(adapter, taesd, transformer, z_enc, vae_h, vae_w):
            latent = adapter(z_enc).view(z_enc.shape[0], 4, vae_h, vae_w)
            taesd_out = torch.sigmoid(taesd.decode(latent).sample)
            # transformer refines at 256×256, upsamples to 512×512 internally
            return transformer(taesd_out)

        stream_outputs = []
        for i, (adapter, taesd, transformer) in enumerate(
            zip(self.adapters, self.taesd_decoders, self.transformers)
        ):
            if active_streams is not None and i not in active_streams:
                stream_outputs.append(
                    torch.zeros(B, 3, self.patch_height, self.patch_width,
                                device=z.device, dtype=z.dtype)
                )
            else:
                stream_outputs.append(_run_stream(adapter, taesd, transformer, z_enriched,
                                                  self.vae_latent_h, self.vae_latent_w))

        # Concatenate all stream outputs: [B, 18, H, W]
        combined = torch.cat(stream_outputs, dim=1)

        # Channel mixing with spatial attention: [B, 3, H, W]
        return self.channel_mixer(combined, z_enriched)

    def forward_clean(self, z: Tensor) -> Tensor:
        """Convenience alias: forward without task conditioning (z_enriched=z)."""
        return self.forward(z)
