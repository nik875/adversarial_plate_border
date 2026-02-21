"""
FoundationPatchGenerator — self-contained reproduction of the generator architecture.

No imports from progressive_patch.py.  Architecture is identical to progressive_patch.py
lines 58–868 so checkpoints saved by either copy are mutually compatible.

Contains:
  - LoRALinear
  - LoRAConv2d
  - inject_lora_into_vae_decoder
  - DilatedResidualSmoother
  - BottleneckDenseRefiner
  - FoundationPatchGenerator
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from diffusers import AutoencoderKL


# ---------------------------------------------------------------------------
# LoRA modules
# ---------------------------------------------------------------------------

class LoRALinear(nn.Module):
    """Linear layer with LoRA adaptation (base weights stored as buffers)."""

    def __init__(self, linear_layer: nn.Linear, r: int = 8, lora_alpha: int = 16):
        super().__init__()
        self.r = r
        self.lora_alpha = lora_alpha

        self.register_buffer('weight', linear_layer.weight.detach())
        if linear_layer.bias is not None:
            self.register_buffer('bias', linear_layer.bias.detach())
        else:
            self.register_buffer('bias', None)

        self.lora_A = nn.Parameter(torch.zeros(r, linear_layer.in_features))
        self.lora_B = nn.Parameter(torch.zeros(linear_layer.out_features, r))
        self.scaling = lora_alpha / r

        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: Tensor) -> Tensor:
        result = F.linear(x, self.weight, self.bias)
        lora_out = (x @ self.lora_A.T @ self.lora_B.T) * self.scaling
        return result + lora_out


class LoRAConv2d(nn.Module):
    """Conv2d with LoRA using 1×1 convolutions (base weights stored as buffers)."""

    def __init__(self, conv_layer: nn.Conv2d, r: int = 8, lora_alpha: int = 16):
        super().__init__()
        self.r = r
        self.lora_alpha = lora_alpha
        self.padding = conv_layer.padding
        self.stride = conv_layer.stride
        self.dilation = conv_layer.dilation
        self.groups = conv_layer.groups

        self.register_buffer('weight', conv_layer.weight.detach())
        if conv_layer.bias is not None:
            self.register_buffer('bias', conv_layer.bias.detach())
        else:
            self.register_buffer('bias', None)

        self.lora_down = nn.Conv2d(conv_layer.in_channels, r, kernel_size=1, bias=False)
        self.lora_up = nn.Conv2d(r, conv_layer.out_channels, kernel_size=1, bias=False)
        self.scaling = lora_alpha / r

        nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_up.weight)

    def forward(self, x: Tensor) -> Tensor:
        result = F.conv2d(x, self.weight, self.bias,
                          stride=self.stride, padding=self.padding,
                          dilation=self.dilation, groups=self.groups)
        lora_out = self.lora_up(self.lora_down(x)) * self.scaling
        return result + lora_out


def inject_lora_into_vae_decoder(vae, r: int = 8, lora_alpha: int = 16) -> dict:
    """
    Inject LoRA into ALL Conv2d and Linear layers in the VAE decoder.

    Returns dict mapping module path → LoRA module.
    """
    lora_modules = {}

    def wrap_all(module, prefix):
        for name, child in module.named_children():
            full_name = f"{prefix}.{name}" if prefix else name
            if isinstance(child, nn.Conv2d):
                wrapped = LoRAConv2d(child, r, lora_alpha)
                setattr(module, name, wrapped)
                lora_modules[full_name] = wrapped
            elif isinstance(child, nn.Linear):
                wrapped = LoRALinear(child, r, lora_alpha)
                setattr(module, name, wrapped)
                lora_modules[full_name] = wrapped
            else:
                wrap_all(child, full_name)

    wrap_all(vae.decoder, 'decoder')
    return lora_modules


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
        x = F.silu(self.conv1(x)) + self.skip1(x)
        x = F.silu(self.conv2(x)) + x
        x = F.silu(self.conv3(x)) + x
        x = F.silu(self.conv4(x)) + x
        x = F.silu(self.conv5(x)) + x
        x = self.conv6(x) + self.skip6(x)
        return x


# ---------------------------------------------------------------------------
# BottleneckDenseRefiner
# ---------------------------------------------------------------------------

class BottleneckDenseRefiner(nn.Module):
    """
    Bottleneck dense refiner for patch refinement with seed conditioning.

    Architecture: compress spatial dims → dense bottleneck (+ z injection) → expand back.
    """

    def __init__(
        self,
        patch_height: int = 256,
        patch_width: int = 512,
        latent_dim: int = 16,
        bottleneck_dim: int = 256,
        use_omniglot: bool = True,
    ):
        super().__init__()

        self.patch_height = patch_height
        self.patch_width = patch_width
        self.latent_dim = latent_dim
        self.bottleneck_dim = bottleneck_dim
        self.use_omniglot = use_omniglot

        self.compress = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=4, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.SiLU(inplace=True),
        )

        self.bottleneck = nn.Sequential(
            nn.AdaptiveAvgPool2d((4, 8)),
        )

        self.seed_embed_dim = 64
        self.seed_projection = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.SiLU(inplace=True),
            nn.Linear(128, self.seed_embed_dim),
        )

        bottleneck_with_seed_dim = 4096 + self.seed_embed_dim
        self.dense = nn.Sequential(
            nn.Linear(bottleneck_with_seed_dim, 512),
            nn.SiLU(inplace=True),
            nn.Linear(512, self.bottleneck_dim),
            nn.SiLU(inplace=True),
            nn.Linear(self.bottleneck_dim, 4096),
            nn.SiLU(inplace=True),
        )

        self.expand = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.SiLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=4, padding=0),
            nn.SiLU(inplace=True),
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=4, padding=0),
            nn.Tanh()
        )

        self.post_expansion_smooth = DilatedResidualSmoother()
        self.final_activation = nn.Sigmoid()

        self.scales = [8, 16, 32, 64]

        if use_omniglot:
            print("Loading Omniglot decoder for character conditioning...")
            omniglot_decoder_path = Path(__file__).parent.parent / "omniglot_ae_export" / "decoder_traced.pt"
            if omniglot_decoder_path.exists():
                self.omniglot_decoder = torch.jit.load(str(omniglot_decoder_path), map_location="cpu")
                self.omniglot_decoder.train()
                print(f"✓ Omniglot decoder loaded from {omniglot_decoder_path} (trainable)")
            else:
                raise FileNotFoundError(
                    f"Omniglot decoder not found at {omniglot_decoder_path}. "
                    f"Please ensure omniglot_ae_export/decoder_traced.pt exists."
                )

            self.scale_mlps = nn.ModuleDict()
            self.char_embed_dim = 32

            for scale in self.scales:
                num_patches_h = patch_height // scale
                num_patches_w = patch_width // scale
                output_size = self.char_embed_dim * num_patches_h * num_patches_w
                mlp = nn.Sequential(
                    nn.Linear(latent_dim, 64),
                    nn.SiLU(inplace=True),
                    nn.Linear(64, 64),
                    nn.SiLU(inplace=True),
                    nn.Linear(64, output_size),
                )
                self.scale_mlps[str(scale)] = mlp

            self.spatial_layers = nn.Sequential(
                nn.Conv2d(10, 32, kernel_size=3, padding=1),
                nn.SiLU(inplace=True),
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.SiLU(inplace=True),
                nn.Conv2d(64, 32, kernel_size=3, padding=1),
                nn.SiLU(inplace=True),
            )
        else:
            print("Omniglot character conditioning disabled.")
            self.spatial_layers = nn.Sequential(
                nn.Conv2d(6, 32, kernel_size=3, padding=1),
                nn.SiLU(inplace=True),
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.SiLU(inplace=True),
                nn.Conv2d(64, 32, kernel_size=3, padding=1),
                nn.SiLU(inplace=True),
            )

        self.num_modes = 16
        self.proj_modes = nn.ModuleList([nn.Conv2d(32, 3, kernel_size=1) for _ in range(self.num_modes)])

        self.attn_grid_h = patch_height // 32
        self.attn_grid_w = patch_width // 32
        self.attention_proj = nn.Linear(latent_dim, self.num_modes * self.attn_grid_h * self.attn_grid_w)
        self.attention_upsample = nn.Sequential(
            nn.ConvTranspose2d(self.num_modes, 32, kernel_size=4, stride=4),
            nn.SiLU(inplace=True),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=4),
            nn.SiLU(inplace=True),
            nn.ConvTranspose2d(16, self.num_modes, kernel_size=4, stride=2, padding=1),
        )

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
        print(f"BottleneckDenseRefiner initialized: ~{total_params:,} parameters")

    def forward(self, patches: Tensor, z: Optional[Tensor] = None) -> Tensor:
        batch_size = patches.shape[0]

        patches_logit = torch.logit(patches.clamp(1e-6, 1 - 1e-6))

        compressed = self.compress(patches)
        pooled = self.bottleneck(compressed)
        bottleneck_flat = pooled.view(batch_size, -1)

        if z is not None:
            seed_embed = self.seed_projection(z)
            bottleneck_with_seed = torch.cat([bottleneck_flat, seed_embed], dim=1)
        else:
            bottleneck_with_seed = bottleneck_flat

        refined_features = self.dense(bottleneck_with_seed)
        refined_features = refined_features.view(batch_size, 128, 4, 8)
        refined = self.expand(refined_features)

        if refined.shape[2] != self.patch_height or refined.shape[3] != self.patch_width:
            refined = F.interpolate(refined, size=(self.patch_height, self.patch_width),
                                    mode='bilinear', align_corners=True)

        if self.use_omniglot:
            char_features_list = []
            for scale in self.scales:
                mlp = self.scale_mlps[str(scale)]
                num_patches_h = self.patch_height // scale
                num_patches_w = self.patch_width // scale
                num_patches = num_patches_h * num_patches_w

                mlp_output = mlp(z)
                char_embeddings = mlp_output.view(batch_size, num_patches, self.char_embed_dim)
                char_embeddings_flat = char_embeddings.view(batch_size * num_patches, self.char_embed_dim)

                self.omniglot_decoder = self.omniglot_decoder.to(char_embeddings_flat.device)
                characters = self.omniglot_decoder(char_embeddings_flat)

                characters_resized = F.interpolate(characters, size=(scale, scale),
                                                   mode='bilinear', align_corners=True)
                characters_resized = characters_resized.view(batch_size, num_patches_h, num_patches_w, 1, scale, scale)
                characters_resized = characters_resized.permute(0, 3, 1, 4, 2, 5).contiguous()
                char_scale = characters_resized.view(batch_size, 1, self.patch_height, self.patch_width)
                char_features_list.append(char_scale)

            combined = torch.cat([patches, refined] + char_features_list, dim=1)
        else:
            combined = torch.cat([patches, refined], dim=1)

        spatial_features = self.spatial_layers(combined)
        mode_outputs = torch.stack([m(spatial_features) for m in self.proj_modes], dim=1)

        attn_grid = self.attention_proj(z).view(batch_size, self.num_modes, self.attn_grid_h, self.attn_grid_w)
        blend_weights = self.attention_upsample(attn_grid)
        blend_weights = torch.softmax(blend_weights, dim=1)
        blend_weights = blend_weights.unsqueeze(2)

        refined_patches = (mode_outputs * blend_weights).sum(dim=1)
        refined_patches = F.silu(refined_patches)
        refined_patches = self.post_expansion_smooth(refined_patches)
        refined_patches = refined_patches + patches_logit
        refined_patches = self.final_activation(refined_patches)

        return refined_patches


# ---------------------------------------------------------------------------
# FoundationPatchGenerator
# ---------------------------------------------------------------------------

class FoundationPatchGenerator(nn.Module):
    """
    Patch generator: latent code z → adversarial patch in [0, 1].

    Pipeline: z → adapter → SD VAE decoder (LoRA) → CNN refiner → patch projector → BottleneckDenseRefiner
    """

    def __init__(
        self,
        latent_dim: int,
        patch_height: int = 256,
        patch_width: int = 512,
        num_layers: int = 11,
        use_vae_lora: bool = True,
        lora_rank: int = 8,
        lora_alpha: int = 16,
        use_bottleneck_refiner: bool = True,
        bottleneck_dim: int = 256,
        use_omniglot: bool = True,
    ):
        super().__init__()

        self.latent_dim = latent_dim
        self.patch_height = patch_height
        self.patch_width = patch_width

        self.vae_latent_h = patch_height // 8
        self.vae_latent_w = patch_width // 8
        self.vae_latent_channels = 4
        self.vae_latent_dim = self.vae_latent_channels * self.vae_latent_h * self.vae_latent_w

        print("Loading Stable Diffusion VAE decoder...")
        self.vae = AutoencoderKL.from_pretrained(
            "stabilityai/sdxl-vae",
            torch_dtype=torch.float32
        )
        del self.vae.encoder
        self.vae.encoder = None

        self.use_vae_lora = use_vae_lora
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.use_omniglot = use_omniglot

        if self.use_vae_lora:
            print(f"Injecting LoRA (rank={self.lora_rank}, alpha={self.lora_alpha})...")
            self.vae_lora_modules = inject_lora_into_vae_decoder(
                self.vae, r=self.lora_rank, lora_alpha=self.lora_alpha)

            vae_lora_params = sum(p.numel() for p in self.vae.parameters() if p.requires_grad)
            vae_base_params = sum(b.numel() for b in self.vae.buffers())
            print(f"  LoRA injected into {len(self.vae_lora_modules)} modules")
            print(f"  VAE base (frozen): {vae_base_params:,}")
            print(f"  VAE LoRA (trainable): {vae_lora_params:,}")
            if vae_base_params > 0:
                print(f"  Reduction: {100 * (1 - vae_lora_params / vae_base_params):.2f}%")
        else:
            print("VAE full fine-tuning enabled")
            self.vae_lora_modules = None

        self.vae.train()
        print(f"VAE loaded. Latent space: [{self.vae_latent_channels}, {self.vae_latent_h}, {self.vae_latent_w}]")

        self.adapter = nn.Linear(latent_dim, self.vae_latent_dim)
        nn.init.kaiming_normal_(self.adapter.weight, mode='fan_out', nonlinearity='relu')
        if self.adapter.bias is not None:
            nn.init.constant_(self.adapter.bias, 0)

        self.skip_projection = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.SiLU(inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

        self.cnn_refiner = nn.Sequential(
            nn.Conv2d(4, 64, kernel_size=3, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.GroupNorm(8, 128),
            nn.SiLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.GroupNorm(8, 128),
            nn.SiLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(inplace=True),
        )

        for m in self.cnn_refiner.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        self.patch_projector = nn.Sequential(
            nn.Conv2d(65, 32, kernel_size=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(32, 3, kernel_size=1),
            nn.Sigmoid()
        )

        for m in self.patch_projector.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        self.bottleneck_refiner = BottleneckDenseRefiner(
            patch_height, patch_width, latent_dim, bottleneck_dim,
            use_omniglot=use_omniglot)
        print(f"Bottleneck dense refiner enabled (bottleneck_dim={bottleneck_dim})")
        print("CNN refiner initialized: 4 → 64 → 64 → 128 → 128 → 64 → 64 channels")
        print("Patch projector: 65 → 32 → 3 channels (1×1 convolutions)")

    def forward(self, z: Tensor) -> Tensor:
        """
        Args:
            z: [B, latent_dim]
        Returns:
            patches: [B, 3, patch_height, patch_width] in [0, 1]
        """
        batch_size = z.shape[0]

        vae_latent_flat = self.adapter(z)
        vae_latent = vae_latent_flat.view(
            batch_size, self.vae_latent_channels, self.vae_latent_h, self.vae_latent_w)

        vae_output = self.vae.decode(vae_latent).sample
        vae_output = torch.clamp(vae_output, 0.0, 1.0)

        if vae_output.shape[2] != self.patch_height or vae_output.shape[3] != self.patch_width:
            vae_output = F.interpolate(vae_output, size=(self.patch_height, self.patch_width),
                                       mode='bilinear', align_corners=True)

        skip_scale = self.skip_projection(z).view(batch_size, 1, 1, 1)
        skip_features = skip_scale.expand(batch_size, 1, self.patch_height, self.patch_width)

        cnn_input = torch.cat([vae_output, skip_features], dim=1)
        cnn_output = self.cnn_refiner(cnn_input)

        projector_input = torch.cat([cnn_output, skip_features], dim=1)
        patches = self.patch_projector(projector_input)

        patches = self.bottleneck_refiner(patches, z)

        return patches
