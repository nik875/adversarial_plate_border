"""
GenericPatchTrainer — self-contained progressive layer trainer.

No imports from progressive_patch.py. Uses:
  - framework/generator.py   : FoundationPatchGenerator
  - framework/losses.py      : TV, spectrum, diversity losses
  - framework/base/          : DomainAdapter, AttackStrategy, LayerConfig
"""
from __future__ import annotations

import csv
import math
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import torchvision.transforms as T

from framework.generator import FoundationPatchGenerator
from framework.losses import (
    total_variation_loss,
    compute_spectrum_loss,
    compute_activation_diversity,
)
from framework.base.domain import DomainAdapter, LayerConfig
from framework.base.attack_strategy import AttackStrategy


class GenericPatchTrainer:
    """
    Domain-agnostic progressive layer patch trainer.

    Trains a FoundationPatchGenerator to produce adversarial patches by
    progressively targeting deeper layers of the target model.

    Talks ONLY to:
      - DomainAdapter  (model, dataset, layer schedule, preprocessing)
      - AttackStrategy (compositing, visibility mask)

    Hyperparameters mirror ProgressivePatchTrainer for familiarity.
    """

    def __init__(
        self,
        domain: DomainAdapter,
        strategy: AttackStrategy,
        basis_dim: int = 16,
        patches_per_image: int = 4,
        images_per_batch: int = 1,
        diversity_weight: float = 0.0,
        quality_weight: float = 1.0,
        performance_weight: float = 1.0,
        tv_weight: float = 0.0,
        spectrum_weight: float = 0.0,
        num_taesd: int = 1,
        transformer_d_model: int = 256,
        transformer_nhead: int = 4,
        transformer_d_ff: int = 1024,
        transformer_enc_layers: int = 2,
        transformer_dec_layers: int = 2,
        output_dir: str = "framework_output",
        save_examples_every: Optional[int] = None,
        learning_rate: float = 1e-4,
        taesd_lr_ratio: float = 0.1,
        lr_min: float = 1e-6,
        max_epochs: int = 100,
        val_split: float = 0.2,
        num_workers: int = 0,
        device: Optional[str] = None,
    ):
        self.domain = domain
        self.strategy = strategy
        self.basis_dim = basis_dim
        self.patches_per_image = patches_per_image
        self.images_per_batch = images_per_batch
        self.diversity_weight = diversity_weight
        self.quality_weight = quality_weight
        self.performance_weight = performance_weight
        self.tv_weight = tv_weight
        self.spectrum_weight = spectrum_weight
        self.output_dir = Path(output_dir)
        self.save_examples_every = save_examples_every
        self.learning_rate = learning_rate
        self.taesd_lr_ratio = taesd_lr_ratio
        self.lr_min = lr_min
        self.max_epochs = max_epochs

        # Device resolution
        if device is None:
            _device = domain.device
        else:
            _device = torch.device(device)
        self._device = _device

        # Patch spatial dimensions come from domain.input_shape
        H, W = domain.input_shape
        self.patch_height = H
        self.patch_width = W

        # Layer progression from domain
        self.layer_configs: List[LayerConfig] = domain.get_layer_progression()
        self.current_layer_idx: int = 0
        self.current_layer_epoch: int = 0

        # Build dataset / dataloaders
        full_dataset = domain.build_dataset(split='train')
        n = len(full_dataset)
        val_n = max(1, int(n * val_split))
        train_n = n - val_n
        from torch.utils.data import random_split
        train_ds, val_ds = random_split(
            full_dataset, [train_n, val_n],
            generator=torch.Generator().manual_seed(42))

        pin = (str(_device).startswith('cuda'))
        self.train_loader = DataLoader(train_ds, batch_size=1, shuffle=True,
                                       num_workers=num_workers, pin_memory=pin)
        self.val_loader = DataLoader(val_ds, batch_size=1, shuffle=False,
                                     num_workers=num_workers, pin_memory=pin)

        print(f"Dataset: {train_n} train / {val_n} val")

        # Build generator
        self.generator = FoundationPatchGenerator(
            latent_dim=basis_dim,
            patch_height=H,
            patch_width=W,
            num_taesd=num_taesd,
            transformer_d_model=transformer_d_model,
            transformer_nhead=transformer_nhead,
            transformer_d_ff=transformer_d_ff,
            transformer_enc_layers=transformer_enc_layers,
            transformer_dec_layers=transformer_dec_layers,
        ).to(_device)

        # Hook state for capturing activations from target model
        self._current_activations: Optional[Tensor] = None
        self._activation_hook = None
        self.layer_activation_stddev: Dict[int, Tensor] = {}
        self.baseline_activations_cache: Dict[int, Tensor] = {}

        # Statistics
        self.epoch_stats: List[dict] = []
        self.checkpoint_base = str(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Sampling / generation helpers
    # ------------------------------------------------------------------

    def sample_z(self, batch_size: int) -> Tensor:
        """Sample latent codes z ~ Uniform(0, 1)."""
        return torch.rand(batch_size, self.basis_dim, device=self._device)

    def generate_patches(self, z: Tensor) -> Tensor:
        """Run generator: [B, D] → [B, 3, H, W]."""
        return self.generator(z)

    # ------------------------------------------------------------------
    # Activation capture helpers
    # ------------------------------------------------------------------

    def _find_layer(self, name: str):
        """Find a module in domain.model by name (or name substring)."""
        for n, m in self.domain.model.named_modules():
            if n == name or name in n:
                return m
        return None

    def _register_hook(self, layer_name: str):
        """Register a forward hook on the named layer."""
        if self._activation_hook is not None:
            self._activation_hook.remove()
            self._activation_hook = None
        self._current_activations = None

        layer = self._find_layer(layer_name)
        if layer is None:
            raise RuntimeError(f"Layer not found: {layer_name}")

        def hook_fn(module, inp, output):
            if isinstance(output, Tensor):
                self._current_activations = output.squeeze(0)

        self._activation_hook = layer.register_forward_hook(hook_fn)
        print(f"✓ Registered activation hook on: {layer_name}")

    def _get_activations(
        self,
        model_input: Tensor,
        use_grad: bool = False,
    ) -> Optional[Tensor]:
        """Run model forward pass; return captured activations."""
        ctx = torch.no_grad() if not use_grad else torch.enable_grad()
        with ctx:
            self.domain.model(model_input)
        act = self._current_activations
        self._current_activations = None
        return act

    def _composited_model_input(
        self, image: Tensor, patch: Tensor, **strategy_kwargs
    ) -> Tensor:
        """
        Apply strategy + domain preprocessing to get model-ready input.

        strategy_kwargs are forwarded to strategy.apply() (e.g. bbox for StickerStrategy).

        Returns: model input Tensor
        """
        composited, _ = self.strategy.apply(image, patch, **strategy_kwargs)
        return self.domain.preprocess_for_model(composited)

    def _neutral_model_input(self, image: Tensor) -> Tensor:
        """Apply neutral strategy + preprocessing for baseline."""
        baseline_img = self.domain.get_baseline_image(image)
        return self.domain.preprocess_for_model(baseline_img)

    def _visibility_mask_for(self, patch: Tensor, **strategy_kwargs) -> Tensor:
        """Get the visibility mask for a dummy application of the strategy."""
        H, W = self.patch_height, self.patch_width
        dummy = torch.zeros(1, 3, H, W, device=self._device)
        _, mask = self.strategy.apply(dummy, patch, **strategy_kwargs)
        return mask

    # ------------------------------------------------------------------
    # Train one epoch
    # ------------------------------------------------------------------

    def train_epoch(
        self,
        optimizer: optim.Optimizer,
        scheduler,
        epoch: int,
    ) -> Tuple[float, float, float]:
        """
        Train for one epoch targeting the current layer.

        Returns (avg_diversity_loss, avg_tv_loss, avg_spectrum_loss).
        """
        target_layer_idx = self.current_layer_idx
        layer_cfg = self.layer_configs[target_layer_idx]

        # Ensure hook is on the right layer
        self._register_hook(layer_cfg.name)

        total_div_loss = 0.0
        total_tv_loss = 0.0
        total_spec_loss = 0.0
        num_updates = 0

        n_images = len(self.train_loader)
        n_batches = math.ceil(n_images / self.images_per_batch)
        print(f"  Train epoch {epoch}: {n_images} images / "
              f"{self.images_per_batch} per batch = {n_batches} batches")

        history_path = self.output_dir / 'training_history.csv'

        batch_global = 0
        loader_iter = iter(self.train_loader)

        with tqdm(total=n_batches, desc=f"Epoch {epoch}", leave=False) as pbar:
            while True:
                batch_div_loss = 0.0
                batch_tv_loss = 0.0
                batch_spec_loss = 0.0
                batch_count = 0

                try:
                    for _ in range(self.images_per_batch):
                        raw = next(loader_iter)
                        # Unwrap batch dim 1 → item
                        image = raw['image']  # [1, 3, H, W]
                        if image.dim() == 4:
                            image = image.squeeze(0)  # [3, H, W]
                        image = image.to(self._device).unsqueeze(0)  # [1, 3, H, W]

                        # ---- baseline activation (neutral composite) ----
                        with torch.no_grad():
                            baseline_inp = self._neutral_model_input(image)
                            self.domain.model(baseline_inp)
                            baseline_act = self._current_activations
                            self._current_activations = None
                            if baseline_act is None:
                                continue

                        # ---- sample placement once for this image ----
                        # All patches_per_image patches share the same placement so
                        # diversity is measured at a consistent composite position.
                        strategy_kwargs = self.strategy.sample_kwargs(
                            image, self.patch_height, self.patch_width
                        )

                        # ---- generate patches ----
                        patches = []
                        patch_acts = []
                        for _ in range(self.patches_per_image):
                            z = self.sample_z(1)
                            patch = self.generate_patches(z)[0]  # [3, H, W]
                            patches.append(patch)

                            # activation with patch (needs grad)
                            model_inp = self._composited_model_input(
                                image, patch, **strategy_kwargs
                            )
                            act = self._get_activations(model_inp, use_grad=True)
                            patch_acts.append(act if act is not None else baseline_act.detach())

                        if not patches:
                            continue

                        # ---- diversity loss ----
                        baseline_list = [baseline_act.detach()] * len(patches)
                        div_score = compute_activation_diversity(
                            patch_acts, baseline_list, device=self._device)

                        # ---- quality score (if std calibrated) ----
                        if target_layer_idx in self.layer_activation_stddev:
                            std = self.layer_activation_stddev[target_layer_idx]
                            q_scores = []
                            for pa in patch_acts:
                                delta = (pa - baseline_act.detach()).reshape(-1)
                                norm_delta = delta / (std.reshape(-1) + 1e-8)
                                q_scores.append((norm_delta ** 2).mean().sqrt())
                            quality_score = torch.stack(q_scores).mean()
                        else:
                            quality_score = torch.tensor(1.0, device=self._device)

                        combined = (self.diversity_weight * div_score
                                    + self.quality_weight * quality_score)
                        total_loss = -(self.performance_weight * combined)

                        # ---- TV + spectrum losses ----
                        patches_stacked = torch.stack(patches, dim=0)  # [P, 3, H, W]
                        vis_mask = self._visibility_mask_for(patches[0], **strategy_kwargs)

                        tv_val = self.tv_weight * total_variation_loss(patches_stacked, vis_mask)
                        spec_val = self.spectrum_weight * compute_spectrum_loss(patches_stacked, vis_mask)

                        final_loss = total_loss + tv_val + spec_val
                        final_loss.backward()

                        batch_div_loss += total_loss.item()
                        batch_tv_loss += tv_val.item()
                        batch_spec_loss += spec_val.item()
                        batch_count += 1

                        if str(self._device).startswith('cuda'):
                            torch.cuda.empty_cache()
                        elif str(self._device) == 'mps':
                            torch.mps.empty_cache()

                except StopIteration:
                    pass

                if batch_count > 0:
                    torch.nn.utils.clip_grad_norm_(self.generator.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    num_updates += 1

                    total_div_loss += batch_div_loss / batch_count
                    total_tv_loss += batch_tv_loss / batch_count
                    total_spec_loss += batch_spec_loss / batch_count

                    # Live CSV row
                    avg_d = total_div_loss / num_updates
                    avg_t = total_tv_loss / num_updates
                    avg_s = total_spec_loss / num_updates
                    lr_now = (scheduler.get_last_lr()[0]
                              if hasattr(scheduler, 'get_last_lr') else 0)
                    write_header = not history_path.exists()
                    with open(history_path, 'a', newline='') as f:
                        w = csv.DictWriter(
                            f, fieldnames=['epoch', 'batch', 'div_loss', 'tv_loss',
                                           'ssim_loss', 'lr'])
                        if write_header:
                            w.writeheader()
                        w.writerow({'epoch': epoch, 'batch': batch_global,
                                    'div_loss': f'{avg_d:.6f}',
                                    'tv_loss': f'{avg_t:.6f}',
                                    'ssim_loss': f'{avg_s:.6f}',
                                    'lr': lr_now})

                    pbar.set_postfix({
                        'DivLoss': f'{avg_d:.4f}',
                        'TVLoss': f'{avg_t:.4f}',
                        'SpecLoss': f'{avg_s:.4f}',
                    })

                batch_global += 1
                pbar.update(1)

                # Optionally save example patches
                if (self.save_examples_every is not None
                        and batch_global % self.save_examples_every == 0):
                    ex_dir = self.output_dir / "example_samples" / f"epoch_{epoch:04d}_batch_{batch_global:06d}"
                    self._save_examples(epoch, str(ex_dir), num_samples=10, save_generator=False)

                # Check if we've exhausted the loader
                if batch_global >= n_batches:
                    break

        return (
            total_div_loss / max(num_updates, 1),
            total_tv_loss / max(num_updates, 1),
            total_spec_loss / max(num_updates, 1),
        )

    # ------------------------------------------------------------------
    # Checkpoint helpers
    # ------------------------------------------------------------------

    def _save_examples(
        self,
        epoch: int,
        save_dir: str,
        num_samples: int = 5,
        save_generator: bool = True,
    ):
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        with torch.no_grad():
            if save_generator:
                gen = self.generator
                ckpt = {
                    'generator_state_dict': gen.state_dict(),
                    'basis_dim':            gen.latent_dim,
                    'patch_size':           [gen.patch_height, gen.patch_width],
                    'num_taesd':            gen.num_taesd,
                    'transformer_d_model':  gen.transformer_d_model,
                    'transformer_nhead':    gen.transformer_nhead,
                    'transformer_d_ff':     gen.transformer_d_ff,
                    'transformer_enc_layers': gen.transformer_enc_layers,
                    'transformer_dec_layers': gen.transformer_dec_layers,
                    'training_info':        {'epoch': epoch},
                }
                torch.save(ckpt, f"{save_dir}/generator_epoch_{epoch:04d}.pt")

            z = self.sample_z(num_samples)
            patches = self.generate_patches(z)
            for i, patch in enumerate(patches):
                T.ToPILImage()(patch.cpu()).save(
                    f"{save_dir}/patch_epoch_{epoch:04d}_sample_{i}.png")

    def save_checkpoint(self, epoch: int, subdir: str = "checkpoint"):
        ckpt_dir = self.output_dir / subdir
        self._save_examples(epoch, str(ckpt_dir), num_samples=10, save_generator=True)
        print(f"✓ Saved checkpoint to: {ckpt_dir}")

    # ------------------------------------------------------------------
    # Full training loop
    # ------------------------------------------------------------------

    def train(self, resume_from: Optional[str] = None):
        """
        Train through the full layer progression.

        Args:
            resume_from: optional path to a checkpoint directory to resume from.
        """
        # Build optimizer: lower LR for TAESD decoders (pretrained), full LR for rest
        taesd_params = [p for n, p in self.generator.named_parameters()
                        if 'taesd_decoders.' in n and p.requires_grad]
        taesd_ids = {id(p) for p in taesd_params}
        custom_params = [p for n, p in self.generator.named_parameters()
                         if p.requires_grad and id(p) not in taesd_ids]

        taesd_lr = self.learning_rate * self.taesd_lr_ratio
        optimizer = optim.AdamW([
            {'params': taesd_params, 'lr': taesd_lr,            'name': 'taesd_decoders'},
            {'params': custom_params, 'lr': self.learning_rate, 'name': 'custom'},
        ])

        n_images = len(self.train_loader)
        batches_per_epoch = math.ceil(n_images / self.images_per_batch)
        total_steps = batches_per_epoch * self.max_epochs + 1

        scheduler = optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=[
                lambda step: (
                    self.lr_min / taesd_lr +
                    (1 - self.lr_min / taesd_lr) *
                    (1 + math.cos(math.pi * step / total_steps)) / 2
                ),
                lambda step: (
                    self.lr_min / self.learning_rate +
                    (1 - self.lr_min / self.learning_rate) *
                    (1 + math.cos(math.pi * step / total_steps)) / 2
                ),
            ],
        )

        start_epoch = 1

        # Resume if requested
        if resume_from is not None:
            ckpt_files = sorted(Path(resume_from).glob("generator_epoch_*.pt"))
            if ckpt_files:
                ckpt = torch.load(ckpt_files[-1], map_location='cpu')
                self.generator.load_state_dict(ckpt['generator_state_dict'])
                start_epoch = ckpt.get('epoch', 0) + 1
                print(f"Resumed from {ckpt_files[-1]} at epoch {start_epoch}")

        print(f"\n{'='*80}")
        print(f"Starting GenericPatchTrainer — {len(self.layer_configs)} layers")
        print(f"{'='*80}")

        layer_idx = self.current_layer_idx
        for epoch in range(start_epoch, self.max_epochs + 1):
            cfg = self.layer_configs[layer_idx]
            print(f"\n[Epoch {epoch}] Layer {layer_idx+1}/{len(self.layer_configs)}: {cfg.description}")

            div_loss, tv_loss, spec_loss = self.train_epoch(optimizer, scheduler, epoch)
            print(f"  div_loss={div_loss:.4f}  tv_loss={tv_loss:.4f}  spec_loss={spec_loss:.4f}")

            # Periodic checkpoint
            self.save_checkpoint(epoch, subdir=f"checkpoint_epoch_{epoch:04d}")

            # Advance layer when max_epochs reached for current layer
            self.current_layer_epoch += 1
            if self.current_layer_epoch >= cfg.max_epochs:
                if layer_idx < len(self.layer_configs) - 1:
                    layer_idx += 1
                    self.current_layer_idx = layer_idx
                    self.current_layer_epoch = 0
                    self.baseline_activations_cache.clear()
                    print(f"\n→ Advancing to layer {layer_idx+1}: {self.layer_configs[layer_idx].description}")
                else:
                    print(f"\n✓ All layers complete at epoch {epoch}.")
                    break

        # Final save
        final_dir = self.output_dir / "training_complete_final_model"
        self._save_examples(epoch, str(final_dir), num_samples=10, save_generator=True)
        print(f"\n✓ Training complete. Final model saved to: {final_dir}")
