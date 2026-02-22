"""
Ensemble adversarial training system.

EnsembleModelPool
    Pool of frozen target models, each paired with an attack strategy.
    Models are always stored on CPU; on_device() temporarily moves one to
    compute_device for a forward pass and returns it to CPU afterward.

EnsembleTrainer
    Trains a FoundationPatchGenerator against the full ensemble pool using
    10,000 uniformly-sampled neurons per step (no fixed layer progression).

    Loss = -(diversity_weight * log_det_gram
             + quality_weight * log(mean_rms_normalized_delta))
           + tv_weight * TV_loss
           + spectrum_weight * SSIM_loss

Gradient flow:
    - All model params have requires_grad=False (frozen in register()).
    - Gradients flow through composited_input (from generator) → activations.
    - PyTorch does not allocate grad buffers for frozen params.
"""
from __future__ import annotations

import math
import os
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Generator, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor, optim
import torchvision.transforms as T
from tqdm import tqdm

from framework.base.attack_strategy import AttackStrategy
from framework.dataset_pool import LazyDatasetPool
from framework.generator import FoundationPatchGenerator
from framework.losses import total_variation_loss, compute_spectrum_loss
from framework.neuron_sampler import NeuronSampler
from framework.task_encoder import TaskEncoder


# ---------------------------------------------------------------------------
# EnsembleModelPool
# ---------------------------------------------------------------------------

@dataclass
class ModelEntry:
    """One registered model entry in the ensemble pool."""
    model_id: int
    name: str
    domain_type: str
    _model: nn.Module           # always on CPU
    input_shape: Tuple[int, int]   # (H, W)
    preprocess_fn: Callable[[Tensor], Tensor]
    strategy: AttackStrategy
    strategy_id: int


class EnsembleModelPool:
    """
    Pool of frozen target models, each paired with an attack strategy.

    Models are kept on CPU between steps; on_device() temporarily moves
    one model to the compute device for a forward pass.

    Usage::

        pool = EnsembleModelPool(compute_device=torch.device('cuda'))
        pool.register('resnet50', model, 'classification',
                      strategy=BorderStrategy(), strategy_id=0,
                      input_shape=(224, 224), preprocess_fn=fn)
        entry = pool.sample_entry()
        with pool.on_device(entry) as model:
            out = model(inp)
    """

    def __init__(self, compute_device: Optional[torch.device] = None):
        self.compute_device = compute_device or torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        self._entries: List[ModelEntry] = []
        self._next_id: int = 0

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(
        self,
        name: str,
        model: nn.Module,
        domain_type: str,
        strategy: AttackStrategy,
        strategy_id: int,
        input_shape: Tuple[int, int],
        preprocess_fn: Callable[[Tensor], Tensor],
    ) -> int:
        """
        Register a model in the ensemble pool.

        The model is moved to CPU, set to eval mode, and all parameters are
        frozen (requires_grad=False).  Gradients flow through composited
        inputs (from the generator) but NOT into model weights.

        Args:
            name:          human-readable identifier
            model:         nn.Module to register (will be frozen and CPU-resident)
            domain_type:   arbitrary tag (e.g. 'classification', 'detection')
            strategy:      AttackStrategy to use when attacking via this model
            strategy_id:   integer strategy identifier (for TaskEncoder one-hot)
            input_shape:   (H, W) expected by this model (after preprocessing)
            preprocess_fn: callable (Tensor [B,3,H,W]) → Tensor [B,3,H',W']

        Returns:
            model_id (int)
        """
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        model = model.cpu()

        model_id = self._next_id
        self._next_id += 1
        self._entries.append(ModelEntry(
            model_id=model_id,
            name=name,
            domain_type=domain_type,
            _model=model,
            input_shape=input_shape,
            preprocess_fn=preprocess_fn,
            strategy=strategy,
            strategy_id=strategy_id,
        ))
        return model_id

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def sample_entry(self) -> ModelEntry:
        """Sample a uniformly random model entry."""
        if not self._entries:
            raise RuntimeError("No models registered. Call register() first.")
        import random
        return random.choice(self._entries)

    def get_entry(self, model_id: int) -> ModelEntry:
        for e in self._entries:
            if e.model_id == model_id:
                return e
        raise KeyError(f"model_id={model_id} not registered.")

    # ------------------------------------------------------------------
    # Device management
    # ------------------------------------------------------------------

    @contextmanager
    def on_device(self, entry: ModelEntry) -> Generator[nn.Module, None, None]:
        """
        Context manager: move model CPU → compute_device, yield, move back.

        Memory-safe: only one model lives on GPU at a time.
        Clears CUDA cache after moving back to CPU.
        """
        model = entry._model
        try:
            model = model.to(self.compute_device)
            yield model
        finally:
            model = model.cpu()
            entry._model = model
            if self.compute_device.type == 'cuda':
                torch.cuda.empty_cache()
            elif self.compute_device.type == 'mps':
                torch.mps.empty_cache()

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    def num_models(self) -> int:
        return len(self._entries)

    def num_strategies(self) -> int:
        return len(set(e.strategy_id for e in self._entries))

    def __repr__(self) -> str:
        names = [e.name for e in self._entries]
        return f"EnsembleModelPool(models={names}, device={self.compute_device})"


# ---------------------------------------------------------------------------
# EnsembleTrainer
# ---------------------------------------------------------------------------

class EnsembleTrainer:
    """
    Trains a FoundationPatchGenerator against an ensemble of target models.

    Key differences from GenericPatchTrainer:
        - Random model + strategy each step (any pairing)
        - 10,000 random neurons drawn from ALL layers (no fixed layer progression)
        - Per-step control baseline (no precomputed profiles)
        - TaskEncoder conditions z on model/strategy/dataset metadata
        - Two on_device() calls per step (ctrl_acts no_grad, adv_acts with_grad)

    Optimizer groups (AdamW):
        - VAE LoRA params:         lr = learning_rate * vae_lr_ratio
        - TaskEncoder params:      lr = learning_rate
        - PriorRegistry scale_mlps:lr = learning_rate * vae_lr_ratio
        - All other generator params: lr = learning_rate
    """

    def __init__(
        self,
        ensemble: EnsembleModelPool,
        dataset_pool: LazyDatasetPool,
        task_encoder: TaskEncoder,
        generator: FoundationPatchGenerator,
        neuron_sampler: NeuronSampler,
        k_neurons: int = 10_000,
        patches_per_batch: int = 4,
        diversity_weight: float = 1.0,
        quality_weight: float = 1.0,
        tv_weight: float = 2.5,
        spectrum_weight: float = 1.0,
        learning_rate: float = 1e-4,
        vae_lr_ratio: float = 0.1,
        lr_min: float = 1e-6,
        max_epochs: int = 100,
        output_dir: str = 'ensemble_output',
        save_every_epochs: int = 5,
        device: Optional[torch.device] = None,
    ):
        self.ensemble = ensemble
        self.dataset_pool = dataset_pool
        self.task_encoder = task_encoder
        self.generator = generator
        self.neuron_sampler = neuron_sampler

        self.k_neurons = k_neurons
        self.patches_per_batch = patches_per_batch
        self.diversity_weight = diversity_weight
        self.quality_weight = quality_weight
        self.tv_weight = tv_weight
        self.spectrum_weight = spectrum_weight
        self.learning_rate = learning_rate
        self.vae_lr_ratio = vae_lr_ratio
        self.lr_min = lr_min
        self.max_epochs = max_epochs
        self.output_dir = Path(output_dir)
        self.save_every_epochs = save_every_epochs

        self._device = device or ensemble.compute_device
        self.generator = self.generator.to(self._device)
        self.task_encoder = self.task_encoder.to(self._device)

        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Optimizer construction
    # ------------------------------------------------------------------

    def _build_optimizer(self, total_steps: int):
        """Build AdamW with 4 parameter groups + cosine annealing scheduler."""
        gen = self.generator

        # Group 1: VAE LoRA params
        vae_params = [p for p in gen.vae.parameters() if p.requires_grad]

        # Group 2: TaskEncoder params
        task_params = list(self.task_encoder.parameters())

        # Group 3: PriorRegistry scale_mlps (inside bottleneck_refiner)
        prior_params: List[nn.Parameter] = []
        br = getattr(gen, 'bottleneck_refiner', None)
        if br is not None:
            pr = getattr(br, 'prior_registry', None)
            if pr is not None and hasattr(pr, 'scale_mlps'):
                prior_params = list(pr.scale_mlps.parameters())

        # Group 4: all other generator params
        vae_ids = {id(p) for p in vae_params}
        prior_ids = {id(p) for p in prior_params}
        other_gen_params = [
            p for n, p in gen.named_parameters()
            if p.requires_grad
            and id(p) not in vae_ids
            and id(p) not in prior_ids
        ]

        vae_lr = self.learning_rate * self.vae_lr_ratio
        optimizer = optim.AdamW([
            {'params': vae_params,      'lr': vae_lr,              'name': 'vae_lora'},
            {'params': task_params,     'lr': self.learning_rate,   'name': 'task_encoder'},
            {'params': prior_params,    'lr': vae_lr,              'name': 'prior_mlps'},
            {'params': other_gen_params,'lr': self.learning_rate,   'name': 'generator_custom'},
        ])

        lrs = [vae_lr, self.learning_rate, vae_lr, self.learning_rate]
        scheduler = optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=[
                (lambda base: lambda step: (
                    self.lr_min / base
                    + (1 - self.lr_min / base)
                    * (1 + math.cos(math.pi * step / total_steps)) / 2
                ))(lr) for lr in lrs
            ],
        )

        return optimizer, scheduler

    # ------------------------------------------------------------------
    # Training step
    # ------------------------------------------------------------------

    def _train_step(
        self,
        optimizer: optim.Optimizer,
    ) -> Dict[str, float]:
        """
        One gradient update across a randomly sampled (model, strategy, dataset, image).

        Returns dict of scalar loss values for logging.
        """
        gen = self.generator
        device = self._device

        # --- 1. Sample a random model entry and image ---
        entry = self.ensemble.sample_entry()
        item = self.dataset_pool.sample()

        # image: [3, H, W] → [1, 3, H, W] on device
        image = item.image.unsqueeze(0).to(device)

        # --- 2. Sample strategy placement kwargs ---
        strategy_kwargs = entry.strategy.sample_kwargs(
            image, gen.patch_height, gen.patch_width
        )

        # --- 3. Control activation (no_grad, no patch) ---
        with self.ensemble.on_device(entry) as model:
            model = model.to(device)
            neutral_inp = entry.preprocess_fn(
                entry.strategy.apply_neutral(image, **strategy_kwargs)
            )
            sample_shape = (3, *entry.input_shape)
            sampled_neurons = self.neuron_sampler.sample_neurons(
                model, self.k_neurons, sample_shape
            )
            ctrl_acts = self.neuron_sampler.capture_sampled_activations(
                model, neutral_inp, sampled_neurons, no_grad=True
            ).to(device)   # [k], detached, no grad

        # --- 4. Generate P patches conditioned on task metadata ---
        P = self.patches_per_batch
        z = torch.randn(P, gen.latent_dim, device=device)

        model_idx    = torch.full((P,), entry.model_id,    device=device, dtype=torch.long)
        strategy_idx = torch.full((P,), entry.strategy_id, device=device, dtype=torch.long)
        dataset_idx  = torch.full((P,), item.dataset_id,   device=device, dtype=torch.long)

        # Clamp indices to valid range (defensive)
        model_idx    = model_idx.clamp(0, self.task_encoder.num_models - 1)
        strategy_idx = strategy_idx.clamp(0, self.task_encoder.num_strategies - 1)
        dataset_idx  = dataset_idx.clamp(0, self.task_encoder.num_datasets - 1)

        z_enriched = self.task_encoder(z, model_idx, strategy_idx, dataset_idx)  # [P, D]
        patches = gen(z, z_enriched)   # [P, 3, H, W]

        # --- 5. Adversarial activations (with grad) for each patch ---
        adv_acts_list: List[Tensor] = []
        with self.ensemble.on_device(entry) as model:
            model = model.to(device)
            for p_idx in range(P):
                patch = patches[p_idx]   # [3, pH, pW]
                composited, _ = entry.strategy.apply(image, patch, **strategy_kwargs)
                adv_inp = entry.preprocess_fn(composited)
                adv_acts = self.neuron_sampler.capture_sampled_activations(
                    model, adv_inp, sampled_neurons, no_grad=False
                )   # [k], has grad_fn
                adv_acts_list.append(adv_acts)

        # --- 6. Compute losses ---
        # Deltas: [P, k]
        deltas = torch.stack([a - ctrl_acts for a in adv_acts_list], dim=0)

        # Diversity: log-det of Gram matrix of unit-normalised deltas
        eps = max(1e-6, 1e-2 / P)
        normalized = F.normalize(deltas, p=2, dim=1)   # [P, k]
        gram = normalized @ normalized.T               # [P, P]
        gram = gram + eps * torch.eye(P, device=device)
        sign, log_det = torch.slogdet(gram)
        if torch.isnan(log_det) or sign <= 0:
            log_det = torch.tensor(-20.0, device=device, dtype=deltas.dtype)

        # Quality: mean RMS of normalized deltas across patches
        ctrl_std = ctrl_acts.std() + 1e-8
        norm_deltas_sq = (deltas / ctrl_std) ** 2      # [P, k]
        per_patch_rms = norm_deltas_sq.mean(dim=1).sqrt()  # [P]
        quality = per_patch_rms.mean() + 1e-8           # scalar

        total_act_loss = -(
            self.diversity_weight * log_det
            + self.quality_weight * torch.log(quality)
        )

        # Visibility mask for TV/spectrum (use first patch; geometry is per-image)
        dummy = torch.zeros(1, 3, gen.patch_height, gen.patch_width, device=device)
        _, vis_mask = entry.strategy.apply(dummy, patches[0].detach(), **strategy_kwargs)

        patches_stacked = patches  # [P, 3, H, W]
        tv_val = self.tv_weight * total_variation_loss(patches_stacked, vis_mask)
        spec_val = self.spectrum_weight * compute_spectrum_loss(patches_stacked, vis_mask)

        loss = total_act_loss + tv_val + spec_val
        loss.backward()

        # Release the grad graph immediately
        del deltas, adv_acts_list

        return {
            'loss':         loss.item(),
            'diversity':    log_det.item(),
            'quality':      quality.item(),
            'tv':           tv_val.item(),
            'spectrum':     spec_val.item(),
            'model':        entry.name,
        }

    # ------------------------------------------------------------------
    # Checkpoint
    # ------------------------------------------------------------------

    def _save_checkpoint(self, epoch: int, subdir: str = 'checkpoint') -> None:
        ckpt_dir = self.output_dir / subdir
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        pr = None
        br = getattr(self.generator, 'bottleneck_refiner', None)
        if br is not None:
            pr = getattr(br, 'prior_registry', None)

        ckpt = {
            'generator_state_dict':    self.generator.state_dict(),
            'task_encoder_state_dict': self.task_encoder.state_dict(),
            'epoch': epoch,
            'config': {
                'latent_dim':       self.generator.latent_dim,
                'patch_height':     self.generator.patch_height,
                'patch_width':      self.generator.patch_width,
                'k_neurons':        self.k_neurons,
                'patches_per_batch':self.patches_per_batch,
            },
        }
        if pr is not None:
            ckpt['prior_registry_state_dict'] = pr.state_dict()

        ckpt_path = ckpt_dir / f'ensemble_epoch_{epoch:04d}.pt'
        torch.save(ckpt, ckpt_path)
        print(f"  ✓ Checkpoint saved: {ckpt_path}")

        # Save a few example patches
        with torch.no_grad():
            z = torch.randn(4, self.generator.latent_dim, device=self._device)
            patches = self.generator(z)
            for i, patch in enumerate(patches):
                T.ToPILImage()(patch.cpu()).save(
                    ckpt_dir / f'patch_epoch_{epoch:04d}_sample_{i}.png'
                )

    # ------------------------------------------------------------------
    # Full training loop
    # ------------------------------------------------------------------

    def train(
        self,
        steps_per_epoch: int = 100,
        resume_from: Optional[str] = None,
        max_steps: Optional[int] = None,
    ) -> None:
        """
        Train the generator for max_epochs, steps_per_epoch steps each.

        Args:
            steps_per_epoch: number of gradient updates per epoch
            resume_from:     optional checkpoint path to resume from
            max_steps:       if set, stop after this many total steps (for smoke tests)
        """
        total_steps = steps_per_epoch * self.max_epochs + 1
        optimizer, scheduler = self._build_optimizer(total_steps)

        start_epoch = 1
        if resume_from is not None:
            ckpt_files = sorted(Path(resume_from).glob("ensemble_epoch_*.pt"))
            if ckpt_files:
                ckpt = torch.load(ckpt_files[-1], map_location='cpu')
                self.generator.load_state_dict(ckpt['generator_state_dict'])
                self.task_encoder.load_state_dict(ckpt['task_encoder_state_dict'])
                start_epoch = ckpt.get('epoch', 0) + 1
                print(f"Resumed from {ckpt_files[-1]} at epoch {start_epoch}")

        print(f"\n{'='*70}")
        print(f"EnsembleTrainer — {self.ensemble.num_models()} models, "
              f"{self.dataset_pool.num_datasets()} datasets, "
              f"k={self.k_neurons} neurons, P={self.patches_per_batch} patches/step")
        print(f"{'='*70}\n")

        global_step = 0

        for epoch in range(start_epoch, self.max_epochs + 1):
            self.generator.train()
            self.task_encoder.train()

            epoch_losses: Dict[str, float] = {
                'loss': 0.0, 'diversity': 0.0, 'quality': 0.0,
                'tv': 0.0, 'spectrum': 0.0,
            }

            with tqdm(total=steps_per_epoch, desc=f"Epoch {epoch}", leave=False) as pbar:
                for step in range(steps_per_epoch):
                    optimizer.zero_grad()

                    try:
                        info = self._train_step(optimizer)
                    except Exception as e:
                        print(f"\n  Warning: step {step} failed ({e}); skipping.")
                        optimizer.zero_grad()
                        continue

                    torch.nn.utils.clip_grad_norm_(
                        list(self.generator.parameters())
                        + list(self.task_encoder.parameters()), 1.0
                    )
                    optimizer.step()
                    scheduler.step()

                    for k in epoch_losses:
                        epoch_losses[k] += info.get(k, 0.0)

                    pbar.set_postfix({
                        'loss':  f"{info['loss']:.4f}",
                        'div':   f"{info['diversity']:.2f}",
                        'model': info['model'],
                    })
                    pbar.update(1)

                    global_step += 1
                    if max_steps is not None and global_step >= max_steps:
                        print(f"\n  max_steps={max_steps} reached; stopping.")
                        break

            n = steps_per_epoch
            print(
                f"[Epoch {epoch}] "
                f"loss={epoch_losses['loss']/n:.4f}  "
                f"div={epoch_losses['diversity']/n:.3f}  "
                f"qual={epoch_losses['quality']/n:.3f}  "
                f"tv={epoch_losses['tv']/n:.4f}  "
                f"spec={epoch_losses['spectrum']/n:.4f}"
            )

            if epoch % self.save_every_epochs == 0 or epoch == self.max_epochs:
                self._save_checkpoint(epoch, subdir=f'checkpoint_epoch_{epoch:04d}')

            if max_steps is not None and global_step >= max_steps:
                break

        # Final checkpoint
        self._save_checkpoint(self.max_epochs, subdir='ensemble_final')
        print(f"\n✓ Ensemble training complete. Output: {self.output_dir}")
