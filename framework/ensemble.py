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

import io
import math
import os
import random
import signal
import tarfile
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, Generator, List, Optional, Tuple

import PIL.Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor, optim
import torchvision.transforms as T
from tqdm import tqdm

from framework.base.attack_strategy import AttackStrategy, BorderStrategy, StickerStrategy
from framework.dataset_pool import LazyDatasetPool
from framework.generator import FoundationPatchGenerator
from framework.losses import total_variation_loss, compute_spectrum_loss
from framework.neuron_sampler import NeuronSampler
from framework.task_encoder import TaskEncoder


# ---------------------------------------------------------------------------
# Prefetch dataset for parallel training image loading
# ---------------------------------------------------------------------------

class _TrainingImageDataset(torch.utils.data.IterableDataset):
    """
    Infinite random-sampling dataset for parallel image prefetching during training.

    Each DataLoader worker independently calls pool.sample() in a tight loop,
    so N workers load N images in parallel. The DataLoader batches these into
    [images_per_batch, 3, H, W] tensors that are handed to _train_step.
    """
    def __init__(self, pool: LazyDatasetPool):
        self.pool = pool

    def __iter__(self):
        while True:
            item = self.pool.sample()
            yield item.image, item.dataset_id


# ---------------------------------------------------------------------------
# EnsembleModelPool
# ---------------------------------------------------------------------------

@dataclass
class ModelEntry:
    """One registered model entry in the ensemble pool."""
    model_id: int
    name: str
    domain_type: str
    _model: nn.Module
    input_shape: Tuple[int, int]   # (H, W)
    preprocess_fn: Callable[[Tensor], Tensor]


@dataclass
class StrategyEntry:
    """One registered attack strategy."""
    strategy_id: int
    name: str
    strategy: AttackStrategy


class EnsembleModelPool:
    """
    Pool of frozen target models and a separate pool of attack strategies.

    Any strategy can be applied to any model — they are sampled independently.
    Models are permanently on compute_device after registration.

    Usage::

        pool = EnsembleModelPool(compute_device=torch.device('cuda'))
        pool.register('resnet50', model, 'classification',
                      input_shape=(224, 224), preprocess_fn=fn)
        pool.register_strategy('border', BorderStrategy(), strategy_id=0)
        entry = pool.sample_entry()
        strat = pool.sample_strategy()
        with pool.on_device(entry) as model:
            out = model(inp)
    """

    def __init__(self, compute_device: Optional[torch.device] = None):
        self.compute_device = compute_device or torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        self._entries: List[ModelEntry] = []
        self._strategies: List[StrategyEntry] = []
        self._next_id: int = 0

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(
        self,
        name: str,
        model: nn.Module,
        domain_type: str,
        input_shape: Tuple[int, int],
        preprocess_fn: Callable[[Tensor], Tensor],
    ) -> int:
        """
        Register a model in the ensemble pool.

        The model is set to eval mode, all parameters frozen, and moved
        to compute_device permanently.

        Args:
            name:          human-readable identifier
            model:         nn.Module to register
            domain_type:   arbitrary tag (e.g. 'classification', 'detection')
            input_shape:   (H, W) expected by this model (after preprocessing)
            preprocess_fn: callable (Tensor [B,3,H,W]) → Tensor [B,3,H',W']

        Returns:
            model_id (int)
        """
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        model = model.to(self.compute_device)

        model_id = self._next_id
        self._next_id += 1
        self._entries.append(ModelEntry(
            model_id=model_id,
            name=name,
            domain_type=domain_type,
            _model=model,
            input_shape=input_shape,
            preprocess_fn=preprocess_fn,
        ))
        return model_id

    def register_strategy(
        self,
        name: str,
        strategy: AttackStrategy,
        strategy_id: int,
    ) -> None:
        """Register an attack strategy. Any strategy can be used with any model."""
        self._strategies.append(StrategyEntry(
            strategy_id=strategy_id,
            name=name,
            strategy=strategy,
        ))

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def sample_entry(self) -> ModelEntry:
        """Sample a uniformly random model entry."""
        if not self._entries:
            raise RuntimeError("No models registered. Call register() first.")
        return random.choice(self._entries)

    def sample_strategy(self) -> StrategyEntry:
        """Sample a uniformly random strategy entry."""
        if not self._strategies:
            raise RuntimeError("No strategies registered. Call register_strategy() first.")
        return random.choice(self._strategies)

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
        """No-op context manager — all models are permanently on compute_device."""
        yield entry._model

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    def num_models(self) -> int:
        return len(self._entries)

    def num_strategies(self) -> int:
        return len(self._strategies)

    def __repr__(self) -> str:
        model_names = [e.name for e in self._entries]
        strat_names = [s.name for s in self._strategies]
        return (f"EnsembleModelPool(models={model_names}, "
                f"strategies={strat_names}, device={self.compute_device})")


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
        - Gradient accumulation over images_per_batch images per optimizer step

    Each optimizer step:
        for i in range(images_per_batch):
            sample image + model
            generate patches_per_image patches
            compute loss / images_per_batch
            loss.backward()           ← accumulates into .grad
        optimizer.step()              ← one update per images_per_batch images

    Optimizer groups (AdamW):
        - TAESD decoder params:    lr = learning_rate * vae_lr_ratio
        - TaskEncoder params:      lr = learning_rate
        - All other generator params (adapters, transformers, channel_mixer): lr = learning_rate
    """

    def __init__(
        self,
        ensemble: EnsembleModelPool,
        dataset_pool: LazyDatasetPool,
        task_encoder: TaskEncoder,
        generator: FoundationPatchGenerator,
        neuron_sampler: NeuronSampler,
        k_neurons: int = 10_000,
        patches_per_image: int = 8,
        images_per_batch: int = 32,
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
        warmup_steps: int = 0,
        warmup_model: Optional[str] = None,
        num_prefetch_workers: int = 8,
        device: Optional[torch.device] = None,
    ):
        self.ensemble = ensemble
        self.dataset_pool = dataset_pool
        self.task_encoder = task_encoder
        self.generator = generator
        self.neuron_sampler = neuron_sampler

        self.k_neurons = k_neurons
        self.patches_per_image = patches_per_image
        self.images_per_batch = images_per_batch
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
        self.warmup_steps = warmup_steps
        self.warmup_model = warmup_model
        self.num_prefetch_workers = num_prefetch_workers

        self._device = device or ensemble.compute_device
        self.generator = self.generator.to(self._device)
        self.task_encoder = self.task_encoder.to(self._device)


        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Optimizer construction
    # ------------------------------------------------------------------

    def _build_optimizer(self, total_steps: int):
        """Build AdamW with 3 parameter groups + cosine annealing scheduler."""
        gen = self.generator

        # Group 1: TAESD decoder params (lower LR — pretrained)
        taesd_params = [
            p for name, p in gen.named_parameters()
            if 'taesd_decoders.' in name and p.requires_grad
        ]

        # Group 2: TaskEncoder params
        task_params = list(self.task_encoder.parameters())

        # Group 3: all other generator params (adapters, transformers, channel_mixer)
        taesd_ids = {id(p) for p in taesd_params}
        other_gen_params = [
            p for n, p in gen.named_parameters()
            if p.requires_grad and id(p) not in taesd_ids
        ]

        taesd_lr = self.learning_rate * self.vae_lr_ratio
        optimizer = optim.AdamW([
            {'params': taesd_params,     'lr': taesd_lr,            'name': 'taesd_decoders'},
            {'params': task_params,      'lr': self.learning_rate,   'name': 'task_encoder'},
            {'params': other_gen_params, 'lr': self.learning_rate,   'name': 'generator_custom'},
        ])

        warmup_steps = 5
        lrs = [taesd_lr, self.learning_rate, self.learning_rate]
        scheduler = optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=[
                (lambda base: lambda step: (
                    (step + 1) / warmup_steps  # linear warmup: 0.2, 0.4, 0.6, 0.8, 1.0
                    if step < warmup_steps
                    else (
                        self.lr_min / base
                        + (1 - self.lr_min / base)
                        * (1 + math.cos(math.pi * (step - warmup_steps) / (total_steps - warmup_steps))) / 2
                    )
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
        prefetched: Optional[Tuple[Tensor, Tensor]] = None,
        entries_override=None,
        use_stream_gating: bool = False,
    ) -> Dict[str, float]:
        """
        One optimizer update accumulating gradients over images_per_batch images.

        Loop structure: outer = models, inner = images per model.
        Per image: ctrl + P patches are batched into one target-model forward pass
        (P+1 inputs, same neuron sample), then backward() is called immediately
        so only one image's activation stack lives at a time.

        Args:
            prefetched: (images [N,3,H,W], dataset_ids [N]) pre-loaded by the
                        background DataLoader. If None, falls back to synchronous
                        dataset_pool.sample() (used in smoke-test / CPU mode).
        """
        gen = self.generator
        device = self._device
        P = self.patches_per_image
        N = self.images_per_batch

        acc: Dict[str, float] = {
            'loss': 0.0, 'diversity': 0.0, 'quality': 0.0, 'final_quality': 0.0,
            'tv': 0.0, 'spectrum': 0.0, 'model': '',
        }

        # Balanced distribution: N images split evenly across all models
        all_entries = entries_override if entries_override is not None else list(self.ensemble._entries)
        n_models = len(all_entries)
        images_per_model = N // n_models
        remainder = N % n_models

        img_cursor = 0  # index into prefetched batch

        # Outer loop: models. Inner loop: images per model.
        # Per-image forward pass batches ctrl + P patches together (P+1 inputs,
        # same neuron sample) so the target model is called once per image
        # instead of P+1 times. Gradient accumulation: backward after each image.
        for model_idx_outer, entry in enumerate(all_entries):
            n_images = images_per_model + (1 if model_idx_outer < remainder else 0)

            with self.ensemble.on_device(entry) as model:
                sample_shape = (3, *entry.input_shape)
                sampled_neurons = self.neuron_sampler.sample_neurons(
                    model, self.k_neurons, sample_shape
                )
                final_neurons    = self.neuron_sampler.get_final_layer_neurons(entry.model_id)
                k_rand           = len(sampled_neurons)
                combined_neurons = sampled_neurons + final_neurons

                for _ in range(n_images):
                    # --- 1. Get image (from prefetch queue or synchronous fallback) ---
                    strat_entry = self.ensemble.sample_strategy()
                    if prefetched is not None:
                        image      = prefetched[0][img_cursor].unsqueeze(0).to(device)
                        dataset_id = int(prefetched[1][img_cursor])
                        img_cursor += 1
                    else:
                        item       = self.dataset_pool.sample()
                        image      = item.image.unsqueeze(0).to(device)
                        dataset_id = item.dataset_id

                    strategy_kwargs = strat_entry.strategy.sample_kwargs(
                        image, gen.patch_height, gen.patch_width,
                        model_input_shape=entry.input_shape,
                    )

                    # --- 2. Generate P patches ---
                    z = torch.randn(P, gen.latent_dim, device=device)

                    midx = torch.full((P,), entry.model_id,          device=device, dtype=torch.long)
                    sidx = torch.full((P,), strat_entry.strategy_id, device=device, dtype=torch.long)
                    didx = torch.full((P,), dataset_id,              device=device, dtype=torch.long)

                    midx = midx.clamp(0, self.task_encoder.num_models    - 1)
                    sidx = sidx.clamp(0, self.task_encoder.num_strategies - 1)
                    didx = didx.clamp(0, self.task_encoder.num_datasets  - 1)

                    z_enriched = self.task_encoder(z, midx, sidx, didx)

                    # Stream gating: during warmup, each strategy uses a dedicated
                    # subset of TAESD streams so each specialises early on.
                    active_streams = None
                    if use_stream_gating:
                        num_strats = self.task_encoder.num_strategies
                        streams_per_strat = max(1, gen.num_taesd // num_strats)
                        start = strat_entry.strategy_id * streams_per_strat
                        active_streams = list(range(
                            start, min(start + streams_per_strat, gen.num_taesd)
                        ))

                    patches = gen(z, z_enriched, active_streams=active_streams)  # [P, 3, H, W]

                    # --- 3. Build [P+1, 3, H', W'] batch: ctrl first, then P adv ---
                    ctrl_inp = entry.preprocess_fn(
                        strat_entry.strategy.apply_neutral(image, **strategy_kwargs)
                    )                                                           # [1, 3, H', W']

                    vis_mask  = None
                    adv_inps  = []
                    for p_idx in range(P):
                        composited, mask = strat_entry.strategy.apply(
                            image, patches[p_idx], **strategy_kwargs
                        )
                        if vis_mask is None:
                            vis_mask = mask
                        adv_inps.append(entry.preprocess_fn(composited))       # [1, 3, H', W']

                    batch_inp = torch.cat([ctrl_inp] + adv_inps, dim=0)        # [P+1, 3, H', W']

                    # --- 4. One batched forward with robust neuron extraction ---
                    # The neuron sampler now uses the shape cache to infer per-sample
                    # size and correctly extract neurons even if batch dims are folded
                    # with spatial or head dimensions in intermediate layers.
                    combined = self.neuron_sampler.capture_sampled_activations(
                        model, batch_inp, combined_neurons, no_grad=False,
                    ).to(device)                                                # [P+1, k_rand+k_final]

                    ctrl_acts  = combined[0, :k_rand].detach()                 # [k_rand], detached
                    ctrl_final = combined[0, k_rand:].detach()                 # [k_final], detached
                    adv_acts   = combined[1:]                                  # [P, k_rand+k_final]

                    # --- 5. Compute losses ---
                    deltas = adv_acts[:, :k_rand] - ctrl_acts                  # [P, k_rand]

                    neuron_stds    = self.neuron_sampler.lookup_neuron_stds(
                        entry.model_id, sampled_neurons
                    ).to(device)
                    beta           = self.neuron_sampler.lookup_beta(entry.model_id)
                    effective_stds = neuron_stds.clamp(min=beta)
                    norm_deltas    = deltas / effective_stds                    # [P, k_rand] — kept for quality logging

                    norm_deltas_sq = norm_deltas ** 2
                    quality        = norm_deltas_sq.mean(dim=1).sqrt().mean() + 1e-8

                    # Final-layer activations — used for quality only
                    if final_neurons:
                        final_deltas      = adv_acts[:, k_rand:] - ctrl_final  # [P, k_final]
                        final_neuron_stds = self.neuron_sampler.lookup_neuron_stds(
                            entry.model_id, final_neurons
                        ).to(device)
                        final_eff_stds    = final_neuron_stds.clamp(min=beta)
                        final_norm_deltas = final_deltas / final_eff_stds       # [P, k_final]
                        final_quality     = final_norm_deltas.pow(2).mean(dim=1).sqrt().mean() + 1e-8
                    else:
                        final_quality = torch.tensor(1e-8, device=device)

                    # Diversity always uses randomly sampled neurons (broad coverage)
                    div_vecs = norm_deltas

                    eps        = max(1e-6, 1e-2 / P)
                    normalized = F.normalize(div_vecs, p=2, dim=1)
                    gram       = normalized @ normalized.T + eps * torch.eye(P, device=device)
                    sign, log_det = torch.slogdet(gram)
                    if torch.isnan(log_det) or sign <= 0:
                        log_det = torch.tensor(-20.0, device=device, dtype=div_vecs.dtype)

                    # Resize patches to actual display size before computing TV/SSIM.
                    # For StickerStrategy the bbox gives the on-image sticker dimensions.
                    bbox = strategy_kwargs.get('bbox')
                    if bbox is not None and isinstance(strat_entry.strategy, StickerStrategy):
                        x0, y0, x1, y1 = bbox
                        sh, sw = max(1, y1 - y0), max(1, x1 - x0)
                        patches_scaled = F.interpolate(patches, size=(sh, sw), mode='bilinear', align_corners=False)
                        vis_mask_scaled = torch.ones(1, 1, sh, sw, device=device)
                    else:
                        patches_scaled  = patches
                        vis_mask_scaled = vis_mask

                    # Zero out invisible (center) pixels so SSIM sliding windows that
                    # overlap the border/center boundary don't see arbitrary generator values.
                    patches_for_loss = patches_scaled * vis_mask_scaled

                    # Apply TV loss to both BorderStrategy and StickerStrategy (penalize high-frequency noise)
                    if isinstance(strat_entry.strategy, (BorderStrategy, StickerStrategy)):
                        tv_raw = total_variation_loss(patches_for_loss, vis_mask_scaled)
                    else:
                        tv_raw = torch.tensor(0.0, device=device)

                    spec_raw = compute_spectrum_loss(patches_for_loss, vis_mask_scaled)

                    per_image_loss = -(
                        self.diversity_weight * log_det
                        + self.quality_weight * final_quality
                    ) + self.tv_weight * tv_raw + self.spectrum_weight * spec_raw

                    (per_image_loss / N).backward()

                    acc['loss']          += per_image_loss.item()
                    acc['diversity']     += log_det.item()
                    acc['quality']       += quality.item()
                    acc['final_quality'] += final_quality.item()
                    acc['tv']            += (self.tv_weight * tv_raw).item()
                    acc['spectrum']      += spec_raw.item()
                    acc['model']          = entry.name

        return {
            'loss':          acc['loss']          / N,
            'diversity':     acc['diversity']     / N,
            'quality':       acc['quality']       / N,
            'final_quality': acc['final_quality'] / N,
            'tv':            acc['tv']            / N,
            'spectrum':      acc['spectrum']      / N,
            'model':         acc['model'],
        }

    # ------------------------------------------------------------------
    # Neuron profiling
    # ------------------------------------------------------------------

    def _profile_all_models(self, n_images: int = 1000) -> None:
        """
        Precompute per-neuron activation std for every registered model.

        For each model, checks ~/.cache/adversarial_plate_profiles/ for a saved
        profile matching {model_name}_n{n_images}.pt.  If found, loads it
        instantly.  Otherwise runs Welford profiling (batched, with progress bar)
        and saves the result for future runs.

        Args:
            n_images: number of images to profile with (default 1024, cache key includes this)
        """
        import os
        cache_dir = Path(os.path.expanduser('~/.cache/adversarial_plate_profiles'))
        cache_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nProfiling {self.ensemble.num_models()} models "
              f"({n_images} images each) for per-neuron std ...")

        patch_h = self.generator.patch_height
        patch_w = self.generator.patch_width
        all_strategies = [se.strategy for se in self.ensemble._strategies]

        for entry in self.ensemble._entries:
            # Cache key includes '_delta' to distinguish from old raw-activation profiles
            cache_path = cache_dir / f'{entry.name}_n{n_images}_delta.pt'

            # --- Try loading from cache ---
            if self.neuron_sampler.load_profile(entry.model_id, cache_path):
                print(f"  {entry.name} ... loaded from cache ({cache_path.name})")
                continue

            # --- Cache miss: profile std of activation DELTAS (patched - clean) ---
            # Workers load raw images on CPU; main thread composites and preprocesses on GPU.
            # Each batch: neutral-border images vs randomly-attacked images → one 2B forward pass.
            class _ProfilingDataset(torch.utils.data.Dataset):
                def __init__(self, pool, n):
                    self.pool = pool
                    self.n = n

                def __len__(self):
                    return self.n

                def __getitem__(self, idx):
                    # Return raw image — compositing + preprocessing happens on GPU in main thread
                    return self.pool.sample().image  # [3, H, W]

            dataset = _ProfilingDataset(self.dataset_pool, n_images)
            loader = torch.utils.data.DataLoader(
                dataset,
                batch_size=10,
                num_workers=4,
                shuffle=False,
                pin_memory=(self._device.type == 'cuda'),
            )

            num_batches = len(loader)
            sample_shape = (3, *entry.input_shape)
            with self.ensemble.on_device(entry) as model:
                state = self.neuron_sampler.init_profile(model, sample_shape, device=self._device)
                with tqdm(total=num_batches, desc=f"    {entry.name}", leave=False) as pbar:
                    for raw_batch in loader:
                        raw_batch = raw_batch.to(self._device)  # [B, 3, H, W]

                        # Randomly pick an attack strategy for this batch
                        strategy = random.choice(all_strategies) if all_strategies else BorderStrategy()
                        strategy_kwargs = strategy.sample_kwargs(
                            raw_batch, patch_h, patch_w,
                            model_input_shape=entry.input_shape,
                        )

                        # Clean: neutral-border composite → preprocess
                        clean_composited = strategy.apply_neutral(raw_batch, **strategy_kwargs)
                        clean_batch = entry.preprocess_fn(clean_composited)

                        # Attacked: uniform-random patch → preprocess
                        rand_patch = torch.rand(3, patch_h, patch_w, device=self._device)
                        patched_composited, _ = strategy.apply(raw_batch, rand_patch, **strategy_kwargs)
                        patched_batch = entry.preprocess_fn(patched_composited)

                        self.neuron_sampler.update_profile_delta(model, clean_batch, patched_batch, state)
                        pbar.update(1)
                self.neuron_sampler.finish_profile(entry.model_id, state)

            self.neuron_sampler.save_profile(entry.model_id, cache_path)
            print(f"  {entry.name} done (saved to {cache_path.name})")

        print("Profiling complete.\n")

    # ------------------------------------------------------------------
    # Checkpoint
    # ------------------------------------------------------------------

    def _save_checkpoint(
        self,
        epoch: int,
        global_step: int,
        run_dir: Path,
        optimizer: optim.Optimizer,
        scheduler,
        subdir: str = 'checkpoint',
    ) -> None:
        ckpt_dir = run_dir / subdir
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        gen = self.generator
        ckpt = {
            # Top-level keys read directly by resume logic
            'epoch':                   epoch,
            'global_step':             global_step,
            # Model weights
            'generator_state_dict':    gen.state_dict(),
            'task_encoder_state_dict': self.task_encoder.state_dict(),
            # Optimizer + scheduler state — needed for correct resume
            'optimizer_state_dict':    optimizer.state_dict(),
            'scheduler_state_dict':    scheduler.state_dict(),
            # Architecture metadata
            'basis_dim':               gen.latent_dim,
            'patch_size':              [gen.patch_height, gen.patch_width],
            'num_taesd':               gen.num_taesd,
            'transformer_d_model':     gen.transformer_d_model,
            'transformer_nhead':       gen.transformer_nhead,
            'transformer_d_ff':        gen.transformer_d_ff,
            'transformer_enc_layers':  gen.transformer_enc_layers,
            'transformer_dec_layers':  gen.transformer_dec_layers,
            'run_dir': str(run_dir),
            'training_info': {
                'epoch':             epoch,
                'global_step':       global_step,
                'k_neurons':         self.k_neurons,
                'patches_per_image': self.patches_per_image,
                'images_per_batch':  self.images_per_batch,
            },
        }

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
    # Debug visualizations
    # ------------------------------------------------------------------

    def _save_debug_visualizations(self, debug_dir: Path) -> None:
        """Generate debug images showing clean and attacked image pairs."""
        debug_dir.mkdir(parents=True, exist_ok=True)

        if not self.ensemble._strategies:
            print("  No strategies registered; skipping debug visualizations")
            return

        num_samples = 4
        patch_h = self.generator.patch_height
        patch_w = self.generator.patch_width

        model_input_shape = self.ensemble._entries[0].input_shape if self.ensemble._entries else None

        print(f"\nGenerating debug visualizations ({num_samples} pairs × {len(self.ensemble._strategies)} strategies)...")

        for strat_entry in self.ensemble._strategies:
            strategy = strat_entry.strategy
            for sample_idx in range(num_samples):
                item = self.dataset_pool.sample()
                raw_image = item.image.to(self._device)  # [3, H, W]

                strategy_kwargs = strategy.sample_kwargs(
                    raw_image.unsqueeze(0), patch_h, patch_w,
                    model_input_shape=model_input_shape,
                )

                # Clean: apply neutral
                clean_composited = strategy.apply_neutral(
                    raw_image.unsqueeze(0), **strategy_kwargs
                )  # [1, 3, ?, ?]

                # Attacked: apply random patch
                rand_patch = torch.rand(3, patch_h, patch_w, device=self._device)
                attacked_composited, _ = strategy.apply(
                    raw_image.unsqueeze(0), rand_patch, **strategy_kwargs
                )  # [1, 3, ?, ?]

                clean_pil = T.ToPILImage()(clean_composited[0].cpu().clamp(0, 1))
                attacked_pil = T.ToPILImage()(attacked_composited[0].cpu().clamp(0, 1))

                combined_width = clean_pil.width + attacked_pil.width
                combined_height = max(clean_pil.height, attacked_pil.height)
                combined = PIL.Image.new('RGB', (combined_width, combined_height))
                combined.paste(clean_pil, (0, 0))
                combined.paste(attacked_pil, (clean_pil.width, 0))

                combined.save(debug_dir / f'debug_pair_{strat_entry.name}_{sample_idx:02d}_clean_vs_attacked.png')

        print(f"  Saved debug image pairs to {debug_dir}\n")

    # ------------------------------------------------------------------
    # Sample saving
    # ------------------------------------------------------------------

    def _save_samples(self, global_step: int, samples_dir: Path) -> None:
        """Save 10 patches per strategy × model into a tar at samples_dir/step_NNNNNNN.tar."""
        step_name = f'step_{global_step:07d}'
        tar_path  = samples_dir / f'{step_name}.tar'

        self.generator.eval()
        self.task_encoder.eval()
        try:
            with torch.no_grad():
                with tarfile.open(tar_path, 'w') as tar:
                    for strat_entry in self.ensemble._strategies:
                        for model_entry in self.ensemble._entries:
                            dataset_idx = random.randint(0, self.task_encoder.num_datasets - 1)

                            z            = torch.randn(10, self.generator.latent_dim, device=self._device)
                            model_idx    = torch.full((10,), model_entry.model_id,    dtype=torch.long, device=self._device)
                            strategy_idx = torch.full((10,), strat_entry.strategy_id, dtype=torch.long, device=self._device)
                            dataset_idx_t = torch.full((10,), dataset_idx,            dtype=torch.long, device=self._device)

                            z_enriched     = self.task_encoder(z, model_idx, strategy_idx, dataset_idx_t)
                            sample_patches = self.generator(z, z_enriched)

                            # Resize to actual on-image sticker size for display
                            if isinstance(strat_entry.strategy, StickerStrategy):
                                ref_h, ref_w = model_entry.input_shape
                                target_area = strat_entry.strategy.area_fraction * ref_h * ref_w
                                sh = sw = max(1, int(target_area ** 0.5))
                                sample_patches = F.interpolate(
                                    sample_patches, size=(sh, sw), mode='bilinear', align_corners=False
                                )

                            for i, patch in enumerate(sample_patches):
                                buf = io.BytesIO()
                                T.ToPILImage()(patch.cpu()).save(buf, format='PNG')
                                buf.seek(0)

                                tar_info = tarfile.TarInfo(
                                    name=f'{step_name}/{strat_entry.name}/{model_entry.name}/{i}.png'
                                )
                                tar_info.size = len(buf.getvalue())
                                tar.addfile(tar_info, buf)
        finally:
            self.generator.train()
            self.task_encoder.train()

    # ------------------------------------------------------------------
    # Full training loop
    # ------------------------------------------------------------------

    def train(
        self,
        resume_from: Optional[str] = None,
        max_steps: Optional[int] = None,
    ) -> None:
        """
        Train the generator for max_epochs epochs.

        One epoch = one full pass over the dataset pool (dataset_pool.total_images()
        images, grouped into steps of images_per_batch each).

        The LR scheduler steps every optimizer update (not every epoch), so the
        cosine schedule stays correct even when training is stopped mid-epoch.
        Checkpoints are saved every save_every_epochs completed epochs, and also
        whenever max_steps is hit mid-epoch.

        Args:
            resume_from: optional checkpoint directory to resume from
            max_steps:   stop after this many total optimizer steps (partial epoch ok)
        """
        # One epoch = ceil(total_images / images_per_batch) optimizer steps
        total_images = self.dataset_pool.total_images()
        steps_per_epoch = max(1, math.ceil(total_images / self.images_per_batch))
        total_steps = self.warmup_steps + steps_per_epoch * self.max_epochs + 1

        # If max_steps is set, design the LR schedule for that budget
        schedule_steps = max_steps if max_steps is not None else total_steps
        optimizer, scheduler = self._build_optimizer(schedule_steps)

        start_epoch = 1
        start_step  = 0
        run_dir: Optional[Path] = None

        if resume_from is not None:
            ckpt_files = sorted(Path(resume_from).glob("ensemble_epoch_*.pt"))
            if ckpt_files:
                ckpt = torch.load(ckpt_files[-1], map_location='cpu')
                self.generator.load_state_dict(ckpt['generator_state_dict'])
                self.task_encoder.load_state_dict(ckpt['task_encoder_state_dict'])
                # epoch and global_step are at top level (training_info was a bug)
                start_epoch = ckpt.get('epoch', ckpt.get('training_info', {}).get('epoch', 0)) + 1
                start_step  = ckpt.get('global_step', ckpt.get('training_info', {}).get('global_step', 0))
                # Restore optimizer moments + scheduler position (no fast-forward needed)
                if 'optimizer_state_dict' in ckpt:
                    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
                if 'scheduler_state_dict' in ckpt:
                    scheduler.load_state_dict(ckpt['scheduler_state_dict'])
                else:
                    # Legacy checkpoint: fast-forward scheduler
                    for _ in range(start_step):
                        scheduler.step()
                # Restore the original run directory so outputs stay together
                saved_run_dir = ckpt.get('run_dir')
                if saved_run_dir and Path(saved_run_dir).exists():
                    run_dir = Path(saved_run_dir)
                else:
                    run_dir = Path(resume_from).parent
                print(f"Resumed from {ckpt_files[-1]} — epoch {start_epoch}, step {start_step}")
                print(f"  Run directory: {run_dir}")

        if run_dir is None:
            # Fresh run: create a timestamped subdirectory
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            run_dir = self.output_dir / f'run_{timestamp}'
            run_dir.mkdir(parents=True, exist_ok=True)
            print(f"New run directory: {run_dir}")

        print(f"\n{'='*70}")
        print(f"EnsembleTrainer — {self.ensemble.num_models()} models, "
              f"{self.dataset_pool.num_datasets()} datasets, "
              f"k={self.k_neurons} neurons, "
              f"{self.patches_per_image} patches/image × {self.images_per_batch} images/batch")
        print(f"  {total_images:,} dataset images → {steps_per_epoch:,} steps/epoch")
        print(f"{'='*70}\n")

        global_step = start_step
        samples_dir = run_dir / 'samples'
        samples_dir.mkdir(parents=True, exist_ok=True)

        # Precompute per-neuron activation statistics for all models (once per run)
        self._profile_all_models(n_images=1000)

        # Save debug visualizations (clean vs attacked image pairs)
        debug_dir = run_dir / 'debug_visualizations'
        self._save_debug_visualizations(debug_dir)

        # Raise KeyboardInterrupt immediately on Ctrl+C so we can save and exit
        signal.signal(signal.SIGINT, signal.default_int_handler)

        # Persistent background DataLoader: workers pre-load images while the GPU
        # processes the previous batch, eliminating synchronous disk I/O from the
        # hot path. pin_memory=True enables fast DMA transfers on CUDA.
        prefetch_loader = torch.utils.data.DataLoader(
            _TrainingImageDataset(self.dataset_pool),
            batch_size=self.images_per_batch,
            num_workers=self.num_prefetch_workers,
            prefetch_factor=2,
            persistent_workers=True,
            pin_memory=(self._device.type == 'cuda'),
        )
        prefetch_iter = iter(prefetch_loader)
        print(f"Prefetch DataLoader: {self.num_prefetch_workers} workers, "
              f"batch_size={self.images_per_batch}, pin_memory={self._device.type == 'cuda'}\n")

        # --- Warmup phase ---
        if self.warmup_steps > 0 and self.warmup_model and global_step == 0:
            warmup_entries = [e for e in self.ensemble._entries if e.name == self.warmup_model]
            if not warmup_entries:
                print(f"WARNING: warmup_model '{self.warmup_model}' not found in ensemble — skipping warmup")
            else:
                print(f"\n{'='*60}")
                print(f"Warmup phase: {self.warmup_steps} steps on '{self.warmup_model}'")
                print(f"{'='*60}")
                self.generator.train(); self.task_encoder.train()
                all_params = list(self.generator.parameters()) + list(self.task_encoder.parameters())
                try:
                    with tqdm(total=self.warmup_steps, desc="Warmup") as wpbar:
                        for _ in range(self.warmup_steps):
                            optimizer.zero_grad()
                            prefetched = next(prefetch_iter)
                            info = self._train_step(optimizer, prefetched=prefetched, entries_override=warmup_entries, use_stream_gating=True)
                            torch.nn.utils.clip_grad_norm_(all_params, 1.0)
                            optimizer.step()
                            scheduler.step()
                            global_step += 1
                            if global_step % 10 == 0:
                                self._save_samples(global_step, samples_dir)
                            wpbar.set_postfix({
                                'loss': f"{info['loss']:.4f}",
                                'div':  f"{info['diversity']:.2f}",
                                'qual': f"{info['final_quality']:.4f}",
                                'tv':   f"{info['tv']:.4f}",
                                'ssim': f"{info['spectrum']:.4f}",
                                'lr':   f"{optimizer.param_groups[1]['lr']:.2e}",
                            })
                            wpbar.update(1)
                except KeyboardInterrupt:
                    print(f"\n  Interrupted during warmup at step {global_step}; saving checkpoint.")
                    self._save_checkpoint(
                        0, global_step, run_dir, optimizer, scheduler,
                        subdir=f'checkpoint_step_{global_step:07d}'
                    )
                    return
                print(f"Warmup complete. global_step={global_step}\n")

        for epoch in range(start_epoch, self.max_epochs + 1):
            self.generator.train()
            self.task_encoder.train()

            epoch_losses: Dict[str, float] = {
                'loss': 0.0, 'diversity': 0.0, 'quality': 0.0, 'final_quality': 0.0,
                'tv': 0.0, 'spectrum': 0.0,
            }
            epoch_steps = 0

            steps_this_epoch = steps_per_epoch
            if max_steps is not None:
                steps_this_epoch = min(steps_per_epoch, max_steps - global_step)

            try:
                with tqdm(total=steps_this_epoch, desc=f"Epoch {epoch}", leave=False) as pbar:
                    for step in range(steps_this_epoch):
                        optimizer.zero_grad()
                        prefetched = next(prefetch_iter)
                        info = self._train_step(optimizer, prefetched=prefetched)

                        torch.nn.utils.clip_grad_norm_(
                            list(self.generator.parameters())
                            + list(self.task_encoder.parameters()), 1.0
                        )
                        optimizer.step()
                        scheduler.step()

                        for k in epoch_losses:
                            epoch_losses[k] += info.get(k, 0.0)
                        epoch_steps += 1

                        current_lr = optimizer.param_groups[1]['lr']  # Task encoder & generator LR (fastest)
                        pbar.set_postfix({
                            'loss': f"{info['loss']:.4f}",
                            'div':  f"{info['diversity']:.2f}",
                            'qual': f"{info['final_quality']:.4f}",
                            'tv':   f"{info['tv']:.4f}",
                            'ssim': f"{info['spectrum']:.4f}",
                            'lr':   f"{current_lr:.2e}",
                        })
                        pbar.update(1)

                        global_step += 1

                        # Save sample patches every 10 optimizer steps
                        if global_step % 10 == 0:
                            self._save_samples(global_step, samples_dir)

                        if max_steps is not None and global_step >= max_steps:
                            print(f"\n  max_steps={max_steps} reached; saving checkpoint.")
                            self._save_checkpoint(
                                epoch, global_step, run_dir, optimizer, scheduler,
                                subdir=f'checkpoint_step_{global_step:07d}'
                            )
                            return

            except KeyboardInterrupt:
                print(f"\n  Interrupted at step {global_step}; saving checkpoint.")
                self._save_checkpoint(
                    epoch, global_step, run_dir, optimizer, scheduler,
                    subdir=f'checkpoint_step_{global_step:07d}'
                )
                return

            n = max(epoch_steps, 1)
            print(
                f"[Epoch {epoch}/{self.max_epochs}] "
                f"loss={epoch_losses['loss']/n:.4f}  "
                f"div={epoch_losses['diversity']/n:.3f}  "
                f"qual={epoch_losses['quality']/n:.3f}  "
                f"tv={epoch_losses['tv']/n:.4f}  "
                f"spec={epoch_losses['spectrum']/n:.4f}"
            )

            if epoch % self.save_every_epochs == 0 or epoch == self.max_epochs:
                self._save_checkpoint(epoch, global_step, run_dir, optimizer, scheduler,
                                      subdir=f'checkpoint_epoch_{epoch:04d}')

        # Final checkpoint
        self._save_checkpoint(self.max_epochs, global_step, run_dir, optimizer, scheduler,
                              subdir='ensemble_final')
        print(f"\n✓ Ensemble training complete. Output: {run_dir}")
