#!/usr/bin/env python3
"""
train_ensemble.py — CLI entry point for ensemble adversarial training.

Usage::

    # Smoke test (CPU, synthetic data, 2 steps)
    python framework/scripts/train_ensemble.py framework/configs/ensemble_example.yaml \
        --epochs 1 --max-steps 2 --device cpu

    # Full run with real models and data (override YAML device)
    python framework/scripts/train_ensemble.py my_ensemble.yaml \
        --device cuda --output-dir runs/ensemble_001

The script:
    1. Parses the YAML config via load_ensemble_config().
    2. Builds synthetic or real models from the 'models' section.
    3. Creates synthetic image directories if datasets.synthetic=true.
    4. Instantiates FoundationPatchGenerator (multi-TAESD architecture).
    5. Runs EnsembleTrainer.train().
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

# Ensure project root is on PYTHONPATH
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from framework.config_loader import load_ensemble_config
from framework.ensemble import EnsembleModelPool, EnsembleTrainer
from framework.neuron_sampler import NeuronSampler
from framework.models import build_model as _registry_build, REGISTRY


# ---------------------------------------------------------------------------
# Tiny synthetic model (no torchvision dependency)
# ---------------------------------------------------------------------------

class _SyntheticCNN(nn.Module):
    """Small CNN for smoke-testing without real pretrained weights."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool  = nn.AdaptiveAvgPool2d((4, 4))
        self.fc    = nn.Linear(32 * 4 * 4, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.flatten(1)
        return self.fc(x)


# ---------------------------------------------------------------------------
# Synthetic image creation
# ---------------------------------------------------------------------------

def _create_synthetic_images(root: str, n: int = 20, size: int = 128) -> None:
    """Write n random PNG images to root/ for dataset pool discovery."""
    import numpy as np
    from PIL import Image

    root_path = Path(root)
    root_path.mkdir(parents=True, exist_ok=True)

    # Skip if already populated
    existing = list(root_path.glob('*.png'))
    if len(existing) >= n:
        return

    for i in range(n):
        arr = (torch.rand(size, size, 3) * 255).byte().numpy()
        img = Image.fromarray(arr, mode='RGB')
        img.save(root_path / f'synthetic_{i:04d}.png')

    print(f"  Created {n} synthetic images in {root}")


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def _build_model(model_cfg: Dict[str, Any]) -> nn.Module:
    """
    Instantiate a model from a config dict.

    Dispatches to ``framework.models.REGISTRY`` for all real pretrained models.
    Falls back to the tiny ``_SyntheticCNN`` for ``architecture: simple_cnn``.
    """
    arch = model_cfg.get('architecture', 'simple_cnn').lower()

    if arch == 'simple_cnn':
        return _SyntheticCNN(num_classes=model_cfg.get('num_classes', 10))

    return _registry_build(arch)


# ---------------------------------------------------------------------------
# Strategy factory
# ---------------------------------------------------------------------------

def _build_strategy(model_cfg: Dict[str, Any]):
    from framework.base.attack_strategy import BorderStrategy, StickerStrategy, PerturbationStrategy

    strat = model_cfg.get('strategy', 'border').lower()
    if strat == 'border':
        return BorderStrategy(
            center_ratio=model_cfg.get('center_ratio', 0.6),
            neutral_color=model_cfg.get('neutral_color', 0.5),
        )
    elif strat == 'sticker':
        return StickerStrategy(
            sticker_h=model_cfg.get('sticker_h', None),
            sticker_w=model_cfg.get('sticker_w', None),
            neutral_color=model_cfg.get('neutral_color', 0.5),
        )
    elif strat in ('perturbation', 'additive'):
        return PerturbationStrategy(
            budget=model_cfg.get('budget', 0.05),
            norm=model_cfg.get('norm', 'linf'),
        )
    else:
        raise ValueError(f"Unknown strategy '{strat}'.")


# ---------------------------------------------------------------------------
# Generator factory (handles missing SDXL VAE gracefully)
# ---------------------------------------------------------------------------

def _build_generator(gen_cfg: Dict[str, Any]) -> Any:
    """Build a FoundationPatchGenerator (multi-TAESD architecture) from config dict."""
    from framework.generator import FoundationPatchGenerator

    return FoundationPatchGenerator(
        latent_dim=gen_cfg.get('latent_dim', 16),
        patch_height=gen_cfg.get('patch_height', 512),
        patch_width=gen_cfg.get('patch_width', 512),
        num_taesd=gen_cfg.get('num_taesd', 6),
        transformer_d_model=gen_cfg.get('transformer_d_model', 256),
        transformer_nhead=gen_cfg.get('transformer_nhead', 4),
        transformer_d_ff=gen_cfg.get('transformer_d_ff', 1024),
        transformer_enc_layers=gen_cfg.get('transformer_enc_layers', 2),
        transformer_dec_layers=gen_cfg.get('transformer_dec_layers', 2),
    )


# ---------------------------------------------------------------------------
# Preprocessing factory
# ---------------------------------------------------------------------------

def _build_preprocess(model_cfg: Dict[str, Any], input_shape) -> Any:
    """Return a preprocessing function for the given model config.

    For architectures in the registry, delegates to the wrapper's
    ``get_preprocess_fn()`` class method (which applies the correct
    normalization statistics and resize).  Falls back to a simple
    resize+clamp for ``simple_cnn`` and unknown architectures.
    """
    arch = model_cfg.get('architecture', '').lower()

    if arch in REGISTRY:
        cls = REGISTRY[arch]
        if hasattr(cls, 'get_preprocess_fn'):
            return cls.get_preprocess_fn()

    # Fallback for simple_cnn and unknown architectures
    H, W = input_shape

    def preprocess(x: torch.Tensor) -> torch.Tensor:
        if x.shape[2] != H or x.shape[3] != W:
            x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=False)
        return x.clamp(0.0, 1.0)

    return preprocess


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description='Ensemble adversarial patch training')
    p.add_argument('config', type=str, help='Path to ensemble YAML config')
    p.add_argument('--epochs',     type=int, default=None,
                   help='Override max_epochs from config')
    p.add_argument('--max-steps',  type=int, default=None,
                   help='Stop after this many total gradient steps (smoke test)')
    p.add_argument('--device',     type=str, default=None,
                   help='Override compute_device from config (cpu, cuda, mps)')
    p.add_argument('--output-dir', type=str, default=None,
                   help='Override output_dir from config')
    p.add_argument('--resume',     type=str, default=None,
                   help='Path to checkpoint directory to resume from')
    return p.parse_args()


def main():
    args = parse_args()

    print(f"\nLoading ensemble config: {args.config}")
    ensemble_cfg = load_ensemble_config(args.config)
    raw = ensemble_cfg.raw

    # Apply CLI overrides
    device_str = args.device or raw.get('compute_device', 'cpu')
    compute_device = torch.device(device_str)
    ensemble_cfg.ensemble_pool.compute_device = compute_device

    trainer_cfg = dict(ensemble_cfg.trainer_cfg)
    if args.epochs is not None:
        trainer_cfg['max_epochs'] = args.epochs
    if args.output_dir is not None:
        trainer_cfg['output_dir'] = args.output_dir

    # ------------------------------------------------------------------
    # Set up synthetic datasets
    # ------------------------------------------------------------------
    for ds in raw.get('datasets', []):
        if ds.get('synthetic', False):
            root = ds.get('root', '/tmp/synthetic_images')
            n = ds.get('max_samples', 20)
            _create_synthetic_images(root, n=n, size=128)

    # ------------------------------------------------------------------
    # Register models into the ensemble pool
    # ------------------------------------------------------------------
    ensemble_pool = ensemble_cfg.ensemble_pool
    for mc in ensemble_cfg.models_cfg:
        model = _build_model(mc)
        strategy = _build_strategy(mc)
        arch = mc.get('architecture', 'simple_cnn').lower()
        if 'input_shape' in mc:
            input_shape = tuple(mc['input_shape'])
        elif arch in REGISTRY and hasattr(REGISTRY[arch], 'input_size'):
            input_shape = REGISTRY[arch].input_size()
        else:
            input_shape = (64, 64)
        preprocess_fn = _build_preprocess(mc, input_shape)

        ensemble_pool.register(
            name=mc['name'],
            model=model,
            domain_type=mc.get('domain_type', 'generic'),
            strategy=strategy,
            strategy_id=int(mc.get('strategy_id', 0)),
            input_shape=input_shape,
            preprocess_fn=preprocess_fn,
        )
        print(f"  Registered model: {mc['name']}  "
              f"(strategy={mc.get('strategy','border')}, id={mc.get('strategy_id',0)})")

    if ensemble_pool.num_models() == 0:
        print("WARNING: no models registered — check 'models' section in YAML.")

    # ------------------------------------------------------------------
    # Build generator
    # ------------------------------------------------------------------
    generator = _build_generator(ensemble_cfg.generator_cfg)

    # ------------------------------------------------------------------
    # Build NeuronSampler
    # ------------------------------------------------------------------
    sampler = NeuronSampler(device=compute_device)

    # ------------------------------------------------------------------
    # Build EnsembleTrainer
    # ------------------------------------------------------------------
    trainer = EnsembleTrainer(
        ensemble=ensemble_pool,
        dataset_pool=ensemble_cfg.dataset_pool,
        task_encoder=ensemble_cfg.task_encoder,
        generator=generator,
        neuron_sampler=sampler,
        k_neurons=int(trainer_cfg.get('k_neurons', 100)),
        patches_per_batch=int(trainer_cfg.get('patches_per_batch', 2)),
        diversity_weight=float(trainer_cfg.get('diversity_weight', 1.0)),
        quality_weight=float(trainer_cfg.get('quality_weight', 1.0)),
        tv_weight=float(trainer_cfg.get('tv_weight', 2.5)),
        spectrum_weight=float(trainer_cfg.get('spectrum_weight', 1.0)),
        learning_rate=float(trainer_cfg.get('learning_rate', 1e-4)),
        vae_lr_ratio=float(trainer_cfg.get('taesd_lr_ratio', trainer_cfg.get('vae_lr_ratio', 0.1))),
        lr_min=float(trainer_cfg.get('lr_min', 1e-6)),
        max_epochs=int(trainer_cfg.get('max_epochs', 10)),
        output_dir=str(trainer_cfg.get('output_dir', 'ensemble_output')),
        save_every_epochs=int(trainer_cfg.get('save_every_epochs', 5)),
        device=compute_device,
    )

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    trainer.train(
        steps_per_epoch=int(trainer_cfg.get('steps_per_epoch', 5)),
        resume_from=args.resume,
        max_steps=args.max_steps,
    )


if __name__ == '__main__':
    main()
