#!/usr/bin/env python3
"""
train_domain.py — CLI entry point for white-box progressive layer training on any domain.

Usage:
    python framework/scripts/train_domain.py <config.yaml> [options]

Example:
    python framework/scripts/train_domain.py framework/configs/classification_resnet50.yaml \
        --epochs 10 --output-dir runs/resnet50_border
"""
import argparse
import sys
from pathlib import Path

# Ensure project root is on sys.path when running as a script
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def main():
    parser = argparse.ArgumentParser(
        description='Domain-agnostic progressive layer patch training.'
    )
    parser.add_argument('config', help='Path to YAML domain config file')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Override max_epochs from config')
    parser.add_argument('--output-dir', default=None,
                        help='Override output directory from config')
    parser.add_argument('--resume', default=None,
                        help='Path to checkpoint directory to resume from')
    parser.add_argument('--device', default=None,
                        help='Device (cpu, cuda, mps). Overrides config/domain default.')
    args = parser.parse_args()

    from framework.config_loader import load_domain_config

    print(f"Loading config: {args.config}")
    cfg = load_domain_config(args.config)
    raw = cfg.raw

    # Extract trainer kwargs from YAML
    trainer_cfg = raw.get('trainer', {})
    patch_cfg = raw.get('patch', {})

    output_dir = args.output_dir or trainer_cfg.get('output_dir', 'framework_output')
    max_epochs = args.epochs or trainer_cfg.get('max_epochs', 100)

    from framework.trainer import GenericPatchTrainer

    trainer = GenericPatchTrainer(
        domain=cfg.domain_adapter,
        strategy=cfg.strategy,
        basis_dim=trainer_cfg.get('basis_dim', 16),
        patches_per_image=trainer_cfg.get('patches_per_image', 4),
        images_per_batch=trainer_cfg.get('images_per_batch', 1),
        diversity_weight=trainer_cfg.get('diversity_weight', 1.0),
        quality_weight=trainer_cfg.get('quality_weight', 1.0),
        performance_weight=trainer_cfg.get('performance_weight', 1.0),
        tv_weight=trainer_cfg.get('tv_weight', 2.5),
        spectrum_weight=trainer_cfg.get('spectrum_weight', 1.0),
        use_vae_lora=trainer_cfg.get('use_vae_lora', True),
        lora_rank=trainer_cfg.get('lora_rank', 8),
        lora_alpha=trainer_cfg.get('lora_alpha', 16),
        bottleneck_dim=trainer_cfg.get('bottleneck_dim', 256),
        use_omniglot=trainer_cfg.get('use_omniglot', False),
        output_dir=output_dir,
        save_examples_every=trainer_cfg.get('save_examples_every', None),
        learning_rate=trainer_cfg.get('learning_rate', 1e-4),
        vae_lr_ratio=trainer_cfg.get('vae_lr_ratio', 0.1),
        lr_min=trainer_cfg.get('lr_min', 1e-6),
        max_epochs=max_epochs,
        val_split=trainer_cfg.get('val_split', 0.2),
        num_workers=trainer_cfg.get('num_workers', 0),
        device=args.device,
    )

    trainer.train(resume_from=args.resume)


if __name__ == '__main__':
    main()
