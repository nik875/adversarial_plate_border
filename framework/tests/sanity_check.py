#!/usr/bin/env python3
"""
Quick sanity check: verify framework imports and basic operations work.

Usage:
    python framework/tests/sanity_check.py
"""
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def test_imports():
    """Test that all framework modules import successfully."""
    print("Testing imports...")

    try:
        from framework.base.domain import DomainAdapter, LayerConfig
        print("  ✓ framework.base.domain")
    except Exception as e:
        print(f"  ✗ framework.base.domain: {e}")
        return False

    try:
        from framework.base.attack_strategy import (
            AttackStrategy, BorderStrategy, StickerStrategy, PerturbationStrategy
        )
        print("  ✓ framework.base.attack_strategy")
    except Exception as e:
        print(f"  ✗ framework.base.attack_strategy: {e}")
        return False

    try:
        from framework.base.metric import (
            EvalMetric, TopKAccuracyDrop, EditDistanceMetric, DetectionDisruptionMetric
        )
        print("  ✓ framework.base.metric")
    except Exception as e:
        print(f"  ✗ framework.base.metric: {e}")
        return False

    try:
        from framework.generator import FoundationPatchGenerator
        print("  ✓ framework.generator")
    except Exception as e:
        print(f"  ✗ framework.generator: {e}")
        return False

    try:
        from framework.losses import (
            total_variation_loss, compute_spectrum_loss, compute_activation_diversity
        )
        print("  ✓ framework.losses")
    except Exception as e:
        print(f"  ✗ framework.losses: {e}")
        return False

    try:
        from framework.generator_loader import load_generator, generate_patch_from_z
        print("  ✓ framework.generator_loader")
    except Exception as e:
        print(f"  ✗ framework.generator_loader: {e}")
        return False

    try:
        from framework.config_loader import load_domain_config
        print("  ✓ framework.config_loader")
    except Exception as e:
        print(f"  ✗ framework.config_loader: {e}")
        return False

    try:
        from framework.trainer import GenericPatchTrainer
        print("  ✓ framework.trainer")
    except Exception as e:
        print(f"  ✗ framework.trainer: {e}")
        return False

    return True


def test_basic_operations():
    """Test that basic operations work."""
    import torch
    from framework.base.attack_strategy import BorderStrategy, StickerStrategy, PerturbationStrategy
    from framework.base.metric import TopKAccuracyDrop
    from framework.generator import FoundationPatchGenerator
    from framework.losses import total_variation_loss

    print("\nTesting basic operations...")

    # Test BorderStrategy
    try:
        strategy = BorderStrategy(center_ratio=0.6)
        image = torch.rand(2, 3, 64, 64)
        patch = torch.rand(3, 64, 64)
        composited, mask = strategy.apply(image, patch)
        assert composited.shape == image.shape
        assert mask.shape == (1, 1, 64, 64)
        print("  ✓ BorderStrategy.apply()")
    except Exception as e:
        print(f"  ✗ BorderStrategy.apply(): {e}")
        return False

    # Test StickerStrategy
    try:
        strategy = StickerStrategy(sticker_h=32, sticker_w=32)
        image = torch.rand(2, 3, 64, 64)
        patch = torch.rand(3, 64, 64)
        composited, mask = strategy.apply(image, patch, bbox=(10, 10, 42, 42))
        assert composited.shape == image.shape
        print("  ✓ StickerStrategy.apply() with bbox")
    except Exception as e:
        print(f"  ✗ StickerStrategy.apply(): {e}")
        return False

    # Test sample_kwargs
    try:
        strategy = StickerStrategy(sticker_h=32, sticker_w=32)
        image = torch.rand(2, 3, 64, 64)
        kwargs = strategy.sample_kwargs(image, 64, 64)
        assert 'bbox' in kwargs
        x0, y0, x1, y1 = kwargs['bbox']
        assert x1 - x0 == 32 and y1 - y0 == 32
        print("  ✓ StickerStrategy.sample_kwargs()")
    except Exception as e:
        print(f"  ✗ StickerStrategy.sample_kwargs(): {e}")
        return False

    # Test PerturbationStrategy
    try:
        strategy = PerturbationStrategy(budget=0.1, norm='linf')
        image = torch.rand(2, 3, 64, 64)
        patch = torch.rand(3, 64, 64)
        composited, mask = strategy.apply(image, patch)
        assert composited.shape == image.shape
        assert (composited >= 0).all() and (composited <= 1).all()
        print("  ✓ PerturbationStrategy.apply()")
    except Exception as e:
        print(f"  ✗ PerturbationStrategy.apply(): {e}")
        return False

    # Test Generator
    try:
        gen = FoundationPatchGenerator(
            latent_dim=8,
            patch_height=32,
            patch_width=32,
            use_vae_lora=False,
        )
        z = torch.randn(2, 8)
        patches = gen.forward_clean(z)
        assert patches.shape == (2, 3, 32, 32)
        assert (patches >= 0).all() and (patches <= 1).all()
        print("  ✓ FoundationPatchGenerator.forward_clean()")
    except Exception as e:
        print(f"  ✗ FoundationPatchGenerator.forward_clean(): {e}")
        return False

    # Test TV Loss
    try:
        patches = torch.rand(2, 3, 32, 32)
        mask = torch.ones(1, 1, 32, 32)
        loss = total_variation_loss(patches, mask)
        assert loss.item() >= 0
        print("  ✓ total_variation_loss()")
    except Exception as e:
        print(f"  ✗ total_variation_loss(): {e}")
        return False

    return True


def test_config_loading():
    """Test loading classification config."""
    print("\nTesting config loading...")

    try:
        from framework.config_loader import load_domain_config

        config_path = _PROJECT_ROOT / 'framework' / 'configs' / 'classification_resnet50.yaml'
        if not config_path.exists():
            print(f"  ⊘ Config not found at {config_path} (skipping)")
            return True

        cfg = load_domain_config(str(config_path))
        assert cfg.domain_adapter is not None
        assert cfg.strategy is not None
        assert cfg.metric is not None
        assert cfg.target_model is not None
        print("  ✓ load_domain_config() + all components")
        return True
    except Exception as e:
        print(f"  ✗ load_domain_config(): {e}")
        return False


def main():
    print("=" * 80)
    print("FRAMEWORK SANITY CHECK")
    print("=" * 80)
    print()

    all_pass = True

    if not test_imports():
        all_pass = False

    if not test_basic_operations():
        all_pass = False

    if not test_config_loading():
        all_pass = False

    print()
    print("=" * 80)
    if all_pass:
        print("✓ ALL SANITY CHECKS PASSED")
        print("=" * 80)
        return 0
    else:
        print("✗ SOME SANITY CHECKS FAILED")
        print("=" * 80)
        return 1


if __name__ == '__main__':
    sys.exit(main())
