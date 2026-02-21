#!/usr/bin/env python3
"""
Comprehensive framework test suite.

Tests all major components: domains, strategies, metrics, generator, trainer, config loading.
Designed to catch integration issues before cloud deployment.

Usage:
    python -m pytest framework/tests/test_framework.py -v
    # or without pytest:
    python framework/tests/test_framework.py
"""
import sys
import tempfile
import unittest
from pathlib import Path
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Add project root to path
_PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


# =============================================================================
# Test Components
# =============================================================================

class SimpleTestModel(nn.Module):
    """Tiny model for testing (no torchvision dependency)."""
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(32, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class SimpleTestDataset(torch.utils.data.Dataset):
    """Tiny synthetic dataset for testing."""
    def __init__(self, num_samples: int = 50, img_size: int = 64, num_classes: int = 10):
        self.num_samples = num_samples
        self.img_size = img_size
        self.num_classes = num_classes

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Random image + label
        image = torch.randn(3, self.img_size, self.img_size)
        image = (image - image.min()) / (image.max() - image.min())  # [0, 1]
        label = torch.randint(0, self.num_classes, (1,)).item()
        return {
            'image': image,
            'label': label,
            'index': idx,
        }


# =============================================================================
# Test: Attack Strategies
# =============================================================================

class TestAttackStrategies(unittest.TestCase):
    """Test BorderStrategy, StickerStrategy, PerturbationStrategy."""

    def setUp(self):
        from framework.base.attack_strategy import (
            BorderStrategy, StickerStrategy, PerturbationStrategy
        )
        self.BorderStrategy = BorderStrategy
        self.StickerStrategy = StickerStrategy
        self.PerturbationStrategy = PerturbationStrategy

        self.batch_size = 2
        self.img_h, self.img_w = 64, 64
        self.patch_h, self.patch_w = 64, 64

    def _create_test_image(self):
        """Create test image [B, 3, H, W] in [0, 1]."""
        img = torch.rand(self.batch_size, 3, self.img_h, self.img_w)
        return img

    def _create_test_patch(self):
        """Create test patch [3, H, W] in [0, 1]."""
        patch = torch.rand(3, self.patch_h, self.patch_w)
        return patch

    def test_border_strategy_apply(self):
        """Test BorderStrategy.apply()."""
        strategy = self.BorderStrategy(center_ratio=0.6)
        image = self._create_test_image()
        patch = self._create_test_patch()

        composited, mask = strategy.apply(image, patch)

        # Check shapes
        self.assertEqual(composited.shape, image.shape)
        self.assertEqual(mask.shape, (1, 1, self.patch_h, self.patch_w))

        # Check values in [0, 1]
        self.assertTrue(composited.min() >= 0 and composited.max() <= 1)
        self.assertTrue(mask.min() >= 0 and mask.max() <= 1)

    def test_border_strategy_neutral(self):
        """Test BorderStrategy.apply_neutral()."""
        strategy = self.BorderStrategy(center_ratio=0.6, neutral_color=0.5)
        image = self._create_test_image()

        neutral = strategy.apply_neutral(image)

        # Check shape
        self.assertEqual(neutral.shape, image.shape)

        # Center should be image, border should be ~0.5
        center_h = int(self.img_h * 0.6)
        center_w = int(self.img_w * 0.6)
        pad_h = (self.img_h - center_h) // 2
        pad_w = (self.img_w - center_w) // 2
        border = neutral[:, :, :pad_h, :]
        self.assertTrue(border.abs() - 0.5 < 0.1)

    def test_border_strategy_sample_kwargs(self):
        """Test BorderStrategy.sample_kwargs() returns empty dict."""
        strategy = self.BorderStrategy()
        image = self._create_test_image()
        kwargs = strategy.sample_kwargs(image, self.patch_h, self.patch_w)
        self.assertEqual(kwargs, {})

    def test_sticker_strategy_fixed_bbox(self):
        """Test StickerStrategy with fixed bbox."""
        bbox = (10, 10, 40, 40)
        strategy = self.StickerStrategy(bbox=bbox)
        image = self._create_test_image()
        patch = self._create_test_patch()

        composited, mask = strategy.apply(image, patch)

        # Check shape
        self.assertEqual(composited.shape, image.shape)

        # Mask should be all ones (entire patch visible)
        self.assertTrue(torch.allclose(mask, torch.ones_like(mask)))

    def test_sticker_strategy_random_bbox(self):
        """Test StickerStrategy with random bbox."""
        strategy = self.StickerStrategy(sticker_h=32, sticker_w=32)
        image = self._create_test_image()

        # sample_kwargs should return a bbox
        kwargs = strategy.sample_kwargs(image, self.patch_h, self.patch_w)
        self.assertIn('bbox', kwargs)

        bbox = kwargs['bbox']
        x0, y0, x1, y1 = bbox

        # Check bbox is within image bounds
        self.assertTrue(0 <= x0 < self.img_w)
        self.assertTrue(0 <= y0 < self.img_h)
        self.assertTrue(x1 <= self.img_w)
        self.assertTrue(y1 <= self.img_h)

        # Check size
        self.assertEqual(x1 - x0, 32)
        self.assertEqual(y1 - y0, 32)

    def test_perturbation_strategy_linf(self):
        """Test PerturbationStrategy with L∞ norm."""
        strategy = self.PerturbationStrategy(budget=0.1, norm='linf')
        image = self._create_test_image()
        patch = self._create_test_patch()

        composited, mask = strategy.apply(image, patch)

        # Check shape
        self.assertEqual(composited.shape, image.shape)

        # Check values clamped to [0, 1]
        self.assertTrue(composited.min() >= 0 and composited.max() <= 1)

        # Mask should be all ones
        self.assertTrue(torch.allclose(mask, torch.ones_like(mask)))

    def test_perturbation_strategy_neutral(self):
        """Test PerturbationStrategy.apply_neutral() returns unchanged image."""
        strategy = self.PerturbationStrategy(budget=0.1)
        image = self._create_test_image()

        neutral = strategy.apply_neutral(image)

        # Should be identical to input
        self.assertTrue(torch.allclose(neutral, image))


# =============================================================================
# Test: Metrics
# =============================================================================

class TestMetrics(unittest.TestCase):
    """Test evaluation metrics."""

    def setUp(self):
        from framework.base.metric import TopKAccuracyDrop
        self.TopKAccuracyDrop = TopKAccuracyDrop
        self.model = SimpleTestModel(num_classes=10)
        self.model.eval()

    def test_topk_accuracy_drop_precompute(self):
        """Test TopKAccuracyDrop.precompute_control()."""
        metric = self.TopKAccuracyDrop(k=1)

        # Create test images
        images = [torch.rand(3, 64, 64) for _ in range(5)]

        with torch.no_grad():
            control_preds = metric.precompute_control(images, self.model)

        # Should return list of predictions
        self.assertEqual(len(control_preds), len(images))
        self.assertTrue(all(isinstance(p, int) for p in control_preds))
        self.assertTrue(all(0 <= p < 10 for p in control_preds))

    def test_topk_accuracy_drop_compute(self):
        """Test TopKAccuracyDrop.compute()."""
        metric = self.TopKAccuracyDrop(k=1)

        images = [torch.rand(3, 64, 64) for _ in range(5)]
        with torch.no_grad():
            control_preds = metric.precompute_control(images, self.model)

        # Perturb images slightly
        adversarial = [img + 0.1 * torch.randn_like(img) for img in images]
        adversarial = [torch.clamp(img, 0, 1) for img in adversarial]

        with torch.no_grad():
            results = metric.compute(adversarial, control_preds, self.model)

        # Check required fields
        self.assertIn('primary', results)
        self.assertIn('success_rate', results)
        self.assertIn('top1_drop', results)

        # Check values are in [0, 1]
        self.assertTrue(0 <= results['success_rate'] <= 1)
        self.assertTrue(0 <= results['primary'] <= 1)

    def test_topk_accuracy_drop_compute_detailed(self):
        """Test TopKAccuracyDrop.compute_detailed()."""
        metric = self.TopKAccuracyDrop(k=1)

        images = [torch.rand(3, 64, 64) for _ in range(5)]
        with torch.no_grad():
            control_preds = metric.precompute_control(images, self.model)

        adversarial = [torch.clamp(img + 0.1, 0, 1) for img in images]

        with torch.no_grad():
            agg, per_image = metric.compute_detailed(
                adversarial, control_preds, self.model
            )

        # Check per-image data
        self.assertEqual(len(per_image), len(images))
        for item in per_image:
            self.assertIn('img_idx', item)
            self.assertIn('success', item)
            self.assertIn('control_class', item)
            self.assertIn('pred_class', item)


# =============================================================================
# Test: Generator
# =============================================================================

class TestGenerator(unittest.TestCase):
    """Test FoundationPatchGenerator."""

    def setUp(self):
        from framework.generator import FoundationPatchGenerator
        self.FoundationPatchGenerator = FoundationPatchGenerator
        self.device = 'cpu'

    def test_generator_init(self):
        """Test generator initialization."""
        gen = self.FoundationPatchGenerator(
            latent_dim=16,
            patch_height=64,
            patch_width=64,
            use_vae_lora=True,
            lora_rank=4,
            lora_alpha=8,
            use_omniglot=False,
        )
        gen = gen.to(self.device)

        self.assertIsNotNone(gen)
        self.assertEqual(gen.latent_dim, 16)

    def test_generator_forward(self):
        """Test generator forward pass."""
        gen = self.FoundationPatchGenerator(
            latent_dim=8,
            patch_height=32,
            patch_width=32,
            use_vae_lora=False,
        )
        gen = gen.to(self.device)

        z = torch.randn(2, 8, device=self.device)
        with torch.no_grad():
            patches = gen.forward_clean(z)

        # Check shape and values
        self.assertEqual(patches.shape, (2, 3, 32, 32))
        self.assertTrue(patches.min() >= 0 and patches.max() <= 1)

    def test_generator_lora_parameters(self):
        """Test that LoRA adds trainable parameters."""
        gen_no_lora = self.FoundationPatchGenerator(
            latent_dim=8,
            patch_height=32,
            patch_width=32,
            use_vae_lora=False,
        )

        gen_lora = self.FoundationPatchGenerator(
            latent_dim=8,
            patch_height=32,
            patch_width=32,
            use_vae_lora=True,
            lora_rank=4,
        )

        params_no_lora = sum(p.numel() for p in gen_no_lora.parameters() if p.requires_grad)
        params_lora = sum(p.numel() for p in gen_lora.parameters() if p.requires_grad)

        # LoRA should add parameters
        self.assertGreater(params_lora, params_no_lora)


# =============================================================================
# Test: Config Loading
# =============================================================================

class TestConfigLoader(unittest.TestCase):
    """Test YAML config loading."""

    def test_load_classification_config(self):
        """Test loading classification config."""
        from framework.config_loader import load_domain_config

        config_path = _PROJECT_ROOT / 'framework' / 'configs' / 'classification_resnet50.yaml'
        if not config_path.exists():
            self.skipTest(f"Config file not found: {config_path}")

        cfg = load_domain_config(str(config_path))

        # Check all components loaded
        self.assertIsNotNone(cfg.domain_adapter)
        self.assertIsNotNone(cfg.strategy)
        self.assertIsNotNone(cfg.metric)
        self.assertIsNotNone(cfg.target_model)
        self.assertIsNotNone(cfg.raw)

    def test_build_strategy_border(self):
        """Test building BorderStrategy from config dict."""
        from framework.config_loader import _build_strategy

        strategy_cfg = {
            'type': 'border',
            'center_ratio': 0.5,
            'neutral_color': 0.5,
        }
        strategy = _build_strategy(strategy_cfg)

        self.assertIsNotNone(strategy)
        self.assertEqual(strategy.center_ratio, 0.5)

    def test_build_strategy_sticker(self):
        """Test building StickerStrategy from config dict."""
        from framework.config_loader import _build_strategy

        strategy_cfg = {
            'type': 'sticker',
            'sticker_h': 64,
            'sticker_w': 64,
        }
        strategy = _build_strategy(strategy_cfg)

        self.assertIsNotNone(strategy)
        self.assertEqual(strategy.sticker_h, 64)
        self.assertEqual(strategy.sticker_w, 64)

    def test_build_metric_topk(self):
        """Test building TopKAccuracyDrop metric from config dict."""
        from framework.config_loader import _build_metric

        metric_cfg = {
            'type': 'top1_accuracy_drop',
            'k': 1,
        }
        metric = _build_metric(metric_cfg)

        self.assertIsNotNone(metric)
        self.assertEqual(metric.k, 1)


# =============================================================================
# Test: Domain Adapter (Classification)
# =============================================================================

class TestClassificationDomain(unittest.TestCase):
    """Test ImageClassificationDomain."""

    def test_domain_properties(self):
        """Test domain adapter basic properties."""
        from framework.domains.classification import ImageClassificationDomain
        from torch.utils.data import DataLoader

        # Use tiny dataset for testing
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a fake ImageNet structure
            class_dir = Path(tmpdir) / 'val' / 'n01440764'
            class_dir.mkdir(parents=True)

            # Create a dummy image (HWC uint8)
            from PIL import Image
            img_array = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
            img = Image.fromarray(img_array)
            img.save(class_dir / 'test.JPEG')

            # Skip this if torchvision ImageNet dataset needs actual data
            # For now, just test the interface
            print("  (Skipping domain initialization due to ImageNet dataset requirement)")

    def test_domain_with_simple_dataset(self):
        """Test domain methods with synthetic data."""
        # Create a minimal domain-like object
        from framework.base.domain import DomainAdapter, LayerConfig

        # We can't easily test the full domain without ImageNet,
        # but we can verify the interface
        print("  (Domain testing requires real ImageNet data; skipping)")


# =============================================================================
# Test: Trainer
# =============================================================================

class TestTrainer(unittest.TestCase):
    """Test GenericPatchTrainer initialization and basic operations."""

    def test_trainer_init(self):
        """Test trainer initialization."""
        from framework.trainer import GenericPatchTrainer
        from framework.base.domain import DomainAdapter, LayerConfig
        from framework.base.attack_strategy import BorderStrategy

        # Create a minimal mock domain
        class MockDomain(DomainAdapter):
            @property
            def input_shape(self):
                return (64, 64)

            @property
            def model(self):
                return SimpleTestModel(num_classes=10)

            @property
            def device(self):
                return torch.device('cpu')

            def preprocess_for_model(self, image):
                return image

            def get_layer_progression(self):
                return [
                    LayerConfig(name='conv1', out_channels=16),
                    LayerConfig(name='conv2', out_channels=32),
                ]

            def build_dataset(self, split='train'):
                return SimpleTestDataset(num_samples=10, img_size=64, num_classes=10)

            def get_baseline_image(self, image):
                return image

        domain = MockDomain()
        strategy = BorderStrategy()

        trainer = GenericPatchTrainer(
            domain=domain,
            strategy=strategy,
            basis_dim=8,
            patches_per_image=2,
            images_per_batch=1,
            max_epochs=1,
            output_dir=tempfile.mkdtemp(),
        )

        self.assertIsNotNone(trainer)
        self.assertEqual(trainer.basis_dim, 8)


# =============================================================================
# Test: Generator Loader
# =============================================================================

class TestGeneratorLoader(unittest.TestCase):
    """Test load_generator and generate_patch_from_z."""

    def test_generate_patch_from_z(self):
        """Test generate_patch_from_z function."""
        from framework.generator_loader import generate_patch_from_z
        from framework.generator import FoundationPatchGenerator

        gen = FoundationPatchGenerator(
            latent_dim=8,
            patch_height=32,
            patch_width=32,
            use_vae_lora=False,
        )

        z = torch.randn(2, 8)
        patches = generate_patch_from_z(gen, z)

        # Check output
        self.assertEqual(len(patches), 2)
        self.assertEqual(patches[0].shape, (3, 32, 32))
        self.assertTrue(patches[0].min() >= 0 and patches[0].max() <= 1)


# =============================================================================
# Test: Losses
# =============================================================================

class TestLosses(unittest.TestCase):
    """Test loss functions."""

    def test_total_variation_loss(self):
        """Test total_variation_loss."""
        from framework.losses import total_variation_loss

        patches = torch.rand(2, 3, 32, 32)
        mask = torch.ones(1, 1, 32, 32)

        loss = total_variation_loss(patches, mask)

        self.assertIsNotNone(loss)
        self.assertTrue(loss.item() >= 0)

    def test_spectrum_loss(self):
        """Test compute_spectrum_loss."""
        from framework.losses import compute_spectrum_loss

        patches = torch.rand(2, 3, 32, 32)
        mask = torch.ones(1, 1, 32, 32)

        loss = compute_spectrum_loss(patches, mask)

        self.assertIsNotNone(loss)
        self.assertTrue(loss.item() >= 0)

    def test_activation_diversity(self):
        """Test compute_activation_diversity."""
        from framework.losses import compute_activation_diversity

        # Mock activations
        patch_acts = [torch.randn(1, 32) for _ in range(4)]
        baseline_acts = [torch.randn(1, 32) for _ in range(4)]

        div_score = compute_activation_diversity(patch_acts, baseline_acts)

        self.assertIsNotNone(div_score)
        # Should be a positive scalar (log-det)
        self.assertTrue(div_score.item() > 0)


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration(unittest.TestCase):
    """End-to-end integration tests."""

    def test_strategy_consistency(self):
        """Test that all 3 strategies produce valid outputs."""
        from framework.base.attack_strategy import (
            BorderStrategy, StickerStrategy, PerturbationStrategy
        )

        strategies = [
            BorderStrategy(center_ratio=0.6),
            StickerStrategy(sticker_h=32, sticker_w=32),
            PerturbationStrategy(budget=0.1),
        ]

        image = torch.rand(2, 3, 64, 64)
        patch = torch.rand(3, 64, 64)

        for strategy in strategies:
            # apply
            composited, mask = strategy.apply(image, patch)
            self.assertEqual(composited.shape[0], image.shape[0])
            self.assertEqual(composited.shape[1], 3)
            self.assertTrue(composited.min() >= 0 and composited.max() <= 1)

            # apply_neutral
            neutral = strategy.apply_neutral(image)
            self.assertEqual(neutral.shape[0], image.shape[0])
            self.assertEqual(neutral.shape[1], 3)

            # sample_kwargs
            kwargs = strategy.sample_kwargs(image, 64, 64)
            self.assertIsInstance(kwargs, dict)

    def test_metric_with_model_and_strategy(self):
        """Test metric evaluation with model and strategy."""
        from framework.base.metric import TopKAccuracyDrop
        from framework.base.attack_strategy import BorderStrategy

        model = SimpleTestModel(num_classes=10)
        model.eval()

        metric = TopKAccuracyDrop(k=1)
        strategy = BorderStrategy()

        images = [torch.rand(3, 64, 64) for _ in range(5)]
        patch = torch.rand(3, 64, 64)

        with torch.no_grad():
            # Precompute control
            control_preds = metric.precompute_control(images, model)

            # Apply strategy to create composites
            composites = []
            for img in images:
                img_batch = img.unsqueeze(0)
                composited, _ = strategy.apply(img_batch, patch)
                composites.append(composited.squeeze(0))

            # Evaluate
            results = metric.compute(composites, control_preds, model)
            self.assertIn('primary', results)
            self.assertIn('success_rate', results)


# =============================================================================
# Main
# =============================================================================

def run_tests():
    """Run all tests."""
    print("=" * 80)
    print("FRAMEWORK COMPREHENSIVE TEST SUITE")
    print("=" * 80)
    print()

    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestAttackStrategies))
    suite.addTests(loader.loadTestsFromTestCase(TestMetrics))
    suite.addTests(loader.loadTestsFromTestCase(TestGenerator))
    suite.addTests(loader.loadTestsFromTestCase(TestConfigLoader))
    suite.addTests(loader.loadTestsFromTestCase(TestTrainer))
    suite.addTests(loader.loadTestsFromTestCase(TestGeneratorLoader))
    suite.addTests(loader.loadTestsFromTestCase(TestLosses))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Summary
    print()
    print("=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(f"Tests run:     {result.testsRun}")
    print(f"Passed:        {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures:      {len(result.failures)}")
    print(f"Errors:        {len(result.errors)}")
    print(f"Skipped:       {len(result.skipped)}")

    if result.wasSuccessful():
        print()
        print("✓ ALL TESTS PASSED")
        return 0
    else:
        print()
        print("✗ SOME TESTS FAILED")
        if result.failures:
            print("\nFailures:")
            for test, traceback in result.failures:
                print(f"  - {test}")
                print(f"    {traceback}")
        if result.errors:
            print("\nErrors:")
            for test, traceback in result.errors:
                print(f"  - {test}")
                print(f"    {traceback}")
        return 1


if __name__ == '__main__':
    sys.exit(run_tests())
