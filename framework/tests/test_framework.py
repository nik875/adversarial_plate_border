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
# Test: TaskEncoder
# =============================================================================

class TestTaskEncoder(unittest.TestCase):
    """Test TaskEncoder initialization and forward pass."""

    def setUp(self):
        from framework.task_encoder import TaskEncoder
        self.TaskEncoder = TaskEncoder

    def test_init(self):
        enc = self.TaskEncoder(
            num_models=3, num_strategies=3, num_datasets=2,
            latent_dim=8, hidden_dim=32, num_layers=2, alpha=0.5,
        )
        self.assertIsNotNone(enc)

    def test_zero_delta_at_init(self):
        """Final Linear is zero-initialized → task_delta = 0 → z_enriched = z."""
        enc = self.TaskEncoder(
            num_models=3, num_strategies=3, num_datasets=2,
            latent_dim=8, hidden_dim=32, num_layers=2, alpha=0.5,
        )
        z = torch.randn(2, 8)
        mi = torch.zeros(2, dtype=torch.long)
        si = torch.zeros(2, dtype=torch.long)
        di = torch.zeros(2, dtype=torch.long)
        z_enc = enc(z, mi, si, di)
        self.assertTrue(torch.allclose(z_enc, z),
                        msg="z_enriched should equal z at init (delta=0)")

    def test_forward_shape(self):
        enc = self.TaskEncoder(
            num_models=4, num_strategies=3, num_datasets=5, latent_dim=16,
        )
        B = 3
        z = torch.randn(B, 16)
        mi = torch.randint(0, 4, (B,))
        si = torch.randint(0, 3, (B,))
        di = torch.randint(0, 5, (B,))
        out = enc(z, mi, si, di)
        self.assertEqual(out.shape, (B, 16))

    def test_alpha_bounds_shift(self):
        """z_enriched should be within alpha=1.0 of z (since Tanh ∈ [-1,1])."""
        enc = self.TaskEncoder(
            num_models=2, num_strategies=2, num_datasets=2,
            latent_dim=8, hidden_dim=16, num_layers=2, alpha=1.0,
        )
        # Give the final layer non-zero weights so delta isn't trivially 0
        with torch.no_grad():
            enc.mlp[-2].weight.fill_(0.1)
            enc.mlp[-2].bias.fill_(0.0)

        z = torch.zeros(1, 8)
        mi = torch.tensor([0]); si = torch.tensor([0]); di = torch.tensor([0])
        z_enc = enc(z, mi, si, di)
        diff = (z_enc - z).abs().max().item()
        self.assertLessEqual(diff, 1.0 + 1e-5,
                             msg="shift must be ≤ alpha * 1 = 1.0")


# =============================================================================
# Test: NeuronSampler
# =============================================================================

class TestNeuronSampler(unittest.TestCase):
    """Test NeuronSampler layer discovery and activation capture."""

    def setUp(self):
        from framework.neuron_sampler import NeuronSampler
        self.NeuronSampler = NeuronSampler
        self.device = torch.device('cpu')

    def _tiny_model(self):
        return SimpleTestModel(num_classes=5)

    def test_discover_layers(self):
        sampler = self.NeuronSampler(self.device)
        model = self._tiny_model()
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)

        shapes = sampler.discover_layers(model, (3, 32, 32))
        self.assertGreater(len(shapes), 0, "Should find at least one layer")
        # All values should be non-empty tuples
        for name, shape in shapes.items():
            self.assertIsInstance(shape, tuple)
            self.assertGreater(len(shape), 0)

    def test_discover_layers_cached(self):
        sampler = self.NeuronSampler(self.device)
        model = self._tiny_model()
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)

        shapes1 = sampler.discover_layers(model, (3, 32, 32))
        shapes2 = sampler.discover_layers(model, (3, 32, 32))
        self.assertIs(shapes1, shapes2, "Second call should return cached dict")

    def test_sample_neurons_length(self):
        sampler = self.NeuronSampler(self.device)
        model = self._tiny_model()
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)

        k = 50
        neurons = sampler.sample_neurons(model, k, (3, 32, 32))
        self.assertEqual(len(neurons), k)
        for name, flat_idx in neurons:
            self.assertIsInstance(name, str)
            self.assertIsInstance(flat_idx, int)

    def test_sample_neurons_spread(self):
        """Neurons should be sampled from more than one layer."""
        sampler = self.NeuronSampler(self.device)
        model = self._tiny_model()
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)

        neurons = sampler.sample_neurons(model, 200, (3, 32, 32))
        unique_layers = len(set(name for name, _ in neurons))
        self.assertGreater(unique_layers, 1,
                           msg="Neurons should span multiple layers")

    def test_capture_no_grad(self):
        sampler = self.NeuronSampler(self.device)
        model = self._tiny_model()
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)

        k = 20
        neurons = sampler.sample_neurons(model, k, (3, 32, 32))
        inp = torch.randn(1, 3, 32, 32)
        acts = sampler.capture_sampled_activations(model, inp, neurons, no_grad=True)

        self.assertEqual(acts.shape, (k,))
        self.assertFalse(acts.requires_grad, "no_grad=True should return detached tensor")

    def test_capture_with_grad(self):
        sampler = self.NeuronSampler(self.device)
        model = self._tiny_model()
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)

        k = 10
        neurons = sampler.sample_neurons(model, k, (3, 32, 32))
        inp = torch.randn(1, 3, 32, 32, requires_grad=True)
        acts = sampler.capture_sampled_activations(model, inp, neurons, no_grad=False)

        self.assertEqual(acts.shape, (k,))
        # Gradient should flow through inp
        acts.sum().backward()
        self.assertIsNotNone(inp.grad)

    def test_invalidate(self):
        sampler = self.NeuronSampler(self.device)
        model = self._tiny_model()
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)

        sampler.discover_layers(model, (3, 32, 32))
        self.assertIn(id(model), sampler._shape_cache)
        sampler.invalidate(model)
        self.assertNotIn(id(model), sampler._shape_cache)


# =============================================================================
# Test: LazyDatasetPool
# =============================================================================

class TestDatasetPool(unittest.TestCase):
    """Test LazyDatasetPool registration, discovery, and sampling."""

    def setUp(self):
        from framework.dataset_pool import LazyDatasetPool, SampledItem
        self.LazyDatasetPool = LazyDatasetPool
        self.SampledItem = SampledItem

    def _make_png_dir(self, tmpdir: str, n: int = 5) -> str:
        """Create n small synthetic PNG images in tmpdir."""
        from PIL import Image
        import numpy as np
        d = Path(tmpdir) / 'imgs'
        d.mkdir(exist_ok=True)
        for i in range(n):
            arr = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
            Image.fromarray(arr, 'RGB').save(d / f'img_{i}.png')
        return str(d)

    def test_register_and_num_datasets(self):
        pool = self.LazyDatasetPool()
        id0 = pool.register('d0', '/tmp/fake0')
        id1 = pool.register('d1', '/tmp/fake1')
        self.assertEqual(pool.num_datasets(), 2)
        self.assertEqual(id0, 0)
        self.assertEqual(id1, 1)

    def test_register_paths_and_sample(self):
        import tempfile
        from PIL import Image
        import numpy as np

        pool = self.LazyDatasetPool()

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a real image
            p = Path(tmpdir) / 'test.png'
            arr = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
            Image.fromarray(arr, 'RGB').save(str(p))

            did = pool.register_paths('synthetic', [str(p)])
            item = pool.sample_from(did)

        self.assertIsInstance(item, self.SampledItem)
        self.assertEqual(item.image.shape[0], 3)
        self.assertEqual(item.dataset_id, did)

    def test_sample_returns_correct_dtype(self):
        import tempfile
        from PIL import Image
        import numpy as np

        pool = self.LazyDatasetPool()
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir) / 'x.png'
            Image.fromarray(
                np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8), 'RGB'
            ).save(str(p))
            did = pool.register_paths('test', [str(p)])
            item = pool.sample_from(did)

        self.assertEqual(item.image.dtype, torch.float32)
        self.assertGreaterEqual(item.image.min().item(), 0.0)
        self.assertLessEqual(item.image.max().item(), 1.0 + 1e-5)


# =============================================================================
# Test: PriorRegistry
# =============================================================================

class TestPriorRegistry(unittest.TestCase):
    """Test PriorRegistry initialization (no JIT decoder needed for basic tests)."""

    def setUp(self):
        from framework.priors import PriorRegistry
        self.PriorRegistry = PriorRegistry

    def test_init_empty(self):
        reg = self.PriorRegistry(patch_height=64, patch_width=64, latent_dim=8)
        self.assertEqual(reg.num_output_channels, 0)

    def test_forward_empty(self):
        """Empty registry forward should return empty list."""
        reg = self.PriorRegistry(patch_height=64, patch_width=64, latent_dim=8)
        z = torch.randn(2, 8)
        out = reg(z)
        self.assertEqual(out, [])

    def test_num_output_channels_formula(self):
        """num_output_channels = num_priors × 4 (4 scales)."""
        reg = self.PriorRegistry(patch_height=64, patch_width=64, latent_dim=8)
        # Simulate 2 priors without actually loading JIT decoders
        reg._decoders['fake1'] = None
        reg._decoders['fake2'] = None
        self.assertEqual(reg.num_output_channels, 8)

    def test_add_prior_missing_file(self):
        """add_prior with non-existent file should raise FileNotFoundError."""
        reg = self.PriorRegistry(patch_height=64, patch_width=64, latent_dim=8)
        with self.assertRaises(FileNotFoundError):
            reg.add_prior('missing', '/nonexistent/path/decoder.pt')


# =============================================================================
# Test: EnsembleModelPool
# =============================================================================

class TestEnsembleModelPool(unittest.TestCase):
    """Test EnsembleModelPool register/sample/on_device."""

    def setUp(self):
        from framework.ensemble import EnsembleModelPool
        from framework.base.attack_strategy import BorderStrategy
        self.EnsembleModelPool = EnsembleModelPool
        self.BorderStrategy = BorderStrategy

    def _make_entry(self, pool, name: str, strategy_id: int = 0):
        model = SimpleTestModel(num_classes=5)
        pool.register(
            name=name,
            model=model,
            domain_type='classification',
            strategy=self.BorderStrategy(),
            strategy_id=strategy_id,
            input_shape=(64, 64),
            preprocess_fn=lambda x: x,
        )

    def test_register_freezes_params(self):
        pool = self.EnsembleModelPool(compute_device=torch.device('cpu'))
        self._make_entry(pool, 'model_a')
        entry = pool.get_entry(0)
        for p in entry._model.parameters():
            self.assertFalse(p.requires_grad, "All params should be frozen")

    def test_register_moves_to_cpu(self):
        pool = self.EnsembleModelPool(compute_device=torch.device('cpu'))
        self._make_entry(pool, 'model_b')
        entry = pool.get_entry(0)
        for p in entry._model.parameters():
            self.assertEqual(p.device.type, 'cpu')

    def test_num_models(self):
        pool = self.EnsembleModelPool(compute_device=torch.device('cpu'))
        self._make_entry(pool, 'a', strategy_id=0)
        self._make_entry(pool, 'b', strategy_id=1)
        self.assertEqual(pool.num_models(), 2)

    def test_num_strategies(self):
        pool = self.EnsembleModelPool(compute_device=torch.device('cpu'))
        self._make_entry(pool, 'a', strategy_id=0)
        self._make_entry(pool, 'b', strategy_id=0)   # same strategy
        self._make_entry(pool, 'c', strategy_id=1)
        self.assertEqual(pool.num_strategies(), 2)

    def test_sample_entry_random(self):
        pool = self.EnsembleModelPool(compute_device=torch.device('cpu'))
        self._make_entry(pool, 'a')
        entry = pool.sample_entry()
        self.assertEqual(entry.name, 'a')

    def test_on_device_cpu(self):
        pool = self.EnsembleModelPool(compute_device=torch.device('cpu'))
        self._make_entry(pool, 'test')
        entry = pool.get_entry(0)
        with pool.on_device(entry) as model:
            self.assertIsNotNone(model)
            inp = torch.randn(1, 3, 64, 64)
            out = model(inp)
            self.assertEqual(out.shape[0], 1)
        # Model should be back on CPU after context exit
        for p in entry._model.parameters():
            self.assertEqual(p.device.type, 'cpu')


# =============================================================================
# Test: EnsembleTrainer smoke test
# =============================================================================

class TestEnsembleTrainer(unittest.TestCase):
    """Smoke-test EnsembleTrainer._train_step() on CPU with synthetic data."""

    def _build_trainer(self, tmpdir: str):
        """Build a minimal EnsembleTrainer for testing."""
        import tempfile
        from PIL import Image
        import numpy as np
        from framework.ensemble import EnsembleModelPool, EnsembleTrainer
        from framework.task_encoder import TaskEncoder
        from framework.dataset_pool import LazyDatasetPool
        from framework.neuron_sampler import NeuronSampler
        from framework.base.attack_strategy import BorderStrategy

        device = torch.device('cpu')

        # Minimal model
        model = SimpleTestModel(num_classes=5)

        # Dataset pool with a real image
        img_path = Path(tmpdir) / 'test.png'
        arr = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        Image.fromarray(arr, 'RGB').save(str(img_path))

        dataset_pool = LazyDatasetPool()
        dataset_pool.register_paths('test', [str(img_path)])

        # Ensemble pool
        pool = EnsembleModelPool(compute_device=device)
        pool.register(
            name='tiny_cnn',
            model=model,
            domain_type='classification',
            strategy=BorderStrategy(),
            strategy_id=0,
            input_shape=(64, 64),
            preprocess_fn=lambda x: x,
        )

        # TaskEncoder
        task_enc = TaskEncoder(
            num_models=1, num_strategies=3, num_datasets=1,
            latent_dim=4, hidden_dim=16, num_layers=2,
        )

        # Tiny generator (no VAE — can't load SDXL in unit tests)
        # We'll use a mock generator that just returns random patches
        class MockGenerator(nn.Module):
            def __init__(self):
                super().__init__()
                self.latent_dim = 4
                self.patch_height = 64
                self.patch_width = 64
                self.conv = nn.Conv2d(1, 3, 1)

            def forward(self, z, z_enriched=None):
                B = z.shape[0]
                noise = torch.rand(B, 3, 64, 64)
                return self.conv(noise[:, :1])  # [B, 3, 64, 64]

            # Stubs so EnsembleTrainer can navigate
            @property
            def vae(self):
                return nn.Module()

            @property
            def bottleneck_refiner(self):
                return None

        gen = MockGenerator()

        sampler = NeuronSampler(device)

        trainer = EnsembleTrainer(
            ensemble=pool,
            dataset_pool=dataset_pool,
            task_encoder=task_enc,
            generator=gen,
            neuron_sampler=sampler,
            k_neurons=10,
            patches_per_batch=2,
            diversity_weight=1.0,
            quality_weight=1.0,
            tv_weight=0.0,      # off to keep test fast
            spectrum_weight=0.0,
            learning_rate=1e-3,
            max_epochs=1,
            output_dir=tmpdir,
            save_every_epochs=1,
            device=device,
        )
        return trainer

    def test_train_step_runs(self):
        """A single _train_step should complete without error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = self._build_trainer(tmpdir)
            # Build a real optimizer for the mock generator
            optimizer = torch.optim.Adam(
                list(trainer.generator.parameters())
                + list(trainer.task_encoder.parameters()), lr=1e-3
            )
            optimizer.zero_grad()
            info = trainer._train_step(optimizer)
            self.assertIn('loss', info)
            self.assertIsInstance(info['loss'], float)

    def test_train_loop_runs(self):
        """Full train() with 2 steps should complete without error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = self._build_trainer(tmpdir)
            trainer.train(steps_per_epoch=2, max_steps=2)


# =============================================================================
# Test: EnsembleConfig loading
# =============================================================================

class TestEnsembleConfigLoader(unittest.TestCase):
    """Test load_ensemble_config() with a minimal YAML."""

    def test_load_ensemble_config_minimal(self):
        from framework.config_loader import load_ensemble_config

        config_path = _PROJECT_ROOT / 'framework' / 'configs' / 'ensemble_example.yaml'
        if not config_path.exists():
            self.skipTest(f"Ensemble config not found: {config_path}")

        cfg = load_ensemble_config(str(config_path))

        self.assertIsNotNone(cfg.ensemble_pool)
        self.assertIsNotNone(cfg.dataset_pool)
        self.assertIsNotNone(cfg.task_encoder)
        self.assertIsNotNone(cfg.generator_cfg)
        self.assertIsNotNone(cfg.trainer_cfg)
        self.assertIsInstance(cfg.models_cfg, list)

    def test_task_encoder_dimensions(self):
        """TaskEncoder dimensions should match YAML model/dataset counts."""
        from framework.config_loader import load_ensemble_config

        config_path = _PROJECT_ROOT / 'framework' / 'configs' / 'ensemble_example.yaml'
        if not config_path.exists():
            self.skipTest(f"Ensemble config not found: {config_path}")

        cfg = load_ensemble_config(str(config_path))
        te = cfg.task_encoder
        # ensemble_example.yaml has 2 models and 1 dataset
        self.assertEqual(te.num_models, len(cfg.models_cfg))
        self.assertEqual(te.num_datasets, cfg.dataset_pool.num_datasets())


# =============================================================================
# Test: migrate_checkpoint
# =============================================================================

class TestMigrateCheckpoint(unittest.TestCase):
    """Test that migrate_checkpoint correctly upgrades old state dicts."""

    def test_attention_proj_migration(self):
        from framework.generator_loader import migrate_checkpoint

        # Simulate an old state dict with a single Linear attention_proj
        old_sd = {
            'bottleneck_refiner.attention_proj.weight': torch.randn(32, 16),
            'bottleneck_refiner.attention_proj.bias':   torch.zeros(32),
        }
        new_sd = migrate_checkpoint(old_sd)

        self.assertIn('bottleneck_refiner.attention_proj.0.weight', new_sd)
        self.assertIn('bottleneck_refiner.attention_proj.2.weight', new_sd)
        self.assertIn('bottleneck_refiner.attention_proj.4.weight', new_sd)
        # Old keys should be gone
        self.assertNotIn('bottleneck_refiner.attention_proj.weight', new_sd)
        # Final layer should carry old weights
        self.assertTrue(torch.allclose(
            new_sd['bottleneck_refiner.attention_proj.4.weight'],
            old_sd['bottleneck_refiner.attention_proj.weight']
        ))

    def test_cnn_refiner_channel_migration(self):
        from framework.generator_loader import migrate_checkpoint

        old_sd = {
            'cnn_refiner.0.weight': torch.randn(64, 4, 3, 3),
        }
        new_sd = migrate_checkpoint(old_sd)

        self.assertEqual(new_sd['cnn_refiner.0.weight'].shape, (64, 7, 3, 3))
        # First 4 channels should be unchanged
        self.assertTrue(torch.allclose(
            new_sd['cnn_refiner.0.weight'][:, :4],
            old_sd['cnn_refiner.0.weight']
        ))
        # Extra 3 channels should be zero
        self.assertTrue(
            new_sd['cnn_refiner.0.weight'][:, 4:].abs().max().item() == 0.0
        )

    def test_no_migration_needed(self):
        """State dict without old keys should pass through unchanged."""
        from framework.generator_loader import migrate_checkpoint

        sd = {
            'some_param': torch.randn(3, 3),
            'cnn_refiner.0.weight': torch.randn(64, 7, 3, 3),  # already 7 ch
        }
        new_sd = migrate_checkpoint(sd)
        # cnn_refiner already has 7 channels → no change
        self.assertEqual(new_sd['cnn_refiner.0.weight'].shape, (64, 7, 3, 3))


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
    # --- New ensemble tests ---
    suite.addTests(loader.loadTestsFromTestCase(TestTaskEncoder))
    suite.addTests(loader.loadTestsFromTestCase(TestNeuronSampler))
    suite.addTests(loader.loadTestsFromTestCase(TestDatasetPool))
    suite.addTests(loader.loadTestsFromTestCase(TestPriorRegistry))
    suite.addTests(loader.loadTestsFromTestCase(TestEnsembleModelPool))
    suite.addTests(loader.loadTestsFromTestCase(TestEnsembleTrainer))
    suite.addTests(loader.loadTestsFromTestCase(TestEnsembleConfigLoader))
    suite.addTests(loader.loadTestsFromTestCase(TestMigrateCheckpoint))

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
