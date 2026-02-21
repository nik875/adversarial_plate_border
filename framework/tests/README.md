# Framework Tests

Comprehensive test suite to validate all framework components before cloud deployment.

## Quick Sanity Check (≈30 seconds)

Fastest way to verify basic functionality:

```bash
python framework/tests/sanity_check.py
```

**What it tests:**
- All module imports work
- Basic operations: all 3 attack strategies
- Generator initialization and forward pass
- Loss functions
- Config loading (if classification_resnet50.yaml exists)

**Output:**
```
================================================================================
FRAMEWORK SANITY CHECK
================================================================================

Testing imports...
  ✓ framework.base.domain
  ✓ framework.base.attack_strategy
  ✓ framework.base.metric
  ✓ framework.generator
  ✓ framework.losses
  ✓ framework.generator_loader
  ✓ framework.config_loader
  ✓ framework.trainer

Testing basic operations...
  ✓ BorderStrategy.apply()
  ✓ StickerStrategy.apply() with bbox
  ✓ StickerStrategy.sample_kwargs()
  ✓ PerturbationStrategy.apply()
  ✓ FoundationPatchGenerator.forward_clean()
  ✓ total_variation_loss()

Testing config loading...
  ✓ load_domain_config() + all components

================================================================================
✓ ALL SANITY CHECKS PASSED
================================================================================
```

---

## Comprehensive Test Suite (≈2-3 minutes)

Full unit tests for all components. Requires `pytest`:

```bash
# Install pytest (if needed)
pip install pytest

# Run all tests with verbose output
python -m pytest framework/tests/test_framework.py -v

# Or without pytest (uses unittest):
python framework/tests/test_framework.py
```

**What it tests:**

### Attack Strategies (TestAttackStrategies)
- `BorderStrategy.apply()` — composite + visibility mask
- `BorderStrategy.apply_neutral()` — grey canvas with centered subject
- `StickerStrategy` — fixed and random placement
- `StickerStrategy.sample_kwargs()` — random bbox generation
- `PerturbationStrategy` — L∞ clipping, clamping to [0,1]

### Metrics (TestMetrics)
- `TopKAccuracyDrop.precompute_control()` — stores clean predictions
- `TopKAccuracyDrop.compute()` — success rate + drop metrics
- `TopKAccuracyDrop.compute_detailed()` — per-image breakdowns

### Generator (TestGenerator)
- Initialization with/without LoRA
- Forward pass shapes and value ranges [0,1]
- LoRA adds trainable parameters

### Config Loading (TestConfigLoader)
- Parse YAML configs
- Build strategies (border, sticker, perturbation)
- Build metrics
- Load full domain config

### Trainer (TestTrainer)
- Trainer initialization
- Mock domain + strategy integration

### Generator Loader (TestGeneratorLoader)
- `generate_patch_from_z()` function

### Losses (TestLosses)
- `total_variation_loss()` — patch smoothness
- `compute_spectrum_loss()` — structural diversity
- `compute_activation_diversity()` — layer diversity

### Integration Tests (TestIntegration)
- All strategies produce valid [B,3,H,W] composites in [0,1]
- Metrics work with models + strategies
- End-to-end pipeline

**Example output:**
```
test_activation_diversity ... ok
test_apply_neutral ... ok
test_border_strategy_apply ... ok
test_border_strategy_neutral ... ok
test_border_strategy_sample_kwargs ... ok
test_build_metric_topk ... ok
test_build_strategy_border ... ok
test_build_strategy_sticker ... ok
test_compute ... ok
test_compute_detailed ... ok
test_forward ... ok
test_generate_patch_from_z ... ok
test_generator_init ... ok
test_generator_lora_parameters ... ok
test_integration_consistency ... ok
test_load_classification_config ... ok
test_metric_with_model_and_strategy ... ok
test_perturbation_strategy_linf ... ok
test_perturbation_strategy_neutral ... ok
test_precompute ... ok
test_spectrum_loss ... ok
test_sticker_strategy_fixed_bbox ... ok
test_sticker_strategy_random_bbox ... ok
test_strategy_consistency ... ok
test_topk_accuracy_drop_compute ... ok
test_topk_accuracy_drop_compute_detailed ... ok
test_topk_accuracy_drop_precompute ... ok
test_total_variation_loss ... ok
test_trainer_init ... ok

================================================================================
TEST SUMMARY
================================================================================
Tests run:     30
Passed:        30
Failures:      0
Errors:        0

✓ ALL TESTS PASSED
```

---

## Testing Before Cloud Deployment

**Recommended workflow:**

1. **Run sanity check locally** (30 sec)
   ```bash
   python framework/tests/sanity_check.py
   ```

2. **Run full test suite locally** (2-3 min)
   ```bash
   python framework/tests/test_framework.py
   ```

3. **If both pass:** framework is ready for cloud instance

4. **On cloud instance:**
   ```bash
   # Run sanity check to verify environment
   python framework/tests/sanity_check.py

   # Run a tiny training job
   python framework/scripts/train_domain.py framework/configs/classification_resnet50.yaml \
       --epochs 1 \
       --output-dir /tmp/test_run
   ```

---

## Troubleshooting Common Issues

| Error | Cause | Fix |
|-------|-------|-----|
| `ModuleNotFoundError: No module named 'framework'` | Project root not on path | Run from project root: `cd /path/to/Adversarial\ Plate && python framework/tests/sanity_check.py` |
| `ImportError: No module named 'torch'` | PyTorch not installed | `pip install torch torchvision` |
| `FileNotFoundError: classification_resnet50.yaml` | Config missing | Not critical; sanity check will skip config tests |
| `CUDA out of memory` | GPU memory exhausted | Reduce `batch_size`, `patches_per_image`, or `basis_dim` in config |
| `AttributeError: 'NoneType' has no attribute 'model'` | Domain not initialized correctly | Check YAML config paths and model names |

---

## Test Coverage

| Component | Status | Coverage |
|-----------|--------|----------|
| BorderStrategy | ✓ | apply(), apply_neutral(), sample_kwargs() |
| StickerStrategy | ✓ | fixed bbox, random bbox, sample_kwargs() |
| PerturbationStrategy | ✓ | L∞ clipping, neutral |
| TopKAccuracyDrop | ✓ | precompute, compute, compute_detailed |
| FoundationPatchGenerator | ✓ | forward_clean, LoRA, initialization |
| total_variation_loss | ✓ | gradient, shape, value range |
| compute_spectrum_loss | ✓ | SSIM-based |
| compute_activation_diversity | ✓ | log-det |
| load_domain_config | ✓ | YAML parsing, strategy/metric building |
| GenericPatchTrainer | ✓ | initialization, mock domain |
| Config loader | ✓ | all strategies, all metrics |
| Integration | ✓ | multi-strategy pipeline, metric evaluation |

---

## Adding New Tests

When adding new features, add tests to `test_framework.py`:

```python
class TestNewFeature(unittest.TestCase):
    def test_something(self):
        from framework.some_module import SomeClass
        obj = SomeClass()
        # assertions...
        self.assertTrue(...)
```

Then run:
```bash
python -m pytest framework/tests/test_framework.py::TestNewFeature -v
```
