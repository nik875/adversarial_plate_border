"""
test_model_wrappers.py — Forward-pass smoke tests for all 8 model wrappers.

Run lightweight tests (torchvision only, no large downloads):
    pytest framework/tests/test_model_wrappers.py -v

Run all tests including models that download large weights:
    pytest framework/tests/test_model_wrappers.py -v --run-heavy

Each test:
  1. Gets the wrapper class from framework.models.REGISTRY (lazy import)
  2. Calls wrapper.get_preprocess_fn() to build the preprocessing function
  3. Creates a random [2, 3, H, W] input in [0, 1]
  4. Runs forward() and asserts the output is a Tensor with batch dim == 2
"""
from __future__ import annotations

import pytest
import torch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_input(H: int, W: int, B: int = 2) -> torch.Tensor:
    return torch.rand(B, 3, H, W)


def _run_wrapper(wrapper_cls, B: int = 2) -> torch.Tensor:
    """Instantiate wrapper, preprocess a random input, run forward."""
    wrapper = wrapper_cls()
    H, W = wrapper_cls.input_size()
    preprocess = wrapper_cls.get_preprocess_fn()
    x_raw = _make_input(H, W, B=B)
    x = preprocess(x_raw)
    with torch.no_grad():
        out = wrapper(x)
    assert isinstance(out, torch.Tensor), f"Expected Tensor, got {type(out)}"
    assert out.shape[0] == B, f"Expected batch size {B}, got {out.shape[0]}"
    return out


# ---------------------------------------------------------------------------
# Lightweight tests — torchvision only, no large downloads
# ---------------------------------------------------------------------------

class TestClassificationWrappers:
    def test_resnet50_output_shape(self):
        from framework.models.classification import ResNet50Wrapper
        out = _run_wrapper(ResNet50Wrapper)
        assert out.shape == (2, 1000), f"Unexpected shape: {out.shape}"

    def test_convnext_small_output_shape(self):
        from framework.models.classification import ConvNeXtSmallWrapper
        out = _run_wrapper(ConvNeXtSmallWrapper)
        assert out.shape == (2, 1000), f"Unexpected shape: {out.shape}"

    def test_swin_t_output_shape(self):
        from framework.models.swin import SwinTWrapper
        out = _run_wrapper(SwinTWrapper)
        assert out.shape == (2, 1000), f"Unexpected shape: {out.shape}"


# ---------------------------------------------------------------------------
# Heavy tests — require internet access and large model downloads
# Skipped unless --run-heavy is passed.
# ---------------------------------------------------------------------------

@pytest.mark.heavy
class TestHeavyWrappers:
    @pytest.fixture(autouse=True)
    def skip_unless_heavy(self, request):
        if not request.config.getoption('--run-heavy', default=False):
            pytest.skip('Pass --run-heavy to run large model download tests')

    def test_dinov2_output_shape(self):
        from framework.models.dinov2 import DINOv2Wrapper
        out = _run_wrapper(DINOv2Wrapper)
        assert out.shape == (2, 768), f"Unexpected shape: {out.shape}"

    def test_clip_output_shape(self):
        from framework.models.clip_encoder import CLIPVisionWrapper
        out = _run_wrapper(CLIPVisionWrapper)
        assert out.shape == (2, 1024), f"Unexpected shape: {out.shape}"

    def test_yolo_output_shape(self):
        from framework.models.yolo_wrapper import YOLOv8Wrapper
        out = _run_wrapper(YOLOv8Wrapper)
        assert out.ndim == 3, f"Expected 3D output [B,84,8400], got {out.shape}"
        assert out.shape[0] == 2

    def test_trocr_output_shape(self):
        from framework.models.trocr import TrOCREncoderWrapper
        out = _run_wrapper(TrOCREncoderWrapper)
        assert out.shape == (2, 577, 768), f"Unexpected shape: {out.shape}"

    def test_smolvlm_output_shape(self):
        from framework.models.vlm import SmolVLMWrapper
        out = _run_wrapper(SmolVLMWrapper)
        assert out.ndim == 3, f"Expected 3D output [B,seq,vocab], got {out.shape}"
        assert out.shape[0] == 2


# ---------------------------------------------------------------------------
# Registry tests — no model instantiation for the keys/raises tests
# ---------------------------------------------------------------------------

class TestRegistry:
    def test_registry_has_all_keys(self):
        from framework.models import REGISTRY
        expected = {
            'resnet50', 'convnext_small', 'swin_t',
            'dinov2_vitb14', 'clip_vit_l14', 'yolov8s',
            'trocr_base', 'smolvlm_500m',
        }
        actual = set(REGISTRY.keys())
        assert expected == actual, (
            f"Missing: {expected - actual}, Extra: {actual - expected}"
        )

    def test_registry_contains(self):
        from framework.models import REGISTRY
        assert 'resnet50' in REGISTRY
        assert 'unknown_arch' not in REGISTRY

    def test_build_model_raises_on_unknown(self):
        from framework.models import build_model
        with pytest.raises(ValueError, match="Unknown architecture"):
            build_model('this_does_not_exist')

    def test_classification_wrappers_via_registry(self):
        """End-to-end test using REGISTRY for the three torchvision models."""
        from framework.models import REGISTRY
        for key in ('resnet50', 'convnext_small', 'swin_t'):
            cls = REGISTRY[key]
            out = _run_wrapper(cls)
            assert out.shape[0] == 2, f"{key}: bad batch dim {out.shape}"
