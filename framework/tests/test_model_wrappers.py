"""
test_model_wrappers.py — Forward-pass smoke tests for all 8 model wrappers.

Run with pytest (skips heavyweight models unless --run-heavy is passed):
    pytest framework/tests/test_model_wrappers.py -v
    pytest framework/tests/test_model_wrappers.py -v --run-heavy

Each test:
  1. Imports the wrapper from framework.models.REGISTRY
  2. Calls wrapper.get_preprocess_fn() to build the preprocessing function
  3. Creates a random [2, 3, H, W] input in [0, 1]
  4. Runs forward() and asserts the output is a Tensor with the expected batch dim
"""
from __future__ import annotations

import pytest
import torch


def _pytest_addoption_safe(parser):
    """Add --run-heavy option if not already added by another plugin."""
    try:
        parser.addoption(
            '--run-heavy', action='store_true', default=False,
            help='Include tests that download large pretrained models'
        )
    except ValueError:
        pass


def pytest_addoption(parser):
    _pytest_addoption_safe(parser)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_input(H: int, W: int, B: int = 2) -> torch.Tensor:
    return torch.rand(B, 3, H, W)


def _run_wrapper(wrapper_cls, raw_input_size=(64, 64), B=2):
    """Instantiate wrapper, preprocess input, run forward, return output tensor."""
    wrapper = wrapper_cls()
    preprocess = wrapper_cls.get_preprocess_fn()
    x_raw = _make_input(*raw_input_size, B=B)
    x = preprocess(x_raw)
    with torch.no_grad():
        out = wrapper(x)
    assert isinstance(out, torch.Tensor), f"Expected Tensor, got {type(out)}"
    assert out.shape[0] == B, f"Expected batch size {B}, got {out.shape[0]}"
    return out


# ---------------------------------------------------------------------------
# Lightweight tests (torchvision only — no large downloads)
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
# Heavy tests (require internet / large model downloads)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not pytest.config.getoption('--run-heavy', default=False)
    if hasattr(pytest, 'config') else True,
    reason='Pass --run-heavy to run large model download tests'
)
class TestHeavyWrappers:
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
        # [B, 84, 8400]
        assert out.shape[0] == 2, f"Unexpected batch dim: {out.shape}"
        assert out.ndim == 3, f"Expected 3D output, got {out.ndim}D"

    def test_trocr_output_shape(self):
        from framework.models.trocr import TrOCREncoderWrapper
        out = _run_wrapper(TrOCREncoderWrapper)
        # [B, 577, 768]
        assert out.shape == (2, 577, 768), f"Unexpected shape: {out.shape}"

    def test_smolvlm_output_shape(self):
        from framework.models.vlm import SmolVLMWrapper
        out = _run_wrapper(SmolVLMWrapper)
        # [B, seq_len, vocab_size]
        assert out.shape[0] == 2, f"Unexpected batch dim: {out.shape}"
        assert out.ndim == 3, f"Expected 3D output, got {out.ndim}D"


# ---------------------------------------------------------------------------
# Registry smoke test
# ---------------------------------------------------------------------------

class TestRegistry:
    def test_registry_has_all_keys(self):
        from framework.models import REGISTRY
        expected = {
            'resnet50', 'convnext_small', 'swin_t',
            'dinov2_vitb14', 'clip_vit_l14', 'yolov8s',
            'trocr_base', 'smolvlm_500m',
        }
        assert expected == set(REGISTRY.keys()), (
            f"Registry mismatch. Missing: {expected - set(REGISTRY)}, "
            f"Extra: {set(REGISTRY) - expected}"
        )

    def test_build_model_raises_on_unknown(self):
        from framework.models import build_model
        with pytest.raises(ValueError, match="Unknown architecture"):
            build_model('this_does_not_exist')

    def test_classification_wrappers_via_registry(self):
        """Light end-to-end test using the registry for torchvision models."""
        from framework.models import REGISTRY
        for key in ('resnet50', 'convnext_small', 'swin_t'):
            cls = REGISTRY[key]
            out = _run_wrapper(cls)
            assert out.shape[0] == 2, f"{key}: bad batch dim {out.shape}"
