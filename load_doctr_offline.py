"""
Load doctr ViTSTR model from local ONNX cache (no doctr imports needed).

This module provides utilities to load the ViTSTR model from ONNX format
using ONNX Runtime, allowing it to work completely offline without doctr.
The model wrapper provides PyTorch-like functionality.

Usage:
    from load_doctr_offline import load_doctr_model, DoctrLoader

    # Simple loading
    model = load_doctr_model("./doctr_model")

    # Or use the class-based interface
    loader = DoctrLoader("./doctr_model")
    model = loader.model
"""

from pathlib import Path
import torch
import numpy as np


class ONNXModelWrapper(torch.nn.Module):
    """PyTorch-like wrapper around ONNX Runtime model."""

    def __init__(self, onnx_session):
        super().__init__()
        self.session = onnx_session
        self.input_name = onnx_session.get_inputs()[0].name
        self.output_names = [o.name for o in onnx_session.get_outputs()]

    def forward(self, x):
        """Run inference on input tensor."""
        # Convert torch tensor to numpy
        x_np = x.cpu().detach().numpy()

        # Run ONNX inference
        outputs = self.session.run(self.output_names, {self.input_name: x_np})

        # Convert output back to torch tensor
        # Assume single output (logits)
        output_tensor = torch.from_numpy(outputs[0]).to(x.device)
        return output_tensor

    def eval(self):
        """Set to eval mode (no-op for ONNX)."""
        return self

    def to(self, device):
        """Move to device (ONNX runs on CPU, but cache output device)."""
        self.device = device
        return self


class DoctrLoader:
    """Load ONNX ViTSTR model from local directory (no doctr dependency)."""

    def __init__(self, model_dir: str = "./doctr_model"):
        """
        Initialize doctr loader.

        Args:
            model_dir: Path to directory containing saved ONNX model
        """
        self.model_dir = Path(model_dir)
        self._validate_directory()
        self._model = None

    def _validate_directory(self) -> None:
        """Validate that model files exist in the directory."""
        if not self.model_dir.exists():
            raise FileNotFoundError(
                f"Model directory not found: {self.model_dir}\n"
                f"Please run: python download_doctr_model.py --output_dir {self.model_dir}"
            )

        model_file = self.model_dir / "vitstr_small.onnx"
        if not model_file.exists():
            raise FileNotFoundError(
                f"Model file not found: {model_file}\n"
                f"Please run: python download_doctr_model.py --output_dir {self.model_dir}"
            )

    @property
    def model(self):
        """Load and cache the ViTSTR model (ONNX-based)."""
        if self._model is None:
            import onnxruntime as ort

            print(f"Loading ViTSTR model from {self.model_dir / 'vitstr_small.onnx'}...")

            # Create ONNX Runtime session
            session = ort.InferenceSession(
                str(self.model_dir / "vitstr_small.onnx"),
                providers=['CPUExecutionProvider']
            )

            # Wrap in PyTorch-like interface
            self._model = ONNXModelWrapper(session)
            self._model.eval()

        return self._model


def load_doctr_model(model_dir: str = "./doctr_model"):
    """
    Simple function to load doctr ViTSTR model.

    Args:
        model_dir: Path to directory containing model

    Returns:
        Loaded model

    Example:
        model = load_doctr_model("./doctr_model")
    """
    loader = DoctrLoader(model_dir)
    return loader.model


if __name__ == "__main__":
    # Simple test
    print("Testing doctr offline loader...")
    print()

    try:
        loader = DoctrLoader()
        print("✓ Successfully loaded doctr ViTSTR model")
        print(f"  Model: {loader.model.__class__.__name__}")
    except FileNotFoundError as e:
        print(f"✗ Error: {e}")
