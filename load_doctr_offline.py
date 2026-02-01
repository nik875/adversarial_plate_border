"""
Load doctr ViTSTR encoder from local offline cache (ONNX-converted PyTorch).

This module provides utilities to load the ViTSTR feature extractor (encoder)
that has been converted from ONNX format, allowing it to work completely offline
without doctr imports. The model maintains full PyTorch functionality.

Note: We save and load only the encoder (feat_extractor), not the full model
with decoder, since that's what's actually used for profiling.

Usage:
    from load_doctr_offline import load_doctr_model, DoctrLoader

    # Simple loading
    encoder = load_doctr_model("./doctr_model")

    # Or use the class-based interface
    loader = DoctrLoader("./doctr_model")
    encoder = loader.model
"""

from pathlib import Path
import torch


class DoctrLoader:
    """Load ONNX-converted ViTSTR encoder from local directory (no doctr dependency)."""

    def __init__(self, model_dir: str = "./doctr_model"):
        """
        Initialize doctr loader.

        Args:
            model_dir: Path to directory containing saved ONNX-converted encoder
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

        model_file = self.model_dir / "vitstr_small.pt"
        if not model_file.exists():
            raise FileNotFoundError(
                f"Model file not found: {model_file}\n"
                f"Please run: python download_doctr_model.py --output_dir {self.model_dir}"
            )

    @property
    def model(self):
        """Load and cache the ViTSTR encoder (ONNX-converted PyTorch)."""
        if self._model is None:
            print(f"Loading ViTSTR encoder from {self.model_dir / 'vitstr_small.pt'}...")

            # Load ONNX-converted PyTorch encoder (no doctr import needed)
            self._model = torch.load(
                str(self.model_dir / "vitstr_small.pt"),
                map_location='cpu',
                weights_only=False
            )
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
