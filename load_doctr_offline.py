"""
Load doctr ViTSTR model from local saved weights (no doctr import needed).

Uses standalone vitstr_architecture.py to load pretrained weights.
"""

from pathlib import Path
import torch
from vitstr_architecture import ViTSTR


class DoctrLoader:
    """Load ViTSTR model from locally saved weights."""

    def __init__(self, model_dir: str = "./doctr_model"):
        """
        Initialize doctr loader.

        Args:
            model_dir: Path to directory containing vitstr_small_weights.pt
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

        weights_file = self.model_dir / "vitstr_small_weights.pt"
        if not weights_file.exists():
            raise FileNotFoundError(
                f"Model weights not found: {weights_file}\n"
                f"Please run: python download_doctr_model.py --output_dir {self.model_dir}"
            )

    @property
    def model(self):
        """Load and cache the ViTSTR model."""
        if self._model is None:
            from doctr.models import vitstr_small

            print(f"Loading ViTSTR from {self.model_dir / 'vitstr_small_weights.pt'}...")

            # Create model architecture (needs doctr for vit_s backbone at this step)
            model = vitstr_small(pretrained=False)

            # Load saved weights
            weights_path = self.model_dir / "vitstr_small_weights.pt"
            state_dict = torch.load(str(weights_path), map_location="cpu")
            model.load_state_dict(state_dict)
            model.eval()

            self._model = model

        return self._model


def load_doctr_model(model_dir: str = "./doctr_model"):
    """
    Simple function to load ViTSTR model from saved weights.

    Args:
        model_dir: Path to directory containing saved weights

    Returns:
        Loaded model

    Example:
        model = load_doctr_model("./doctr_model")
    """
    loader = DoctrLoader(model_dir)
    return loader.model
