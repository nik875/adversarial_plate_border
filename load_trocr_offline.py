"""
Load TrOCR model from local offline cache.

This module provides utilities to load the TrOCR model from a locally saved
directory without requiring HuggingFace API access. This is useful when running
in offline environments or when dealing with conflicting dependency versions.

Usage:
    from load_trocr_offline import load_trocr_model, TrOCRLoader

    # Simple loading
    model, tokenizer, feature_extractor = load_trocr_model("./trocr_model")

    # Or use the class-based interface
    loader = TrOCRLoader("./trocr_model")
    model = loader.model
    tokenizer = loader.tokenizer
    feature_extractor = loader.feature_extractor
"""

from pathlib import Path
from typing import Optional, Tuple
import warnings


class TrOCRLoader:
    """Load TrOCR model components from local directory."""

    def __init__(self, model_dir: str = "./trocr_model"):
        """
        Initialize TrOCR loader.

        Args:
            model_dir: Path to directory containing model, tokenizer, feature_extractor
        """
        self.model_dir = Path(model_dir)
        self._validate_directory()
        self._model = None
        self._tokenizer = None
        self._feature_extractor = None

    def _validate_directory(self) -> None:
        """Validate that all required components exist in the directory."""
        if not self.model_dir.exists():
            raise FileNotFoundError(
                f"Model directory not found: {self.model_dir}\n"
                f"Please run: python download_trocr_model.py --output_dir {self.model_dir}"
            )

        required_dirs = ["model", "tokenizer", "feature_extractor"]
        missing = [d for d in required_dirs if not (self.model_dir / d).exists()]

        if missing:
            raise FileNotFoundError(
                f"Missing model components in {self.model_dir}: {', '.join(missing)}\n"
                f"Please run: python download_trocr_model.py --output_dir {self.model_dir}"
            )

    @property
    def model(self):
        """Load and cache the TrOCR model."""
        if self._model is None:
            from transformers import AutoModelForImageTextToText

            print(f"Loading TrOCR model from {self.model_dir / 'model'}...")
            self._model = AutoModelForImageTextToText.from_pretrained(
                str(self.model_dir / "model"), local_files_only=True
            )
        return self._model

    @property
    def tokenizer(self):
        """Load and cache the tokenizer."""
        if self._tokenizer is None:
            from transformers import AutoTokenizer

            print(f"Loading tokenizer from {self.model_dir / 'tokenizer'}...")
            self._tokenizer = AutoTokenizer.from_pretrained(
                str(self.model_dir / "tokenizer"), local_files_only=True
            )
        return self._tokenizer

    @property
    def feature_extractor(self):
        """Load and cache the feature extractor."""
        if self._feature_extractor is None:
            from transformers import AutoFeatureExtractor

            print(
                f"Loading feature extractor from {self.model_dir / 'feature_extractor'}..."
            )
            self._feature_extractor = AutoFeatureExtractor.from_pretrained(
                str(self.model_dir / "feature_extractor"), local_files_only=True
            )
        return self._feature_extractor

    def recognize_text(self, image):
        """
        Recognize text in an image using TrOCR.

        Args:
            image: PIL Image or image path

        Returns:
            Recognized text as string
        """
        from PIL import Image

        # Load image if path is provided
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")

        # Preprocess image
        pixel_values = self.feature_extractor(image, return_tensors="pt").pixel_values

        # Generate text
        generated_ids = self.model.generate(pixel_values)
        generated_text = self.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0]

        return generated_text


def load_trocr_model(
    model_dir: str = "./trocr_model",
) -> Tuple:
    """
    Simple function to load all TrOCR components.

    Args:
        model_dir: Path to directory containing model components

    Returns:
        Tuple of (model, tokenizer, feature_extractor)

    Example:
        model, tokenizer, feature_extractor = load_trocr_model("./trocr_model")
    """
    loader = TrOCRLoader(model_dir)
    return loader.model, loader.tokenizer, loader.feature_extractor


if __name__ == "__main__":
    # Simple test
    print("Testing TrOCR offline loader...")
    print()

    try:
        loader = TrOCRLoader()
        print("✓ Successfully loaded TrOCR components")
        print(f"  Model: {loader.model.__class__.__name__}")
        print(f"  Tokenizer: {loader.tokenizer.__class__.__name__}")
        print(f"  Feature Extractor: {loader.feature_extractor.__class__.__name__}")
    except FileNotFoundError as e:
        print(f"✗ Error: {e}")
