#!/usr/bin/env python3
"""
Download and save TrOCR model locally for offline use.

This script downloads the microsoft/trocr-small-printed model and saves it
to a local directory, allowing it to be loaded without HuggingFace API access.

Usage:
    python download_trocr_model.py [--output_dir ./trocr_model]
"""

import argparse
import os
from pathlib import Path

def download_trocr_model(output_dir="./trocr_model"):
    """
    Download and save TrOCR model locally.

    Args:
        output_dir: Directory to save the model (default: ./trocr_model)
    """
    from transformers import (
        AutoTokenizer,
        AutoModelForImageTextToText,
        AutoFeatureExtractor,
    )

    model_name = "microsoft/trocr-small-printed"
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Downloading TrOCR model: {model_name}")
    print(f"Output directory: {output_path.absolute()}")
    print()

    # Download model
    print("Downloading model weights...")
    model = AutoModelForImageTextToText.from_pretrained(model_name)
    model.save_pretrained(str(output_path / "model"))
    print(f"✓ Model saved to {output_path / 'model'}")

    # Download tokenizer
    print("Downloading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.save_pretrained(str(output_path / "tokenizer"))
    print(f"✓ Tokenizer saved to {output_path / 'tokenizer'}")

    # Download feature extractor (for vision preprocessing)
    print("Downloading feature extractor...")
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
    feature_extractor.save_pretrained(str(output_path / "feature_extractor"))
    print(f"✓ Feature extractor saved to {output_path / 'feature_extractor'}")

    print()
    print("=" * 80)
    print("Download complete!")
    print("=" * 80)
    print()
    print("To load the model in your code, use:")
    print()
    print(f"  from transformers import (")
    print(f"      AutoTokenizer,")
    print(f"      AutoModelForImageTextToText,")
    print(f"      AutoFeatureExtractor,")
    print(f"  )")
    print()
    print(f"  model_dir = '{output_path.absolute()}'")
    print(f"  model = AutoModelForImageTextToText.from_pretrained(")
    print(f"      model_dir + '/model',")
    print(f"      local_files_only=True")
    print(f"  )")
    print(f"  tokenizer = AutoTokenizer.from_pretrained(")
    print(f"      model_dir + '/tokenizer',")
    print(f"      local_files_only=True")
    print(f"  )")
    print(f"  feature_extractor = AutoFeatureExtractor.from_pretrained(")
    print(f"      model_dir + '/feature_extractor',")
    print(f"      local_files_only=True")
    print(f"  )")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download and save TrOCR model locally"
    )
    parser.add_argument(
        "--output_dir",
        default="./trocr_model",
        help="Directory to save the model (default: ./trocr_model)",
    )
    args = parser.parse_args()

    try:
        download_trocr_model(args.output_dir)
    except Exception as e:
        print(f"Error downloading model: {e}")
        raise
