#!/usr/bin/env python3
"""
Download and save doctr ViTSTR model locally for offline use.

This script downloads the vitstr_small pretrained model from doctr and saves it
to a local directory, allowing it to be loaded without accessing HuggingFace.

Usage:
    python download_doctr_model.py [--output_dir ./doctr_model]
"""

import argparse
import torch
from pathlib import Path


def download_doctr_model(output_dir="./doctr_model"):
    """
    Download and save doctr ViTSTR model locally.

    Args:
        output_dir: Directory to save the model (default: ./doctr_model)
    """
    from doctr.models import vitstr_small

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Downloading doctr ViTSTR model")
    print(f"Output directory: {output_path.absolute()}")
    print()

    # Load pretrained model
    print("Loading vitstr_small (pretrained)...")
    model = vitstr_small(pretrained=True)
    model.eval()

    # Save only state dict (no doctr class references)
    model_path = output_path / "vitstr_small.pt"
    print(f"Saving model state dict to {model_path}...")
    torch.save(model.state_dict(), str(model_path))
    print(f"✓ Model state dict saved to {model_path}")

    print()
    print("=" * 80)
    print("Download complete!")
    print("=" * 80)
    print()
    print("To load the model in your code, use:")
    print()
    print(f"  from load_doctr_offline import load_doctr_model")
    print()
    print(f"  model = load_doctr_model('{output_path.absolute()}')")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download and save doctr ViTSTR model locally"
    )
    parser.add_argument(
        "--output_dir",
        default="./doctr_model",
        help="Directory to save the model (default: ./doctr_model)",
    )
    args = parser.parse_args()

    try:
        download_doctr_model(args.output_dir)
    except Exception as e:
        print(f"Error downloading model: {e}")
        raise
