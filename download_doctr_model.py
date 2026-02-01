#!/usr/bin/env python3
"""
Download and convert doctr ViTSTR model to ONNX + PyTorch for offline use.

This script:
1. Downloads vitstr_small pretrained model from doctr
2. Exports to ONNX format
3. Converts ONNX back to PyTorch (removes doctr dependencies)
4. Saves the PyTorch model for offline use

Usage:
    python download_doctr_model.py [--output_dir ./doctr_model]
"""

import argparse
import torch
from pathlib import Path
import tempfile


def download_doctr_model(output_dir="./doctr_model"):
    """
    Download and save doctr ViTSTR model locally via ONNX conversion.

    Args:
        output_dir: Directory to save the model (default: ./doctr_model)
    """
    from doctr.models import vitstr_small
    import onnx
    import onnx2torch

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Downloading and converting doctr ViTSTR model")
    print(f"Output directory: {output_path.absolute()}")
    print()

    # Load pretrained model
    print("Loading vitstr_small (pretrained)...")
    model = vitstr_small(pretrained=True)
    model.eval()

    # Export to ONNX via temporary file
    print("Exporting to ONNX...")
    onnx_path = output_path / "vitstr_small.onnx"

    # Create a dummy input for export
    dummy_input = torch.randn(1, 3, 32, 128)

    torch.onnx.export(
        model,
        dummy_input,
        str(onnx_path),
        input_names=['input'],
        output_names=['output'],
        opset_version=12,
        do_constant_folding=True,
        verbose=False
    )
    print(f"✓ ONNX model saved to {onnx_path}")

    # Convert ONNX back to PyTorch (removes doctr dependencies)
    print("Converting ONNX back to PyTorch...")
    onnx_model = onnx.load(str(onnx_path))
    torch_model = onnx2torch.ConvertModel(onnx_model)
    torch_model.eval()

    # Save PyTorch model
    model_path = output_path / "vitstr_small.pt"
    print(f"Saving converted PyTorch model to {model_path}...")
    torch.save(torch_model, str(model_path))
    print(f"✓ PyTorch model saved to {model_path}")

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
