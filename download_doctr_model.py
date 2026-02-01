#!/usr/bin/env python3
"""
Download and convert doctr ViTSTR model to ONNX for offline use.

This script:
1. Downloads vitstr_small pretrained model from doctr
2. Creates a wrapper that outputs logits only (no postprocessor string decoding)
3. Exports wrapper to ONNX format
4. Converts ONNX back to PyTorch (removes doctr dependencies)
5. Saves the converted model

Usage:
    python download_doctr_model.py [--output_dir ./doctr_model]
"""

import argparse
import torch
import torch.nn as nn
from pathlib import Path


class ViTSTRLogitsWrapper(nn.Module):
    """Wrapper that outputs logits only, no postprocessor string decoding."""

    def __init__(self, model):
        super().__init__()
        self.model = model
        # Store original postprocessor
        self.original_postprocessor = model.postprocessor

    def forward(self, x):
        # Temporarily disable postprocessor by replacing it with an identity function
        # that just returns the logits without .numpy() conversion
        def dummy_postprocessor(logits):
            return logits

        self.model.postprocessor = dummy_postprocessor

        try:
            # Call model with postprocessor disabled
            output = self.model(x)
        finally:
            # Restore original postprocessor
            self.model.postprocessor = self.original_postprocessor

        # Extract just the logits (should be the raw tensor output)
        if isinstance(output, dict):
            return output.get('logits', output)
        return output


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

    # Wrap to output logits only (skip postprocessor string decoding)
    print("Creating logits-only wrapper...")
    logits_model = ViTSTRLogitsWrapper(model)
    logits_model.eval()

    # Export to ONNX
    print("Exporting to ONNX...")
    onnx_path = output_path / "vitstr_small.onnx"
    dummy_input = torch.randn(1, 3, 32, 128)

    torch.onnx.export(
        logits_model,
        dummy_input,
        str(onnx_path),
        input_names=['input'],
        output_names=['logits'],
        opset_version=18,
        do_constant_folding=True,
        verbose=False
    )
    print(f"✓ ONNX model saved to {onnx_path}")

    print(f"✓ ONNX export successful!")
    print()
    print("Note: ONNX model saved. Use with ONNX Runtime (no doctr needed)")
    print(f"ONNX model path: {onnx_path}")

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
