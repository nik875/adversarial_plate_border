#!/usr/bin/env python3
"""
Download and convert doctr ViTSTR model to TorchScript for offline use.

This script:
1. Downloads vitstr_small pretrained model from doctr
2. Traces to TorchScript using torch.jit.trace (forgiving of complex ops)
3. Saves the traced model (removes most doctr dependencies)
4. Falls back to full model save if tracing fails

Usage:
    python download_doctr_model.py [--output_dir ./doctr_model]
"""

import argparse
import torch
from pathlib import Path


def download_doctr_model(output_dir="./doctr_model"):
    """
    Download and save doctr ViTSTR model locally via TorchScript tracing.

    Args:
        output_dir: Directory to save the model (default: ./doctr_model)
    """
    from doctr.models import vitstr_small

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Downloading and converting doctr ViTSTR model")
    print(f"Output directory: {output_path.absolute()}")
    print()

    # Load pretrained model
    print("Loading vitstr_small (pretrained)...")
    model = vitstr_small(pretrained=True)
    model.eval()

    # Trace to TorchScript (more forgiving than ONNX export for complex models)
    print("Tracing model to TorchScript...")
    dummy_input = torch.randn(1, 3, 32, 128)

    try:
        traced_model = torch.jit.trace(model, dummy_input, strict=False)
        print(f"✓ Model traced to TorchScript")

        # Save traced model (removes doctr class references)
        model_path = output_path / "vitstr_small.pt"
        print(f"Saving traced PyTorch model to {model_path}...")
        torch.save(traced_model, str(model_path))
        print(f"✓ Traced model saved to {model_path}")

    except Exception as e:
        print(f"Warning: Tracing failed, saving full model instead...")
        print(f"  Error: {e}")

        # Fallback: save full model with weights_only=False
        model_path = output_path / "vitstr_small.pt"
        print(f"Saving full PyTorch model to {model_path}...")
        torch.save(model, str(model_path))
        print(f"✓ Full model saved to {model_path}")
        print(f"  Note: Full model contains doctr class references")

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
