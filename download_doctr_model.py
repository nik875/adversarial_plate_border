#!/usr/bin/env python3
"""
Download and save doctr ViTSTR model weights locally for offline use.

This script:
1. Downloads vitstr_small pretrained model from doctr
2. Saves the model state dict (weights only)
3. Uses local architecture code (vitstr_architecture.py) at load time

Usage:
    python download_doctr_model.py [--output_dir ./doctr_model]

The saved weights can be loaded with vitstr_architecture.ViTSTR without doctr.
"""

import argparse
import torch
from pathlib import Path


def download_doctr_model(output_dir="./doctr_model"):
    """
    Download and save doctr ViTSTR model state dict locally.

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

    # Save state dict (weights only, no class references)
    state_dict_path = output_path / "vitstr_small_weights.pt"
    print(f"Saving state dict to {state_dict_path}...")
    torch.save(model.state_dict(), str(state_dict_path))
    print(f"✓ State dict saved")

    # Also save model config for reference
    config_path = output_path / "model_config.txt"
    with open(config_path, "w") as f:
        f.write("ViTSTR Model Configuration\n")
        f.write("=" * 50 + "\n")
        f.write(f"Vocab size: {len(model.vocab)}\n")
        f.write(f"Max length: {model.max_length}\n")
        f.write(f"Vocab: {model.vocab}\n")
        f.write(f"Config: {model.cfg}\n")
        f.write("\nModel structure:\n")
        f.write(str(model))
    print(f"✓ Config saved")

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
