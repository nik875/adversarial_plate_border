#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.optim as optim
import onnx
import onnx2torch
from open_image_models import LicensePlateDetector
from pathlib import Path


def test_gradient_descent():
    print("Loading model...")
    detector = LicensePlateDetector(detection_model="yolo-v9-t-384-license-plate-end2end")

    # Get ONNX model path from cache
    model_cache_dir = Path.home() / ".cache/open-image-models/yolo-v9-t-384-license-plate-end2end"
    onnx_path = model_cache_dir / "yolo-v9-t-384-license-plates-end2end.onnx"

    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX model not found at: {onnx_path}")

    print(f"Converting ONNX to PyTorch: {onnx_path}")
    onnx_model = onnx.load(str(onnx_path))
    pytorch_model = onnx2torch.convert(onnx_model)
    pytorch_model.train()

    # Test forward pass and gradients
    print("Testing differentiability...")
    input_tensor = torch.randn(1, 3, 384, 384, requires_grad=True)
    output = pytorch_model(input_tensor)

    # Simple MSE loss for testing
    target = torch.randn_like(output)
    loss = nn.MSELoss()(output, target)

    # Test backward pass
    optimizer = optim.SGD(pytorch_model.parameters(), lr=0.001)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"✅ SUCCESS: Loss = {loss.item():.2f}, Gradients computed, Model is differentiable!")
    return pytorch_model


if __name__ == "__main__":
    model = test_gradient_descent()
