#!/usr/bin/env python3
"""
Detailed inspection of TrOCRProcessor configuration and behavior
"""
from transformers import TrOCRProcessor
from PIL import Image
import numpy as np
import json

# Load processor
print("Loading TrOCRProcessor...")
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-small-printed")

# Check image processor
img_proc = processor.image_processor

print("\n" + "="*80)
print("ALL IMAGE PROCESSOR ATTRIBUTES")
print("="*80)
for attr in dir(img_proc):
    if not attr.startswith('_'):
        try:
            val = getattr(img_proc, attr)
            if not callable(val):
                print(f"{attr}: {val}")
        except:
            pass

print("\n" + "="*80)
print("PREPROCESSING STEPS")
print("="*80)

# Create test image with known pattern
test_img = Image.new('RGB', (64, 128), color=(128, 128, 128))
print(f"\nInput image: 64x128 (H x W), all pixels = 128")

# Process step by step
print("\nProcessing with processor...")
processed = processor(images=test_img, return_tensors="pt")
print(f"Output shape: {processed.pixel_values.shape}")
print(f"Output dtype: {processed.pixel_values.dtype}")
print(f"Output value range: [{processed.pixel_values.min():.6f}, {processed.pixel_values.max():.6f}]")

# Get the source code to understand what happens
print("\n" + "="*80)
print("IMAGE PROCESSOR CLASS INFO")
print("="*80)
print(f"Class: {type(img_proc).__name__}")
print(f"Module: {type(img_proc).__module__}")

# Print the __call__ method signature if available
import inspect
try:
    sig = inspect.signature(img_proc.__call__)
    print(f"\n__call__ signature: {sig}")
except:
    pass

# Test with different input sizes to understand resize behavior
print("\n" + "="*80)
print("RESIZE BEHAVIOR TEST")
print("="*80)

test_cases = [
    (64, 128, "Original (64x128)"),
    (100, 200, "Larger (100x200)"),
    (384, 384, "Square (384x384)"),
    (10, 20, "Very small (10x20)"),
]

for h, w, label in test_cases:
    img = Image.new('RGB', (w, h), color=(128, 128, 128))
    proc = processor(images=img, return_tensors="pt")
    out_shape = proc.pixel_values.shape
    print(f"{label:30} -> {out_shape}")

# Check if there are any config files
print("\n" + "="*80)
print("CONFIG CHECK")
print("="*80)
if hasattr(img_proc, 'to_dict'):
    try:
        config = img_proc.to_dict()
        print("Processor config as dict:")
        print(json.dumps(config, indent=2))
    except Exception as e:
        print(f"Could not get config: {e}")

# Check the actual preprocessing order
print("\n" + "="*80)
print("STEP-BY-STEP VERIFICATION")
print("="*80)
from PIL import Image as PILImage
import torch

# Manual preprocessing to understand the order
test_img_manual = Image.new('RGB', (64, 128), color=(200, 100, 50))
proc_result = processor(images=test_img_manual, return_tensors="pt")

print(f"\nInput: RGB image 64x128, pixel=(200, 100, 50)")
print(f"Output shape: {proc_result.pixel_values.shape}")
print(f"First pixel R,G,B: {proc_result.pixel_values[0, :, 0, 0]}")

# Calculate what the values should be if normalized with mean=0.5, std=0.5
pixel_normalized = (np.array([200, 100, 50]) / 255.0 - 0.5) / 0.5
print(f"\nIf formula is (x/255 - 0.5)/0.5:")
print(f"  Expected: {pixel_normalized}")
print(f"  Match: {np.allclose(proc_result.pixel_values[0, :, 0, 0].numpy(), pixel_normalized, atol=0.01)}")
