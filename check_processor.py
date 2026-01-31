#!/usr/bin/env python3
"""
Check what TrOCRProcessor actually does
"""
from transformers import TrOCRProcessor
from PIL import Image
import numpy as np
import torch

# Load processor
print("Loading TrOCRProcessor...")
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-small-printed")

# Check image processor details
img_proc = processor.image_processor
print("\n" + "="*80)
print("IMAGE PROCESSOR DETAILS")
print("="*80)
print(f"Type: {type(img_proc).__name__}")
print(f"Size: {img_proc.size}")
print(f"Image mean: {img_proc.image_mean}")
print(f"Image std: {img_proc.image_std}")
print(f"Do normalize: {img_proc.do_normalize}")
print(f"Do resize: {img_proc.do_resize}")
print(f"Resample: {img_proc.resample}")

# Test with a simple image
print("\n" + "="*80)
print("TEST PREPROCESSING")
print("="*80)

# Create test image (gray, 128 value)
test_img = Image.fromarray(np.ones((64, 128, 3), dtype=np.uint8) * 128)
print(f"\nInput image shape: {test_img.size} (PIL format is W, H)")
print(f"Input pixel values: all 128")

# Process
processed = processor(images=test_img, return_tensors="pt")
print(f"\nOutput tensor shape: {processed.pixel_values.shape}")
print(f"Output value range - min: {processed.pixel_values.min():.6f}, max: {processed.pixel_values.max():.6f}")
print(f"First pixel (all 128 input): {processed.pixel_values[0, :, 0, 0]}")

# Test with 0 and 255
test_black = Image.fromarray(np.zeros((64, 128, 3), dtype=np.uint8))
test_white = Image.fromarray(np.ones((64, 128, 3), dtype=np.uint8) * 255)

proc_black = processor(images=test_black, return_tensors="pt")
proc_white = processor(images=test_white, return_tensors="pt")

print(f"\nBlack (0) input -> output: {proc_black.pixel_values[0, :, 0, 0]}")
print(f"White (255) input -> output: {proc_white.pixel_values[0, :, 0, 0]}")

# Verify normalization formula
print("\n" + "="*80)
print("NORMALIZATION VERIFICATION")
print("="*80)
# Standard formula: (x - mean) / std
img_mean = np.array(img_proc.image_mean)
img_std = np.array(img_proc.image_std)

# For value 128 in [0, 255]
normalized_128 = (128.0 / 255.0 - img_mean) / img_std
print(f"\nManual calculation for 128:")
print(f"  128/255 = {128/255:.6f}")
print(f"  (128/255 - mean) / std = {normalized_128}")
print(f"  Processor output: {processed.pixel_values[0, :, 0, 0]}")
print(f"  Match: {np.allclose(processed.pixel_values[0, :, 0, 0].numpy(), normalized_128)}")
