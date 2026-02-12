#!/usr/bin/env python3
"""
Predict text on composite and control images using fast-alpr OCR-only mode.

Usage:
  python predict_composites.py composite_output -o results.csv
"""

import argparse
import sys
from pathlib import Path
import csv

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

try:
    from fast_alpr import ALPR
except ImportError:
    ALPR = None


def main():
    parser = argparse.ArgumentParser(
        description='Predict text on composite and control images using fast-alpr.'
    )
    parser.add_argument('composite_dir', help='Path to composite_output directory')
    parser.add_argument('-o', '--outfile', default='predictions.csv',
                        help='Output CSV file (default: predictions.csv)')

    args = parser.parse_args()

    composite_dir = Path(args.composite_dir)
    if not composite_dir.exists():
        print(f"Error: Directory not found: {composite_dir}", file=sys.stderr)
        sys.exit(1)

    # Check for image files
    composite_images = sorted(composite_dir.glob("composite_*.jpg"))
    control_images = sorted(composite_dir.glob("control_*.jpg"))

    if not composite_images and not control_images:
        print(f"Error: No composite_*.jpg or control_*.jpg files found in {composite_dir}",
              file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(composite_images)} composite images")
    print(f"Found {len(control_images)} control images")

    # Initialize ALPR in OCR-only mode
    if ALPR is None:
        print("Error: fast-alpr not installed. Install with: pip install fast-alpr",
              file=sys.stderr)
        sys.exit(1)

    print("\nInitializing fast-alpr (OCR-only mode)...")
    alpr = ALPR(
        detector=None,
        ocr_model="cct-s-v1-global-model",
    )
    print("fast-alpr loaded")

    # Process all images
    results = []

    print("\nPredicting on control images...")
    for img_path in tqdm(control_images, desc="Control"):
        try:
            # Load image
            image = Image.open(str(img_path)).convert('RGB')
            image_array = np.array(image)

            # Predict text
            ocr_result = alpr.ocr.predict(image_array)
            detected_text = ocr_result.text if ocr_result is not None else ""

            results.append({
                'image': img_path.name,
                'type': 'control',
                'predicted_text': detected_text,
            })
        except Exception as e:
            print(f"Error processing {img_path.name}: {e}", file=sys.stderr)
            results.append({
                'image': img_path.name,
                'type': 'control',
                'predicted_text': f'ERROR: {e}',
            })

    print("\nPredicting on composite images...")
    for img_path in tqdm(composite_images, desc="Composite"):
        try:
            # Load image
            image = Image.open(str(img_path)).convert('RGB')
            image_array = np.array(image)

            # Predict text
            ocr_result = alpr.ocr.predict(image_array)
            detected_text = ocr_result.text if ocr_result is not None else ""

            results.append({
                'image': img_path.name,
                'type': 'composite',
                'predicted_text': detected_text,
            })
        except Exception as e:
            print(f"Error processing {img_path.name}: {e}", file=sys.stderr)
            results.append({
                'image': img_path.name,
                'type': 'composite',
                'predicted_text': f'ERROR: {e}',
            })

    # Save results to CSV
    output_path = Path(args.outfile)
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['image', 'type', 'predicted_text'])
        writer.writeheader()
        writer.writerows(results)

    print(f"\nResults saved to {output_path}")

    # Print summary
    control_results = [r for r in results if r['type'] == 'control']
    composite_results = [r for r in results if r['type'] == 'composite']

    print(f"\nSummary:")
    print(f"  Control images: {len(control_results)}")
    print(f"  Composite images: {len(composite_results)}")
    print(f"\nControl predictions:")
    for r in control_results:
        print(f"  {r['image']:20s} -> {r['predicted_text']}")
    print(f"\nComposite predictions:")
    for r in composite_results:
        print(f"  {r['image']:20s} -> {r['predicted_text']}")


if __name__ == '__main__':
    main()
