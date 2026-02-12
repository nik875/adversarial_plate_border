#!/usr/bin/env python3
"""
Predict text on composite and control images and analyze patch effectiveness.

Groups results by patch and calculates success rate where success =
detected text differs between control and composite versions.

Usage:
  python predict_composites.py composite_output -n 10 -o results.csv
"""

import argparse
import sys
from pathlib import Path
import csv
from collections import defaultdict

import numpy as np
from PIL import Image
from tqdm import tqdm

try:
    from fast_alpr import ALPR
except ImportError:
    ALPR = None


def main():
    parser = argparse.ArgumentParser(
        description='Analyze patch effectiveness on composite images.'
    )
    parser.add_argument('composite_dir', help='Path to composite_output directory')
    parser.add_argument('-r', '--run-dir', default=None,
                        help='Path to run directory to infer patch count (default: infer from composite_dir parent)')
    parser.add_argument('-o', '--outfile', default='patch_analysis.csv',
                        help='Output CSV file (default: patch_analysis.csv)')
    parser.add_argument('--white-box', action='store_true',
                        help='Use smaller xs model instead of s model')

    args = parser.parse_args()

    composite_dir = Path(args.composite_dir)
    if not composite_dir.exists():
        print(f"Error: Directory not found: {composite_dir}", file=sys.stderr)
        sys.exit(1)

    # Infer number of patches from run directory
    if args.run_dir:
        run_dir = Path(args.run_dir)
    else:
        # Try to find run directory (parent or sibling)
        run_dir = composite_dir.parent
        if not (run_dir / "example_samples").exists():
            run_dir = composite_dir.parent.parent
        if not (run_dir / "example_samples").exists():
            print("Error: Could not find run directory with example_samples/", file=sys.stderr)
            sys.exit(1)

    # Count patches in the latest batch
    example_samples_dir = run_dir / "example_samples"
    batch_dirs = sorted([d for d in example_samples_dir.iterdir() if d.is_dir()])
    if not batch_dirs:
        print("Error: No batch directories found", file=sys.stderr)
        sys.exit(1)

    latest_batch = batch_dirs[-1]
    patch_files = list(latest_batch.glob("patch_epoch_*_sample_*.png"))
    num_patches = len(patch_files)

    if num_patches == 0:
        print("Error: No patch files found", file=sys.stderr)
        sys.exit(1)

    print(f"Inferred {num_patches} patches from {latest_batch.name}")

    # Check for image files
    composite_images = sorted(composite_dir.glob("composite_*.jpg"))
    control_images = sorted(composite_dir.glob("control_*.jpg"))

    if not composite_images or not control_images:
        print(f"Error: No composite_*.jpg or control_*.jpg files found in {composite_dir}",
              file=sys.stderr)
        sys.exit(1)

    if len(composite_images) != len(control_images):
        print(f"Error: Number of composite ({len(composite_images)}) and control ({len(control_images)}) images don't match",
              file=sys.stderr)
        sys.exit(1)

    num_samples = len(composite_images) // num_patches
    total_samples = len(composite_images)

    print(f"Found {total_samples} image pairs")
    print(f"Number of patches: {num_patches}")
    print(f"Samples per patch: {num_samples}")

    # Initialize ALPR in OCR-only mode
    if ALPR is None:
        print("Error: fast-alpr not installed. Install with: pip install fast-alpr",
              file=sys.stderr)
        sys.exit(1)

    print("\nInitializing fast-alpr (OCR-only mode)...")
    ocr_model = "cct-xs-v1-global-model" if args.white_box else "cct-s-v1-global-model"
    print(f"Using OCR model: {ocr_model}")
    alpr = ALPR(
        detector=None,
        ocr_model=ocr_model,
    )
    print("fast-alpr loaded")

    # Process all image pairs
    print("\nRunning OCR on all images...")
    pbar = tqdm(total=total_samples * 2, desc="OCR Progress")

    composite_predictions = {}
    control_predictions = {}

    for idx, (comp_path, ctrl_path) in enumerate(zip(composite_images, control_images)):
        try:
            # Load and predict composite
            image = Image.open(str(comp_path)).convert('RGB')
            image_array = np.array(image)
            ocr_result = alpr.ocr.predict(image_array)
            composite_predictions[idx] = ocr_result.text if ocr_result is not None else ""
            pbar.update(1)
        except Exception as e:
            print(f"Error processing {comp_path.name}: {e}", file=sys.stderr)
            composite_predictions[idx] = f"ERROR: {e}"
            pbar.update(1)

        try:
            # Load and predict control
            image = Image.open(str(ctrl_path)).convert('RGB')
            image_array = np.array(image)
            ocr_result = alpr.ocr.predict(image_array)
            control_predictions[idx] = ocr_result.text if ocr_result is not None else ""
            pbar.update(1)
        except Exception as e:
            print(f"Error processing {ctrl_path.name}: {e}", file=sys.stderr)
            control_predictions[idx] = f"ERROR: {e}"
            pbar.update(1)

    pbar.close()

    # Group results by patch and calculate metrics
    print("\nAnalyzing results by patch...")
    patch_results = defaultdict(list)

    for sample_idx in range(total_samples):
        patch_idx = sample_idx % num_patches
        composite_text = composite_predictions.get(sample_idx, "")
        control_text = control_predictions.get(sample_idx, "")

        # Success = texts differ (patch changed the prediction)
        success = composite_text != control_text

        patch_results[patch_idx].append({
            'sample_idx': sample_idx,
            'control_text': control_text,
            'composite_text': composite_text,
            'success': success,
        })

    # Calculate and display metrics per patch
    print("\n" + "=" * 80)
    print("PATCH EFFECTIVENESS ANALYSIS")
    print("=" * 80)

    summary_results = []
    overall_success = 0
    overall_total = 0

    for patch_idx in range(num_patches):
        results = patch_results[patch_idx]
        successes = sum(1 for r in results if r['success'])
        total = len(results)
        success_rate = (successes / total * 100) if total > 0 else 0

        print(f"\nPatch {patch_idx}:")
        print(f"  Success Rate: {successes}/{total} ({success_rate:.1f}%)")
        print(f"  Samples:")
        for r in results:
            status = "✓ SUCCESS" if r['success'] else "✗ FAILED"
            print(f"    {status} | Sample {r['sample_idx']:3d} | Control: '{r['control_text']:15s}' → Composite: '{r['composite_text']:15s}'")

        summary_results.append({
            'patch_idx': patch_idx,
            'successes': successes,
            'total': total,
            'success_rate': f"{success_rate:.1f}%",
        })

        overall_success += successes
        overall_total += total

    # Print overall summary
    overall_rate = (overall_success / overall_total * 100) if overall_total > 0 else 0
    print(f"\n{'=' * 80}")
    print(f"OVERALL SUMMARY")
    print(f"{'=' * 80}")
    print(f"Total Success Rate: {overall_success}/{overall_total} ({overall_rate:.1f}%)")

    # Save results to CSV
    output_path = Path(args.outfile)
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['patch_idx', 'successes', 'total', 'success_rate'])
        writer.writeheader()
        writer.writerows(summary_results)

    print(f"\nResults saved to {output_path}")


if __name__ == '__main__':
    main()
