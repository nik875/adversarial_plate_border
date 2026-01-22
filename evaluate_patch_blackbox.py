#!/usr/bin/env python3
"""
Evaluate a patch on physical_world_test control images using black-box ALPR.
Applies the patch to all control images, queries fast-alpr, and generates a pie chart.
"""

import argparse
import pandas as pd
import numpy as np
import torch
import cv2
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm

try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
except ImportError:
    print("Warning: pillow_heif not installed. HEIC images may not load.")

try:
    from fast_alpr import ALPR
except ImportError:
    ALPR = None

import kornia.geometry as K


def load_patch(patch_path: str) -> torch.Tensor:
    """Load patch image and convert to tensor [3, H, W] in range [0, 1]"""
    img = Image.open(patch_path).convert('RGB')
    img_np = np.array(img)
    # Convert to torch tensor [3, H, W]
    patch_tensor = torch.from_numpy(img_np).float().permute(2, 0, 1) / 255.0
    return patch_tensor


def apply_patch_to_image(image: np.ndarray, corners: np.ndarray, patch: torch.Tensor,
                         border_scale: float = 1.4, device: str = 'cpu') -> np.ndarray:
    """
    Apply patch to image using homography (matches refine_generator.py logic exactly).

    Args:
        image: [H, W, 3] numpy array, uint8, range [0, 255]
        corners: [4, 2] plate corner coordinates
        patch: [3, H_patch, W_patch] tensor, range [0, 1]
        border_scale: Scale factor for border (default: 1.4)
        device: torch device

    Returns:
        patched_image: [H, W, 3] numpy array, uint8, range [0, 255]
    """
    # Convert image to tensor [1, 3, H, W] in range [0, 1]
    image_tensor = torch.from_numpy(image).float().permute(2, 0, 1) / 255.0
    image_tensor = image_tensor.unsqueeze(0).to(device)

    batch_size = 1
    image_height, image_width = image.shape[:2]
    dsize = (image_height, image_width)

    # Patch is already in [0, 1] range
    patch_normalized = patch.to(device)

    # Convert corners to tensor
    plate_corners = torch.from_numpy(corners).float().to(device)

    # Calculate center and create larger border quad
    center_x = plate_corners[:, 0].mean()
    center_y = plate_corners[:, 1].mean()
    center = torch.tensor([center_x, center_y], device=device)

    border_corners = center.unsqueeze(0) + (plate_corners - center.unsqueeze(0)) * border_scale
    border_corners = border_corners.unsqueeze(0)  # [1, 4, 2]

    # Create patch corner coordinates in patch space
    patch_h, patch_w = patch.shape[1], patch.shape[2]
    src_corners = torch.tensor([
        [0, 0], [patch_w, 0], [patch_w, patch_h], [0, patch_h]
    ], dtype=torch.float32, device=device).unsqueeze(0)

    # Compute perspective transforms
    M_border = K.get_perspective_transform(src_corners, border_corners)
    M_plate = K.get_perspective_transform(src_corners, plate_corners.unsqueeze(0))

    # Prepare patch batch
    patch_batch = patch_normalized.unsqueeze(0).repeat(batch_size, 1, 1, 1)

    # Warp patch to border region
    warped_patch = K.warp_perspective(
        patch_batch, M_border, dsize=dsize,
        mode='bilinear', padding_mode='zeros', align_corners=True
    )

    # Create masks
    patch_mask = torch.ones(batch_size, 1, patch_h, patch_w,
                            dtype=torch.float32, device=device)

    warped_border_mask = K.warp_perspective(
        patch_mask, M_border, dsize=dsize,
        mode='bilinear', padding_mode='zeros', align_corners=True
    )

    warped_plate_mask = K.warp_perspective(
        patch_mask, M_plate, dsize=dsize,
        mode='bilinear', padding_mode='zeros', align_corners=True
    )

    # Final mask is border minus plate (ring around plate)
    final_mask = torch.clamp(warped_border_mask - warped_plate_mask, 0, 1)
    final_mask = final_mask.expand(-1, 3, -1, -1)

    # Apply patch
    result_image = image_tensor * (1 - final_mask) + warped_patch * final_mask
    result_image = torch.clamp(result_image, 0, 1)

    # Convert back to numpy
    result_np = (result_image.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

    return result_np


def categorize_result(detected_text, true_plate, impersonation_target):
    """Categorize detection result"""
    if detected_text is None or detected_text == "":
        return 'Failed detection'

    detected_text = str(detected_text).strip()

    if detected_text == true_plate:
        return 'Correct read'

    if impersonation_target and detected_text == impersonation_target:
        return 'Successful impersonation'

    return 'Misread'


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate a patch using black-box ALPR on physical_world_test control images'
    )
    parser.add_argument(
        '--patch',
        required=True,
        help='Path to patch image file (PNG/JPG)'
    )
    parser.add_argument(
        '--corners-csv',
        default='control_plate_corners.csv',
        help='CSV with image paths and plate corners (default: control_plate_corners.csv)'
    )
    parser.add_argument(
        '--impersonation-target',
        default=None,
        help='Impersonation target plate text (default: None for disruption mode)'
    )
    parser.add_argument(
        '--true-plate',
        default='VRJ7774',
        help='True plate text (default: VRJ7774)'
    )
    parser.add_argument(
        '--output',
        default='patch_evaluation_pie.png',
        help='Output pie chart filename (default: patch_evaluation_pie.png)'
    )
    parser.add_argument(
        '--results-csv',
        default=None,
        help='Save detailed results to CSV (default: None, no CSV output)'
    )

    args = parser.parse_args()

    if ALPR is None:
        print("Error: fast-alpr not installed. Install with: pip install fast-alpr")
        return

    # Load patch
    print(f"Loading patch from {args.patch}...")
    patch = load_patch(args.patch)
    print(f"Patch loaded: shape={patch.shape}")

    # Load control images and corners from CSV
    print(f"\nLoading control images from {args.corners_csv}...")
    df = pd.read_csv(args.corners_csv)

    if len(df) == 0:
        print("Error: No images found in CSV")
        return

    print(f"Found {len(df)} control images")

    # Initialize ALPR
    print("\nInitializing fast-alpr...")
    alpr = ALPR(
        detector_model="yolo-v9-s-608-license-plate-end2end",
        ocr_model="cct-s-v1-global-model",
    )
    print("fast-alpr loaded")

    # Process each image
    results = []

    print("\nEvaluating patch on all control images...")
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
        # Get image path
        img_path = row['filename']

        if not Path(img_path).exists():
            print(f"Warning: Image not found: {img_path}")
            continue

        # Load image
        try:
            # Support HEIC images
            image = Image.open(img_path).convert('RGB')
            image = np.array(image)
        except Exception as e:
            print(f"Warning: Failed to load {img_path}: {e}")
            continue

        # Get corners from CSV
        corners = np.array([
            [row['p1_x'], row['p1_y']],
            [row['p2_x'], row['p2_y']],
            [row['p3_x'], row['p3_y']],
            [row['p4_x'], row['p4_y']]
        ], dtype=np.float32)

        # Apply patch
        patched_image = apply_patch_to_image(image, corners, patch)

        # Convert to BGR for fast-alpr
        patched_bgr = cv2.cvtColor(patched_image, cv2.COLOR_RGB2BGR)

        # Query ALPR
        try:
            predictions = alpr.predict(patched_bgr)
        except Exception as e:
            print(f"Warning: ALPR failed on image {idx}: {e}")
            detected_text = None
        else:
            if predictions and len(predictions) > 0:
                # Get highest confidence detection
                best_pred = max(predictions, key=lambda p: p.ocr.confidence if p.ocr else 0.0)
                detected_text = best_pred.ocr.text if best_pred.ocr else None
            else:
                detected_text = None

        # Categorize result
        category = categorize_result(detected_text, args.true_plate, args.impersonation_target)

        results.append({
            'image_index': idx,
            'detected_text': detected_text,
            'category': category
        })

    # Convert to DataFrame
    results_df = pd.DataFrame(results)

    if len(results_df) == 0:
        print("Error: No results generated. Check CSV format and image paths.")
        return

    # Save detailed results if requested
    if args.results_csv:
        results_df.to_csv(args.results_csv, index=False)
        print(f"\n✓ Detailed results saved to: {args.results_csv}")

    # Count categories
    counts = results_df['category'].value_counts()

    # Define colors
    colors = {
        'Correct read': '#5cb85c',
        'Failed detection': '#e57373',
        'Misread': '#ff9800',
        'Successful impersonation': '#ffd54f'
    }

    # Ensure all categories present
    all_categories = ['Correct read', 'Successful impersonation', 'Misread', 'Failed detection']
    for cat in all_categories:
        if cat not in counts.index:
            counts[cat] = 0

    counts = counts.reindex(all_categories, fill_value=0)
    counts_nonzero = counts[counts > 0]
    color_list = [colors[cat] for cat in counts_nonzero.index]

    # Create pie chart
    fig, ax = plt.subplots(figsize=(10, 8))

    wedges, texts, autotexts = ax.pie(
        counts_nonzero.values,
        labels=counts_nonzero.index,
        autopct='%1.1f%%',
        colors=color_list,
        startangle=90,
        textprops={'fontsize': 12, 'weight': 'bold'}
    )

    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontsize(13)
        autotext.set_weight('bold')

    # Title
    mode = "IMPERSONATION" if args.impersonation_target else "DISRUPTION"
    ax.set_title(f'{mode} Patch Evaluation - Detection Outcomes',
                fontsize=16, weight='bold', pad=20)

    plt.text(0.5, 0.95, f'(n={len(results_df)} control images evaluated)',
            ha='center', transform=fig.transFigure, fontsize=11, style='italic')

    # Save
    plt.tight_layout()
    plt.savefig(args.output, dpi=300, bbox_inches='tight')
    print(f"\n✓ Pie chart saved to: {args.output}")

    # Print summary
    print("\n" + "="*70)
    print(f"PATCH EVALUATION SUMMARY: {mode} MODE")
    print("="*70)
    print(f"Patch: {args.patch}")
    print(f"True plate: {args.true_plate}")
    if args.impersonation_target:
        print(f"Impersonation target: {args.impersonation_target}")
    print(f"Total images evaluated: {len(results_df)}")
    print("-" * 70)

    for cat in all_categories:
        count = counts[cat]
        pct = count / len(results_df) * 100
        print(f"  {cat:30s}: {count:4d} ({pct:5.1f}%)")

    print("="*70)


if __name__ == '__main__':
    main()
