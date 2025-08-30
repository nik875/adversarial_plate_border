#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Process images from CSV with plate-aware square cropping and ALPR verification.

Takes a CSV with image filenames and corner coordinates, crops to square while
preserving license plate area, scales to 384x384, and runs ALPR verification.
"""

import os
import cv2
import csv
import argparse
import numpy as np
from PIL import Image

# HEIC support
try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
    HEIC_OK = True
except Exception:
    HEIC_OK = False

# fast_alpr
try:
    from fast_alpr import ALPR
except ImportError as e:
    raise SystemExit("fast_alpr is required. Install with: pip install fast-alpr") from e


def load_image_any(path):
    """Load with Pillow (handles HEIC if opener registered), return RGB np.array and BGR np.array."""
    ext = os.path.splitext(path)[1].lower()
    if ext in (".heic", ".heif") and not HEIC_OK:
        raise RuntimeError("HEIC file but pillow-heif isn't installed. pip install pillow-heif")

    pil_img = Image.open(path).convert("RGB")
    rgb = np.array(pil_img)  # HxWx3 RGB
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    return rgb, bgr


def get_plate_bbox(corners):
    """Get bounding box of the 4 corner points"""
    xs = [p[0] for p in corners]
    ys = [p[1] for p in corners]
    return min(xs), min(ys), max(xs), max(ys)


def crop_to_square_preserving_plate(image, corners, target_size=384):
    """
    Crop image to square while preserving plate area, then scale to target_size.
    If cropping would cut into plate, pad with white instead.

    Returns: processed_image, transformation_matrix, (crop_x, crop_y, pad_left, pad_top)
    """
    h, w = image.shape[:2]

    # Get plate bounding box
    plate_x1, plate_y1, plate_x2, plate_y2 = get_plate_bbox(corners)
    plate_w = plate_x2 - plate_x1
    plate_h = plate_y2 - plate_y1

    # Find the largest square we can crop without cutting the plate
    min_size_for_plate = max(plate_w, plate_h) * 1.1  # Add 10% margin
    max_size_possible = min(w, h)

    best_crop = None

    # Try different square sizes, starting from largest possible
    for size in range(max_size_possible, int(min_size_for_plate) - 1, -1):
        # Try different center points
        centers_to_try = [
            (w // 2, h // 2),  # Image center
            ((plate_x1 + plate_x2) // 2, (plate_y1 + plate_y2) // 2),  # Plate center
        ]

        for center_x, center_y in centers_to_try:
            x1 = max(0, min(w - size, center_x - size // 2))
            y1 = max(0, min(h - size, center_y - size // 2))
            x2 = x1 + size
            y2 = y1 + size

            # Check if this crop contains the entire plate
            if (x1 <= plate_x1 and y1 <= plate_y1 and
                    x2 >= plate_x2 and y2 >= plate_y2):
                best_crop = (x1, y1, x2, y2, size)
                break

        if best_crop:
            break

    if best_crop:
        # We found a good crop - crop and scale
        x1, y1, x2, y2, size = best_crop
        cropped = image[y1:y2, x1:x2]
        scaled = cv2.resize(cropped, (target_size, target_size), interpolation=cv2.INTER_AREA)
        scale_factor = target_size / size

        # Transformation matrix: translate then scale
        T = np.array([
            [scale_factor, 0, -scale_factor * x1],
            [0, scale_factor, -scale_factor * y1],
            [0, 0, 1]
        ], dtype=np.float64)

        return scaled, T, (x1, y1, 0, 0)

    else:
        # Need to pad to make square - pad in the direction that preserves most content
        square_size = max(w, h)

        # Calculate required padding
        pad_w = square_size - w
        pad_h = square_size - h

        # Distribute padding to center the image, but bias toward keeping the plate centered
        if pad_w > 0:
            plate_center_x = (plate_x1 + plate_x2) / 2
            ideal_offset = plate_center_x - square_size / 2
            pad_left = max(0, min(pad_w, int(-ideal_offset)))
            pad_right = pad_w - pad_left
        else:
            pad_left = pad_right = 0

        if pad_h > 0:
            plate_center_y = (plate_y1 + plate_y2) / 2
            ideal_offset = plate_center_y - square_size / 2
            pad_top = max(0, min(pad_h, int(-ideal_offset)))
            pad_bottom = pad_h - pad_top
        else:
            pad_top = pad_bottom = 0

        # Apply padding with white
        padded = cv2.copyMakeBorder(image, pad_top, pad_bottom, pad_left, pad_right,
                                    cv2.BORDER_CONSTANT, value=(255, 255, 255))

        scaled = cv2.resize(padded, (target_size, target_size), interpolation=cv2.INTER_AREA)
        scale_factor = target_size / square_size

        # Transformation matrix: pad then scale
        T = np.array([
            [scale_factor, 0, scale_factor * pad_left],
            [0, scale_factor, scale_factor * pad_top],
            [0, 0, 1]
        ], dtype=np.float64)

        return scaled, T, (0, 0, pad_left, pad_top)


def transform_points(points, T):
    """Transform list of (x,y) points using transformation matrix T"""
    points_homo = np.array([[x, y, 1] for x, y in points]).T  # 3xN
    transformed = T @ points_homo  # 3xN
    return [(float(transformed[0, i]), float(transformed[1, i]))
            for i in range(transformed.shape[1])]


def transform_homography(H, T):
    """Transform homography matrix H by transformation T"""
    return T @ H


def detect_plates_bgr(alpr, bgr_img):
    """Return list of plate detections using fast_alpr"""
    preds = alpr.predict(bgr_img)
    results = []
    for p in preds or []:
        bb = p.detection.bounding_box
        ocr = p.ocr
        results.append({
            'text': (ocr.text or "").strip().upper(),
            'conf': float(ocr.confidence),
            'x1': int(bb.x1), 'y1': int(bb.y1),
            'x2': int(bb.x2), 'y2': int(bb.y2)
        })
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Process images from CSV with plate-aware square cropping")
    parser.add_argument("--csv", required=True, help="Input CSV file")
    parser.add_argument(
        "--output-dir",
        default="preprocessed_images",
        help="Output directory for images")
    parser.add_argument("--output-csv", required=True, help="Output CSV file")
    parser.add_argument("--target-size", type=int, default=384,
                        help="Target image size (default: 384)")
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize ALPR
    print("Initializing ALPR...")
    alpr = ALPR(
        detector_model="yolo-v9-t-384-license-plate-end2end",
        ocr_model="cct-xs-v1-global-model",
    )
    print("✓ ALPR ready")

    # Read input CSV
    print(f"Reading input CSV: {args.csv}")
    with open(args.csv, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    print(f"Found {len(rows)} rows to process")

    # Process each image
    output_rows = []

    for i, row in enumerate(rows, 1):
        filename = row['filename']
        base_name = os.path.splitext(os.path.basename(filename))[0]
        print(f"\nProcessing {i}/{len(rows)}: {base_name}")

        # Check if file exists
        if not os.path.exists(filename):
            print(f"ERROR: File not found: {filename}")
            continue

        # Load image
        try:
            rgb, bgr = load_image_any(filename)
            print(f"  Loaded image: {bgr.shape[1]}x{bgr.shape[0]}")
        except Exception as e:
            print(f"ERROR loading {filename}: {e}")
            continue

        # Extract corner points from CSV
        try:
            corners = [
                (float(row['p1_x']), float(row['p1_y'])),
                (float(row['p2_x']), float(row['p2_y'])),
                (float(row['p3_x']), float(row['p3_y'])),
                (float(row['p4_x']), float(row['p4_y']))
            ]
        except (KeyError, ValueError) as e:
            print(f"ERROR parsing corner coordinates: {e}")
            continue

        # Extract homography matrix
        try:
            H = np.array([
                [float(row['H00']), float(row['H01']), float(row['H02'])],
                [float(row['H10']), float(row['H11']), float(row['H12'])],
                [float(row['H20']), float(row['H21']), float(row['H22'])]
            ], dtype=np.float64)
        except (KeyError, ValueError) as e:
            print(f"ERROR parsing homography matrix: {e}")
            continue

        # Get plate bounding box for reference
        plate_bbox = get_plate_bbox(corners)
        print(
            f"  Original plate bbox: ({plate_bbox[0]:.0f}, {plate_bbox[1]:.0f}) to ({plate_bbox[2]:.0f}, {plate_bbox[3]:.0f})")

        # Process image (crop to square and scale)
        try:
            processed_bgr, T, transform_info = crop_to_square_preserving_plate(
                bgr, corners, args.target_size)
            print(f"  Processed to: {processed_bgr.shape[1]}x{processed_bgr.shape[0]}")
            print(
                f"  Transform info: crop=({transform_info[0]}, {transform_info[1]}), pad=({transform_info[2]}, {transform_info[3]})")
        except Exception as e:
            print(f"ERROR processing image: {e}")
            continue

        # Transform corners and homography to new coordinate system
        new_corners = transform_points(corners, T)
        new_H = transform_homography(H, T)

        new_bbox = get_plate_bbox(new_corners)
        print(
            f"  New plate bbox: ({new_bbox[0]:.0f}, {new_bbox[1]:.0f}) to ({new_bbox[2]:.0f}, {new_bbox[3]:.0f})")

        # Run ALPR on processed image to verify detection
        try:
            plates = detect_plates_bgr(alpr, processed_bgr)
            print(f"  ALPR detected {len(plates)} plates")
            if plates:
                best_plate = max(plates, key=lambda p: p['conf'])
                print(f"  Best plate: '{best_plate['text']}' (conf={best_plate['conf']:.3f})")
        except Exception as e:
            print(f"ERROR running ALPR: {e}")
            plates = []

        # Save processed image
        output_filename = f"{base_name}_processed.png"
        output_path = os.path.join(args.output_dir, output_filename)
        success = cv2.imwrite(output_path, processed_bgr)
        if not success:
            print(f"ERROR saving processed image to {output_path}")
            continue
        print(f"  Saved: {output_filename}")

        # Create output row with all transformed data
        output_row = {
            'original_filename': os.path.abspath(filename),
            'processed_filename': os.path.abspath(output_path),
            'p1_x': new_corners[0][0], 'p1_y': new_corners[0][1],
            'p2_x': new_corners[1][0], 'p2_y': new_corners[1][1],
            'p3_x': new_corners[2][0], 'p3_y': new_corners[2][1],
            'p4_x': new_corners[3][0], 'p4_y': new_corners[3][1],
            'H00': new_H[0, 0], 'H01': new_H[0, 1], 'H02': new_H[0, 2],
            'H10': new_H[1, 0], 'H11': new_H[1, 1], 'H12': new_H[1, 2],
            'H20': new_H[2, 0], 'H21': new_H[2, 1], 'H22': new_H[2, 2],
            'out_w': int(row.get('out_w', 600)),  # Use original or default
            'out_h': int(row.get('out_h', 400)),
            'processed_size': args.target_size
        }

        # Add ALPR verification results
        if plates:
            best_plate = max(plates, key=lambda p: p['conf'])
            output_row.update({
                'alpr_text': best_plate['text'],
                'alpr_conf': best_plate['conf'],
                'alpr_x1': best_plate['x1'],
                'alpr_y1': best_plate['y1'],
                'alpr_x2': best_plate['x2'],
                'alpr_y2': best_plate['y2']
            })
        else:
            output_row.update({
                'alpr_text': '',
                'alpr_conf': 0.0,
                'alpr_x1': -1, 'alpr_y1': -1,
                'alpr_x2': -1, 'alpr_y2': -1
            })

        output_rows.append(output_row)
        print(f"✓ Successfully processed {base_name}")

    # Write output CSV
    if output_rows:
        print(f"\nWriting output CSV with {len(output_rows)} rows...")
        fieldnames = [
            'original_filename', 'processed_filename',
            'p1_x', 'p1_y', 'p2_x', 'p2_y', 'p3_x', 'p3_y', 'p4_x', 'p4_y',
            'H00', 'H01', 'H02', 'H10', 'H11', 'H12', 'H20', 'H21', 'H22',
            'out_w', 'out_h', 'processed_size',
            'alpr_text', 'alpr_conf', 'alpr_x1', 'alpr_y1', 'alpr_x2', 'alpr_y2'
        ]

        with open(args.output_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(output_rows)

        print(f"✓ Output CSV written to: {args.output_csv}")
        print(f"✓ Processed images saved to: {args.output_dir}/")

        # Summary statistics
        successful_alpr = sum(1 for row in output_rows if row['alpr_conf'] > 0)
        print(f"\nSummary:")
        print(f"  Total images processed: {len(output_rows)}")
        print(f"  Successful ALPR detections: {successful_alpr}")
        print(f"  ALPR success rate: {successful_alpr/len(output_rows)*100:.1f}%")
    else:
        print("ERROR: No rows were successfully processed")


if __name__ == "__main__":
    main()
