#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Process images from CSV with plate-aware square cropping and ALPR verification.
Includes built-in validation to drop invalid rows and handle duplicates.

Takes a CSV with image filenames and corner coordinates, crops to square while
preserving license plate area, scales to 384x384, and runs ALPR verification.
"""

import os
import cv2
import csv
import argparse
import numpy as np
from PIL import Image
from pathlib import Path

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


def validate_row_data(row, row_index):
    """
    Validate a single CSV row for required fields and data integrity.

    Args:
        row: Dictionary representing CSV row
        row_index: Row number for error reporting

    Returns:
        (is_valid, error_messages)
    """
    errors = []

    # Check required fields exist
    required_fields = ['filename', 'p1_x', 'p1_y', 'p2_x', 'p2_y',
                       'p3_x', 'p3_y', 'p4_x', 'p4_y',
                       'H00', 'H01', 'H02', 'H10', 'H11', 'H12', 'H20', 'H21', 'H22']

    missing_fields = [
        field for field in required_fields if field not in row or not row[field].strip()]
    if missing_fields:
        errors.append(f"Missing required fields: {missing_fields}")
        return False, errors

    try:
        # Validate corner coordinates
        corners = []
        for i in range(1, 5):
            try:
                x = float(row[f'p{i}_x'])
                y = float(row[f'p{i}_y'])

                if not np.isfinite(x) or not np.isfinite(y):
                    errors.append(f"Invalid coordinate p{i}: ({x}, {y}) - not finite")
                    continue

                if x < 0 or y < 0:
                    errors.append(f"Negative coordinate p{i}: ({x}, {y})")

                corners.append((x, y))
            except (ValueError, TypeError) as e:
                errors.append(f"Cannot parse coordinate p{i}: {e}")

        if len(corners) == 4:
            # Check if corners form a valid quadrilateral (not degenerate)
            xs = [c[0] for c in corners]
            ys = [c[1] for c in corners]

            bbox_width = max(xs) - min(xs)
            bbox_height = max(ys) - min(ys)

            if bbox_width <= 0 or bbox_height <= 0:
                errors.append(f"Degenerate bounding box: {bbox_width}x{bbox_height}")
            elif bbox_width < 5 or bbox_height < 5:
                errors.append(f"Bounding box too small: {bbox_width:.1f}x{bbox_height:.1f}")

            # Check for reasonable coordinate ranges (assuming typical image sizes)
            if max(max(xs), max(ys)) > 10000:
                errors.append(f"Coordinates suspiciously large: max_x={max(xs)}, max_y={max(ys)}")

        # Validate homography matrix
        H_values = []
        for i in range(3):
            for j in range(3):
                try:
                    val = float(row[f'H{i}{j}'])
                    if not np.isfinite(val):
                        errors.append(f"Invalid homography value H{i}{j}: {val}")
                    H_values.append(val)
                except (ValueError, TypeError) as e:
                    errors.append(f"Cannot parse homography H{i}{j}: {e}")

        if len(H_values) == 9:
            H = np.array(H_values).reshape(3, 3)
            try:
                # Check if homography is invertible (determinant != 0)
                det = np.linalg.det(H)
                if abs(det) < 1e-10:
                    errors.append(f"Singular homography matrix (det={det})")
            except np.linalg.LinAlgError:
                errors.append("Cannot compute homography determinant")

    except Exception as e:
        errors.append(f"Unexpected validation error: {e}")

    # Check if image file exists
    filename = row['filename']
    if not os.path.exists(filename):
        errors.append(f"Image file not found: {filename}")

    return len(errors) == 0, errors


def validate_and_filter_csv_data(rows):
    """
    Validate CSV data and filter out invalid rows and duplicates.

    Args:
        rows: List of dictionaries from CSV reader

    Returns:
        (filtered_rows, validation_report)
    """
    print("\n" + "=" * 60)
    print("VALIDATING INPUT DATA")
    print("=" * 60)

    # Track filenames to detect duplicates
    seen_filenames = {}
    valid_rows = []
    validation_issues = []

    for i, row in enumerate(rows):
        filename = row.get('filename', 'UNKNOWN')

        # Check for duplicates
        if filename in seen_filenames:
            validation_issues.append({
                'row': i,
                'filename': filename,
                'issues': [f"Duplicate filename (first seen at row {seen_filenames[filename]})"],
                'action': 'DROPPED'
            })
            continue

        # Validate row data
        is_valid, errors = validate_row_data(row, i)

        if is_valid:
            seen_filenames[filename] = i
            valid_rows.append(row)
        else:
            validation_issues.append({
                'row': i,
                'filename': filename,
                'issues': errors,
                'action': 'DROPPED'
            })

    # Report validation results
    print(f"Input validation complete:")
    print(f"  Total input rows: {len(rows)}")
    print(f"  Valid rows: {len(valid_rows)}")
    print(f"  Dropped rows: {len(validation_issues)}")

    if validation_issues:
        duplicate_issues = sum(
            1 for issue in validation_issues if 'Duplicate filename' in str(
                issue['issues']))
        coordinate_issues = len(validation_issues) - duplicate_issues
        print(f"    - Duplicates: {duplicate_issues}")
        print(f"    - Invalid data: {coordinate_issues}")

        # Show first few problematic rows
        print(f"\nFirst {min(5, len(validation_issues))} dropped rows:")
        for issue in validation_issues[:5]:
            print(f"  Row {issue['row']}: {Path(issue['filename']).name}")
            for error in issue['issues'][:3]:  # Show first 3 errors
                print(f"    - {error}")
            if len(issue['issues']) > 3:
                print(f"    - ... and {len(issue['issues']) - 3} more issues")

        if len(validation_issues) > 5:
            print(f"  ... and {len(validation_issues) - 5} more dropped rows")

    # Save validation report if there were issues
    if validation_issues:
        report_path = 'preprocessing_validation_report.csv'
        with open(report_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['row', 'filename', 'action', 'issues'])
            writer.writeheader()
            for issue in validation_issues:
                writer.writerow({
                    'row': issue['row'],
                    'filename': issue['filename'],
                    'action': issue['action'],
                    'issues': '; '.join(issue['issues'])
                })
        print(f"\nValidation report saved to: {report_path}")

    if len(valid_rows) == 0:
        raise RuntimeError("No valid rows remaining after validation!")

    print("=" * 60)

    return valid_rows, {
        'total_input': len(rows),
        'valid_rows': len(valid_rows),
        'dropped_rows': len(validation_issues),
        'issues': validation_issues
    }


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


def preprocess(img: np.ndarray, img_size: tuple[int, int]
               | int, corners: list, homography: np.ndarray):
    """
    Preprocess the input image for model inference.

    :param img: Input image in BGR format.
    :param img_size: Desired size to resize the image.
    :return: Preprocessed image tensor, resize ratio, and padding (dw, dh).
    """
    # Resize the input image to match training format
    im, corners, homography = letterbox(
        img, corners, homography, new_shape=img_size)
    # HWC to CHW, BGR to RGB
    im = im.transpose((2, 0, 1))[::-1]
    # 0 - 255 to 0.0 - 1.0
    im = im / 255.0
    # Model precision is FP32
    im = im.astype(np.float32)
    # Add batch dimension
    im = np.expand_dims(im, 0)
    return im, corners, homography


def letterbox(
    im: np.ndarray,
    corners: list,
    homography: np.ndarray,
    new_shape: tuple[int, int] | int = (640, 640),
    color: tuple[int, int, int] = (114, 114, 114),
    scaleup: bool = True,
) -> tuple[np.ndarray, tuple[float, float], tuple[float, float]]:
    """
    Simplified letterbox function with fixed behavior for YOLOv9 preprocessing.

    Resizes and pads the input image to the desired size while maintaining aspect ratio.
    """
    shape = im.shape[:2]  # current shape [height, width]

    # Convert integer new_shape to a tuple (new_shape, new_shape)
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Calculate the scaling ratio and resize the image
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:
        r = min(r, 1.0)

    # Calculate new unpadded dimensions and padding
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw = (new_shape[1] - new_unpad[0]) / 2  # divide padding into 2 sides
    dh = (new_shape[0] - new_unpad[1]) / 2

    corners *= r
    corners[[0, 2]] += dw
    corners[[1, 3]] += dh

    T = np.array([
        [r, 0, dw],
        [0, r, dh],
        [0, 0, 1]
    ])
    homography = homography @ T

    # Resize the image to the new unpadded dimensions
    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)

    # Add padding to maintain the new shape with the specified color
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(
        im,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=color)  # add border

    return im, corners, homography


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
        description="Process images from CSV with plate-aware square cropping and validation")
    parser.add_argument("--csv", required=True, help="Input CSV file")
    parser.add_argument(
        "--output-dir",
        default="preprocessed_images",
        help="Output directory for images")
    parser.add_argument("--output-csv", required=True, help="Output CSV file")
    parser.add_argument("--target-size", type=int, default=384,
                        help="Target image size (default: 384)")
    parser.add_argument("--skip-validation", action="store_true",
                        help="Skip input data validation (not recommended)")
    parser.add_argument("--min-alpr-confidence", type=float, default=0.2,
                        help="Minimum ALPR confidence to keep row (default: 0.2)")
    parser.add_argument("--target-plate-text", type=str, default="VRJ7774",
                        help="Only keep rows with this exact ALPR text (default: VRJ7774)")
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Read input CSV
    print(f"Reading input CSV: {args.csv}")
    with open(args.csv, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    print(f"Found {len(rows)} total rows in CSV")

    # Validate and filter data unless skipped
    if not args.skip_validation:
        rows, validation_report = validate_and_filter_csv_data(rows)
        print(f"Proceeding with {len(rows)} validated rows")
    else:
        print("WARNING: Skipping validation - this may cause processing failures!")

    # Initialize ALPR
    print("\nInitializing ALPR...")
    alpr = ALPR(
        detector_model="yolo-v9-t-384-license-plate-end2end",
        ocr_model="cct-xs-v1-global-model",
    )
    print("✓ ALPR ready")

    # Process each image
    output_rows = []
    processing_failures = []

    print(f"\n" + "=" * 60)
    print("PROCESSING IMAGES")
    print("=" * 60)

    for i, row in enumerate(rows, 1):
        filename = row['filename']
        base_name = os.path.splitext(os.path.basename(filename))[0]
        print(f"\nProcessing {i}/{len(rows)}: {base_name}")

        try:
            # Load image
            rgb, bgr = load_image_any(filename)
            print(f"  Loaded image: {bgr.shape[1]}x{bgr.shape[0]}")

            # Extract corner points from CSV
            corners = [
                (float(row['p1_x']), float(row['p1_y'])),
                (float(row['p2_x']), float(row['p2_y'])),
                (float(row['p3_x']), float(row['p3_y'])),
                (float(row['p4_x']), float(row['p4_y']))
            ]

            # Extract homography matrix
            H = np.array([
                [float(row['H00']), float(row['H01']), float(row['H02'])],
                [float(row['H10']), float(row['H11']), float(row['H12'])],
                [float(row['H20']), float(row['H21']), float(row['H22'])]
            ], dtype=np.float64)

            # Get plate bounding box for reference
            plate_bbox = get_plate_bbox(corners)
            print(
                f"  Original plate bbox: ({plate_bbox[0]:.0f}, {plate_bbox[1]:.0f}) to ({plate_bbox[2]:.0f}, {plate_bbox[3]:.0f})")

            # Process image (crop to square and scale)
            processed_bgr, new_corners, new_H = preprocess(
                bgr, args.target_size, corners, H)
            print(f"  Processed to: {processed_bgr.shape[1]}x{processed_bgr.shape[0]}")

            new_bbox = get_plate_bbox(new_corners)
            print(
                f"  New plate bbox: ({new_bbox[0]:.0f}, {new_bbox[1]:.0f}) to ({new_bbox[2]:.0f}, {new_bbox[3]:.0f})")

            # Run ALPR on processed image to verify detection
            plates = detect_plates_bgr(alpr, processed_bgr)
            print(f"  ALPR detected {len(plates)} plates")
            if plates:
                best_plate = max(plates, key=lambda p: p['conf'])
                print(f"  Best plate: '{best_plate['text']}' (conf={best_plate['conf']:.3f})")

            # Save processed image
            output_filename = f"{base_name}_processed.png"
            output_path = os.path.join(args.output_dir, output_filename)
            success = cv2.imwrite(output_path, processed_bgr)
            if not success:
                raise RuntimeError(f"Failed to save processed image to {output_path}")
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

        except Exception as e:
            error_msg = f"PROCESSING FAILED for {base_name}: {type(e).__name__}: {str(e)}"
            print(f"  {error_msg}")
            processing_failures.append({
                'filename': filename,
                'error': error_msg
            })

    # Write output CSV
    if output_rows:
        print(f"\n" + "=" * 60)
        print("FILTERING BY ALPR CONFIDENCE AND TEXT")
        print("=" * 60)

        # Filter rows by ALPR confidence and text match
        pre_filter_count = len(output_rows)
        filtered_rows = []
        low_confidence_rows = []
        wrong_text_rows = []

        target_text = args.target_plate_text.upper().strip()

        for row in output_rows:
            detected_text = row['alpr_text'].upper().strip()

            # Check confidence first
            if row['alpr_conf'] < args.min_alpr_confidence:
                low_confidence_rows.append({
                    'filename': Path(row['original_filename']).name,
                    'confidence': row['alpr_conf'],
                    'text': row['alpr_text'],
                    'reason': 'Low confidence'
                })
                continue

            # Check text match
            if detected_text != target_text:
                wrong_text_rows.append({
                    'filename': Path(row['original_filename']).name,
                    'confidence': row['alpr_conf'],
                    'text': row['alpr_text'],
                    'expected': target_text,
                    'reason': 'Wrong text'
                })
                continue

            # Passed both filters
            filtered_rows.append(row)

        print(
            f"ALPR filtering (min confidence: {args.min_alpr_confidence}, target text: '{target_text}'):")
        print(f"  Rows before filtering: {pre_filter_count}")
        print(f"  Rows after filtering: {len(filtered_rows)}")
        print(f"  Dropped low confidence: {len(low_confidence_rows)}")
        print(f"  Dropped wrong text: {len(wrong_text_rows)}")
        print(f"  Total dropped: {len(low_confidence_rows) + len(wrong_text_rows)}")

        # Show dropped samples
        all_dropped = low_confidence_rows + wrong_text_rows
        if all_dropped:
            print(f"\nFirst {min(10, len(all_dropped))} dropped rows:")
            for row in all_dropped[:10]:
                if row['reason'] == 'Low confidence':
                    print(
                        f"  - {row['filename']}: conf={row['confidence']:.3f}, text='{row['text']}' (LOW CONFIDENCE)")
                else:
                    print(
                        f"  - {row['filename']}: conf={row['confidence']:.3f}, text='{row['text']}' ≠ '{row['expected']}' (WRONG TEXT)")
            if len(all_dropped) > 10:
                print(f"  ... and {len(all_dropped) - 10} more dropped rows")

            # Save combined filtering report
            filter_report_path = 'preprocessing_filtering_report.csv'
            with open(filter_report_path, 'w', newline='') as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        'filename',
                        'confidence',
                        'text',
                        'expected',
                        'reason'])
                writer.writeheader()
                for row in all_dropped:
                    if row['reason'] == 'Low confidence':
                        writer.writerow({
                            'filename': row['filename'],
                            'confidence': row['confidence'],
                            'text': row['text'],
                            'expected': target_text,
                            'reason': row['reason']
                        })
                    else:
                        writer.writerow(row)
            print(f"\nFiltering report saved to: {filter_report_path}")

        if not filtered_rows:
            raise RuntimeError(f"No rows remaining after ALPR filtering! "
                               f"(min confidence: {args.min_alpr_confidence}, target text: '{target_text}')")

        # Use filtered rows for output
        output_rows = filtered_rows

        print(f"\n" + "=" * 60)
        print("WRITING OUTPUT")
        print("=" * 60)
        print(f"Writing output CSV with {len(output_rows)} rows...")

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

        # Final summary statistics
        # All remaining rows have good ALPR confidence and correct text
        successful_alpr = len(output_rows)

        print(f"\n" + "=" * 60)
        print("FINAL SUMMARY")
        print("=" * 60)
        print(f"Input validation:")
        if not args.skip_validation:
            print(f"  Total input rows: {validation_report['total_input']}")
            print(f"  Dropped invalid/duplicate: {validation_report['dropped_rows']}")
            print(f"  Validated for processing: {validation_report['valid_rows']}")
        else:
            print(f"  Validation skipped - processed all {len(rows)} input rows")

        print(f"Image processing:")
        print(f"  Successfully processed: {pre_filter_count}")
        print(f"  Processing failures: {len(processing_failures)}")
        print(
            f"  Processing success rate: {pre_filter_count/(pre_filter_count+len(processing_failures))*100:.1f}%")

        print(f"ALPR filtering (≥{args.min_alpr_confidence} conf, text='{target_text}'):")
        print(f"  Rows after processing: {pre_filter_count}")
        print(f"  Dropped low confidence: {len(low_confidence_rows)}")
        print(f"  Dropped wrong text: {len(wrong_text_rows)}")
        print(f"  Final output rows: {len(output_rows)}")
        print(
            f"  Overall pipeline success: {len(output_rows)/validation_report.get('total_input', len(rows))*100:.1f}%")

        if processing_failures:
            print(f"\nProcessing failures:")
            for failure in processing_failures[:5]:
                print(f"  - {Path(failure['filename']).name}: {failure['error']}")
            if len(processing_failures) > 5:
                print(f"  ... and {len(processing_failures) - 5} more failures")

        print("=" * 60)

    else:
        raise RuntimeError("No images were successfully processed!")


if __name__ == "__main__":
    main()
