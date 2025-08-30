#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Simple test script to visualize corner quadrilateral and ALPR bounding box
on a processed image from the output directory.
"""

import os
import csv
import cv2
import argparse
import numpy as np


def load_csv_data(csv_path):
    """Load CSV data and return as list of dictionaries"""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        raise ValueError("CSV file is empty")

    return rows


def draw_annotations(image, row):
    """Draw corner quadrilateral and ALPR bounding box on image"""
    # Make a copy to avoid modifying original
    annotated = image.copy()

    # Extract corner points
    try:
        corners = [
            (int(float(row['p1_x'])), int(float(row['p1_y']))),
            (int(float(row['p2_x'])), int(float(row['p2_y']))),
            (int(float(row['p3_x'])), int(float(row['p3_y']))),
            (int(float(row['p4_x'])), int(float(row['p4_y'])))
        ]

        # Draw corner quadrilateral in green
        quad_points = np.array(corners, dtype=np.int32)
        cv2.polylines(annotated, [quad_points], True, (0, 255, 0), 2)

        # Label each corner
        for i, (x, y) in enumerate(corners):
            cv2.circle(annotated, (x, y), 4, (0, 255, 0), -1)
            cv2.putText(annotated, f"P{i+1}", (x + 8, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        print(f"Corner quadrilateral: {corners}")

    except (KeyError, ValueError) as e:
        print(f"Error parsing corner coordinates: {e}")

    # Extract ALPR bounding box
    try:
        alpr_x1 = int(float(row['alpr_x1']))
        alpr_y1 = int(float(row['alpr_y1']))
        alpr_x2 = int(float(row['alpr_x2']))
        alpr_y2 = int(float(row['alpr_y2']))
        alpr_text = row.get('alpr_text', '')
        alpr_conf = float(row.get('alpr_conf', 0))

        # Only draw if valid detection (confidence > 0 and coordinates >= 0)
        if alpr_conf > 0 and alpr_x1 >= 0 and alpr_y1 >= 0:
            # Draw ALPR bounding box in red
            cv2.rectangle(annotated, (alpr_x1, alpr_y1), (alpr_x2, alpr_y2), (0, 0, 255), 2)

            # Add text label
            label = f"ALPR: {alpr_text} ({alpr_conf:.2f})"
            label_y = max(alpr_y1 - 10, 20)  # Position above box, or at top if too high
            cv2.putText(annotated, label, (alpr_x1, label_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            print(f"ALPR bbox: ({alpr_x1}, {alpr_y1}) to ({alpr_x2}, {alpr_y2})")
            print(f"ALPR text: '{alpr_text}' (confidence: {alpr_conf:.3f})")
        else:
            print("No valid ALPR detection to display")

    except (KeyError, ValueError) as e:
        print(f"Error parsing ALPR coordinates: {e}")

    # Add legend
    legend_y = 30
    cv2.putText(annotated, "Green: Corner Quadrilateral", (10, legend_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(annotated, "Red: ALPR Bounding Box", (10, legend_y + 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    return annotated


def main():
    parser = argparse.ArgumentParser(
        description="Visualize corner quadrilateral and ALPR bounding box")
    parser.add_argument("--csv", default="preproc_labels.csv", help="CSV file with processed data")
    parser.add_argument("--output-dir", default="preprocessed_images",
                        help="Directory containing processed images")
    parser.add_argument("--output", default="test.png", help="Output image file")
    parser.add_argument(
        "--index",
        type=int,
        default=0,
        help="Index of image to visualize (default: 0)")
    args = parser.parse_args()

    print(f"Loading CSV data from: {args.csv}")

    # Load CSV data
    try:
        rows = load_csv_data(args.csv)
        print(f"Found {len(rows)} rows in CSV")
    except Exception as e:
        print(f"ERROR: {e}")
        return 1

    # Select image to visualize
    if args.index >= len(rows):
        print(f"ERROR: Index {args.index} out of range (0-{len(rows)-1})")
        return 1

    row = rows[args.index]
    processed_filename = row.get('processed_filename', '')

    if not processed_filename:
        print("ERROR: No processed_filename found in CSV row")
        return 1

    # Check if file exists
    if not os.path.exists(processed_filename):
        print(f"ERROR: Processed image file not found: {processed_filename}")
        print(f"Looking in output directory: {args.output_dir}")

        # Try finding it in the output directory by basename
        basename = os.path.basename(processed_filename)
        alt_path = os.path.join(args.output_dir, basename)
        if os.path.exists(alt_path):
            processed_filename = alt_path
            print(f"Found alternative path: {alt_path}")
        else:
            print(f"Alternative path also not found: {alt_path}")
            return 1

    print(f"Loading image: {processed_filename}")

    # Load image
    try:
        image = cv2.imread(processed_filename)
        if image is None:
            raise ValueError("Failed to load image")
        print(f"Image dimensions: {image.shape[1]}x{image.shape[0]}")
    except Exception as e:
        print(f"ERROR loading image: {e}")
        return 1

    # Draw annotations
    try:
        annotated_image = draw_annotations(image, row)
    except Exception as e:
        print(f"ERROR drawing annotations: {e}")
        return 1

    # Save result
    try:
        success = cv2.imwrite(args.output, annotated_image)
        if not success:
            raise ValueError("cv2.imwrite returned False")
        print(f"✓ Saved annotated image to: {args.output}")
    except Exception as e:
        print(f"ERROR saving output image: {e}")
        return 1

    # Print summary
    original_filename = row.get('original_filename', 'unknown')
    print(f"\nSummary:")
    print(f"  Original file: {os.path.basename(original_filename)}")
    print(f"  Processed file: {os.path.basename(processed_filename)}")
    print(f"  Output: {args.output}")

    return 0


if __name__ == "__main__":
    exit(main())
