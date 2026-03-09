#!/usr/bin/env python3
"""
Crop license plate regions from Mercosur dataset based on CSV annotations.
Creates a new dataset of cropped license plate images organized by image source class.
"""

from pathlib import Path
from PIL import Image
from tqdm import tqdm
import csv

# Configuration
MERCOSUR_DIR = Path.home() / ".cache" / "Mercosur"
OUTPUT_DIR = Path.home() / ".cache" / "mercosur_crops"

# Cropping options
PADDING = 5  # pixels to pad around bounding box


def get_image_class(filename):
    """Extract the image class from filename prefix."""
    if filename.startswith("monitoring_system_"):
        return "monitoring_system"
    elif filename.startswith("parking_lot1_"):
        return "parking_lot1"
    elif filename.startswith("parking_lot2_"):
        return "parking_lot2"
    elif filename.startswith("parking_lot3_"):
        return "parking_lot3"
    elif filename.startswith("cropped_parking_lot"):
        return "cropped_parking_lot"
    else:
        return "unknown"


def normalize_to_pixel_coords(img_width, img_height, norm_bbox):
    """Convert normalized YOLO coordinates to pixel coordinates."""
    x_center = norm_bbox['x_center'] * img_width
    y_center = norm_bbox['y_center'] * img_height
    width = norm_bbox['width'] * img_width
    height = norm_bbox['height'] * img_height

    # Convert center coordinates to top-left, bottom-right
    xmin = x_center - width / 2
    ymin = y_center - height / 2
    xmax = x_center + width / 2
    ymax = y_center + height / 2

    return [xmin, ymin, xmax, ymax]


def crop_image_region(img_path, bbox, padding=PADDING):
    """Crop image to bounding box with padding."""
    try:
        img = Image.open(img_path).convert('RGB')
    except Exception as e:
        return None

    xmin, ymin, xmax, ymax = bbox

    # Add padding
    x1 = max(0, int(xmin - padding))
    y1 = max(0, int(ymin - padding))
    x2 = min(img.width, int(xmax + padding))
    y2 = min(img.height, int(ymax + padding))

    # Validate crop region
    if x1 >= x2 or y1 >= y2:
        return None

    # Crop
    crop = img.crop((x1, y1, x2, y2))
    return crop


def process_mercosur(output_dir=OUTPUT_DIR, mercosur_dir=MERCOSUR_DIR, max_crops=None):
    """Process Mercosur dataset and crop license plate regions."""

    if not mercosur_dir.exists():
        raise FileNotFoundError(f"Mercosur dataset not found: {mercosur_dir}")

    csv_file = mercosur_dir / "dataset.csv"
    if not csv_file.exists():
        raise FileNotFoundError(f"dataset.csv not found: {csv_file}")

    images_dir = mercosur_dir / "images"
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Processing Mercosur dataset...")
    print(f"  Dataset directory: {mercosur_dir}")
    print(f"  Output directory: {output_dir}")
    print()

    # Create single labels file for all crops
    labels_file = output_dir / "labels.txt"
    crop_count = 0

    # Tracking counters
    counters = {
        'total': 0,
        'image_file_not_found': 0,
        'crop_error': 0,
        'success': 0,
    }

    # Read CSV
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        counters['total'] = len(rows)
        print(f"Found {len(rows)} annotations")
        print()

    # Process all rows
    with open(labels_file, 'w') as labels:
        for row in tqdm(rows, desc="Processing images"):
            if max_crops and crop_count >= max_crops:
                break

            img_filename = row['image']
            img_path = images_dir / img_filename

            if not img_path.exists():
                counters['image_file_not_found'] += 1
                continue

            # Get image class from filename
            img_class = get_image_class(img_filename)

            # Load image to get dimensions
            try:
                img = Image.open(img_path)
                img_width, img_height = img.size
            except Exception as e:
                counters['image_file_not_found'] += 1
                continue

            # Parse YOLO coordinates
            norm_bbox = {
                'x_center': float(row['x_center']),
                'y_center': float(row['y_center']),
                'width': float(row['width']),
                'height': float(row['height']),
            }

            # Convert to pixel coordinates
            bbox = normalize_to_pixel_coords(img_width, img_height, norm_bbox)

            # Crop image
            crop = crop_image_region(img_path, bbox)
            if crop is None:
                counters['crop_error'] += 1
                continue

            # Save crop
            crop_filename = f"mercosur_{crop_count:06d}.png"
            crop_path = output_dir / crop_filename
            crop.save(crop_path)

            # Write label with metadata
            # Format: filename source_class source_image
            labels.write(f"{crop_filename} source_class={img_class} source_image={img_filename}\n")

            crop_count += 1
            counters['success'] += 1

    # Print summary
    print(f"\n{'='*60}")
    print(f"CROP SUMMARY")
    print(f"{'='*60}")
    print(f"Total annotations processed:      {counters['total']:>8}")
    print(f"  ✓ Successfully cropped:         {counters['success']:>8}")
    print(f"  ✗ Skipped:")
    print(f"    - Image file not found:       {counters['image_file_not_found']:>8}")
    print(f"    - Crop error:                 {counters['crop_error']:>8}")
    print(f"{'='*60}")
    print(f"✓ Created {crop_count} cropped license plate images")
    print(f"✓ Labels saved to {labels_file}")
    return crop_count


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Crop Mercosur license plate images")
    parser.add_argument("--mercosur-dir", type=Path, default=MERCOSUR_DIR, help="Mercosur dataset directory")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR, help="Output directory")
    parser.add_argument("--max-crops", type=int, help="Maximum number of crops to create per class")

    args = parser.parse_args()

    process_mercosur(
        output_dir=args.output_dir,
        mercosur_dir=args.mercosur_dir,
        max_crops=args.max_crops,
    )
