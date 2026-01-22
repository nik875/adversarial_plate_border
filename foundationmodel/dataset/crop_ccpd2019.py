#!/usr/bin/env python3
"""
Crop license plate regions from CCPD2019 dataset based on filename annotations.
Creates a new dataset of cropped license plate images.
"""

from pathlib import Path
from PIL import Image
from tqdm import tqdm
import re

# Configuration
CCPD2019_DIR = Path.home() / ".cache" / "CCPD2019"
OUTPUT_DIR = Path.home() / ".cache" / "ccpd2019_crops"

# Cropping options
PADDING = 5  # pixels to pad around bounding box


def parse_ccpd_filename(filename):
    """
    Parse CCPD2019 filename format.
    Format: area-tilt_degree-bbox-vertices-license_plate-brightness-blurriness.jpg
    Example: 025-95_113-154&383_386&473-386&473_177&454_154&383_363&402-0_0_22_27_27_33_16-37-15.jpg

    Returns:
        dict with parsed fields or None if parsing fails
    """
    try:
        # Remove .jpg extension
        name = filename.rsplit('.', 1)[0]

        # Split by dash (but be careful with the negative numbers potentially)
        parts = name.split('-')

        if len(parts) < 7:
            return None

        area = parts[0]
        tilt_degree = parts[1]
        bbox = parts[2]  # "154&383_386&473" format
        vertices = parts[3]
        license_plate = parts[4]
        brightness = parts[5]
        blurriness = parts[6]

        # Parse bounding box: "x1&y1_x2&y2"
        bbox_parts = bbox.split('_')
        if len(bbox_parts) != 2:
            return None

        top_left = bbox_parts[0].split('&')
        bottom_right = bbox_parts[1].split('&')

        if len(top_left) != 2 or len(bottom_right) != 2:
            return None

        x1 = int(top_left[0])
        y1 = int(top_left[1])
        x2 = int(bottom_right[0])
        y2 = int(bottom_right[1])

        return {
            'area': area,
            'tilt_degree': tilt_degree,
            'bbox': [x1, y1, x2, y2],
            'vertices': vertices,
            'license_plate': license_plate,
            'brightness': brightness,
            'blurriness': blurriness,
        }
    except Exception as e:
        return None


def crop_image_region(img_path, bbox, padding=PADDING):
    """Crop image to bounding box with padding."""
    try:
        img = Image.open(img_path).convert('RGB')
    except Exception as e:
        return None

    x1, y1, x2, y2 = bbox

    # Add padding
    x1_pad = max(0, int(x1 - padding))
    y1_pad = max(0, int(y1 - padding))
    x2_pad = min(img.width, int(x2 + padding))
    y2_pad = min(img.height, int(y2 + padding))

    # Validate crop region
    if x1_pad >= x2_pad or y1_pad >= y2_pad:
        return None

    # Crop
    crop = img.crop((x1_pad, y1_pad, x2_pad, y2_pad))
    return crop


def process_ccpd2019(output_dir=OUTPUT_DIR, ccpd_dir=CCPD2019_DIR, max_crops=None):
    """Process CCPD2019 dataset and crop license plate regions."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create label file
    labels_file = output_dir / "labels.txt"
    crop_count = 0

    # Tracking counters
    counters = {
        'total': 0,
        'parse_error': 0,
        'image_file_not_found': 0,
        'crop_error': 0,
        'success': 0,
    }

    print(f"Processing CCPD2019 dataset...")
    print(f"  Output directory: {output_dir}")
    print(f"  Dataset directory: {ccpd_dir}")
    print()

    if not ccpd_dir.exists():
        raise FileNotFoundError(f"CCPD2019 dataset not found: {ccpd_dir}")

    # Get all image files
    img_files = sorted(ccpd_dir.glob("*.jpg"))
    print(f"Found {len(img_files)} image files")
    print()

    with open(labels_file, 'w') as labels:
        for img_path in tqdm(img_files, desc="Processing images"):
            if max_crops and crop_count >= max_crops:
                break

            # Parse filename annotation
            annotation = parse_ccpd_filename(img_path.name)
            if annotation is None:
                counters['parse_error'] += 1
                continue

            counters['total'] += 1
            bbox = annotation['bbox']

            # Verify image exists
            if not img_path.exists():
                counters['image_file_not_found'] += 1
                continue

            # Crop image
            crop = crop_image_region(img_path, bbox)
            if crop is None:
                counters['crop_error'] += 1
                continue

            # Save crop
            crop_filename = f"ccpd2019_{crop_count:06d}.png"
            crop_path = output_dir / crop_filename
            crop.save(crop_path)

            # Write label with metadata
            # Format: filename brightness blurriness
            labels.write(f"{crop_filename} brightness={annotation['brightness']} blurriness={annotation['blurriness']}\n")

            crop_count += 1
            counters['success'] += 1

    # Print summary
    print(f"\n{'='*60}")
    print(f"CROP SUMMARY")
    print(f"{'='*60}")
    print(f"Total annotations processed:      {counters['total']:>8}")
    print(f"  ✓ Successfully cropped:         {counters['success']:>8}")
    print(f"  ✗ Skipped:")
    print(f"    - Parse error:                {counters['parse_error']:>8}")
    print(f"    - Image file not found:       {counters['image_file_not_found']:>8}")
    print(f"    - Crop error:                 {counters['crop_error']:>8}")
    print(f"{'='*60}")
    print(f"✓ Created {crop_count} cropped license plate images")
    print(f"✓ Labels saved to {labels_file}")
    return crop_count


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Crop CCPD2019 license plate images")
    parser.add_argument("--ccpd-dir", type=Path, default=CCPD2019_DIR, help="CCPD2019 dataset directory")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR, help="Output directory")
    parser.add_argument("--max-crops", type=int, help="Maximum number of crops to create")

    args = parser.parse_args()

    process_ccpd2019(
        output_dir=args.output_dir,
        ccpd_dir=args.ccpd_dir,
        max_crops=args.max_crops,
    )
