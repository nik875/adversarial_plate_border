#!/usr/bin/env python3
"""
Crop license plate regions from Indian Plates Kaggle dataset based on YOLO annotations.
Creates a new dataset of cropped license plate images.
"""

from pathlib import Path
from PIL import Image
from tqdm import tqdm

# Configuration
INDIAN_PLATES_DIR = Path.home() / ".cache" / "indian_plates_kaggle"
OUTPUT_DIR = Path.home() / ".cache" / "indian_plates_kaggle_crops"

# Cropping options
PADDING = 5  # pixels to pad around bounding box


def parse_yolo_annotation(label_path):
    """
    Parse YOLO format annotation file.
    Format: class x_center y_center width height (normalized 0-1)
    """
    try:
        with open(label_path, 'r') as f:
            line = f.readline().strip()
            if not line:
                return None

            parts = line.split()
            if len(parts) < 5:
                return None

            # class_id = int(parts[0])  # We ignore the class
            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])

            return {
                'x_center': x_center,
                'y_center': y_center,
                'width': width,
                'height': height,
            }
    except Exception as e:
        return None


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


def process_indian_plates(output_dir=OUTPUT_DIR, indian_plates_dir=INDIAN_PLATES_DIR, max_crops=None):
    """Process Indian Plates Kaggle dataset and crop license plate regions."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create label file
    labels_file = output_dir / "labels.txt"
    crop_count = 0

    # Tracking counters
    counters = {
        'total': 0,
        'invalid_bbox': 0,
        'label_file_not_found': 0,
        'image_file_not_found': 0,
        'crop_error': 0,
        'success': 0,
    }

    print(f"Processing Indian Plates Kaggle dataset...")
    print(f"  Output directory: {output_dir}")
    print(f"  Dataset directory: {indian_plates_dir}")
    print()

    images_dir = indian_plates_dir / "images"
    labels_dir = indian_plates_dir / "labels"

    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")

    if not labels_dir.exists():
        raise FileNotFoundError(f"Labels directory not found: {labels_dir}")

    # Get all image files
    img_files = sorted(images_dir.glob("*.png"))
    print(f"Found {len(img_files)} image files")
    print()

    with open(labels_file, 'w') as labels:
        for img_path in tqdm(img_files, desc="Processing images"):
            if max_crops and crop_count >= max_crops:
                break

            # Find corresponding label file
            label_filename = img_path.stem + ".txt"
            label_path = labels_dir / label_filename

            if not label_path.exists():
                counters['label_file_not_found'] += 1
                continue

            # Parse annotation
            norm_bbox = parse_yolo_annotation(label_path)
            if norm_bbox is None:
                counters['invalid_bbox'] += 1
                continue

            counters['total'] += 1

            # Load image to get dimensions
            try:
                img = Image.open(img_path)
                img_width, img_height = img.size
            except Exception as e:
                counters['image_file_not_found'] += 1
                continue

            # Convert normalized to pixel coordinates
            bbox = normalize_to_pixel_coords(img_width, img_height, norm_bbox)

            # Crop image
            crop = crop_image_region(img_path, bbox)
            if crop is None:
                counters['crop_error'] += 1
                continue

            # Save crop
            crop_filename = f"indian_plates_{crop_count:06d}.png"
            crop_path = output_dir / crop_filename
            crop.save(crop_path)

            # Write label
            labels.write(f"{crop_filename} dataset=indian_plates\n")

            crop_count += 1
            counters['success'] += 1

    # Print summary
    print(f"\n{'='*60}")
    print(f"CROP SUMMARY")
    print(f"{'='*60}")
    print(f"Total annotations processed:      {counters['total']:>8}")
    print(f"  ✓ Successfully cropped:         {counters['success']:>8}")
    print(f"  ✗ Skipped:")
    print(f"    - Invalid bbox:               {counters['invalid_bbox']:>8}")
    print(f"    - Label file not found:       {counters['label_file_not_found']:>8}")
    print(f"    - Image file not found:       {counters['image_file_not_found']:>8}")
    print(f"    - Crop error:                 {counters['crop_error']:>8}")
    print(f"{'='*60}")
    print(f"✓ Created {crop_count} cropped license plate images")
    print(f"✓ Labels saved to {labels_file}")
    return crop_count


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Crop Indian Plates Kaggle license plate images")
    parser.add_argument("--indian-plates-dir", type=Path, default=INDIAN_PLATES_DIR, help="Indian Plates Kaggle dataset directory")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR, help="Output directory")
    parser.add_argument("--max-crops", type=int, help="Maximum number of crops to create")

    args = parser.parse_args()

    process_indian_plates(
        output_dir=args.output_dir,
        indian_plates_dir=args.indian_plates_dir,
        max_crops=args.max_crops,
    )
