#!/usr/bin/env python3
"""
Crop license plate regions from CRPD dataset based on quadrilateral annotations.
Creates a new dataset of cropped license plate images from all variants and splits.
"""

from pathlib import Path
from PIL import Image
from tqdm import tqdm

# Configuration
CRPD_DIR = Path.home() / ".cache" / "CRPD"
OUTPUT_DIR = Path.home() / ".cache" / "crpd_crops"

# Cropping options
PADDING = 5  # pixels to pad around bounding box


def parse_crpd_annotation(label_path):
    """
    Parse CRPD annotation file.
    Format: x1 y1 x2 y2 x3 y3 x4 y4 type content
    where x1-y4 are corner points of a quadrilateral.
    """
    annotations = []
    try:
        with open(label_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                parts = line.split()
                if len(parts) < 10:
                    continue

                try:
                    # Parse corner points (quadrilateral)
                    x1, y1, x2, y2, x3, y3, x4, y4 = [int(p) for p in parts[:8]]
                    type_id = int(parts[8])
                    content = parts[9]  # License plate text

                    # Calculate bounding box from corner points
                    xs = [x1, x2, x3, x4]
                    ys = [y1, y2, y3, y4]
                    xmin, xmax = min(xs), max(xs)
                    ymin, ymax = min(ys), max(ys)

                    annotations.append({
                        'bbox': [xmin, ymin, xmax, ymax],
                        'type': type_id,
                        'content': content,
                    })
                except (ValueError, IndexError):
                    continue

        return annotations
    except Exception as e:
        return None


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


def process_crpd(output_dir=OUTPUT_DIR, crpd_dir=CRPD_DIR, max_crops=None):
    """Process CRPD dataset and crop license plate regions from all variants and splits."""

    if not crpd_dir.exists():
        raise FileNotFoundError(f"CRPD dataset not found: {crpd_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Processing CRPD dataset...")
    print(f"  Dataset directory: {crpd_dir}")
    print(f"  Output directory: {output_dir}")
    print()

    # Create single labels file for all crops
    labels_file = output_dir / "labels.txt"
    crop_count = 0

    # Tracking counters
    counters = {
        'total': 0,
        'image_file_not_found': 0,
        'label_parse_error': 0,
        'crop_error': 0,
        'success': 0,
    }

    # Get all variants
    variants = ['CRPD_multi', 'CRPD_single', 'CRPD_double']
    splits = ['train', 'test', 'val']

    with open(labels_file, 'w') as labels:
        for variant in variants:
            variant_dir = crpd_dir / variant

            if not variant_dir.exists():
                print(f"Warning: Variant directory not found: {variant_dir}")
                continue

            for split in splits:
                split_dir = variant_dir / split

                if not split_dir.exists():
                    continue

                images_dir = split_dir / "images"
                labels_dir = split_dir / "labels"

                if not images_dir.exists() or not labels_dir.exists():
                    continue

                # Get all images in this split
                img_files = sorted(images_dir.glob("*"))
                print(f"Processing {variant}/{split}: {len(img_files)} images")

                for img_path in tqdm(img_files, desc=f"  {variant}/{split}"):
                    if max_crops and crop_count >= max_crops:
                        break

                    # Find corresponding label file
                    label_filename = img_path.stem + ".txt"
                    label_path = labels_dir / label_filename

                    if not label_path.exists():
                        continue

                    # Parse annotations
                    annotations = parse_crpd_annotation(label_path)
                    if annotations is None:
                        counters['label_parse_error'] += 1
                        continue

                    if not annotations:
                        continue

                    # Verify image exists
                    if not img_path.exists():
                        counters['image_file_not_found'] += 1
                        continue

                    # Process each license plate in the image
                    for ann_idx, annotation in enumerate(annotations):
                        if max_crops and crop_count >= max_crops:
                            break

                        counters['total'] += 1
                        bbox = annotation['bbox']

                        # Crop image
                        crop = crop_image_region(img_path, bbox)
                        if crop is None:
                            counters['crop_error'] += 1
                            continue

                        # Save crop
                        crop_filename = f"crpd_{crop_count:06d}.png"
                        crop_path = output_dir / crop_filename
                        crop.save(crop_path)

                        # Write label with metadata
                        # Format: filename variant=X split=Y type=Z content=W
                        labels.write(
                            f"{crop_filename} variant={variant} split={split} "
                            f"type={annotation['type']} content={annotation['content']}\n"
                        )

                        crop_count += 1
                        counters['success'] += 1

    # Print summary
    print(f"\n{'='*60}")
    print(f"CROP SUMMARY")
    print(f"{'='*60}")
    print(f"Total annotations processed:      {counters['total']:>8}")
    print(f"  ✓ Successfully cropped:         {counters['success']:>8}")
    print(f"  ✗ Skipped:")
    print(f"    - Label parse error:          {counters['label_parse_error']:>8}")
    print(f"    - Image file not found:       {counters['image_file_not_found']:>8}")
    print(f"    - Crop error:                 {counters['crop_error']:>8}")
    print(f"{'='*60}")
    print(f"✓ Created {crop_count} cropped license plate images")
    print(f"✓ Labels saved to {labels_file}")
    return crop_count


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Crop CRPD license plate images")
    parser.add_argument("--crpd-dir", type=Path, default=CRPD_DIR, help="CRPD dataset directory")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR, help="Output directory")
    parser.add_argument("--max-crops", type=int, help="Maximum number of crops to create")

    args = parser.parse_args()

    process_crpd(
        output_dir=args.output_dir,
        crpd_dir=args.crpd_dir,
        max_crops=args.max_crops,
    )
