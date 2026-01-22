#!/usr/bin/env python3
"""
Crop license plate regions from Kaggle LP Detection dataset based on XML annotations.
Creates a new dataset of cropped license plate images.
"""

import xml.etree.ElementTree as ET
from pathlib import Path
from PIL import Image
from tqdm import tqdm

# Configuration
KAGGLE_LP_DIR = Path.home() / ".cache" / "kaggle_lp_detection"
OUTPUT_DIR = Path.home() / ".cache" / "kaggle_lp_crops"

# Cropping options
PADDING = 5  # pixels to pad around bounding box


def parse_xml_annotation(xml_path):
    """Parse Pascal VOC XML annotation file."""
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # Extract image filename and size
        filename = root.find('filename').text
        width = int(root.find('size/width').text)
        height = int(root.find('size/height').text)

        # Extract all objects (license plates)
        objects = []
        for obj in root.findall('object'):
            name = obj.find('name').text
            if name.lower() not in ['licence', 'license', 'plate']:
                continue

            bndbox = obj.find('bndbox')
            xmin = int(bndbox.find('xmin').text)
            ymin = int(bndbox.find('ymin').text)
            xmax = int(bndbox.find('xmax').text)
            ymax = int(bndbox.find('ymax').text)

            objects.append({
                'xmin': xmin,
                'ymin': ymin,
                'xmax': xmax,
                'ymax': ymax,
            })

        return filename, width, height, objects
    except Exception as e:
        return None, None, None, None


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

    # Crop
    crop = img.crop((x1, y1, x2, y2))
    return crop


def process_kaggle_lp(output_dir=OUTPUT_DIR, kaggle_dir=KAGGLE_LP_DIR, max_crops=None):
    """Process Kaggle LP Detection dataset and crop license plate regions."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create label file
    labels_file = output_dir / "labels.txt"
    crop_count = 0

    # Tracking counters
    counters = {
        'total': 0,
        'invalid_bbox': 0,
        'image_file_not_found': 0,
        'crop_error': 0,
        'success': 0,
    }

    print(f"Processing Kaggle LP Detection dataset...")
    print(f"  Output directory: {output_dir}")
    print(f"  Kaggle directory: {kaggle_dir}")
    print()

    annotations_dir = kaggle_dir / "annotations"
    images_dir = kaggle_dir / "images"

    if not annotations_dir.exists():
        raise FileNotFoundError(f"Annotations directory not found: {annotations_dir}")

    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")

    # Get all XML annotation files
    xml_files = sorted(annotations_dir.glob("*.xml"))
    print(f"Found {len(xml_files)} annotation files")
    print()

    with open(labels_file, 'w') as labels:
        for xml_path in tqdm(xml_files, desc="Processing annotations"):
            if max_crops and crop_count >= max_crops:
                break

            # Parse XML annotation
            img_filename, width, height, objects = parse_xml_annotation(xml_path)

            if img_filename is None:
                counters['invalid_bbox'] += 1
                continue

            # Find image file
            img_path = images_dir / img_filename

            if not img_path.exists():
                counters['image_file_not_found'] += 1
                continue

            # Process each license plate in the image
            for bbox_dict in objects:
                if max_crops and crop_count >= max_crops:
                    break

                bbox = [bbox_dict['xmin'], bbox_dict['ymin'],
                       bbox_dict['xmax'], bbox_dict['ymax']]

                # Validate bbox
                if bbox[0] >= bbox[2] or bbox[1] >= bbox[3]:
                    counters['invalid_bbox'] += 1
                    continue

                counters['total'] += 1

                # Crop image
                crop = crop_image_region(img_path, bbox)
                if crop is None:
                    counters['crop_error'] += 1
                    continue

                # Save crop
                crop_filename = f"kaggle_lp_{crop_count:06d}.png"
                crop_path = output_dir / crop_filename
                crop.save(crop_path)

                # Write label
                labels.write(f"{crop_filename} dataset=kaggle_lp\n")

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
    print(f"    - Image file not found:       {counters['image_file_not_found']:>8}")
    print(f"    - Crop error:                 {counters['crop_error']:>8}")
    print(f"{'='*60}")
    print(f"✓ Created {crop_count} cropped license plate images")
    print(f"✓ Labels saved to {labels_file}")
    return crop_count


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Crop Kaggle LP Detection license plate images")
    parser.add_argument("--kaggle-dir", type=Path, default=KAGGLE_LP_DIR, help="Kaggle LP Detection dataset directory")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR, help="Output directory")
    parser.add_argument("--max-crops", type=int, help="Maximum number of crops to create")

    args = parser.parse_args()

    process_kaggle_lp(
        output_dir=args.output_dir,
        kaggle_dir=args.kaggle_dir,
        max_crops=args.max_crops,
    )
