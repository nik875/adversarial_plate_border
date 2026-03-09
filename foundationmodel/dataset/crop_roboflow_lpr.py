#!/usr/bin/env python3
"""
Crop license plate regions from Roboflow LPR dataset based on COCO annotations.
Creates a new dataset of cropped license plate images.
"""

import json
import os
from pathlib import Path
from PIL import Image
from tqdm import tqdm

# Configuration
ROBOFLOW_LPR_DIR = Path.home() / ".cache" / "roboflow_lpr_dataset"
OUTPUT_DIR = Path.home() / ".cache" / "roboflow_lpr_crops"

# Cropping options
PADDING = 5  # pixels to pad around bounding box


def load_coco_annotations(json_path):
    """Load COCO format annotations."""
    json_path = Path(json_path)
    if not json_path.exists():
        raise FileNotFoundError(f"Annotations JSON not found: {json_path}")

    with open(json_path, 'r') as f:
        data = json.load(f)

    return data


def crop_image_region(img_path, bbox, padding=PADDING):
    """Crop image to bounding box with padding."""
    try:
        img = Image.open(img_path).convert('RGB')
    except Exception as e:
        return None

    x, y, w, h = bbox

    # Add padding
    x1 = max(0, int(x - padding))
    y1 = max(0, int(y - padding))
    x2 = min(img.width, int(x + w + padding))
    y2 = min(img.height, int(y + h + padding))

    # Crop
    crop = img.crop((x1, y1, x2, y2))
    return crop


def process_roboflow_lpr(output_dir=OUTPUT_DIR, roboflow_dir=ROBOFLOW_LPR_DIR, max_crops=None):
    """Process Roboflow LPR dataset and crop license plate regions."""
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

    print(f"Processing Roboflow LPR dataset...")
    print(f"  Output directory: {output_dir}")
    print(f"  Roboflow directory: {roboflow_dir}")
    print()

    # Process each split (train, test, valid)
    splits = ['train', 'test', 'valid']

    with open(labels_file, 'w') as labels:
        for split in splits:
            split_dir = roboflow_dir / split

            if not split_dir.exists():
                print(f"Warning: Split directory not found: {split_dir}")
                continue

            anno_file = split_dir / "_annotations.coco.json"
            if not anno_file.exists():
                print(f"Warning: Annotations file not found: {anno_file}")
                continue

            print(f"Processing {split} split...")

            # Load annotations
            coco_data = load_coco_annotations(anno_file)
            images = {img['id']: img for img in coco_data.get('images', [])}
            annotations = coco_data.get('annotations', [])

            counters['total'] += len(annotations)

            for ann in tqdm(annotations, desc=f"  {split}"):
                if max_crops and crop_count >= max_crops:
                    break

                # Check bbox validity
                bbox = ann.get('bbox')
                if not bbox or len(bbox) != 4:
                    counters['invalid_bbox'] += 1
                    continue

                # Get image info
                image_id = ann['image_id']
                if image_id not in images:
                    counters['image_file_not_found'] += 1
                    continue

                img_info = images[image_id]
                img_filename = img_info['file_name']
                img_path = split_dir / img_filename

                if not img_path.exists():
                    counters['image_file_not_found'] += 1
                    continue

                # Crop image
                crop = crop_image_region(img_path, bbox)
                if crop is None:
                    counters['crop_error'] += 1
                    continue

                # Save crop
                crop_filename = f"roboflow_lpr_{crop_count:06d}.png"
                crop_path = output_dir / crop_filename
                crop.save(crop_path)

                # Write label (just the split, license plates don't have text labels)
                labels.write(f"{crop_filename} split={split}\n")

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

    parser = argparse.ArgumentParser(description="Crop Roboflow LPR license plate images")
    parser.add_argument("--roboflow-dir", type=Path, default=ROBOFLOW_LPR_DIR, help="Roboflow LPR dataset directory")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR, help="Output directory")
    parser.add_argument("--max-crops", type=int, help="Maximum number of crops to create")

    args = parser.parse_args()

    process_roboflow_lpr(
        output_dir=args.output_dir,
        roboflow_dir=args.roboflow_dir,
        max_crops=args.max_crops,
    )
