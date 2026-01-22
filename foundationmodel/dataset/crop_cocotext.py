#!/usr/bin/env python3
"""
Crop images from COCO dataset based on text annotations.
Creates a new dataset of cropped text regions from COCO Text.
"""

import json
import os
from pathlib import Path
from PIL import Image
import numpy as np
from tqdm import tqdm

# Configuration
COCOTEXT_JSON = Path(__file__).parent / "coco_text" / "cocotext.v2.json"
COCO_IMAGES_DIR = Path.home() / ".cache" / "coco" / "val2014"  # Adjust as needed
OUTPUT_DIR = Path.home() / ".cache" / "cocotext_crops"

# Cropping options
PADDING = 5  # pixels to pad around bounding box
MIN_WIDTH = 20  # minimum crop width
MIN_HEIGHT = 15  # minimum crop height
MAX_WIDTH = 500  # maximum crop width
MAX_HEIGHT = 200  # maximum crop height


def load_cocotext(json_path=COCOTEXT_JSON):
    """Load COCO Text annotations."""
    json_path = Path(json_path)
    if not json_path.exists():
        raise FileNotFoundError(f"COCO Text JSON not found: {json_path}")

    with open(json_path, 'r') as f:
        return json.load(f)


def is_valid_annotation(ann):
    """Check if annotation has a valid bounding box."""
    bbox = ann.get('bbox')
    if not bbox or len(bbox) != 4:
        return False, "invalid_bbox"
    return True, "valid"


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


def process_cocotext(output_dir=OUTPUT_DIR, coco_images_dir=COCO_IMAGES_DIR, max_crops=None, cocotext_json=COCOTEXT_JSON):
    """Process COCO Text and crop images."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load annotations
    data = load_cocotext(cocotext_json)
    anns = data['anns']
    imgs = data['imgs']
    img_to_anns = data['imgToAnns']

    # Create label file
    labels_file = output_dir / "labels.txt"
    crop_count = 0

    # Tracking counters
    counters = {
        'total': len(anns),
        'invalid_bbox': 0,
        'image_id_not_found': 0,
        'image_file_not_found': 0,
        'crop_error': 0,
        'success': 0,
    }

    print(f"Processing COCO Text annotations...")
    print(f"  Total annotations: {len(anns)}")
    print(f"  Output directory: {output_dir}")
    print(f"  COCO directory: {coco_images_dir}")
    print()

    with open(labels_file, 'w') as labels:
        for ann_id, ann in tqdm(anns.items(), total=len(anns)):
            if max_crops and crop_count >= max_crops:
                break

            valid, reason = is_valid_annotation(ann)
            if not valid:
                counters['invalid_bbox'] += 1
                continue

            image_id = ann['image_id']
            if image_id not in imgs:
                counters['image_id_not_found'] += 1
                continue

            # Construct image path
            img_info = imgs[image_id]
            img_filename = img_info['name']

            # Try different possible paths
            possible_paths = [
                coco_images_dir / img_filename,
                coco_images_dir / "val2014" / img_filename,
                coco_images_dir / "train2014" / img_filename,
            ]

            img_path = None
            for p in possible_paths:
                if p.exists():
                    img_path = p
                    break

            if img_path is None:
                counters['image_file_not_found'] += 1
                continue

            # Crop image
            crop = crop_image_region(img_path, ann['bbox'])
            if crop is None:
                counters['crop_error'] += 1
                continue

            # Save crop
            crop_filename = f"cocotext_{crop_count:06d}.png"
            crop_path = output_dir / crop_filename
            crop.save(crop_path)

            # Write label
            text = ann['utf8_string']
            legibility = ann.get('legibility', 'unknown')
            class_type = ann.get('class', 'unknown')
            labels.write(f"{crop_filename} {text} legibility={legibility} class={class_type}\n")

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
    print(f"    - Image ID not found:         {counters['image_id_not_found']:>8}")
    print(f"    - Image file not found:       {counters['image_file_not_found']:>8}")
    print(f"    - Crop error:                 {counters['crop_error']:>8}")
    print(f"{'='*60}")
    print(f"✓ Created {crop_count} cropped images")
    print(f"✓ Labels saved to {labels_file}")
    return crop_count


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Crop COCO Text images")
    parser.add_argument("--cocotext-json", type=Path, default=COCOTEXT_JSON, help="Path to cocotext.v2.json file")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR, help="Output directory")
    parser.add_argument("--coco-dir", type=Path, default=COCO_IMAGES_DIR, help="COCO images directory")
    parser.add_argument("--max-crops", type=int, help="Maximum number of crops to create")
    parser.add_argument("--padding", type=int, default=PADDING, help="Padding around bbox")

    args = parser.parse_args()

    process_cocotext(
        output_dir=args.output_dir,
        coco_images_dir=args.coco_dir,
        max_crops=args.max_crops,
        cocotext_json=args.cocotext_json,
    )
