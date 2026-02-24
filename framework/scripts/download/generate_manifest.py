#!/usr/bin/env python3
"""
Generate a manifest CSV for all training datasets.

Scans dataset directories and writes ~/.cache/adversarial_plate_manifest.csv.
Only images that are already on disk (previously downloaded) are included;
datasets that haven't been downloaded yet are skipped with a warning.

CSV columns: path, dataset, split, label, label_type

Split assignment:
  val   — imagenet_val, coco_val  (labeled, used for evaluation)
  train — everything else

Usage:
  python generate_manifest.py
  python generate_manifest.py --data-root /custom/cache
  python generate_manifest.py --output /tmp/manifest.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Per-dataset scanners
# ---------------------------------------------------------------------------

def _scan_imagenet_split(imagenet_root: Path, split: str) -> list[dict]:
    """Scan imagenet/{val,train}/<NNNN>/*.JPEG — labeled by class index."""
    rows: list[dict] = []
    split_dir = imagenet_root / split
    if not split_dir.exists():
        return rows
    for cls_dir in sorted(split_dir.iterdir()):
        if not cls_dir.is_dir():
            continue
        try:
            label = int(cls_dir.name)
        except ValueError:
            continue
        for img in sorted(cls_dir.glob('*.JPEG')):
            rows.append({
                'path': str(img.resolve()),
                'dataset': f'imagenet_{split}',
                'split': split,
                'label': str(label),
                'label_type': 'imagenet_class',
            })
    return rows


def _scan_imagenet_test(imagenet_root: Path) -> list[dict]:
    """Scan imagenet/test/*.JPEG — flat dir, no labels."""
    rows: list[dict] = []
    test_dir = imagenet_root / 'test'
    if not test_dir.exists():
        return rows
    for img in sorted(test_dir.glob('*.JPEG')):
        rows.append({
            'path': str(img.resolve()),
            'dataset': 'imagenet_test',
            'split': 'train',   # no labels → used as unlabeled train diversity
            'label': '',
            'label_type': 'none',
        })
    return rows


def _scan_coco_split(coco_root: Path, split: str) -> list[dict]:
    """Scan coco/{train,val}2017/*.jpg with COCO category labels."""
    rows: list[dict] = []
    img_dir = coco_root / f'{split}2017'
    if not img_dir.exists():
        return rows

    # Build filename → comma-separated category IDs from annotation JSON
    label_map: dict[str, str] = {}
    ann_file = coco_root / 'annotations' / f'instances_{split}2017.json'
    if ann_file.exists():
        with open(ann_file) as f:
            ann = json.load(f)
        id_to_fname: dict[int, str] = {
            img['id']: img['file_name'] for img in ann['images']
        }
        cats_per_image: dict[str, set[int]] = {}
        for item in ann['annotations']:
            fname = id_to_fname.get(item['image_id'], '')
            if fname:
                cats_per_image.setdefault(fname, set()).add(item['category_id'])
        label_map = {
            fname: ','.join(str(c) for c in sorted(cats))
            for fname, cats in cats_per_image.items()
        }

    ds_split = 'val' if split == 'val' else 'train'
    ds_name = f'coco_{split}'

    for img in sorted(img_dir.glob('*.jpg')):
        fname = img.name
        label = label_map.get(fname, '')
        rows.append({
            'path': str(img.resolve()),
            'dataset': ds_name,
            'split': ds_split,
            'label': label,
            'label_type': 'coco_categories' if label else 'none',
        })
    return rows


def _scan_flat(root: Path, name: str, glob_pattern: str = '**/*.jpg') -> list[dict]:
    """Scan a flat/nested dataset with no labels (train split)."""
    rows: list[dict] = []
    if not root.exists():
        return rows
    for img in sorted(root.glob(glob_pattern)):
        rows.append({
            'path': str(img.resolve()),
            'dataset': name,
            'split': 'train',
            'label': '',
            'label_type': 'none',
        })
    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description='Generate adversarial plate training manifest CSV'
    )
    ap.add_argument(
        '--data-root',
        default=os.path.expanduser('~/.cache'),
        help='Root directory containing all downloaded datasets (default: ~/.cache)',
    )
    ap.add_argument(
        '--output',
        default=os.path.expanduser('~/.cache/adversarial_plate_manifest.csv'),
        help='Output CSV path (default: ~/.cache/adversarial_plate_manifest.csv)',
    )
    args = ap.parse_args()

    data_root = Path(args.data_root)
    output = Path(args.output)

    # Each entry: (display_label, scanner_fn)
    # The scanner returns [] if the dataset dir doesn't exist (skip gracefully).
    datasets: list[tuple[str, object]] = [
        ('ImageNet val',   lambda: _scan_imagenet_split(data_root / 'imagenet', 'val')),
        ('ImageNet train', lambda: _scan_imagenet_split(data_root / 'imagenet', 'train')),
        ('ImageNet test',  lambda: _scan_imagenet_test(data_root / 'imagenet')),
        ('COCO val',       lambda: _scan_coco_split(data_root / 'coco', 'val')),
        ('COCO train',     lambda: _scan_coco_split(data_root / 'coco', 'train')),
        ('TextVQA',        lambda: _scan_flat(
                               data_root / 'textvqa' / 'train_val_images',
                               'textvqa', '*.jpg')),
        ('CC3M',           lambda: _scan_flat(
                               data_root / 'cc3m' / 'images',
                               'cc3m', '**/*.jpg')),
        ('OpenImages',     lambda: _scan_flat(
                               data_root / 'openimages' / 'images',
                               'openimages', '**/*.jpg')),
    ]

    all_rows: list[dict] = []
    for display_label, scanner in datasets:
        print(f'Scanning {display_label}...', end=' ', flush=True)
        rows = scanner()
        if rows:
            print(f'{len(rows):,} images')
            all_rows.extend(rows)
        else:
            print('SKIP (not found or empty)')

    if not all_rows:
        print(
            'WARNING: no images found in any dataset. '
            'Run download_all.sh first.',
            file=sys.stderr,
        )
        sys.exit(1)

    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, 'w', newline='') as f:
        writer = csv.DictWriter(
            f, fieldnames=['path', 'dataset', 'split', 'label', 'label_type']
        )
        writer.writeheader()
        writer.writerows(all_rows)

    train_count = sum(1 for r in all_rows if r['split'] == 'train')
    val_count = sum(1 for r in all_rows if r['split'] == 'val')
    print(f'\nManifest written: {output}')
    print(f'  Total : {len(all_rows):,} images')
    print(f'  train : {train_count:,}')
    print(f'  val   : {val_count:,}')


if __name__ == '__main__':
    main()
