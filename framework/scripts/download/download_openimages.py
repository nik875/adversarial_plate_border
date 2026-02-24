#!/usr/bin/env python3
"""
Download an Open Images V7 subset via FiftyOne Zoo.

FiftyOne handles shard selection, parallel downloading from AWS S3,
and deduplication automatically.

Requires:
  pip install fiftyone

Output layout (flat per-split dirs, compatible with LazyDatasetPool):
  <output_dir>/data/*.jpg    (FiftyOne default export structure)

Usage:
  python download_openimages.py --output-dir /data/openimages
  python download_openimages.py --output-dir /data/openimages --num-samples 150000
  python download_openimages.py --output-dir /data/openimages --num-samples 150000 --split validation
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output-dir',  default='/data/openimages')
    ap.add_argument('--num-samples', type=int, default=150_000,
                    help='Number of images to download (default: 150k)')
    ap.add_argument('--split',       default='train',
                    choices=['train', 'validation', 'test'],
                    help='Dataset split (default: train)')
    ap.add_argument('--seed',        type=int, default=42)
    args = ap.parse_args()

    try:
        import fiftyone.zoo as foz          # noqa: PLC0415
        import fiftyone as fo               # noqa: PLC0415
    except ImportError:
        sys.exit(
            'ERROR: fiftyone not installed.\n'
            '  pip install fiftyone\n'
            '  (fiftyone handles AWS S3 download + deduplication automatically)'
        )

    out = Path(args.output_dir)
    done_flag = out / '.done'
    if done_flag.exists():
        print(f'Open Images already downloaded → {out}  (delete .done to re-run)')
        return

    out.mkdir(parents=True, exist_ok=True)

    print(f'==> Downloading Open Images V7  ({args.num_samples:,} images, split={args.split})...')
    print('    First run downloads index files (~200 MB). Subsequent runs are incremental.')

    dataset = foz.load_zoo_dataset(
        'open-images-v7',
        split=args.split,
        max_samples=args.num_samples,
        shuffle=True,
        seed=args.seed,
        label_types=[],         # images only — no annotation download
        dataset_name=f'openimages_{args.split}_{args.num_samples}',
    )

    # Export as a flat directory of images (LazyDatasetPool just needs files)
    print(f'\n==> Exporting images to {out}...')
    dataset.export(
        export_dir=str(out),
        dataset_type=fo.types.ImageDirectory,
        overwrite=True,
    )

    n = len(list(out.rglob('*.jpg'))) + len(list(out.rglob('*.png')))
    done_flag.touch()
    print(f'   Saved {n:,} images → {out}')

    # Clean up fiftyone internal dataset to free cache
    dataset.delete()


if __name__ == '__main__':
    main()
