#!/usr/bin/env python3
"""
Download an Open Images V7 subset via img2dataset.

img2dataset downloads and resizes to 640px on-the-fly, writing final JPEGs
directly — no full-resolution intermediate copies are ever stored on disk.

Pipeline:
  1. Download the Open Images V7 train image list CSV from GCS (public, no auth)
  2. Optionally shuffle and truncate to --num-samples rows
  3. Run img2dataset to fetch and resize images in parallel

Requires:
  pip install img2dataset

Stats: ~1.7M train images available in Open Images V7

Output layout (sharded, compatible with LazyDatasetPool recursive glob):
  <output_dir>/00000/*.jpg
  <output_dir>/00001/*.jpg
  ...

Usage:
  python download_openimages.py --output-dir ~/.cache/openimages
  python download_openimages.py --output-dir ~/.cache/openimages --num-samples 1000000 --processes 16
"""
from __future__ import annotations

import argparse
import csv
import os
import random
import subprocess
import sys
import urllib.request
from pathlib import Path


# Public GCS — no auth required
_OI_TRAIN_CSV_URL = (
    'https://storage.googleapis.com/openimages/2018_04/train/'
    'train-images-boxable-with-rotation.csv'
)


def _progress(count, block_size, total_size):
    if total_size <= 0:
        return
    pct = min(count * block_size / total_size * 100, 100)
    print(f'\r   {pct:5.1f}%', end='', flush=True)


def download_image_list(csv_path: Path) -> None:
    print(f'==> Downloading Open Images V7 train image list...')
    urllib.request.urlretrieve(_OI_TRAIN_CSV_URL, csv_path, reporthook=_progress)
    print()
    n = sum(1 for _ in csv_path.open()) - 1  # subtract header
    print(f'   {n:,} images listed → {csv_path}')


def prepare_url_list(full_csv: Path, url_list: Path, num_samples: int, seed: int) -> None:
    """Read OriginalURL column, shuffle, truncate, write a plain URL-per-line file."""
    print(f'\n==> Sampling {num_samples:,} URLs...')
    urls = []
    with full_csv.open(newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            url = row.get('OriginalURL', '')
            if url:
                urls.append(url)

    random.seed(seed)
    random.shuffle(urls)
    urls = urls[:num_samples]

    with url_list.open('w') as f:
        f.write('url\n')
        for u in urls:
            f.write(u + '\n')

    print(f'   Wrote {len(urls):,} URLs → {url_list}')


def run_img2dataset(url_list: Path, output_dir: Path, processes: int) -> None:
    try:
        import img2dataset  # noqa: F401
    except ImportError:
        sys.exit('ERROR: pip install img2dataset')

    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, '-m', 'img2dataset',
        '--url_list',                str(url_list),
        '--input_format',            'csv',
        '--url_col',                 'url',
        '--output_dir',              str(output_dir),
        '--output_format',           'files',
        '--image_size',              '640',
        '--resize_mode',             'keep_ratio',
        '--min_image_size',          '64',
        '--number_sample_per_shard', '10000',
        '--processes_count',         str(processes),
        '--thread_count',            '64',
        '--retries',                 '2',
        '--enable_wandb',            'False',
        '--save_additional_columns', '[]',
    ]
    print(f'\n==> Running img2dataset  ({processes} processes, 64 threads each)...')
    print('    Images are resized to 640px long-edge on-the-fly — no full-res intermediates.')
    result = subprocess.run(cmd)
    if result.returncode != 0:
        sys.exit(f'img2dataset exited with code {result.returncode}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output-dir',   default=os.path.expanduser('~/.cache/openimages'))
    ap.add_argument('--num-samples',  type=int, default=1_000_000,
                    help='Images to download (default: 1M, ~75 GB at 640px)')
    ap.add_argument('--processes',    type=int, default=16)
    ap.add_argument('--seed',         type=int, default=42)
    ap.add_argument('--skip-csv',     action='store_true',
                    help='Skip CSV download if openimages_train.csv already exists')
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    full_csv = out / 'openimages_train_full.csv'
    url_list = out / 'openimages_urls.csv'

    if not args.skip_csv or not full_csv.exists():
        download_image_list(full_csv)
    else:
        print(f'Skipping CSV download, using existing {full_csv}')

    prepare_url_list(full_csv, url_list, args.num_samples, args.seed)

    images_dir = out / 'images'
    run_img2dataset(url_list, images_dir, args.processes)

    n = sum(1 for _ in images_dir.rglob('*.jpg'))
    print(f'\nDone. Open Images subset: {n:,} images → {images_dir}')
    if n < args.num_samples * 0.5:
        print(f'WARNING: low yield ({n:,}/{args.num_samples:,}). '
              f'Open Images original URLs (Flickr etc.) may have liveness issues. '
              f'Consider increasing --num-samples.')


if __name__ == '__main__':
    main()
