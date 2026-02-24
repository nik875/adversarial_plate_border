#!/usr/bin/env python3
"""
Download a CC3M subset (~200-300k images) via img2dataset.

Pipeline:
  1. Stream image URLs from HuggingFace (google-research-datasets/conceptual_captions)
  2. Write a URL list TSV
  3. Call img2dataset to fetch images in parallel

Requires:
  pip install img2dataset datasets

Notes:
  - CC3M URL liveness is ~50-60% in 2025. To reliably obtain 200k images,
    the script requests 400k URLs. img2dataset skips dead URLs gracefully.
  - Downloads are retried automatically by img2dataset.
  - Output images are saved as JPEG at up to 512px on the long edge.

Output layout (flat shards, compatible with LazyDatasetPool recursive glob):
  <output_dir>/00000/*.jpg
  <output_dir>/00001/*.jpg
  ...

Usage:
  python download_cc3m.py --output-dir /data/cc3m
  python download_cc3m.py --output-dir /data/cc3m --url-count 400000 --processes 16
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def export_url_list(out_tsv: Path, n_urls: int) -> None:
    """Stream CC3M metadata from HF and write a URL list TSV for img2dataset."""
    try:
        from datasets import load_dataset  # noqa: PLC0415
    except ImportError:
        sys.exit('ERROR: pip install datasets')

    print(f'==> Streaming CC3M metadata from HuggingFace ({n_urls:,} URLs)...')
    ds = load_dataset(
        'google-research-datasets/conceptual_captions',
        'unlabeled',
        split='train',
        streaming=True,
    )

    out_tsv.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with out_tsv.open('w') as f:
        f.write('url\tcaption\n')
        for sample in ds:
            url = sample.get('image_url') or sample.get('url') or ''
            cap = (sample.get('caption') or '').replace('\t', ' ').replace('\n', ' ')
            if url:
                f.write(f'{url}\t{cap}\n')
                written += 1
                if written >= n_urls:
                    break
            if written % 50_000 == 0 and written > 0:
                print(f'   {written:,} URLs written...')

    print(f'   Wrote {written:,} URLs → {out_tsv}')


def run_img2dataset(url_tsv: Path, output_dir: Path, processes: int) -> None:
    """Call img2dataset CLI to download images from the URL list."""
    try:
        import img2dataset  # noqa: F401
    except ImportError:
        sys.exit('ERROR: pip install img2dataset')

    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        'img2dataset',
        '--url_list',         str(url_tsv),
        '--input_format',     'tsv',
        '--url_col',          'url',
        '--caption_col',      'caption',
        '--output_dir',       str(output_dir),
        '--output_format',    'files',         # plain JPEG files, no WebDataset
        '--image_size',       '640',
        '--resize_mode',      'keep_ratio',
        '--min_image_size',   '64',
        '--number_sample_per_shard', '10000',
        '--processes_count',  str(processes),
        '--thread_count',     '64',
        '--retries',          '2',
        '--enable_wandb',     'False',
    ]
    print(f'\n==> Running img2dataset  ({processes} processes, 64 threads each)...')
    print('    This will take a while. Progress is logged by img2dataset.')
    result = subprocess.run(cmd)
    if result.returncode != 0:
        sys.exit(f'img2dataset exited with code {result.returncode}')


def count_images(output_dir: Path) -> int:
    return sum(1 for _ in output_dir.rglob('*.jpg'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output-dir', default=os.path.expanduser('~/.cache/cc3m'))
    ap.add_argument('--url-count',  type=int, default=2_500_000,
                    help='URLs to fetch (~50%% success rate; 2.5M → ~1.2M images, ~90 GB at 640px)')
    ap.add_argument('--processes',  type=int, default=16,
                    help='Parallel download processes for img2dataset')
    ap.add_argument('--skip-url-export', action='store_true',
                    help='Skip URL export if cc3m_urls.tsv already exists')
    args = ap.parse_args()

    out = Path(args.output_dir)
    url_tsv = out / 'cc3m_urls.tsv'

    if not args.skip_url_export or not url_tsv.exists():
        export_url_list(url_tsv, args.url_count)
    else:
        print(f'Skipping URL export, using existing {url_tsv}')

    images_dir = out / 'images'
    run_img2dataset(url_tsv, images_dir, args.processes)

    n = count_images(images_dir)
    print(f'\nDone. CC3M subset: {n:,} images → {images_dir}')
    if n < 800_000:
        print(f'WARNING: only {n:,} images downloaded (many CC3M URLs are dead).')
        print('         Consider increasing --url-count or using a mirror.')


if __name__ == '__main__':
    main()
