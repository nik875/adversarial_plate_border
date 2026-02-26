#!/usr/bin/env python3
"""
Download TextVQA train+val images from Facebook's public CDN.

No authentication required.

TextVQA images are sourced from Open Images and cover natural scenes
with embedded text — ideal for the TrOCR surrogate in the ensemble.

Stats: ~34,602 unique images (train + val)

Output layout:
  <output_dir>/train_val_images/  — all images as *.jpg

Usage:
  python download_textvqa.py --output-dir /data/textvqa
"""
from __future__ import annotations

import argparse
import os
import sys
import urllib.request
import zipfile
from pathlib import Path


# Public Facebook CDN — no auth required
_URL = 'https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip'  # ~7 GB


def _progress(count, block_size, total_size):
    if total_size <= 0:
        return
    pct = min(count * block_size / total_size * 100, 100)
    gb = total_size / 1e9
    print(f'\r   {pct:5.1f}%  ({gb:.1f} GB total)', end='', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output-dir', default=os.path.expanduser('~/.cache/textvqa'))
    ap.add_argument('--max-samples', type=int, default=None,
                    help='Randomly sample this many images')
    args = ap.parse_args()

    out = Path(args.output_dir)
    done_flag = out / '.done'

    if done_flag.exists():
        n = len(list(out.rglob('*.jpg')))
        print(f'TextVQA already downloaded ({n:,} images) → {out}  (delete .done to re-run)')
        return

    out.mkdir(parents=True, exist_ok=True)
    zip_path = out / 'train_val_images.zip'

    if not zip_path.exists():
        print(f'==> Downloading TextVQA images  ({_URL})')
        try:
            urllib.request.urlretrieve(_URL, zip_path, reporthook=_progress)
        except Exception as e:
            zip_path.unlink(missing_ok=True)
            sys.exit(f'\nERROR: {e}')
        print()

    print(f'   Extracting {zip_path.name}...')
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(out)
    zip_path.unlink()

    # --- Random sampling if --max-samples is set ---
    if args.max_samples is not None:
        print(f'\n==> Randomly sampling {args.max_samples:,} images...')
        import random

        all_images = list(out.rglob('*.jpg'))

        if len(all_images) > args.max_samples:
            sampled = random.sample(all_images, args.max_samples)
            sampled_set = set(sampled)

            deleted = 0
            for img in all_images:
                if img not in sampled_set:
                    img.unlink()
                    deleted += 1

            print(f'   Kept {len(sampled):,} images, deleted {deleted:,} images')
        else:
            print(f'   Dataset has {len(all_images):,} images (≤ {args.max_samples:,}), keeping all')

    done_flag.touch()
    n = len(list(out.rglob('*.jpg')))
    print(f'   Saved {n:,} images → {out}')


if __name__ == '__main__':
    main()
