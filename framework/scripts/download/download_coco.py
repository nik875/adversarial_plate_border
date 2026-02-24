#!/usr/bin/env python3
"""
Download COCO 2017 train + val images directly from COCO servers.

No authentication required.

Output layout:
  <output_dir>/train2017/  — 118,287 images
  <output_dir>/val2017/    —   5,000 images

Usage:
  python download_coco.py --output-dir /data/coco
  python download_coco.py --output-dir /data/coco --val-only
"""
from __future__ import annotations

import argparse
import os
import sys
import urllib.request
import zipfile
from pathlib import Path


_URLS = {
    'train2017': 'http://images.cocodataset.org/zips/train2017.zip',   # ~18 GB
    'val2017':   'http://images.cocodataset.org/zips/val2017.zip',     #  ~1 GB
}


def _progress(label: str):
    def hook(count, block_size, total_size):
        if total_size <= 0:
            return
        pct = min(count * block_size / total_size * 100, 100)
        gb = total_size / 1e9
        print(f'\r   {label}: {pct:5.1f}%  ({gb:.1f} GB)', end='', flush=True)
    return hook


def download_split(name: str, url: str, out_dir: Path) -> None:
    done_flag = out_dir / name / '.done'
    if done_flag.exists():
        print(f'{name}: already downloaded (delete {name}/.done to re-run)')
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    zip_path = out_dir / f'{name}.zip'

    if not zip_path.exists():
        print(f'\n==> Downloading {name}  ({url})')
        try:
            urllib.request.urlretrieve(url, zip_path, reporthook=_progress(name))
        except Exception as e:
            zip_path.unlink(missing_ok=True)
            sys.exit(f'\nERROR downloading {name}: {e}')
        print()  # newline after progress

    print(f'   Extracting {zip_path.name}...')
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(out_dir)
    zip_path.unlink()

    done_flag.touch()
    n = len(list((out_dir / name).glob('*.jpg')))
    print(f'   {name}: {n:,} images → {out_dir / name}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output-dir', default=os.path.expanduser('~/.cache/coco'))
    ap.add_argument('--val-only',   action='store_true',
                    help='Download validation set only (5k images, ~1 GB)')
    args = ap.parse_args()

    out = Path(args.output_dir)

    splits = ['val2017'] if args.val_only else ['train2017', 'val2017']
    for name in splits:
        download_split(name, _URLS[name], out)

    print(f'\nDone. COCO saved to {out}')


if __name__ == '__main__':
    main()
