#!/usr/bin/env python3
"""
Download ImageNet-1K validation set (~50k images) and a random training subset.

REQUIRES — manual one-time step:
  1. Accept the dataset license at https://huggingface.co/datasets/ILSVRC/imagenet-1k
  2. Create a HF token at https://huggingface.co/settings/tokens
  3. export HF_TOKEN=hf_...

Output layout (flat by synset index, compatible with LazyDatasetPool):
  <output_dir>/val/<0000..0999>/<image>.JPEG
  <output_dir>/train/<0000..0999>/<image>.JPEG

Usage:
  python download_imagenet.py --output-dir /data/imagenet
  python download_imagenet.py --output-dir /data/imagenet --train-samples 150000
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output-dir',    default='/data/imagenet')
    ap.add_argument('--train-samples', type=int, default=150_000,
                    help='Number of training images to download (default: 150k)')
    ap.add_argument('--val-only',      action='store_true',
                    help='Skip training subset, download validation only')
    args = ap.parse_args()

    token = os.environ.get('HF_TOKEN')
    if not token:
        sys.exit(
            'ERROR: HF_TOKEN not set.\n'
            '  1. Accept the license: https://huggingface.co/datasets/ILSVRC/imagenet-1k\n'
            '  2. Create a token:     https://huggingface.co/settings/tokens\n'
            '  3. Run: export HF_TOKEN=hf_...'
        )

    from datasets import load_dataset  # noqa: PLC0415

    out = Path(args.output_dir)

    # ------------------------------------------------------------------
    # Validation set (~50k images)
    # ------------------------------------------------------------------
    val_dir = out / 'val'
    val_done = val_dir / '.done'
    if val_done.exists():
        print(f'Validation already downloaded → {val_dir}  (delete .done to re-run)')
    else:
        print('==> ImageNet-1K validation (50k images)...')
        val_dir.mkdir(parents=True, exist_ok=True)
        ds = load_dataset('ILSVRC/imagenet-1k', split='validation',
                          token=token, trust_remote_code=True)
        for i, sample in enumerate(ds):
            cls_dir = val_dir / f'{sample["label"]:04d}'
            cls_dir.mkdir(exist_ok=True)
            img = sample['image']
            if img.mode != 'RGB':
                img = img.convert('RGB')
            img.save(cls_dir / f'{i:08d}.JPEG')
            if (i + 1) % 5_000 == 0:
                print(f'   val {i+1}/{len(ds)}')
        val_done.touch()
        print(f'   Saved {len(ds)} images → {val_dir}')

    if args.val_only:
        return

    # ------------------------------------------------------------------
    # Training subset (streaming — avoids loading full 1.28M into RAM)
    # ------------------------------------------------------------------
    train_dir = out / 'train'
    train_done = train_dir / '.done'
    if train_done.exists():
        print(f'Training subset already downloaded → {train_dir}  (delete .done to re-run)')
        return

    print(f'\n==> ImageNet-1K training subset ({args.train_samples:,} images, streaming)...')
    train_dir.mkdir(parents=True, exist_ok=True)
    ds_train = load_dataset('ILSVRC/imagenet-1k', split='train',
                            token=token, trust_remote_code=True, streaming=True)
    saved = 0
    for sample in ds_train:
        if saved >= args.train_samples:
            break
        cls_dir = train_dir / f'{sample["label"]:04d}'
        cls_dir.mkdir(exist_ok=True)
        img = sample['image']
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img.save(cls_dir / f'{saved:08d}.JPEG')
        saved += 1
        if saved % 10_000 == 0:
            print(f'   train {saved:,}/{args.train_samples:,}')

    train_done.touch()
    print(f'   Saved {saved:,} images → {train_dir}')


if __name__ == '__main__':
    main()
