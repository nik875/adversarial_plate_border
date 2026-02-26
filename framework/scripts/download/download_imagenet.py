#!/usr/bin/env python3
"""
Download ImageNet-1K validation set (~50k images), training set, and test set.

REQUIRES — manual one-time step:
  1. Accept the dataset license at https://huggingface.co/datasets/ILSVRC/imagenet-1k
  2. Create a HF token at https://huggingface.co/settings/tokens
  3. export HF_TOKEN=hf_...

Output layout (flat by synset index, compatible with LazyDatasetPool):
  <output_dir>/val/<0000..0999>/<image>.JPEG
  <output_dir>/train/<0000..0999>/<image>.JPEG
  <output_dir>/test/<image>.JPEG          (flat, no class subdirs — no labels)

Usage:
  python download_imagenet.py --output-dir /data/imagenet
  python download_imagenet.py --output-dir /data/imagenet --train-samples 150000
  python download_imagenet.py --output-dir /data/imagenet --skip-test
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from PIL import Image as PILImage


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output-dir',    default=os.path.expanduser('~/.cache/imagenet'))
    ap.add_argument('--train-samples', type=int, default=1_000_000,
                    help='Number of training images to download (default: 1M of 1.28M, ~75 GB at 640px)')
    ap.add_argument('--val-only',      action='store_true',
                    help='Skip training subset, download validation only')
    ap.add_argument('--skip-test',     action='store_true',
                    help='Skip test set download (~100k images, no labels)')
    ap.add_argument('--max-size',      type=int, default=640,
                    help='Downscale long edge to this size preserving aspect ratio (default: 640)')
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
    max_size = args.max_size

    def save_image(img: PILImage.Image, path: Path) -> None:
        """Convert to RGB, downscale long edge to max_size, save as JPEG."""
        if img.mode != 'RGB':
            img = img.convert('RGB')
        w, h = img.size
        if max(w, h) > max_size:
            scale = max_size / max(w, h)
            img = img.resize((int(w * scale), int(h * scale)), PILImage.LANCZOS)
        img.save(path, format='JPEG', quality=90)

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
                          token=token)
        for i, sample in enumerate(ds):
            cls_dir = val_dir / f'{sample["label"]:04d}'
            cls_dir.mkdir(exist_ok=True)
            save_image(sample['image'], cls_dir / f'{i:08d}.JPEG')
            if (i + 1) % 5_000 == 0:
                print(f'   val {i+1}/{len(ds)}')
        val_done.touch()
        print(f'   Saved {len(ds)} images → {val_dir}')

    if args.val_only:
        return

    # ------------------------------------------------------------------
    # Training subset (streaming + .take() — only downloads what's needed)
    # ------------------------------------------------------------------
    train_dir = out / 'train'
    train_done = train_dir / '.done'
    if train_done.exists():
        print(f'Training subset already downloaded → {train_dir}  (delete .done to re-run)')
    else:
        print(f'\n==> ImageNet-1K training subset ({args.train_samples:,} images, streaming)...')
        train_dir.mkdir(parents=True, exist_ok=True)
        ds_train = load_dataset('ILSVRC/imagenet-1k', split='train',
                                token=token, streaming=True)
        # Use .take() to only fetch the exact number of samples needed
        ds_train = ds_train.take(args.train_samples)
        saved = 0
        for sample in ds_train:
            cls_dir = train_dir / f'{sample["label"]:04d}'
            cls_dir.mkdir(exist_ok=True)
            save_image(sample['image'], cls_dir / f'{saved:08d}.JPEG')
            saved += 1
            if saved % 10_000 == 0:
                print(f'   train {saved:,}/{args.train_samples:,}')

        train_done.touch()
        print(f'   Saved {saved:,} images → {train_dir}')

    if args.skip_test:
        return

    # ------------------------------------------------------------------
    # Test set (~100k images, streaming — no labels, flat directory)
    # ------------------------------------------------------------------
    test_dir = out / 'test'
    test_done = test_dir / '.done'
    if test_done.exists():
        print(f'Test set already downloaded → {test_dir}  (delete .done to re-run)')
        return

    print('\n==> ImageNet-1K test set (~100k images, no labels, streaming)...')
    test_dir.mkdir(parents=True, exist_ok=True)
    ds_test = load_dataset('ILSVRC/imagenet-1k', split='test',
                           token=token, streaming=True)
    # Stream and save all test images (no limit)
    saved = 0
    for sample in ds_test:
        save_image(sample['image'], test_dir / f'{saved:08d}.JPEG')
        saved += 1
        if saved % 10_000 == 0:
            print(f'   test {saved:,}')

    test_done.touch()
    print(f'   Saved {saved:,} images → {test_dir}')


if __name__ == '__main__':
    main()
