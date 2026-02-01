#!/usr/bin/env python3
"""
Unified license plate crops dataset loader.

This is a streamlined version of load_datasets.py that loads ONLY cropped
license plate images from local cache directories.

Provides:
- uniform image access
- lightweight iteration for anchor sampling
- identical interface to load_datasets.py

Datasets included:
- cocotext_crops
- roboflow_lpr_crops
- kaggle_lp_crops
- indian_plates_kaggle_crops
- ccpd2019_crops (all variants)
- mercosur_crops
- crpd_crops
"""

from typing import Dict, Iterator, Tuple
from pathlib import Path

from PIL import Image


# ---------------------------------------------------------
# Dataset registry (LP crops only)
# ---------------------------------------------------------

DATASETS = {
    "cocotext": {
        "source": "local",
        "cache_dir": Path.home() / ".cache" / "cocotext_crops",
        "splits": ["train"],
    },
    "roboflow_lpr": {
        "source": "local",
        "cache_dir": Path.home() / ".cache" / "roboflow_lpr_crops",
        "splits": ["train", "test", "valid"],
    },
    "kaggle_lp": {
        "source": "local",
        "cache_dir": Path.home() / ".cache" / "kaggle_lp_crops",
        "splits": ["train"],
    },
    "indian_plates_kaggle": {
        "source": "local",
        "cache_dir": Path.home() / ".cache" / "indian_plates_kaggle_crops",
        "splits": ["train"],
    },
    "ccpd2019_base": {
        "source": "local",
        "cache_dir": Path.home() / ".cache" / "ccpd2019_crops" / "ccpd_base",
        "splits": ["train"],
    },
    "ccpd2019_blur": {
        "source": "local",
        "cache_dir": Path.home() / ".cache" / "ccpd2019_crops" / "ccpd_blur",
        "splits": ["train"],
    },
    "ccpd2019_challenge": {
        "source": "local",
        "cache_dir": Path.home() / ".cache" / "ccpd2019_crops" / "ccpd_challenge",
        "splits": ["train"],
    },
    "ccpd2019_db": {
        "source": "local",
        "cache_dir": Path.home() / ".cache" / "ccpd2019_crops" / "ccpd_db",
        "splits": ["train"],
    },
    "ccpd2019_fn": {
        "source": "local",
        "cache_dir": Path.home() / ".cache" / "ccpd2019_crops" / "ccpd_fn",
        "splits": ["train"],
    },
    "ccpd2019_np": {
        "source": "local",
        "cache_dir": Path.home() / ".cache" / "ccpd2019_crops" / "ccpd_np",
        "splits": ["train"],
    },
    "ccpd2019_rotate": {
        "source": "local",
        "cache_dir": Path.home() / ".cache" / "ccpd2019_crops" / "ccpd_rotate",
        "splits": ["train"],
    },
    "ccpd2019_tilt": {
        "source": "local",
        "cache_dir": Path.home() / ".cache" / "ccpd2019_crops" / "ccpd_tilt",
        "splits": ["train"],
    },
    "ccpd2019_weather": {
        "source": "local",
        "cache_dir": Path.home() / ".cache" / "ccpd2019_crops" / "ccpd_weather",
        "splits": ["train"],
    },
    "mercosur": {
        "source": "local",
        "cache_dir": Path.home() / ".cache" / "mercosur_crops",
        "splits": ["train"],
    },
    "crpd": {
        "source": "local",
        "cache_dir": Path.home() / ".cache" / "crpd_crops",
        "splits": ["train", "test", "val"],
    },
}


# ---------------------------------------------------------
# COCO Text cropped images
# ---------------------------------------------------------

def _iter_cocotext(split: str, max_samples: int | None = None) -> Iterator[Tuple[Image.Image, str, Dict]]:
    """
    Iterate over COCO Text cropped images from local directory.
    Split should be 'train' (only split available from crop_cocotext.py).

    Expects directory structure:
    - ~/.cache/cocotext_crops/
      - labels.txt (format: filename text legibility=X class=Y)
      - cocotext_*.png (cropped images)
    """
    cache_dir = DATASETS["cocotext"]["cache_dir"]

    if not cache_dir.exists():
        return

    labels_file = cache_dir / "labels.txt"
    if not labels_file.exists():
        return

    # Load labels
    labels_dict = {}
    with open(labels_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # Format: filename text legibility=X class=Y
            parts = line.split(None, 2)
            if len(parts) < 2:
                continue

            filename = parts[0]
            text = parts[1]
            labels_dict[filename] = text

    count = 0
    # Iterate through all cropped images
    for img_path in sorted(cache_dir.glob("cocotext_*.png")):
        filename = img_path.name

        if filename not in labels_dict:
            continue

        text = labels_dict[filename]

        try:
            img = Image.open(img_path).convert('RGB')

            meta = {
                "dataset": "cocotext",
                "split": split,
            }

            yield img, text, meta

            count += 1
            if max_samples is not None and count >= max_samples:
                break
        except Exception as e:
            print(f"Warning: Could not load image {img_path}: {e}")
            continue


# ---------------------------------------------------------
# Roboflow LPR cropped images
# ---------------------------------------------------------

def _iter_roboflow_lpr(split: str, max_samples: int | None = None) -> Iterator[Tuple[Image.Image, str, Dict]]:
    """
    Iterate over Roboflow LPR cropped license plate images from local directory.
    Split should be 'train', 'test', or 'valid'.

    Expects directory structure:
    - ~/.cache/roboflow_lpr_crops/
      - labels.txt (format: filename split=X)
      - roboflow_lpr_*.png (cropped license plate images)
    """
    cache_dir = DATASETS["roboflow_lpr"]["cache_dir"]

    if not cache_dir.exists():
        return

    labels_file = cache_dir / "labels.txt"
    if not labels_file.exists():
        return

    # Load labels and filter by split
    labels_dict = {}
    with open(labels_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # Format: filename split=X
            parts = line.split()
            if len(parts) < 2:
                continue

            filename = parts[0]
            split_info = parts[1]

            # Parse split=X
            if '=' in split_info:
                _, file_split = split_info.split('=', 1)
                if file_split == split:
                    labels_dict[filename] = file_split

    count = 0
    # Iterate through cropped images
    for img_path in sorted(cache_dir.glob("roboflow_lpr_*.png")):
        filename = img_path.name

        if filename not in labels_dict:
            continue

        try:
            img = Image.open(img_path).convert('RGB')

            # Use split as the "text" label since these are images of objects, not text
            meta = {
                "dataset": "roboflow_lpr",
                "split": split,
            }

            yield img, split, meta

            count += 1
            if max_samples is not None and count >= max_samples:
                break
        except Exception as e:
            print(f"Warning: Could not load image {img_path}: {e}")
            continue


# ---------------------------------------------------------
# Kaggle LP cropped images
# ---------------------------------------------------------

def _iter_kaggle_lp(split: str, max_samples: int | None = None) -> Iterator[Tuple[Image.Image, str, Dict]]:
    """
    Iterate over Kaggle LP cropped license plate images from local directory.
    Split should be 'train' (only split available).

    Expects directory structure:
    - ~/.cache/kaggle_lp_crops/
      - labels.txt (format: filename dataset=kaggle_lp)
      - kaggle_lp_*.png (cropped license plate images)
    """
    cache_dir = DATASETS["kaggle_lp"]["cache_dir"]

    if not cache_dir.exists():
        return

    labels_file = cache_dir / "labels.txt"
    if not labels_file.exists():
        return

    # Load labels
    labels_dict = {}
    with open(labels_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # Format: filename dataset=kaggle_lp
            parts = line.split()
            if len(parts) < 1:
                continue

            filename = parts[0]
            labels_dict[filename] = True

    count = 0
    # Iterate through cropped images
    for img_path in sorted(cache_dir.glob("kaggle_lp_*.png")):
        filename = img_path.name

        if filename not in labels_dict:
            continue

        try:
            img = Image.open(img_path).convert('RGB')

            # Use split as the "text" label since these are images of objects, not text
            meta = {
                "dataset": "kaggle_lp",
                "split": split,
            }

            yield img, split, meta

            count += 1
            if max_samples is not None and count >= max_samples:
                break
        except Exception as e:
            print(f"Warning: Could not load image {img_path}: {e}")
            continue


# ---------------------------------------------------------
# Indian Plates Kaggle cropped images
# ---------------------------------------------------------

def _iter_indian_plates_kaggle(split: str, max_samples: int | None = None) -> Iterator[Tuple[Image.Image, str, Dict]]:
    """
    Iterate over Indian Plates Kaggle cropped license plate images from local directory.
    Split should be 'train' (only split available).

    Expects directory structure:
    - ~/.cache/indian_plates_kaggle_crops/
      - labels.txt (format: filename dataset=indian_plates)
      - indian_plates_*.png (cropped license plate images)
    """
    cache_dir = DATASETS["indian_plates_kaggle"]["cache_dir"]

    if not cache_dir.exists():
        return

    labels_file = cache_dir / "labels.txt"
    if not labels_file.exists():
        return

    # Load labels
    labels_dict = {}
    with open(labels_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # Format: filename dataset=indian_plates
            parts = line.split()
            if len(parts) < 1:
                continue

            filename = parts[0]
            labels_dict[filename] = True

    count = 0
    # Iterate through cropped images
    for img_path in sorted(cache_dir.glob("indian_plates_*.png")):
        filename = img_path.name

        if filename not in labels_dict:
            continue

        try:
            img = Image.open(img_path).convert('RGB')

            # Use split as the "text" label since these are images of objects, not text
            meta = {
                "dataset": "indian_plates_kaggle",
                "split": split,
            }

            yield img, split, meta

            count += 1
            if max_samples is not None and count >= max_samples:
                break
        except Exception as e:
            print(f"Warning: Could not load image {img_path}: {e}")
            continue


# ---------------------------------------------------------
# CCPD2019 variant cropped images
# ---------------------------------------------------------

def _iter_ccpd2019_variant(dataset_name: str, split: str, max_samples: int | None = None) -> Iterator[Tuple[Image.Image, str, Dict]]:
    """
    Iterate over CCPD2019 variant cropped license plate images from local directory.
    Split should be 'train' (only split available).

    Expects directory structure:
    - ~/.cache/ccpd2019_crops/ccpd_*/
      - labels.txt (format: filename brightness=X blurriness=Y)
      - ccpd_*_*.png (cropped license plate images)
    """
    cache_dir = DATASETS[dataset_name]["cache_dir"]

    if not cache_dir.exists():
        return

    labels_file = cache_dir / "labels.txt"
    if not labels_file.exists():
        return

    # Load labels
    labels_dict = {}
    with open(labels_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # Format: filename brightness=X blurriness=Y
            parts = line.split()
            if len(parts) < 1:
                continue

            filename = parts[0]
            labels_dict[filename] = True

    count = 0
    # Iterate through cropped images
    for img_path in sorted(cache_dir.glob("*.png")):
        filename = img_path.name

        if filename not in labels_dict:
            continue

        try:
            img = Image.open(img_path).convert('RGB')

            # Use split as the "text" label since these are images of objects, not text
            meta = {
                "dataset": dataset_name,
                "split": split,
            }

            yield img, split, meta

            count += 1
            if max_samples is not None and count >= max_samples:
                break
        except Exception as e:
            print(f"Warning: Could not load image {img_path}: {e}")
            continue


# ---------------------------------------------------------
# Mercosur cropped images
# ---------------------------------------------------------

def _iter_mercosur(split: str, max_samples: int | None = None) -> Iterator[Tuple[Image.Image, str, Dict]]:
    """
    Iterate over Mercosur cropped license plate images from local directory.
    Split should be 'train' (only split available).

    Expects directory structure:
    - ~/.cache/mercosur_crops/
      - labels.txt (format: filename source_class=X source_image=Y)
      - mercosur_*.png (cropped license plate images)

    Mercosur dataset contains 5 source classes based on image origin:
    - monitoring_system (2925 images)
    - parking_lot1 (566 images)
    - parking_lot2 (23 images)
    - parking_lot3 (11 images)
    - cropped_parking_lot (315 images)
    """
    cache_dir = DATASETS["mercosur"]["cache_dir"]

    if not cache_dir.exists():
        return

    labels_file = cache_dir / "labels.txt"
    if not labels_file.exists():
        return

    # Load labels
    labels_dict = {}
    with open(labels_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # Format: filename source_class=X source_image=Y
            parts = line.split()
            if len(parts) < 1:
                continue

            filename = parts[0]
            labels_dict[filename] = True

    count = 0
    # Iterate through cropped images
    for img_path in sorted(cache_dir.glob("mercosur_*.png")):
        filename = img_path.name

        if filename not in labels_dict:
            continue

        try:
            img = Image.open(img_path).convert('RGB')

            # Use split as the "text" label since these are images of objects, not text
            meta = {
                "dataset": "mercosur",
                "split": split,
            }

            yield img, split, meta

            count += 1
            if max_samples is not None and count >= max_samples:
                break
        except Exception as e:
            print(f"Warning: Could not load image {img_path}: {e}")
            continue


# ---------------------------------------------------------
# CRPD cropped images
# ---------------------------------------------------------

def _iter_crpd(split: str, max_samples: int | None = None) -> Iterator[Tuple[Image.Image, str, Dict]]:
    """
    Iterate over CRPD cropped license plate images from local directory.
    Split should be 'train', 'test', or 'val'.

    Expects directory structure:
    - ~/.cache/crpd_crops/
      - labels.txt (format: filename variant=X split=Y type=Z content=W)
      - crpd_*.png (cropped license plate images)

    CRPD dataset contains 3 variants:
    - CRPD_multi (multi-plate images)
    - CRPD_single (single-plate images)
    - CRPD_double (double-plate images)

    Each variant has train, test, val splits.
    Type field indicates plate type (0 or 1).
    Content field contains the actual license plate text (in Chinese).
    """
    cache_dir = DATASETS["crpd"]["cache_dir"]

    if not cache_dir.exists():
        return

    labels_file = cache_dir / "labels.txt"
    if not labels_file.exists():
        return

    # Load labels and filter by split
    labels_dict = {}
    with open(labels_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # Format: filename variant=X split=Y type=Z content=W
            parts = line.split()
            if len(parts) < 4:
                continue

            filename = parts[0]

            # Find split=Y in the parts
            file_split = None
            for part in parts[1:]:
                if part.startswith('split='):
                    _, file_split = part.split('=', 1)
                    break

            if file_split == split:
                labels_dict[filename] = True

    count = 0
    # Iterate through cropped images
    for img_path in sorted(cache_dir.glob("crpd_*.png")):
        filename = img_path.name

        if filename not in labels_dict:
            continue

        try:
            img = Image.open(img_path).convert('RGB')

            # Use split as the "text" label since these are images of objects, not text
            meta = {
                "dataset": "crpd",
                "split": split,
            }

            yield img, split, meta

            count += 1
            if max_samples is not None and count >= max_samples:
                break
        except Exception as e:
            print(f"Warning: Could not load image {img_path}: {e}")
            continue


# ---------------------------------------------------------
# Unified sample iterator
# ---------------------------------------------------------

def iter_dataset(
    name: str,
    split: str,
    max_samples: int | None = None,
) -> Iterator[Tuple[Image.Image, str, Dict]]:
    """
    Iterate over (image, text, metadata) tuples.

    - image: PIL.Image
    - text: string (split name for most LP crops, actual text for cocotext)
    - metadata: dict with dataset + split info

    Available datasets: cocotext, roboflow_lpr, kaggle_lp, indian_plates_kaggle,
                       ccpd2019_{base,blur,challenge,db,fn,np,rotate,tilt,weather},
                       mercosur, crpd
    """
    cfg = DATASETS[name]

    # Handle COCO Text from local directory
    if name == "cocotext":
        yield from _iter_cocotext(split, max_samples)
        return

    # Handle Roboflow LPR from local directory
    if name == "roboflow_lpr":
        yield from _iter_roboflow_lpr(split, max_samples)
        return

    # Handle Kaggle LP from local directory
    if name == "kaggle_lp":
        yield from _iter_kaggle_lp(split, max_samples)
        return

    # Handle Indian Plates Kaggle from local directory
    if name == "indian_plates_kaggle":
        yield from _iter_indian_plates_kaggle(split, max_samples)
        return

    # Handle CCPD2019 variants from local directory
    if name.startswith("ccpd2019_"):
        yield from _iter_ccpd2019_variant(name, split, max_samples)
        return

    # Handle Mercosur from local directory
    if name == "mercosur":
        yield from _iter_mercosur(split, max_samples)
        return

    # Handle CRPD from local directory
    if name == "crpd":
        yield from _iter_crpd(split, max_samples)
        return

    raise ValueError(f"Unknown dataset: {name}")


# ---------------------------------------------------------
# Convenience: build an anchor pool
# ---------------------------------------------------------

def build_anchor_pool(
    max_per_dataset: int = 1024,
) -> Iterator[Tuple[Image.Image, str, Dict]]:
    """
    Yield a mixed stream of samples across all LP crop datasets,
    suitable for RDM anchor selection.
    """
    for name, cfg in DATASETS.items():
        for split in cfg["splits"]:
            yield from iter_dataset(
                name,
                split,
                max_samples=max_per_dataset,
            )


# ---------------------------------------------------------
# Example usage
# ---------------------------------------------------------

if __name__ == "__main__":
    print("Iterating LP crops datasets...\n")

    total = 0
    for img, text, meta in build_anchor_pool(max_per_dataset=10):
        assert isinstance(img, Image.Image)
        assert isinstance(text, str)

        print(
            f"{meta['dataset']:>20} | {meta['split']:<10} | "
            f"size={img.size} | text='{text[:30]}'"
        )

        total += 1

    print(f"\nLoaded {total} samples successfully.")
