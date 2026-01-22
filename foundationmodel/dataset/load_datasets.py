#!/usr/bin/env python3
"""
Unified OCR dataset loader for MI / representational profiling.

Datasets are loaded via Hugging Face Datasets or direct downloads.
This script provides:
- automatic download
- uniform image access
- lightweight iteration for anchor sampling

Intended use:
- activation profiling
- RDM construction
- layer phenotype estimation

NOT intended for training loops.
"""

from typing import Dict, Iterator, Tuple
from pathlib import Path
import subprocess
import tarfile
import os

import torch
from PIL import Image
from datasets import load_dataset


# ---------------------------------------------------------
# Dataset registry
# ---------------------------------------------------------

DATASETS = {
    "iiit5k": {
        "source": "direct",
        "url": "https://cdn.iiit.ac.in/cdn/cvit.iiit.ac.in/images/Projects/SceneTextUnderstanding/IIIT5K-Word_V3.0.tar.gz",
        "splits": ["train", "test"],
    },
    "mjsynth": {
        "hf_id": "Synth90k",
        "image_key": "image",
        "text_key": "label",
        "splits": ["train"],
    },
    "iam_line": {
        "hf_id": "Teklia/IAM-line",
        "image_key": "image",
        "text_key": "text",
        "splits": ["train", "validation", "test"],
    },
    "icdar2015": {
        "hf_id": "MiXaiLL76/ICDAR2015_OCR",
        "image_key": "image",
        "text_key": "text",
        "splits": ["train", "test"],
    },
    "funsd": {
        "hf_id": "funsd",
        "image_key": "image",
        "text_key": "words",
        "splits": ["train", "test"],
    },
}


# ---------------------------------------------------------
# IIIT5K direct download and loading
# ---------------------------------------------------------

def _ensure_iiit5k_extracted(cache_dir: Path = Path.home() / ".cache" / "iiit5k") -> Path:
    """
    Download and extract IIIT5K dataset if not already present.
    Returns the path to the extracted dataset directory.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    extracted_dir = cache_dir / "IIIT5K_3000"

    if extracted_dir.exists():
        return extracted_dir

    tar_path = cache_dir / "IIIT5K-Word_V3.0.tar.gz"

    # Download if not present
    if not tar_path.exists():
        print(f"Downloading IIIT5K from CDN...")
        url = DATASETS["iiit5k"]["url"]
        subprocess.run([
            "curl", "-L", "--output", str(tar_path), url
        ], check=True)

    # Extract
    print(f"Extracting IIIT5K to {extracted_dir}...")
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(cache_dir)

    return extracted_dir


def _iter_iiit5k(split: str, max_samples: int | None = None) -> Iterator[Tuple[Image.Image, str, Dict]]:
    """
    Iterate over IIIT5K dataset samples.
    Split should be 'train' or 'test'.
    """
    extracted_dir = _ensure_iiit5k_extracted()
    words_dir = extracted_dir / "words_v001d"

    # Map split names to annotation files
    split_files = {
        "train": "IIIT5K_3000_train.txt",
        "test": "IIIT5K_3000_test.txt",
    }

    anno_file = extracted_dir / split_files[split]

    if not anno_file.exists():
        raise FileNotFoundError(f"Annotation file not found: {anno_file}")

    count = 0
    with open(anno_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # Parse annotation line: filename label
            parts = line.split()
            if len(parts) < 2:
                continue

            filename = parts[0]
            label = parts[1]

            # Construct image path
            img_path = words_dir / filename

            if not img_path.exists():
                continue

            try:
                img = Image.open(img_path).convert('RGB')
            except Exception as e:
                print(f"Warning: Could not load image {img_path}: {e}")
                continue

            meta = {
                "dataset": "iiit5k",
                "split": split,
            }

            yield img, label, meta

            count += 1
            if max_samples is not None and count >= max_samples:
                break


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
    - text: string (or joined words for FUNSD)
    - metadata: dict with dataset + split info
    """
    cfg = DATASETS[name]

    # Handle IIIT5K with direct download
    if name == "iiit5k":
        yield from _iter_iiit5k(split, max_samples)
        return

    # Handle other datasets via Hugging Face
    ds = load_dataset(cfg["hf_id"], split=split)

    count = 0
    for sample in ds:
        img = sample[cfg["image_key"]]

        # normalize text field
        if name == "funsd":
            text = " ".join(sample[cfg["text_key"]])
        else:
            text = sample[cfg["text_key"]]

        meta = {
            "dataset": name,
            "split": split,
        }

        yield img, text, meta

        count += 1
        if max_samples is not None and count >= max_samples:
            break


# ---------------------------------------------------------
# Convenience: build an anchor pool
# ---------------------------------------------------------

def build_anchor_pool(
    max_per_dataset: int = 1024,
) -> Iterator[Tuple[Image.Image, str, Dict]]:
    """
    Yield a mixed stream of samples across datasets,
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
    print("Downloading and iterating datasets...\n")

    total = 0
    for img, text, meta in build_anchor_pool(max_per_dataset=10):
        assert isinstance(img, Image.Image)
        assert isinstance(text, str)

        print(
            f"{meta['dataset']:>10} | {meta['split']:<10} | "
            f"size={img.size} | text='{text[:30]}'"
        )

        total += 1

    print(f"\nLoaded {total} samples successfully.")
