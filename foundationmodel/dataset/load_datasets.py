#!/usr/bin/env python3
"""
Unified OCR dataset loader for MI / representational profiling.

All datasets are loaded via Hugging Face Datasets.
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

import torch
from PIL import Image
from datasets import load_dataset


# ---------------------------------------------------------
# Dataset registry
# ---------------------------------------------------------

DATASETS = {
    "iiit5k": {
        "hf_id": "IIIT-5K",
        "image_key": "image",
        "text_key": "label",
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
