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
import scipy.io


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
        "hf_id": "priyank-m/MJSynth_text_recognition",
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
    "icdar2013": {
        "source": "local",
        "cache_dir": Path.home() / ".cache" / "icdar2013",
        "splits": ["train"],
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
    extracted_dir = cache_dir / "IIIT5K"

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
    print(f"Extracting IIIT5K to {cache_dir}...")
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(cache_dir)

    return extracted_dir


def _iter_iiit5k(split: str, max_samples: int | None = None) -> Iterator[Tuple[Image.Image, str, Dict]]:
    """
    Iterate over IIIT5K dataset samples.
    Split should be 'train' or 'test'.

    Loads ground truth labels from MATLAB annotation files:
    - traindata.mat / testdata.mat contain image names and ground truth labels
    - Image files are located in train/ and test/ subdirectories
    """
    extracted_dir = _ensure_iiit5k_extracted()

    # Map split names to MATLAB annotation files
    split_files = {
        "train": "traindata.mat",
        "test": "testdata.mat",
    }

    anno_file = extracted_dir / split_files[split]

    if not anno_file.exists():
        raise FileNotFoundError(f"Annotation file not found: {anno_file}")

    # Load MATLAB annotations
    mat_data = scipy.io.loadmat(str(anno_file), simplify_cells=True)
    samples = mat_data[split + "data"]  # 'traindata' or 'testdata'

    if not isinstance(samples, list):
        samples = [samples]

    count = 0
    for sample in samples:
        # Each sample is a dict with: ImgName, GroundTruth, smallLexi, mediumLexi
        img_rel_path = sample.get("ImgName")
        label = sample.get("GroundTruth")

        if img_rel_path is None or label is None:
            continue

        # Construct full image path
        img_path = extracted_dir / img_rel_path

        if not img_path.exists():
            print(f"Warning: Image not found: {img_path}")
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
# ICDAR 2013 Challenge 1 local dataset loading
# ---------------------------------------------------------

def _iter_icdar2013(split: str, max_samples: int | None = None) -> Iterator[Tuple[Image.Image, str, Dict]]:
    """
    Iterate over ICDAR 2013 Challenge 1 dataset samples from local directory.
    Split should be 'train' (only split available).

    Expects directory structure:
    - ~/.cache/icdar2013/Challenge1/
      - gt/ (ground truth text files)
      - *.png (image files)
    """
    cache_dir = DATASETS["icdar2013"]["cache_dir"]
    challenge_dir = cache_dir / "Challenge1"

    if not challenge_dir.exists():
        raise FileNotFoundError(f"ICDAR 2013 dataset not found at: {challenge_dir}")

    gt_dir = challenge_dir / "gt"
    if not gt_dir.exists():
        raise FileNotFoundError(f"Ground truth directory not found: {gt_dir}")

    count = 0
    # Iterate through all image files
    for img_path in sorted(challenge_dir.glob("*.png")):
        # Find corresponding ground truth file
        gt_file = gt_dir / f"{img_path.stem}.txt"

        if not gt_file.exists():
            continue

        try:
            # Read ground truth text
            with open(gt_file, 'r') as f:
                label = f.read().strip()

            if not label:
                continue

            # Load image
            img = Image.open(img_path).convert('RGB')

            meta = {
                "dataset": "icdar2013",
                "split": split,
            }

            yield img, label, meta

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
    - text: string
    - metadata: dict with dataset + split info
    """
    cfg = DATASETS[name]

    # Handle IIIT5K with direct download
    if name == "iiit5k":
        yield from _iter_iiit5k(split, max_samples)
        return

    # Handle ICDAR 2013 from local directory
    if name == "icdar2013":
        yield from _iter_icdar2013(split, max_samples)
        return

    # Handle other datasets via Hugging Face
    ds = load_dataset(cfg["hf_id"], split=split)

    count = 0
    for sample in ds:
        try:
            img = sample[cfg["image_key"]]

            # Ensure image is valid by attempting to access it
            if hasattr(img, 'convert'):
                img = img.convert('RGB')

            text = sample[cfg["text_key"]]

            meta = {
                "dataset": name,
                "split": split,
            }

            yield img, text, meta

            count += 1
            if max_samples is not None and count >= max_samples:
                break
        except Exception as e:
            # Log error details but skip this sample
            import traceback
            print(f"[{name}/{split}] Error loading sample: {type(e).__name__}: {str(e)}", file=__import__('sys').stderr)
            continue


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
