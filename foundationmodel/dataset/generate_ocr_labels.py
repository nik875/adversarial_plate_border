#!/usr/bin/env python3
"""
Generate OCR text labels for all cropped license plate datasets.

Uses fast-plate-ocr's cct-s-v1-global-model (largest CCT model) to run
inference on every cropped plate image and writes results to ocr_labels.csv
inside each dataset's cache directory.

Usage:
    python foundationmodel/dataset/generate_ocr_labels.py [--batch-size N] [--device auto|cpu|cuda]

Output per dataset (e.g. ~/.cache/roboflow_lpr_crops/ocr_labels.csv):
    filename,ocr_text,confidence
    roboflow_lpr_00001.png,ABC123,0.9231
    ...
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from fast_plate_ocr import LicensePlateRecognizer

# ---------------------------------------------------------------------------
# Dataset registry  (same directories as load_lp_crops.py)
# ---------------------------------------------------------------------------

DATASETS: dict[str, dict] = {
    "cocotext": {
        "cache_dir": Path.home() / ".cache" / "cocotext_crops",
        "glob": "cocotext_*.png",
    },
    "roboflow_lpr": {
        "cache_dir": Path.home() / ".cache" / "roboflow_lpr_crops",
        "glob": "roboflow_lpr_*.png",
    },
    "kaggle_lp": {
        "cache_dir": Path.home() / ".cache" / "kaggle_lp_crops",
        "glob": "kaggle_lp_*.png",
    },
    "indian_plates_kaggle": {
        "cache_dir": Path.home() / ".cache" / "indian_plates_kaggle_crops",
        "glob": "indian_plates_*.png",
    },
    "ccpd2019_base": {
        "cache_dir": Path.home() / ".cache" / "ccpd2019_crops" / "ccpd_base",
        "glob": "*.png",
    },
    "ccpd2019_blur": {
        "cache_dir": Path.home() / ".cache" / "ccpd2019_crops" / "ccpd_blur",
        "glob": "*.png",
    },
    "ccpd2019_challenge": {
        "cache_dir": Path.home() / ".cache" / "ccpd2019_crops" / "ccpd_challenge",
        "glob": "*.png",
    },
    "ccpd2019_db": {
        "cache_dir": Path.home() / ".cache" / "ccpd2019_crops" / "ccpd_db",
        "glob": "*.png",
    },
    "ccpd2019_fn": {
        "cache_dir": Path.home() / ".cache" / "ccpd2019_crops" / "ccpd_fn",
        "glob": "*.png",
    },
    "ccpd2019_np": {
        "cache_dir": Path.home() / ".cache" / "ccpd2019_crops" / "ccpd_np",
        "glob": "*.png",
    },
    "ccpd2019_rotate": {
        "cache_dir": Path.home() / ".cache" / "ccpd2019_crops" / "ccpd_rotate",
        "glob": "*.png",
    },
    "ccpd2019_tilt": {
        "cache_dir": Path.home() / ".cache" / "ccpd2019_crops" / "ccpd_tilt",
        "glob": "*.png",
    },
    "ccpd2019_weather": {
        "cache_dir": Path.home() / ".cache" / "ccpd2019_crops" / "ccpd_weather",
        "glob": "*.png",
    },
    "mercosur": {
        "cache_dir": Path.home() / ".cache" / "mercosur_crops",
        "glob": "mercosur_*.png",
    },
    "crpd": {
        "cache_dir": Path.home() / ".cache" / "crpd_crops",
        "glob": "crpd_*.png",
    },
}


def process_dataset(
    name: str,
    cfg: dict,
    recognizer: LicensePlateRecognizer,
    batch_size: int,
    overwrite: bool,
) -> int:
    """Run OCR on all crops in one dataset and write ocr_labels.csv."""
    cache_dir: Path = cfg["cache_dir"]

    if not cache_dir.exists():
        print(f"  [skip] {name}: cache dir not found ({cache_dir})", file=sys.stderr)
        return 0

    image_paths = sorted(cache_dir.glob(cfg["glob"]))
    if not image_paths:
        print(f"  [skip] {name}: no images found in {cache_dir}", file=sys.stderr)
        return 0

    out_csv = cache_dir / "ocr_labels.csv"
    if out_csv.exists() and not overwrite:
        print(f"  [skip] {name}: {out_csv} already exists (use --overwrite to redo)")
        return 0

    total = len(image_paths)
    written = 0

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "ocr_text", "confidence"])

        pbar = tqdm(
            total=total,
            desc=f"  {name}",
            unit="img",
            leave=True,
            dynamic_ncols=True,
        )

        # Process in batches
        for batch_start in range(0, total, batch_size):
            batch_paths = image_paths[batch_start : batch_start + batch_size]
            batch_arrays: list[np.ndarray] = []
            valid_paths: list[Path] = []

            for p in batch_paths:
                try:
                    img = Image.open(p).convert("RGB")
                    batch_arrays.append(np.array(img))
                    valid_paths.append(p)
                except Exception as e:
                    tqdm.write(f"    Warning: could not load {p.name}: {e}")
                    pbar.update(1)

            if not batch_arrays:
                continue

            try:
                plates, confidences = recognizer.run(
                    batch_arrays, return_confidence=True
                )
            except Exception as e:
                tqdm.write(f"    Warning: OCR failed on batch at {batch_start}: {e}")
                pbar.update(len(batch_arrays))
                continue

            for path, plate_text, conf_arr in zip(valid_paths, plates, confidences):
                # conf_arr is shape (plate_slots,) — mean over slots
                mean_conf = float(np.mean(conf_arr))
                writer.writerow([path.name, plate_text, f"{mean_conf:.4f}"])
                written += 1
                pbar.update(1)

        pbar.close()

    print(f"  -> wrote {written}/{total} labels to {out_csv}")
    return written


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate CCT OCR text labels for all LP crop datasets"
    )
    parser.add_argument(
        "--batch-size", type=int, default=64,
        help="Images per OCR batch (default: 64)",
    )
    parser.add_argument(
        "--device", default="auto", choices=["auto", "cpu", "cuda"],
        help="Inference device (default: auto)",
    )
    parser.add_argument(
        "--datasets", nargs="*", default=None,
        help="Specific dataset names to process (default: all)",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Overwrite existing ocr_labels.csv files",
    )
    args = parser.parse_args()

    print("Loading cct-s-v1-global-model (largest CCT, global plates)...")
    recognizer = LicensePlateRecognizer(
        hub_ocr_model="cct-s-v1-global-model",
        device=args.device,
    )
    print("Model loaded.\n")

    datasets_to_run = (
        {k: DATASETS[k] for k in args.datasets if k in DATASETS}
        if args.datasets
        else DATASETS
    )

    if args.datasets:
        missing = [k for k in args.datasets if k not in DATASETS]
        if missing:
            print(f"Warning: unknown datasets ignored: {missing}", file=sys.stderr)

    total_written = 0
    for name, cfg in datasets_to_run.items():
        print(f"\nProcessing dataset: {name}")
        n = process_dataset(name, cfg, recognizer, args.batch_size, args.overwrite)
        total_written += n

    print(f"\nDone. Total labels written: {total_written}")


if __name__ == "__main__":
    main()
