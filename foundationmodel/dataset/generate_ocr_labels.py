#!/usr/bin/env python3
"""
Generate OCR text labels for license plate datasets.

Reads annotations directly from the raw datasets (no pre-cropped images),
crops each plate region in memory, and runs fast-plate-ocr's
cct-s-v1-global-model (largest CCT model) for inference.

Datasets: roboflow_lpr, kaggle_lp, indian_plates_kaggle, mercosur

Output:
  ocr_labels.csv        — one row per plate detection
  viz/                  — 100 sample images with bbox + OCR text overlaid

CSV columns:
  dataset, image_path, x1, y1, x2, y2, ocr_text, confidence

Usage:
  python foundationmodel/dataset/generate_ocr_labels.py
  python foundationmodel/dataset/generate_ocr_labels.py --output ocr_labels.csv --viz-dir viz
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

from fast_plate_ocr import LicensePlateRecognizer

# ---------------------------------------------------------------------------
# Raw dataset directories
# ---------------------------------------------------------------------------

ROBOFLOW_LPR_DIR     = Path.home() / ".cache" / "roboflow_lpr_dataset"
KAGGLE_LP_DIR        = Path.home() / ".cache" / "kaggle_lp_detection"
INDIAN_PLATES_DIR    = Path.home() / ".cache" / "indian_plates_kaggle"
MERCOSUR_DIR         = Path.home() / ".cache" / "Mercosur"

# ---------------------------------------------------------------------------
# Annotation iterators  →  yield (image_path: Path, x1, y1, x2, y2)
# ---------------------------------------------------------------------------

def _iter_roboflow_lpr() -> Iterator[tuple[Path, int, int, int, int]]:
    """COCO JSON annotations; bbox is [x, y, w, h]."""
    if not ROBOFLOW_LPR_DIR.exists():
        print(f"  [skip] roboflow_lpr: {ROBOFLOW_LPR_DIR} not found")
        return

    for split in ("train", "test", "valid"):
        split_dir = ROBOFLOW_LPR_DIR / split
        anno_file = split_dir / "_annotations.coco.json"
        if not anno_file.exists():
            continue

        with open(anno_file) as f:
            coco = json.load(f)

        id_to_img = {img["id"]: img for img in coco.get("images", [])}

        for ann in coco.get("annotations", []):
            bbox = ann.get("bbox")
            if not bbox or len(bbox) != 4:
                continue
            img_info = id_to_img.get(ann["image_id"])
            if img_info is None:
                continue
            img_path = split_dir / img_info["file_name"]
            if not img_path.exists():
                continue
            x, y, w, h = bbox
            yield img_path, int(x), int(y), int(x + w), int(y + h)


def _iter_kaggle_lp() -> Iterator[tuple[Path, int, int, int, int]]:
    """Pascal VOC XML annotations; bbox is [xmin, ymin, xmax, ymax]."""
    anno_dir  = KAGGLE_LP_DIR / "annotations"
    image_dir = KAGGLE_LP_DIR / "images"
    if not anno_dir.exists() or not image_dir.exists():
        print(f"  [skip] kaggle_lp: {KAGGLE_LP_DIR} not found")
        return

    for xml_path in sorted(anno_dir.glob("*.xml")):
        try:
            root = ET.parse(xml_path).getroot()
            img_filename = root.find("filename").text
            img_path = image_dir / img_filename
            if not img_path.exists():
                continue
            for obj in root.findall("object"):
                name = obj.find("name").text.lower()
                if name not in ("licence", "license", "plate"):
                    continue
                bb = obj.find("bndbox")
                x1 = int(bb.find("xmin").text)
                y1 = int(bb.find("ymin").text)
                x2 = int(bb.find("xmax").text)
                y2 = int(bb.find("ymax").text)
                yield img_path, x1, y1, x2, y2
        except Exception:
            continue


def _iter_indian_plates() -> Iterator[tuple[Path, int, int, int, int]]:
    """YOLO normalized annotations; one label file per image."""
    image_dir = INDIAN_PLATES_DIR / "images"
    label_dir = INDIAN_PLATES_DIR / "labels"
    if not image_dir.exists() or not label_dir.exists():
        print(f"  [skip] indian_plates_kaggle: {INDIAN_PLATES_DIR} not found")
        return

    for img_path in sorted(image_dir.glob("*.png")):
        label_path = label_dir / (img_path.stem + ".txt")
        if not label_path.exists():
            continue
        try:
            with Image.open(img_path) as im:
                iw, ih = im.size
            with open(label_path) as f:
                line = f.readline().strip().split()
            if len(line) < 5:
                continue
            xc, yc, w, h = float(line[1]), float(line[2]), float(line[3]), float(line[4])
            x1 = int((xc - w / 2) * iw)
            y1 = int((yc - h / 2) * ih)
            x2 = int((xc + w / 2) * iw)
            y2 = int((yc + h / 2) * ih)
            yield img_path, x1, y1, x2, y2
        except Exception:
            continue


def _iter_mercosur() -> Iterator[tuple[Path, int, int, int, int]]:
    """CSV annotations with normalized YOLO-style bbox columns."""
    csv_file  = MERCOSUR_DIR / "dataset.csv"
    image_dir = MERCOSUR_DIR / "images"
    if not csv_file.exists() or not image_dir.exists():
        print(f"  [skip] mercosur: {MERCOSUR_DIR} not found")
        return

    with open(csv_file) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    for row in rows:
        img_path = image_dir / row["image"]
        if not img_path.exists():
            continue
        try:
            with Image.open(img_path) as im:
                iw, ih = im.size
            xc = float(row["x_center"])
            yc = float(row["y_center"])
            w  = float(row["width"])
            h  = float(row["height"])
            x1 = int((xc - w / 2) * iw)
            y1 = int((yc - h / 2) * ih)
            x2 = int((xc + w / 2) * iw)
            y2 = int((yc + h / 2) * ih)
            yield img_path, x1, y1, x2, y2
        except Exception:
            continue


DATASET_ITERS = {
    "roboflow_lpr":       _iter_roboflow_lpr,
    "kaggle_lp":          _iter_kaggle_lp,
    "indian_plates_kaggle": _iter_indian_plates,
    "mercosur":           _iter_mercosur,
}

# ---------------------------------------------------------------------------
# Crop helper — replicates what fast-plate-ocr does internally
# ---------------------------------------------------------------------------

def crop_plate(img_path: Path, x1: int, y1: int, x2: int, y2: int) -> np.ndarray | None:
    """
    Open the full image, crop the plate region, return as uint8 RGB numpy array.
    The LicensePlateRecognizer.run() call will handle resize + normalisation.
    """
    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if img is None:
        return None
    h, w = img.shape[:2]
    x1c, y1c = max(0, x1), max(0, y1)
    x2c, y2c = min(w, x2), min(h, y2)
    if x2c <= x1c or y2c <= y1c:
        return None
    crop_bgr = img[y1c:y2c, x1c:x2c]
    return cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)

# ---------------------------------------------------------------------------
# Visualisation helper
# ---------------------------------------------------------------------------

def draw_sample(img_path: Path, x1: int, y1: int, x2: int, y2: int,
                ocr_text: str, confidence: float) -> Image.Image:
    """Draw bbox + OCR text on the original image; return PIL Image."""
    img = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    # Draw bbox
    draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=3)

    # Label above bbox
    label = f"{ocr_text}  ({confidence:.2f})"
    # Try a slightly larger font; fall back to default
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
    except Exception:
        font = ImageFont.load_default()

    # Background rectangle for legibility
    try:
        tx0, ty0, tx1, ty1 = draw.textbbox((x1, max(0, y1 - 24)), label, font=font)
    except AttributeError:
        tx0, ty0 = x1, max(0, y1 - 24)
        tx1, ty1 = tx0 + len(label) * 10, ty0 + 20

    draw.rectangle([tx0, ty0, tx1, ty1], fill=(0, 255, 0))
    draw.text((tx0, ty0), label, fill=(0, 0, 0), font=font)

    return img

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate CCT OCR labels from raw LP datasets (no pre-crops)"
    )
    parser.add_argument("--output",   default="ocr_labels.csv",
                        help="Output CSV path (default: ocr_labels.csv)")
    parser.add_argument("--viz-dir",  default="viz",
                        help="Directory for visualisation samples (default: viz)")
    parser.add_argument("--viz-n",    type=int, default=100,
                        help="Number of visualisation samples (default: 100)")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Crops per OCR batch (default: 64)")
    parser.add_argument("--device",   default="auto",
                        choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()

    output_path = Path(args.output)
    viz_dir     = Path(args.viz_dir)
    viz_dir.mkdir(parents=True, exist_ok=True)

    print("Loading cct-s-v1-global-model (largest CCT, global plates)...")
    recognizer = LicensePlateRecognizer(
        hub_ocr_model="cct-s-v1-global-model",
        device=args.device,
    )
    print("Model loaded.\n")

    # -----------------------------------------------------------------------
    # First pass: collect all (dataset, img_path, x1, y1, x2, y2) entries
    # so we know the total for the progress bar.
    # -----------------------------------------------------------------------
    print("Scanning annotations...")
    all_entries: list[tuple[str, Path, int, int, int, int]] = []
    for ds_name, iter_fn in DATASET_ITERS.items():
        before = len(all_entries)
        for img_path, x1, y1, x2, y2 in iter_fn():
            all_entries.append((ds_name, img_path, x1, y1, x2, y2))
        print(f"  {ds_name}: {len(all_entries) - before} detections")
    print(f"  Total: {len(all_entries)}\n")

    # -----------------------------------------------------------------------
    # Second pass: run OCR in batches, stream CSV rows
    # -----------------------------------------------------------------------
    results: list[tuple[str, Path, int, int, int, int, str, float]] = []

    with open(output_path, "w", newline="", encoding="utf-8") as csv_f:
        writer = csv.writer(csv_f)
        writer.writerow(["dataset", "image_path", "x1", "y1", "x2", "y2",
                         "ocr_text", "confidence"])

        batch_size = args.batch_size
        total = len(all_entries)

        with tqdm(total=total, unit="plate", dynamic_ncols=True) as pbar:
            for batch_start in range(0, total, batch_size):
                batch = all_entries[batch_start : batch_start + batch_size]

                crops: list[np.ndarray] = []
                valid: list[tuple[str, Path, int, int, int, int]] = []

                for ds_name, img_path, x1, y1, x2, y2 in batch:
                    crop = crop_plate(img_path, x1, y1, x2, y2)
                    if crop is None:
                        pbar.update(1)
                        continue
                    crops.append(crop)
                    valid.append((ds_name, img_path, x1, y1, x2, y2))

                if not crops:
                    continue

                try:
                    plates, confs = recognizer.run(crops, return_confidence=True)
                except Exception as e:
                    tqdm.write(f"  Warning: OCR batch failed: {e}")
                    pbar.update(len(crops))
                    continue

                for (ds_name, img_path, x1, y1, x2, y2), plate, conf_arr in \
                        zip(valid, plates, confs):
                    mean_conf = float(np.mean(conf_arr))
                    writer.writerow([ds_name, str(img_path),
                                     x1, y1, x2, y2, plate, f"{mean_conf:.4f}"])
                    results.append((ds_name, img_path, x1, y1, x2, y2,
                                    plate, mean_conf))
                    pbar.update(1)

    print(f"\nWrote {len(results)} rows to {output_path}")

    # -----------------------------------------------------------------------
    # Visualisation: sample viz_n evenly across results
    # -----------------------------------------------------------------------
    n_viz = min(args.viz_n, len(results))
    indices = sorted(random.sample(range(len(results)), n_viz))
    print(f"Saving {n_viz} visualisation images to {viz_dir}/")

    for i, idx in enumerate(tqdm(indices, unit="img", dynamic_ncols=True)):
        ds_name, img_path, x1, y1, x2, y2, plate, conf = results[idx]
        try:
            vis = draw_sample(img_path, x1, y1, x2, y2, plate, conf)
            out_name = f"{i:04d}_{ds_name}_{img_path.stem}.jpg"
            vis.save(viz_dir / out_name, quality=85)
        except Exception as e:
            tqdm.write(f"  Warning: viz failed for {img_path.name}: {e}")

    print("Done.")


if __name__ == "__main__":
    main()
