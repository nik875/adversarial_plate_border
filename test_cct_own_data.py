#!/usr/bin/env python3
"""
test_cct_own_data.py

Tests the CCT-XS-V1 Global plate OCR model (from fast-plate-ocr) on our own
dataset (preproc_labels.csv). Ground truth is always "VRJ7774". Crops a
rectangular bounding box from the 4 plate corners in the CSV.

The CCT model is the primary OCR target in the foundationmodel training pipeline.

Usage:
    python test_cct_own_data.py
    python test_cct_own_data.py --labels preproc_labels.csv --padding 4

Requires:
    pip install fast-plate-ocr pillow numpy pandas tqdm pillow-heif
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except ImportError:
    pass

GT = "VRJ7774"


def corners_to_bbox(row, padding: int = 4):
    xs = [row.p1_x, row.p2_x, row.p3_x, row.p4_x]
    ys = [row.p1_y, row.p2_y, row.p3_y, row.p4_y]
    return (min(xs) - padding, min(ys) - padding,
            max(xs) + padding, max(ys) + padding)


def main():
    parser = argparse.ArgumentParser(
        description="Test CCT-XS-V1 plate OCR on our own dataset"
    )
    parser.add_argument("--labels", default="preproc_labels.csv")
    parser.add_argument("--padding", type=int, default=4)
    args = parser.parse_args()

    try:
        from fast_plate_ocr import ONNXPlateRecognizer
    except ImportError:
        print("ERROR: fast-plate-ocr not found.  pip install fast-plate-ocr")
        raise SystemExit(1)

    print("\n[CCT-XS-V1 Global — Own Dataset]")
    print("  Model  : cct-xs-v1-global-model (fast-plate-ocr)")
    print(f"  Labels : {args.labels}")
    print(f"  GT     : '{GT}' (all images)")

    print("\n[Loading model]")
    model = ONNXPlateRecognizer("cct-xs-v1-global-model")
    print("  OK")

    df = pd.read_csv(args.labels)
    print(f"  {len(df)} rows")

    exact_correct = 0
    char_correct  = 0
    char_total    = 0
    skipped       = 0
    evaluated     = 0
    examples      = []
    first_errors  = []

    pbar = tqdm(df.itertuples(index=False), total=len(df),
                desc="Evaluating", unit="img")

    for row in pbar:
        img_path = Path(row.filename)
        if not img_path.exists():
            if len(first_errors) < 3:
                first_errors.append(f"File not found: {img_path}")
            skipped += 1
            continue

        try:
            pil_full = Image.open(img_path).convert("RGB")
            iw, ih = pil_full.size
            x1, y1, x2, y2 = corners_to_bbox(row, args.padding)
            x1, y1 = max(0, int(x1)), max(0, int(y1))
            x2, y2 = min(iw, int(x2)), min(ih, int(y2))
            crop_np = np.array(pil_full.crop((x1, y1, x2, y2)))   # uint8 RGB
            preds = model.run(crop_np)
            pred = preds[0].strip().upper() if preds else ""
        except Exception as e:
            if len(first_errors) < 3:
                first_errors.append(f"{type(e).__name__}: {e}")
            skipped += 1
            continue

        evaluated += 1
        match = pred == GT
        exact_correct += int(match)
        n_right = sum(a == b for a, b in zip(pred, GT))
        char_correct += n_right
        char_total   += len(GT)

        if len(examples) < 40:
            examples.append((pred, match))

        pbar.set_postfix({"exact_acc": f"{exact_correct/evaluated:.1%}",
                          "skip": skipped})

    pbar.close()

    if first_errors:
        print(f"\n  [First skip reasons]")
        for msg in first_errors:
            print(f"    {msg}")

    exact_acc = exact_correct / evaluated if evaluated else 0.0
    char_acc  = char_correct  / char_total if char_total else 0.0

    print(f"\n{'='*60}")
    print(f"  RESULTS — CCT-XS-V1 Global  [Own Dataset]")
    print(f"{'='*60}")
    print(f"  GT plate text        : '{GT}'")
    print(f"  Evaluated            : {evaluated}")
    print(f"  Skipped              : {skipped}")
    print(f"  Exact match accuracy : {exact_acc:.1%}  ({exact_correct}/{evaluated})")
    print(f"  Character accuracy   : {char_acc:.1%}  ({char_correct}/{char_total})")

    print(f"\n  Examples  (pred  [✓/✗]):")
    for pred, match in examples:
        print(f"    {'✓' if match else '✗'}  pred='{pred}'")


if __name__ == "__main__":
    main()
