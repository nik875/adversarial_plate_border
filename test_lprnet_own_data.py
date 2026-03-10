#!/usr/bin/env python3
"""
test_lprnet_own_data.py

Tests NVIDIA TAO US LPRNet on our own dataset (preproc_labels.csv).
Ground truth is always "VRJ7774". Crops a rectangular bounding box around
the plate from the 4 corner points (p1–p4) stored in the CSV.

Usage:
    python test_lprnet_own_data.py \
        --model weights/lprnet_deployable_onnx_v1.1/us_lprnet_baseline18_deployable.onnx

Requires:
    pip install onnxruntime pillow numpy pandas tqdm pillow-heif
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except ImportError:
    pass  # HEIC support unavailable; non-HEIC images will still work

sys.path.insert(0, str(Path(__file__).parent))
from test_lprnet_inference import (
    ALPHABET, BLANK_IDX,
    load_model, preprocess, ctc_decode,
)

GT = "VRJ7774"


def corners_to_bbox(row, padding: int = 4):
    """Return (x1, y1, x2, y2) axis-aligned bbox from four plate corners."""
    xs = [row.p1_x, row.p2_x, row.p3_x, row.p4_x]
    ys = [row.p1_y, row.p2_y, row.p3_y, row.p4_y]
    return (min(xs) - padding, min(ys) - padding,
            max(xs) + padding, max(ys) + padding)


def run_inference(session, input_name, output_name, outputs_probs,
                  pil_img: Image.Image) -> str:
    inp = preprocess(pil_img)
    out = session.run([output_name], {input_name: inp})[0]
    indices = out[0].argmax(axis=-1) if outputs_probs else out[0].flatten()
    return ctc_decode(indices)


def main():
    parser = argparse.ArgumentParser(
        description="Test NVIDIA TAO US LPRNet on our own dataset"
    )
    parser.add_argument("--model", required=True,
                        help="Path to us_lprnet_baseline18_deployable.onnx")
    parser.add_argument("--labels", default="preproc_labels.csv")
    parser.add_argument("--padding", type=int, default=4,
                        help="Pixels of padding around bbox (default: 4)")
    args = parser.parse_args()

    df = pd.read_csv(args.labels)

    print(f"\n[NVIDIA TAO US LPRNet — Own Dataset]")
    print(f"  Model   : {args.model}")
    print(f"  Labels  : {args.labels}  ({len(df)} rows)")
    print(f"  GT text : '{GT}' (all images)")
    print(f"  Alphabet: '{ALPHABET}'  (blank_id={BLANK_IDX})")

    print(f"\n[Model]")
    session, input_name, output_name, outputs_probs = load_model(args.model)

    exact_correct = 0
    char_correct  = 0
    char_total    = 0
    skipped       = 0
    evaluated     = 0
    examples      = []

    pbar = tqdm(df.itertuples(index=False), total=len(df),
                desc="Evaluating", unit="img")

    for row in pbar:
        img_path = Path(row.filename)
        if not img_path.exists():
            skipped += 1
            continue

        try:
            pil_full = Image.open(img_path).convert("RGB")
            iw, ih = pil_full.size
            x1, y1, x2, y2 = corners_to_bbox(row, args.padding)
            x1, y1 = max(0, int(x1)), max(0, int(y1))
            x2, y2 = min(iw, int(x2)), min(ih, int(y2))
            plate_crop = pil_full.crop((x1, y1, x2, y2))
            pred = run_inference(session, input_name, output_name,
                                 outputs_probs, plate_crop)
        except Exception as e:
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

    exact_acc = exact_correct / evaluated if evaluated else 0.0
    char_acc  = char_correct  / char_total if char_total else 0.0

    print(f"\n{'='*60}")
    print(f"  RESULTS — Own Dataset  [NVIDIA TAO LPRNet]")
    print(f"{'='*60}")
    print(f"  GT plate text        : '{GT}'")
    print(f"  Evaluated            : {evaluated}")
    print(f"  Skipped              : {skipped}  (missing / load error)")
    print(f"  Exact match accuracy : {exact_acc:.1%}  ({exact_correct}/{evaluated})")
    print(f"  Character accuracy   : {char_acc:.1%}  ({char_correct}/{char_total})")

    print(f"\n  Examples  (pred  [✓/✗]):")
    for pred, match in examples:
        status = "✓" if match else "✗"
        print(f"    {status}  pred='{pred}'")


if __name__ == "__main__":
    main()
