#!/usr/bin/env python3
"""
test_lprnet_own_data.py

Tests NVIDIA TAO US LPRNet on our own dataset (test_set_labels.csv).
Labels come from ALPR detections on extracted video frames.

Usage:
    python test_lprnet_own_data.py \
        --model weights/lprnet_deployable_onnx_v1.1/us_lprnet_baseline18_deployable.onnx
    python test_lprnet_own_data.py --model <path> --labels preproc_labels.csv
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

# Reuse shared constants and helpers from the OpenALPR test script
sys.path.insert(0, str(Path(__file__).parent))
from test_lprnet_inference import (
    ALPHABET, BLANK_IDX, INPUT_H, INPUT_W,
    load_model, preprocess, ctc_decode,
)


def run_inference(session, input_name, output_name, outputs_probs,
                  pil_img: Image.Image) -> str:
    inp = preprocess(pil_img)
    out = session.run([output_name], {input_name: inp})[0]
    if outputs_probs:
        indices = out[0].argmax(axis=-1)
    else:
        indices = out[0].flatten()
    return ctc_decode(indices)


def main():
    parser = argparse.ArgumentParser(
        description="Test NVIDIA TAO US LPRNet on our own dataset"
    )
    parser.add_argument("--model", required=True,
                        help="Path to us_lprnet_baseline18_deployable.onnx")
    parser.add_argument("--labels", default="test_set_labels.csv",
                        help="CSV with original_filename, alpr_text, alpr_x1/y1/x2/y2 "
                             "(default: test_set_labels.csv)")
    parser.add_argument("--image_root", default=".",
                        help="Root dir to resolve relative image paths (default: .)")
    parser.add_argument("--padding", type=int, default=4,
                        help="Pixels of padding around bbox crop (default: 4)")
    args = parser.parse_args()

    import pandas as pd
    df = pd.read_csv(args.labels)

    # Normalise path column name
    path_col = "original_filename" if "original_filename" in df.columns else "filename"
    gt_col   = "alpr_text"

    # Fix Mac absolute paths to local-relative paths
    mac_prefix = "/Users/NikhilKalidasu/Documents/Adversarial Plate/test_images/"
    df[path_col] = df[path_col].str.replace(mac_prefix, "test_images/", regex=False)

    image_root = Path(args.image_root)

    print(f"\n[NVIDIA TAO US LPRNet — Own Dataset]")
    print(f"  Model   : {args.model}")
    print(f"  Labels  : {args.labels}  ({len(df)} rows)")
    print(f"  Alphabet: '{ALPHABET}'  (blank_id={BLANK_IDX})")

    print(f"\n[Model]")
    session, input_name, output_name, outputs_probs = load_model(args.model)

    # ── Evaluate ──────────────────────────────────────────────────────────────
    exact_correct  = 0
    char_correct   = 0
    char_total     = 0
    skipped        = 0
    evaluated      = 0
    examples       = []
    plate_buckets  = defaultdict(lambda: [0, 0])   # gt_text → [correct, total]
    length_buckets = defaultdict(lambda: [0, 0])

    pbar = tqdm(df.itertuples(index=False), total=len(df),
                desc="Evaluating", unit="img")

    for row in pbar:
        img_path = image_root / getattr(row, path_col)
        gt = str(getattr(row, gt_col)).strip().upper()

        # Skip plates with chars outside the model alphabet
        if not gt or not all(c in set(ALPHABET) for c in gt):
            skipped += 1
            continue

        if not img_path.exists():
            skipped += 1
            continue

        try:
            pil_full = Image.open(img_path).convert("RGB")
            iw, ih = pil_full.size
            x1 = max(0, int(row.alpr_x1) - args.padding)
            y1 = max(0, int(row.alpr_y1) - args.padding)
            x2 = min(iw, int(row.alpr_x2) + args.padding)
            y2 = min(ih, int(row.alpr_y2) + args.padding)
            plate_crop = pil_full.crop((x1, y1, x2, y2))
            pred = run_inference(session, input_name, output_name,
                                 outputs_probs, plate_crop)
        except Exception as e:
            skipped += 1
            continue

        evaluated += 1
        match = pred == gt
        exact_correct += int(match)
        n_right = sum(a == b for a, b in zip(pred, gt))
        char_correct += n_right
        char_total   += len(gt)
        length_buckets[len(gt)][0] += int(match)
        length_buckets[len(gt)][1] += 1
        plate_buckets[gt][0] += int(match)
        plate_buckets[gt][1] += 1

        if len(examples) < 40:
            examples.append((gt, pred, match))

        pbar.set_postfix({"exact_acc": f"{exact_correct/evaluated:.1%}",
                          "skip": skipped})

    pbar.close()

    exact_acc = exact_correct / evaluated if evaluated else 0.0
    char_acc  = char_correct  / char_total if char_total else 0.0

    print(f"\n{'='*60}")
    print(f"  RESULTS — Own Dataset  [NVIDIA TAO LPRNet]")
    print(f"{'='*60}")
    print(f"  Evaluated            : {evaluated}")
    print(f"  Skipped              : {skipped}  (missing files / non-alphabet GT)")
    print(f"  Exact match accuracy : {exact_acc:.1%}  ({exact_correct}/{evaluated})")
    print(f"  Character accuracy   : {char_acc:.1%}  ({char_correct}/{char_total})")

    print(f"\n  Accuracy by plate length:")
    for length in sorted(length_buckets):
        ok, tot = length_buckets[length]
        print(f"    len={length:>2}  {ok/tot:.1%}  ({ok}/{tot})")

    print(f"\n  Accuracy per plate text (GT):")
    for plate_text in sorted(plate_buckets):
        ok, tot = plate_buckets[plate_text]
        print(f"    '{plate_text}'  {ok/tot:.1%}  ({ok}/{tot})")

    print(f"\n  Examples  (gt → pred  [✓/✗]):")
    for gt, pred, match in examples:
        status = "✓" if match else "✗"
        print(f"    {status}  gt='{gt}'  →  pred='{pred}'")


if __name__ == "__main__":
    main()
