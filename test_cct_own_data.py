#!/usr/bin/env python3
"""
test_cct_own_data.py

Tests the CCT-XS-V1 Global plate OCR model on our own dataset
(preproc_labels.csv). Ground truth is always "VRJ7774". Crops a rectangular
bounding box from the 4 plate corners in the CSV.

Loads the model directly from the fast-plate-ocr cache via onnx2torch,
exactly as done in offensive_patch.py.

Preprocessing:
  PIL crop → [1, 3, H, W] float32 [0,1]
           → permute to [1, H, W, 3] (NHWC) × 255
           → resize to [1, 64, 128, 3]

Decoding:
  softmax → argmax over 37-class alphabet per 9 fixed slots
  alphabet = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ_'  (_ = blank/pad)

Usage:
    python test_cct_own_data.py
    python test_cct_own_data.py --device cuda

Requires:
    pip install onnx onnx2torch torch pillow numpy pandas tqdm pillow-heif
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except ImportError:
    pass

ALPHABET       = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ_'
BLANK          = '_'
OCR_H, OCR_W   = 64, 128
OCR_PATH       = Path.home() / ".cache/fast-plate-ocr/cct-s-v1-global-model/cct_s_v1_global.onnx"
GT             = "VRJ7774"


def corners_to_bbox(row, padding: int = 4):
    xs = [row.p1_x, row.p2_x, row.p3_x, row.p4_x]
    ys = [row.p1_y, row.p2_y, row.p3_y, row.p4_y]
    return (min(xs) - padding, min(ys) - padding,
            max(xs) + padding, max(ys) + padding)


def preprocess(pil_crop: Image.Image, device: torch.device) -> torch.Tensor:
    """PIL crop → [1, 64, 128, 3] float32 [0, 255] NHWC on device."""
    rgb    = pil_crop.convert("RGB")
    arr    = np.array(rgb, dtype=np.float32) / 255.0          # [H, W, 3]
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]
    tensor = F.interpolate(tensor, size=(OCR_H, OCR_W),
                           mode='bilinear', align_corners=False)
    return tensor.permute(0, 2, 3, 1) * 255                   # [1, 64, 128, 3]


def decode(logits: torch.Tensor) -> str:
    """[1, 9, 37] logits → text string (skip blank '_')."""
    probs     = torch.softmax(logits, dim=-1)
    indices   = probs.argmax(dim=-1).squeeze(0)   # [9]
    return "".join(
        ALPHABET[i] for i in indices.tolist()
        if ALPHABET[i] != BLANK
    )


def main():
    parser = argparse.ArgumentParser(
        description="Test CCT-XS-V1 plate OCR on our own dataset"
    )
    parser.add_argument("--labels",  default="preproc_labels.csv")
    parser.add_argument("--padding", type=int, default=4)
    parser.add_argument("--device",  default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--model",   default=str(OCR_PATH),
                        help=f"Path to cct_xs_v1_global.onnx (default: {OCR_PATH})")
    args = parser.parse_args()

    try:
        import onnx, onnx2torch
    except ImportError:
        print("ERROR: pip install onnx onnx2torch")
        raise SystemExit(1)

    print("\n[CCT-S-V1 Global — Own Dataset]")
    print(f"  Model  : {args.model}")
    print(f"  Input  : NHWC [{OCR_H}×{OCR_W}×3] [0,255]")
    print(f"  Labels : {args.labels}")
    print(f"  GT     : '{GT}' (all images)")
    print(f"  Device : {args.device}")

    print("\n[Loading model]")
    ocr_model = onnx.load(args.model)
    model     = onnx2torch.convert(ocr_model).to(args.device).eval()
    print("  OK")

    device = torch.device(args.device)
    df     = pd.read_csv(args.labels)
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
            iw, ih   = pil_full.size
            x1, y1, x2, y2 = corners_to_bbox(row, args.padding)
            x1, y1 = max(0, int(x1)), max(0, int(y1))
            x2, y2 = min(iw, int(x2)), min(ih, int(y2))
            inp = preprocess(pil_full.crop((x1, y1, x2, y2)), device)

            with torch.no_grad():
                logits = model(inp)   # [1, 9, 37]

            pred = decode(logits)

        except Exception as e:
            if len(first_errors) < 3:
                import traceback; first_errors.append(traceback.format_exc())
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
    print(f"  RESULTS — CCT-S-V1 Global  [Own Dataset]")
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
