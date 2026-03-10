#!/usr/bin/env python3
"""
test_trocr_own_data.py

Tests Microsoft TrOCR Small Printed (microsoft/trocr-small-printed) on our
own dataset (preproc_labels.csv). Ground truth is always "VRJ7774". Crops a
rectangular bounding box from the 4 plate corners in the CSV.

TrOCR is a general printed-text recognition model (NOT plate-specific).
Results show the domain gap between printed documents and real plates.

Preprocessing: TrOCRProcessor handles everything (resize to 384×384,
  pixel_values normalization). We pass the raw PIL crop to the processor.

Decoding: VisionEncoderDecoderModel.generate() → processor.batch_decode().

Usage:
    python test_trocr_own_data.py
    python test_trocr_own_data.py --device cuda

Requires:
    pip install transformers pillow numpy pandas tqdm pillow-heif torch
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from PIL import Image
from tqdm import tqdm

try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except ImportError:
    pass

GT = "VRJ7774"
MODEL_ID = "microsoft/trocr-small-printed"


def corners_to_bbox(row, padding: int = 4):
    xs = [row.p1_x, row.p2_x, row.p3_x, row.p4_x]
    ys = [row.p1_y, row.p2_y, row.p3_y, row.p4_y]
    return (min(xs) - padding, min(ys) - padding,
            max(xs) + padding, max(ys) + padding)


def main():
    parser = argparse.ArgumentParser(
        description="Test TrOCR Small Printed on our own dataset"
    )
    parser.add_argument("--labels",  default="preproc_labels.csv")
    parser.add_argument("--padding", type=int,  default=4)
    parser.add_argument("--device",  default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--model",   default=MODEL_ID,
                        help=f"HuggingFace model ID (default: {MODEL_ID})")
    args = parser.parse_args()

    try:
        import torch
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    except ImportError:
        print("ERROR: transformers / torch not found.")
        print("       pip install transformers torch")
        raise SystemExit(1)

    print(f"\n[TrOCR — Own Dataset]")
    print(f"  Model  : {args.model}")
    print(f"  Labels : {args.labels}")
    print(f"  GT     : '{GT}' (all images)")
    print(f"  Device : {args.device}")

    print("\n[Loading model]")
    processor = TrOCRProcessor.from_pretrained(args.model)
    model     = VisionEncoderDecoderModel.from_pretrained(args.model).to(args.device).eval()
    print("  OK")

    df = pd.read_csv(args.labels)
    print(f"  {len(df)} rows")

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
            pil_crop = pil_full.crop((x1, y1, x2, y2))

            pixel_values = processor(
                images=pil_crop, return_tensors="pt"
            ).pixel_values.to(args.device)

            with torch.no_grad():
                generated = model.generate(pixel_values)

            pred = processor.batch_decode(
                generated, skip_special_tokens=True
            )[0].strip().upper()

        except Exception:
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
    print(f"  RESULTS — TrOCR Small Printed  [Own Dataset]")
    print(f"{'='*60}")
    print(f"  GT plate text        : '{GT}'")
    print(f"  Evaluated            : {evaluated}")
    print(f"  Skipped              : {skipped}")
    print(f"  Exact match accuracy : {exact_acc:.1%}  ({exact_correct}/{evaluated})")
    print(f"  Character accuracy   : {char_acc:.1%}  ({char_correct}/{char_total})")
    print(f"\n  Note: TrOCR was trained on printed document text (IAM, SROIE, etc.).")
    print(f"        Domain gap to license plates is expected to be large.")

    print(f"\n  Examples  (pred  [✓/✗]):")
    for pred, match in examples:
        print(f"    {'✓' if match else '✗'}  pred='{pred}'")


if __name__ == "__main__":
    main()
