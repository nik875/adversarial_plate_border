#!/usr/bin/env python3
"""
test_vitstr_own_data.py

Tests the ViTSTR Small model (from python-doctr) on our own dataset
(preproc_labels.csv). Ground truth is always "VRJ7774". Crops a rectangular
bounding box from the 4 plate corners in the CSV.

ViTSTR is a general scene-text recognition model — not plate-specific.
Results show the domain gap between synthetic scene text and real plates.

Preprocessing (doctr convention):
  - Grayscale → RGB (3-channel)
  - Resize to (W=128, H=32) with BILINEAR
  - [0, 1] float32, NCHW

Decoding:
  - Attention-based; doctr's built-in postprocessor

Usage:
    python test_vitstr_own_data.py
    python test_vitstr_own_data.py --device cuda

Requires:
    pip install python-doctr[torch] pillow numpy pandas tqdm pillow-heif
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

GT      = "VRJ7774"
INPUT_H = 32
INPUT_W = 128


def corners_to_bbox(row, padding: int = 4):
    xs = [row.p1_x, row.p2_x, row.p3_x, row.p4_x]
    ys = [row.p1_y, row.p2_y, row.p3_y, row.p4_y]
    return (min(xs) - padding, min(ys) - padding,
            max(xs) + padding, max(ys) + padding)


def preprocess(pil_crop: Image.Image) -> torch.Tensor:
    """PIL crop → [1, 3, 32, 128] float32 [0, 1] (doctr convention)."""
    rgb     = pil_crop.convert("RGB")
    resized = rgb.resize((INPUT_W, INPUT_H), resample=Image.BILINEAR)
    arr     = np.array(resized, dtype=np.float32) / 255.0      # [32, 128, 3]
    tensor  = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # [1, 3, 32, 128]
    return tensor


def decode_output(logits: torch.Tensor, vocab: str) -> str:
    """
    Greedy-decode attention logits [1, T, V] using the model vocabulary.
    doctr vocab: special EOS at index 0, characters at 1..len(vocab).
    """
    EOS = 0
    indices = logits[0].argmax(dim=-1).tolist()   # [T]
    chars = []
    for idx in indices:
        if idx == EOS:
            break
        if 1 <= idx <= len(vocab):
            chars.append(vocab[idx - 1])
    return "".join(chars)


def main():
    parser = argparse.ArgumentParser(
        description="Test ViTSTR Small on our own dataset"
    )
    parser.add_argument("--labels",  default="preproc_labels.csv")
    parser.add_argument("--padding", type=int,  default=4)
    parser.add_argument("--device",  default="cpu", choices=["cpu", "cuda"])
    args = parser.parse_args()

    try:
        from doctr.models import vitstr_small
    except ImportError:
        print("ERROR: python-doctr not found.  pip install python-doctr[torch]")
        raise SystemExit(1)

    print("\n[ViTSTR Small — Own Dataset]")
    print(f"  Source : doctr vitstr_small (pretrained)")
    print(f"  Input  : {INPUT_H}×{INPUT_W} RGB [0,1]")
    print(f"  Labels : {args.labels}")
    print(f"  GT     : '{GT}' (all images)")
    print(f"  Device : {args.device}")

    print("\n[Loading model]")
    model = vitstr_small(pretrained=True).to(args.device).eval()

    # doctr recognition models expose their vocabulary
    vocab = model.vocab if hasattr(model, "vocab") else None
    if vocab is None:
        # Fallback: standard doctr alphanumeric vocab
        import string
        vocab = string.digits + string.ascii_lowercase
    print(f"  Vocab  : '{vocab[:20]}...' ({len(vocab)} chars)")

    df = pd.read_csv(args.labels)
    print(f"  {len(df)} rows")

    exact_correct = 0
    char_correct  = 0
    char_total    = 0
    skipped       = 0
    evaluated     = 0
    examples      = []
    first_errors  = []   # collect first 3 tracebacks for diagnosis
    import traceback

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
            inp = preprocess(pil_full.crop((x1, y1, x2, y2))).to(args.device)

            with torch.no_grad():
                out = model(inp)

            # doctr models may return (logits,) or logits directly
            if isinstance(out, (tuple, list)):
                logits = out[0]
            else:
                logits = out

            # If already decoded to strings by the postprocessor
            if isinstance(logits, list) and isinstance(logits[0], tuple):
                pred = logits[0][0].upper()
            else:
                pred = decode_output(logits.cpu(), vocab).upper()

        except Exception as e:
            if len(first_errors) < 3:
                first_errors.append(traceback.format_exc())
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
    print(f"  RESULTS — ViTSTR Small  [Own Dataset]")
    print(f"{'='*60}")
    print(f"  GT plate text        : '{GT}'")
    print(f"  Evaluated            : {evaluated}")
    print(f"  Skipped              : {skipped}")
    print(f"  Exact match accuracy : {exact_acc:.1%}  ({exact_correct}/{evaluated})")
    print(f"  Character accuracy   : {char_acc:.1%}  ({char_correct}/{char_total})")
    print(f"\n  Note: ViTSTR uses a 37-char lower+digit vocab — plate chars may be")
    print(f"        lowercased in output and 'V' may not be in vocab.")

    print(f"\n  Examples  (pred  [✓/✗]):")
    for pred, match in examples:
        print(f"    {'✓' if match else '✗'}  pred='{pred}'")


if __name__ == "__main__":
    main()
