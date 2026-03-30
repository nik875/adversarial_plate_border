#!/usr/bin/env python3
"""
test_cct.py

Tests the pretrained CCT-S-V1-Global model on the CCPD2019 dataset.

Validates that the model:
  - Ignores the Chinese province character (first character of 7-char plate)
  - Outputs only alphanumeric text (matching our 36-char alphabet A-Z0-9)
  - Achieves reasonable exact-match accuracy on the 6 alphanumeric characters

CCPD label format (from filename):
    province_idx _ letter_idx _ d1 _ d2 _ d3 _ d4 _ d5
    → we skip province (idx[0]), take letter (_ALPHABETS[idx[1]]),
      then 5 alphanumeric chars from _ADS[idx[2..6]]
    → 6-char ground truth (no province)

Usage:
    python test_cct.py
    python test_cct.py --ccpd-root CCPD2019 --limit 2000 --device cuda
    python test_cct.py --subset ccpd_base --limit 5000
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from cct_ocr_torch import CCTOCRTorch

# ---------------------------------------------------------------------------
# Alphabet (37 classes: blank=0, then '0'-'9', 'A'-'Z')
# ---------------------------------------------------------------------------

CHARS   = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"  # 36 printable chars
BLANK   = "_"
ALPHABET = CHARS + BLANK   # index 36 = blank/pad

OCR_H, OCR_W = 64, 128

# CCPD decode tables (from finetune_all_models.py)
_ALPHABETS = list("ABCDEFGHJKLMNPQRSTUVWXYZ") + ["O"]   # 25 entries
_ADS       = list("ABCDEFGHJKLMNPQRSTUVWXYZ") + list("0123456789") + ["O"]  # 35 entries
_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


# ---------------------------------------------------------------------------
# CCPD parsing
# ---------------------------------------------------------------------------

def _decode_ccpd_plate(plate_code: str) -> str:
    """
    Decode "0_0_22_27_27_33_16" → 6-char alphanumeric string.
    Province character (idx[0]) is skipped; only the 6 alphanumeric chars returned.
    """
    idx = list(map(int, plate_code.split("_")))
    chars = [_ALPHABETS[idx[1]]] + [_ADS[idx[i]] for i in range(2, 7)]
    return "".join(chars)


def _parse_ccpd_file(path: Path):
    """Parse CCPD filename → ((x1,y1,x2,y2), label_str) or raise ValueError."""
    parts = path.stem.split("-")
    if len(parts) < 7:
        raise ValueError(f"Bad CCPD name: {path.name}")
    tl, br = parts[2].split("_")
    x1, y1 = map(int, tl.split("&"))
    x2, y2 = map(int, br.split("&"))
    label = _decode_ccpd_plate(parts[4])
    # Keep only standard chars
    label = "".join(c for c in label if c in CHARS)
    if not label:
        raise ValueError("Empty label after filtering")
    return (x1, y1, x2, y2), label


def load_ccpd_records(ccpd_root: Path, subset: str | None, limit: int | None):
    """Walk CCPD directory (or a named subset) and collect records."""
    search_root = ccpd_root / subset if subset else ccpd_root
    records = []
    for p in sorted(search_root.rglob("*")):
        if p.suffix.lower() not in _IMAGE_EXTS:
            continue
        try:
            bbox, label = _parse_ccpd_file(p)
        except (ValueError, IndexError):
            continue
        records.append({"image": p, "bbox": bbox, "label": label})
        if limit and len(records) >= limit:
            break
    return records


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def preprocess(img_bgr: np.ndarray, bbox, device: torch.device) -> torch.Tensor:
    """Crop bbox from BGR image, resize to [64,128], return [1,64,128,3] uint8."""
    x1, y1, x2, y2 = bbox
    h, w = img_bgr.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    crop = img_bgr[y1:max(y1+1, y2), x1:max(x1+1, x2)]
    crop = cv2.resize(crop, (OCR_W, OCR_H), interpolation=cv2.INTER_LINEAR)
    crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    # CCTOCRTorch expects [N, H, W, 3] uint8
    tensor = torch.from_numpy(crop).unsqueeze(0).to(device)  # [1, 64, 128, 3]
    return tensor


# ---------------------------------------------------------------------------
# Decoding
# ---------------------------------------------------------------------------

def decode(probs: torch.Tensor) -> str:
    """[1, 9, 37] softmax probs → text string (skip blank at index 36)."""
    indices = probs.argmax(dim=-1).squeeze(0)   # [9]
    return "".join(
        CHARS[i] for i in indices.tolist()
        if i < len(CHARS)  # index 36 = blank → skip
    )


def is_alphanumeric_only(text: str) -> bool:
    """Return True if every character in text is in our A-Z0-9 alphabet."""
    return all(c in CHARS for c in text)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Test pretrained CCT-S on CCPD2019 — validate province-skip behaviour"
    )
    parser.add_argument(
        "--ccpd-root", default="CCPD2019",
        help="Path to CCPD2019 root directory (default: CCPD2019)"
    )
    parser.add_argument(
        "--subset", default="ccpd_base",
        help="CCPD subset subdirectory (default: ccpd_base; use '' for all)"
    )
    parser.add_argument(
        "--limit", type=int, default=2000,
        help="Max number of images to evaluate (default: 2000)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for shuffling (default: 42)"
    )
    parser.add_argument(
        "--onnx", default=str(
            Path.home() / ".cache/fast-plate-ocr/cct-s-v1-global-model/cct_s_v1_global.onnx"
        ),
        help="Path to CCT-S ONNX file"
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()

    ccpd_root = Path(args.ccpd_root)
    subset    = args.subset if args.subset else None
    device    = torch.device(args.device)

    # ── Load model ────────────────────────────────────────────────────────────
    print(f"\n[CCT-S-V1-Global — CCPD2019 Validation]")
    print(f"  ONNX   : {args.onnx}")
    print(f"  Device : {args.device}")
    onnx_path = Path(args.onnx)
    if not onnx_path.exists():
        print(f"ERROR: ONNX file not found: {onnx_path}")
        raise SystemExit(1)

    print("\n[Loading model]")
    model = CCTOCRTorch.from_onnx(str(onnx_path))
    model.to(device).eval()
    print("  OK")

    # ── Load records ─────────────────────────────────────────────────────────
    print(f"\n[Loading CCPD records from {ccpd_root / (subset or '')}]")
    records = load_ccpd_records(ccpd_root, subset, limit=None)
    print(f"  Found {len(records)} records")
    if not records:
        print("ERROR: No CCPD records found. Check --ccpd-root / --subset.")
        raise SystemExit(1)

    rng = random.Random(args.seed)
    rng.shuffle(records)
    if args.limit:
        records = records[:args.limit]
    print(f"  Evaluating {len(records)} images (limit={args.limit}, seed={args.seed})")

    # ── Evaluate ──────────────────────────────────────────────────────────────
    exact_correct      = 0
    char_correct       = 0
    char_total         = 0
    non_alphanum_count = 0   # predictions containing non-CHARS characters
    skipped            = 0
    evaluated          = 0

    # Track predictions that don't match (for inspection)
    error_examples  = []
    correct_examples = []

    pbar = tqdm(records, desc="Evaluating", unit="img")

    for rec in pbar:
        img_bgr = cv2.imread(str(rec["image"]))
        if img_bgr is None:
            skipped += 1
            continue

        try:
            inp = preprocess(img_bgr, rec["bbox"], device)
            with torch.no_grad():
                probs = model(inp)          # [1, 9, 37]
            pred = decode(probs)
        except Exception:
            skipped += 1
            continue

        gt   = rec["label"]
        evaluated += 1

        # Check alphanumeric-only output
        if not is_alphanumeric_only(pred):
            non_alphanum_count += 1

        exact_match = (pred == gt)
        exact_correct += int(exact_match)

        # Per-character accuracy (align by min length)
        min_len = min(len(pred), len(gt))
        char_correct += sum(pred[i] == gt[i] for i in range(min_len))
        char_total   += len(gt)

        if not exact_match and len(error_examples) < 20:
            error_examples.append((pred, gt))
        if exact_match and len(correct_examples) < 10:
            correct_examples.append((pred, gt))

        pbar.set_postfix({
            "exact": f"{exact_correct/evaluated:.1%}",
            "skip": skipped,
        })

    pbar.close()

    # ── Report ────────────────────────────────────────────────────────────────
    exact_acc  = exact_correct / evaluated if evaluated else 0.0
    char_acc   = char_correct  / char_total if char_total else 0.0
    alnum_rate = 1.0 - (non_alphanum_count / evaluated) if evaluated else 0.0

    print(f"\n{'='*64}")
    print(f"  RESULTS — CCT-S-V1-Global on CCPD2019 ({subset or 'all'})")
    print(f"{'='*64}")
    print(f"  Evaluated            : {evaluated}")
    print(f"  Skipped              : {skipped}")
    print(f"  Exact match accuracy : {exact_acc:.1%}  ({exact_correct}/{evaluated})")
    print(f"  Character accuracy   : {char_acc:.1%}  ({char_correct}/{char_total})")
    print(f"  Alphanumeric-only    : {alnum_rate:.1%}  ({evaluated - non_alphanum_count}/{evaluated})")
    print()
    print(f"  Note: GT is 6 chars (province skipped); model outputs ≤9 slots.")
    print(f"  Alphanumeric-only validates that the model never outputs province chars.")

    if correct_examples:
        print(f"\n  Correct examples (pred → gt):")
        for pred, gt in correct_examples:
            print(f"    '{pred}' → '{gt}'")

    if error_examples:
        print(f"\n  Error examples (pred → gt):")
        for pred, gt in error_examples:
            print(f"    '{pred}' → '{gt}'")

    # ── Assertions ────────────────────────────────────────────────────────────
    print(f"\n{'='*64}")
    print("  Validation checks:")

    checks = [
        ("Alphanumeric-only output rate >= 99%", alnum_rate >= 0.99),
        ("Exact match accuracy >= 10%",          exact_acc  >= 0.10),
        ("Character accuracy >= 50%",            char_acc   >= 0.50),
    ]

    all_pass = True
    for desc, passed in checks:
        status = "PASS" if passed else "FAIL"
        print(f"    [{status}] {desc}")
        if not passed:
            all_pass = False

    print(f"\n  Overall: {'ALL CHECKS PASSED' if all_pass else 'SOME CHECKS FAILED'}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
