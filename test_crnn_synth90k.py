#!/usr/bin/env python3
"""
test_crnn_synth90k.py

Tests the CRNN model on real MJSynth (Synth90K) samples streamed from
Hugging Face — no full dataset download required.

Uses the exact official GitYCC inference config:
  - PIL convert('L') greyscale
  - Fixed resize to (100, 32) with BILINEAR
  - Normalise: pixel / 127.5 - 1.0  → [-1, 1]
  - Prefix beam search, beam_size=10

Usage:
    python test_crnn_synth90k.py --model weights/crnn_synth90k.pt
    python test_crnn_synth90k.py --model weights/crnn_synth90k.pt --n 500
    python test_crnn_synth90k.py --model weights/crnn_synth90k.pt --greedy

Requires:
    pip install datasets torch pillow numpy tqdm
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from ocr_backends import CRNNBackend

# ---------------------------------------------------------------------------
# Constants — official GitYCC config.py
# ---------------------------------------------------------------------------
OFFICIAL_IMG_WIDTH  = 100
OFFICIAL_IMG_HEIGHT = 32
OFFICIAL_BEAM_SIZE  = 10
BLANK_ID            = 0
ALPHABET            = CRNNBackend.DEFAULT_ALPHABET  # '0123456789abcdefghijklmnopqrstuvwxyz'
ALPHABET_SET        = set(ALPHABET)

HF_DATASET_ID = "priyank-m/MJSynth_text_recognition"


# ---------------------------------------------------------------------------
# Official preprocessing (mirrors GitYCC dataset.py exactly)
# ---------------------------------------------------------------------------

def official_preprocess(pil_image: Image.Image) -> torch.Tensor:
    """
    PIL Image (any mode) → [1, 1, 32, 100] float32 tensor in [-1, 1].
    Matches GitYCC dataset.py exactly.
    """
    grey    = pil_image.convert("L")
    resized = grey.resize((OFFICIAL_IMG_WIDTH, OFFICIAL_IMG_HEIGHT),
                          resample=Image.BILINEAR)
    arr     = np.array(resized, dtype=np.float32)              # [32, 100] uint8→float
    arr     = arr.reshape((1, OFFICIAL_IMG_HEIGHT, OFFICIAL_IMG_WIDTH))
    arr     = (arr / 127.5) - 1.0                              # [-1, 1]
    return torch.from_numpy(arr).unsqueeze(0)                  # [1, 1, 32, 100]


# ---------------------------------------------------------------------------
# Prefix beam search (mirrors GitYCC ctc_decoder.py)
# ---------------------------------------------------------------------------

def beam_search_decode(log_probs_np: np.ndarray,
                       beam_size: int = OFFICIAL_BEAM_SIZE,
                       blank_id: int  = BLANK_ID) -> list[int]:
    """
    Prefix beam search CTC decode.
    log_probs_np : [T, C]  log-softmax probabilities
    Returns 1-based label list.
    """
    NEG_INF = float("-inf")
    beams: dict[tuple, list[float]] = {(): [0.0, NEG_INF]}  # prefix → [p_blank, p_nb]

    for t in range(log_probs_np.shape[0]):
        lp = log_probs_np[t]
        new_beams: dict[tuple, list[float]] = defaultdict(lambda: [NEG_INF, NEG_INF])

        for prefix, (p_b, p_nb) in beams.items():
            p_total = np.logaddexp(p_b, p_nb)

            # Emit blank → same prefix
            new_beams[prefix][0] = np.logaddexp(
                new_beams[prefix][0], p_total + lp[blank_id]
            )

            # Emit non-blank c
            for c in range(log_probs_np.shape[1]):
                if c == blank_id:
                    continue
                p_c      = lp[c]
                new_pref = prefix + (c,)

                if prefix and prefix[-1] == c:
                    # Repeated char: only blank-ended path creates new symbol
                    new_beams[new_pref][1] = np.logaddexp(
                        new_beams[new_pref][1], p_b + p_c
                    )
                    # Non-blank path stays on same prefix (collapsed)
                    new_beams[prefix][1] = np.logaddexp(
                        new_beams[prefix][1], p_nb + p_c
                    )
                else:
                    new_beams[new_pref][1] = np.logaddexp(
                        new_beams[new_pref][1], p_total + p_c
                    )

        beams = dict(
            sorted(new_beams.items(),
                   key=lambda x: np.logaddexp(x[1][0], x[1][1]),
                   reverse=True)[:beam_size]
        )

    best, _ = max(beams.items(), key=lambda x: np.logaddexp(x[1][0], x[1][1]))
    return list(best)


def greedy_decode(log_probs_np: np.ndarray, blank_id: int = BLANK_ID) -> list[int]:
    """Greedy CTC decode."""
    labels, prev = [], blank_id
    for t in range(log_probs_np.shape[0]):
        idx = int(log_probs_np[t].argmax())
        if idx != prev and idx != blank_id:
            labels.append(idx)
        prev = idx
    return labels


def labels_to_text(labels: list[int]) -> str:
    return "".join(ALPHABET[i - 1] for i in labels if 1 <= i <= len(ALPHABET))


# ---------------------------------------------------------------------------
# Inference on one PIL image
# ---------------------------------------------------------------------------

def predict(model: torch.nn.Module, pil_image: Image.Image,
            use_beam: bool = True, device: str = "cpu") -> str:
    inp    = official_preprocess(pil_image).to(device)         # [1, 1, 32, 100]
    with torch.no_grad():
        logits = model(inp).squeeze(1)                         # [T, C]
    lp = F.log_softmax(logits, dim=1).cpu().numpy()            # [T, C]
    labels = beam_search_decode(lp) if use_beam else greedy_decode(lp)
    return labels_to_text(labels)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Test CRNN on streamed Synth90K samples from Hugging Face"
    )
    parser.add_argument("--model",  required=True,
                        help="Path to crnn_synth90k.pt checkpoint")
    parser.add_argument("--n",      type=int, default=200,
                        help="Number of valid samples to evaluate (default 200)")
    parser.add_argument("--split",  default="test",
                        choices=["train", "val", "test"],
                        help="Dataset split to stream from (default: test)")
    parser.add_argument("--greedy", action="store_true",
                        help="Use greedy decode instead of beam search")
    parser.add_argument("--device", default="cpu",
                        choices=["cpu", "cuda"])
    parser.add_argument("--max_label_len", type=int, default=12,
                        help="Skip labels longer than this (default 12)")
    args = parser.parse_args()

    # ── Load model ────────────────────────────────────────────────────────────
    print(f"\n[Loading CRNN]  checkpoint={args.model}  device={args.device}")
    backend = CRNNBackend(
        model_path=args.model,
        device=args.device,
        alphabet=ALPHABET,
        img_height=OFFICIAL_IMG_HEIGHT,
        n_hidden=256,
    )
    backend.ensure_loaded()
    model = backend._model
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Params: {n_params:,}")
    print(f"  Decode: {'greedy' if args.greedy else f'beam_search (k={OFFICIAL_BEAM_SIZE})'}")

    # ── Stream dataset ────────────────────────────────────────────────────────
    print(f"\n[Streaming]  {HF_DATASET_ID}  split={args.split}")
    print("  Filtering: labels that are 100% in CRNN alphabet (0-9, a-z)")
    print(f"  Target: {args.n} valid samples\n")

    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: 'datasets' package not found. Install with:  pip install datasets")
        sys.exit(1)

    ds = load_dataset(HF_DATASET_ID, split=args.split, streaming=True,
                      trust_remote_code=True)

    # ── Evaluate ──────────────────────────────────────────────────────────────
    exact_correct  = 0
    char_correct   = 0
    char_total     = 0
    skipped        = 0
    examples       = []   # (gt, pred, match) for printout
    length_buckets = defaultdict(lambda: [0, 0])  # len → [correct, total]

    pbar = tqdm(total=args.n, desc="Evaluating", unit="img")

    for sample in ds:
        raw_label = sample["label"]
        gt        = raw_label.lower()

        # Filter: must only contain alphabet chars and not be too long
        if not gt or len(gt) > args.max_label_len:
            skipped += 1
            continue
        if not all(c in ALPHABET_SET for c in gt):
            skipped += 1
            continue

        # Run inference
        pil_img = sample["image"]
        if not isinstance(pil_img, Image.Image):
            pil_img = Image.fromarray(pil_img)

        try:
            pred = predict(model, pil_img, use_beam=not args.greedy, device=args.device)
        except Exception as e:
            skipped += 1
            continue

        # Accumulate metrics
        match = pred == gt
        exact_correct += int(match)
        n_chars = len(gt)
        n_right = sum(a == b for a, b in zip(pred, gt))
        char_correct += n_right
        char_total   += n_chars
        length_buckets[n_chars][0] += int(match)
        length_buckets[n_chars][1] += 1

        if len(examples) < 20:
            examples.append((gt, pred, match))

        pbar.update(1)
        n_done = exact_correct + (pbar.n - exact_correct)
        pbar.set_postfix({"exact_acc": f"{exact_correct/pbar.n:.1%}",
                          "skipped":   skipped})

        if pbar.n >= args.n:
            break

    pbar.close()
    n_total = pbar.n

    # ── Results ───────────────────────────────────────────────────────────────
    exact_acc = exact_correct / n_total if n_total else 0.0
    char_acc  = char_correct  / char_total if char_total else 0.0

    print(f"\n{'='*60}")
    print(f"  RESULTS on {n_total} Synth90K {args.split} samples")
    print(f"{'='*60}")
    print(f"  Exact match accuracy : {exact_acc:.1%}  ({exact_correct}/{n_total})")
    print(f"  Character accuracy   : {char_acc:.1%}  ({char_correct}/{char_total})")
    print(f"  Samples skipped      : {skipped}  (non-alphabet chars or too long)")

    print(f"\n  Accuracy by label length:")
    for length in sorted(length_buckets):
        ok, tot = length_buckets[length]
        print(f"    len={length:>2}  {ok/tot:.1%}  ({ok}/{tot})")

    print(f"\n  First {len(examples)} examples  (gt → pred  [✓/✗]):")
    for gt, pred, match in examples:
        status = "✓" if match else "✗"
        print(f"    {status}  gt='{gt}'  →  pred='{pred}'")

    print(f"\n  Interpretation:")
    if exact_acc > 0.85:
        print("  → Model works correctly on in-domain Synth90K data.")
        print("    Any errors on plate images are due to domain gap.")
    elif exact_acc > 0.50:
        print("  → Model partially works. Check char accuracy for detail.")
    else:
        print("  → Model accuracy is low even on Synth90K.")
        print("    Likely cause: weights did not load correctly, or")
        print("    wrong architecture variant (map_to_seq vs standard CRNN).")


if __name__ == "__main__":
    main()
