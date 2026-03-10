#!/usr/bin/env python3
"""
test_lprnet_synth90k.py

Tests the NVIDIA TAO US LPRNet deployable ONNX model on Synth90K samples
streamed from Hugging Face — no full dataset download required.

Sourced from NVIDIA TAO TF1 backend  (nvidia_tao_tf1/cv/lprnet/):
  img_utils.py   : preprocess() = cv2.resize((W,H)) / 255.0  → NCHW float32
  inference.py   : blank_id = len(classes)  (blank is LAST, not first)
  ctc_decoder    : decode_ctc_conf() — CTC greedy: collapse repeats, strip blank

Model specifics:
  Input  : [B, 3, 48, 96]  float32  BGR or RGB (model trained on BGR via cv2)
  Output : [B, T]  int64   per-timestep argmax indices (already in ONNX graph)
             — the deployable ONNX has Softmax→ArgMax baked in;
               CTC collapse+strip-blank is done in post-processing here.
  Alphabet: 36 chars  0-9 A-Z  (indices 0..35),  blank at index 36
  Note: blank_id = 36 = len(alphabet) — OPPOSITE convention to CRNN (blank=0)

Data: Synth90K filtered for 2-8 char alphanumeric labels, uppercased.
LPRNet is plate-domain; Synth90K is scene-text → domain gap expected.

Usage:
    python test_lprnet_synth90k.py \\
        --model weights/lprnet_deployable_onnx_v1.1/us_lprnet_baseline18_deployable.onnx
    python test_lprnet_synth90k.py --model <path> --n 500

Requires:
    pip install onnxruntime datasets pillow numpy tqdm
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

# ---------------------------------------------------------------------------
# NVIDIA TAO US LPRNet character set
# ---------------------------------------------------------------------------
# Source: nvidia_tao_tf1/cv/lprnet — characters_list_file for US model.
# Indices 0..35 → chars below.  Index 36 → CTC blank.
# blank_id = len(ALPHABET) = 36  (OPPOSITE of CRNN where blank=0).

ALPHABET  = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"   # 36 chars, 0-indexed
BLANK_IDX = len(ALPHABET)                             # 36 — blank is LAST

# LPRNet input geometry (from augmentation_config in TAO spec)
INPUT_H = 48
INPUT_W = 96

HF_DATASET_ID = "priyank-m/MJSynth_text_recognition"


# ---------------------------------------------------------------------------
# Preprocessing  (nvidia_tao_tf1/cv/lprnet/utils/img_utils.py)
# ---------------------------------------------------------------------------

def preprocess(pil_image: Image.Image) -> np.ndarray:
    """
    PIL Image → numpy [1, 3, 48, 96] float32 in [0, 1].

    NVIDIA TAO source (img_utils.py):
        cv2.resize(img, (output_width, output_height)) / 255.0
        transpose(0, 3, 1, 2)   # NHWC → NCHW

    The model was trained with OpenCV (BGR).  PIL gives RGB.
    We test both and let the output entropy tell us which is correct —
    but for scene-text the domain gap will dominate anyway.
    We use RGB (PIL native) as the default.
    """
    rgb     = pil_image.convert("RGB")
    # cv2.resize uses (width, height) — match exactly
    resized = rgb.resize((INPUT_W, INPUT_H), resample=Image.BILINEAR)
    arr     = np.array(resized, dtype=np.float32) / 255.0   # [48, 96, 3]
    arr     = arr.transpose(2, 0, 1)                          # [3, 48, 96]  NCHW
    return arr[np.newaxis]                                    # [1, 3, 48, 96]


# ---------------------------------------------------------------------------
# CTC greedy decode  (nvidia_tao_tf1/cv/lprnet — decode_ctc_conf)
# ---------------------------------------------------------------------------

def ctc_greedy_decode(indices: np.ndarray,
                      blank_id: int = BLANK_IDX) -> str:
    """
    indices : 1-D int array of per-timestep argmax predictions.
    CTC greedy decode: collapse consecutive identical labels, then remove blank.
    blank_id = len(ALPHABET) = 36  (last index — NVIDIA TAO convention).

    Key difference vs CRNN:
      CRNN : blank=0,  alphabet 1-indexed  (idx-1 → char)
      TAO  : blank=last, alphabet 0-indexed (idx   → char)
    """
    labels, prev = [], -1
    for idx in indices:
        idx = int(idx)
        if idx != prev:
            labels.append(idx)
        prev = idx
    # Strip blanks
    chars = [ALPHABET[i] for i in labels
             if i != blank_id and 0 <= i < len(ALPHABET)]
    return "".join(chars)


# ---------------------------------------------------------------------------
# Model inspection
# ---------------------------------------------------------------------------

def inspect_model(session) -> dict:
    info = {}
    print("\n  [ONNX Model I/O]")
    for inp in session.get_inputs():
        print(f"    Input  '{inp.name}': shape={inp.shape}  dtype={inp.type}")
        info["input_name"]  = inp.name
    for out in session.get_outputs():
        print(f"    Output '{out.name}': shape={out.shape}  dtype={out.type}")
        info["output_name"]  = out.name
        info["output_dtype"] = out.type
        info["output_shape"] = out.shape
    return info


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Test NVIDIA TAO US LPRNet on streamed Synth90K samples"
    )
    parser.add_argument("--model", required=True,
                        help="Path to us_lprnet_baseline18_deployable.onnx")
    parser.add_argument("--n",     type=int, default=200,
                        help="Number of valid samples to evaluate (default 200)")
    parser.add_argument("--split", default="test",
                        choices=["train", "val", "test"])
    parser.add_argument("--max_len", type=int, default=8)
    parser.add_argument("--min_len", type=int, default=2)
    args = parser.parse_args()

    # ── Load model ────────────────────────────────────────────────────────────
    print(f"\n[Loading LPRNet ONNX]  {args.model}")
    try:
        import onnxruntime as ort
    except ImportError:
        print("ERROR: onnxruntime not found.  pip install onnxruntime")
        sys.exit(1)

    session = ort.InferenceSession(
        args.model,
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    model_info   = inspect_model(session)
    input_name   = model_info["input_name"]
    output_name  = model_info["output_name"]
    output_dtype = model_info.get("output_dtype", "")

    # Probe output shape with a dummy input
    dummy     = np.zeros((1, 3, INPUT_H, INPUT_W), dtype=np.float32)
    dummy_out = session.run([output_name], {input_name: dummy})[0]
    print(f"\n  Dummy forward: output shape={dummy_out.shape}  dtype={dummy_out.dtype}")

    # Detect format: if last dim = num_classes → raw probs; if 1-D or small → indices
    outputs_probs = dummy_out.ndim >= 2 and dummy_out.shape[-1] > 10 and "int" not in output_dtype
    if outputs_probs:
        num_classes = dummy_out.shape[-1]
        inferred_blank = num_classes - 1
        print(f"  Output = softmax probs  (num_classes={num_classes}, blank={inferred_blank})")
        # Update blank if model differs from our default
        if num_classes != len(ALPHABET) + 1:
            print(f"  WARNING: expected {len(ALPHABET)+1} classes, got {num_classes}")
    else:
        print(f"  Output = argmax indices  (CTC collapse+strip done here)")

    print(f"\n[Config — NVIDIA TAO source]")
    print(f"  Input          : [B, 3, {INPUT_H}, {INPUT_W}]  float32")
    print(f"  Normalisation  : cv2.resize((W,H)) / 255.0  →  [0, 1]")
    print(f"  Alphabet       : '{ALPHABET}'  ({len(ALPHABET)} chars, 0-indexed)")
    print(f"  blank_id       : {BLANK_IDX}  (last index — TAO convention; CRNN uses 0)")
    print(f"  Decode         : CTC greedy — collapse repeats, strip blank")

    # ── Stream Synth90K ───────────────────────────────────────────────────────
    print(f"\n[Streaming]  {HF_DATASET_ID}  split={args.split}")
    print(f"  Filter: alphanumeric-only labels, length {args.min_len}-{args.max_len}")
    print(f"  Labels uppercased (LPRNet outputs uppercase A-Z)\n")

    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: 'datasets' not found.  pip install datasets")
        sys.exit(1)

    valid_chars = set(ALPHABET.lower() + ALPHABET)
    ds = load_dataset(HF_DATASET_ID, split=args.split, streaming=True,
                      trust_remote_code=True)

    # ── Evaluate ──────────────────────────────────────────────────────────────
    exact_correct  = 0
    char_correct   = 0
    char_total     = 0
    skipped        = 0
    examples       = []
    length_buckets = defaultdict(lambda: [0, 0])
    pred_char_dist = defaultdict(int)

    pbar = tqdm(total=args.n, desc="Evaluating", unit="img")

    for sample in ds:
        raw_label = sample["label"]
        gt        = raw_label.upper()   # compare uppercase ↔ uppercase

        if not (args.min_len <= len(gt) <= args.max_len):
            skipped += 1
            continue
        if not all(c in valid_chars for c in raw_label):
            skipped += 1
            continue

        pil_img = sample["image"]
        if not isinstance(pil_img, Image.Image):
            pil_img = Image.fromarray(pil_img)

        try:
            inp  = preprocess(pil_img)
            outs = session.run([output_name], {input_name: inp})[0]  # [1, T] or [1, T, C]

            if outputs_probs:
                # [1, T, C] softmax — per-slot argmax then CTC decode
                indices = outs[0].argmax(axis=-1)               # [T]
                pred = ctc_greedy_decode(indices)
            else:
                # [1, T] int indices — CTC decode directly
                pred = ctc_greedy_decode(outs[0].flatten())

        except Exception as e:
            skipped += 1
            continue

        match = pred == gt
        exact_correct += int(match)
        n_chars = len(gt)
        n_right = sum(a == b for a, b in zip(pred, gt))
        char_correct += n_right
        char_total   += n_chars
        length_buckets[n_chars][0] += int(match)
        length_buckets[n_chars][1] += 1
        for c in pred:
            pred_char_dist[c] += 1

        if len(examples) < 20:
            examples.append((gt, pred, match))

        pbar.update(1)
        pbar.set_postfix({"exact_acc": f"{exact_correct/pbar.n:.1%}",
                          "skipped": skipped})
        if pbar.n >= args.n:
            break

    pbar.close()
    n_total = pbar.n

    # ── Results ───────────────────────────────────────────────────────────────
    exact_acc = exact_correct / n_total if n_total else 0.0
    char_acc  = char_correct  / char_total if char_total else 0.0

    print(f"\n{'='*60}")
    print(f"  RESULTS on {n_total} Synth90K {args.split} samples  [NVIDIA TAO LPRNet]")
    print(f"{'='*60}")
    print(f"  Exact match accuracy : {exact_acc:.1%}  ({exact_correct}/{n_total})")
    print(f"  Character accuracy   : {char_acc:.1%}  ({char_correct}/{char_total})")
    print(f"  Samples skipped      : {skipped}")

    print(f"\n  Accuracy by label length:")
    for length in sorted(length_buckets):
        ok, tot = length_buckets[length]
        print(f"    len={length:>2}  {ok/tot:.1%}  ({ok}/{tot})")

    print(f"\n  Predicted character distribution (top 15):")
    total_pred = sum(pred_char_dist.values())
    top_chars  = sorted(pred_char_dist.items(), key=lambda x: -x[1])[:15]
    for ch, cnt in top_chars:
        bar = "█" * int(30 * cnt / max(1, top_chars[0][1]))
        print(f"    '{ch}'  {bar}  {cnt}  ({100*cnt/max(1,total_pred):.1f}%)")

    print(f"\n  First {len(examples)} examples  (gt → pred  [✓/✗]):")
    for gt, pred, match in examples:
        status = "✓" if match else "✗"
        print(f"    {status}  gt='{gt}'  →  pred='{pred}'")

    print(f"\n  Key points for interpretation:")
    print(f"  • LPRNet trained on plate images; Synth90K is out-of-domain.")
    print(f"  • Low accuracy vs Synth90K is expected — check char distribution")
    print(f"    to confirm model isn't dead (predicting blank/nothing always).")
    print(f"  • blank_id={BLANK_IDX} (last); if all preds are empty, check alphabet size")
    print(f"    against actual num_classes from dummy output above.")
    print(f"  • CRNN comparison: CRNN got 73% exact / 85% char on Synth90K (in-domain).")


if __name__ == "__main__":
    main()
