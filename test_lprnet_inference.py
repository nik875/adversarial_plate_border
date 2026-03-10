#!/usr/bin/env python3
"""
test_lprnet_inference.py

Tests the NVIDIA TAO US LPRNet deployable ONNX model on synthetic
license plate images with known ground truth.

Sourced from NVIDIA TAO TF1 backend  (nvidia_tao_tf1/cv/lprnet/):
  img_utils.py : preprocess() = cv2.resize((W,H)) / 255.0  NHWC→NCHW
  inference.py : blank_id = len(classes)  (blank is LAST, not first)
  ctc_decoder  : CTC greedy — collapse repeats, strip blank

Model:
  Input  : [B, 3, 48, 96]  float32  [0, 1]
  Output : [B, T]  int64   per-timestep argmax  (already in ONNX graph)
  Alphabet : 0-9 A-Z (36 chars, 0-indexed), blank at index 36

Usage:
    python test_lprnet_inference.py \\
        --model weights/lprnet_deployable_onnx_v1.1/us_lprnet_baseline18_deployable.onnx
    python test_lprnet_inference.py --model <path> --crop plate.png

Requires:
    pip install onnxruntime pillow numpy
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------------------
# NVIDIA TAO US LPRNet constants
# ---------------------------------------------------------------------------
ALPHABET  = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"   # 36 chars, 0-indexed
BLANK_IDX = len(ALPHABET)                             # 36 — blank is LAST

INPUT_H = 48
INPUT_W = 96

# Test plates: cover digits-only, letters-only, mixed, and the target plate
TEST_PLATES = [
    "VRJ7774",    # target plate (uppercase)
    "ABC1234",
    "XYZ9999",
    "7ABC234",
    "ABC",
    "12345",
    "123456",
    "1234567",
    "ABCDEFG",
    "CA12345",
    "NY7ABC4",
]

FONT_PATHS = [
    "/usr/share/fonts/liberation/LiberationMono-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationMono-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
    "/usr/share/fonts/dejavu/DejaVuSansMono-Bold.ttf",
]


# ---------------------------------------------------------------------------
# Synthetic plate renderer
# ---------------------------------------------------------------------------

def _find_font(size: int):
    for path in FONT_PATHS:
        try:
            return ImageFont.truetype(path, size=size)
        except Exception:
            pass
    return ImageFont.load_default()


def render_plate(text: str,
                 width: int = 300, height: int = 80,
                 bg: tuple = (240, 240, 200),
                 fg: tuple = (10, 10, 10),
                 font_size: int = 52) -> Image.Image:
    """
    Render a US-style plate: light yellow background, bold dark text.
    Returns a PIL RGB Image.
    """
    img  = Image.new("RGB", (width, height), color=bg)
    draw = ImageDraw.Draw(img)
    font = _find_font(font_size)

    # Center the text
    bbox   = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x = (width  - tw) // 2
    y = (height - th) // 2
    draw.text((x, y), text, fill=fg, font=font)
    return img


# ---------------------------------------------------------------------------
# Preprocessing  (img_utils.py — /255.0, NCHW)
# ---------------------------------------------------------------------------

def preprocess(pil_image: Image.Image) -> np.ndarray:
    rgb     = pil_image.convert("RGB")
    resized = rgb.resize((INPUT_W, INPUT_H), resample=Image.BILINEAR)
    arr     = np.array(resized, dtype=np.float32) / 255.0
    arr     = arr.transpose(2, 0, 1)        # HWC → CHW
    return arr[np.newaxis]                  # [1, 3, 48, 96]


# ---------------------------------------------------------------------------
# CTC greedy decode  (blank_id = last index — TAO convention)
# ---------------------------------------------------------------------------

def ctc_decode(indices: np.ndarray) -> str:
    chars, prev = [], -1
    for idx in map(int, indices):
        if idx != prev:
            if idx != BLANK_IDX and 0 <= idx < len(ALPHABET):
                chars.append(ALPHABET[idx])
        prev = idx
    return "".join(chars)


# ---------------------------------------------------------------------------
# Model loader and inspector
# ---------------------------------------------------------------------------

def load_model(model_path: str):
    try:
        import onnxruntime as ort
    except ImportError:
        print("ERROR: onnxruntime not found.  pip install onnxruntime")
        sys.exit(1)

    session = ort.InferenceSession(
        model_path,
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    input_name  = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    out_shape   = session.get_outputs()[0].shape
    out_dtype   = session.get_outputs()[0].type

    print(f"  Input  '{input_name}': {session.get_inputs()[0].shape}  {session.get_inputs()[0].type}")
    print(f"  Output '{output_name}': {out_shape}  {out_dtype}")

    # Probe with dummy to get runtime shape and detect probs vs indices
    dummy     = np.zeros((1, 3, INPUT_H, INPUT_W), dtype=np.float32)
    dummy_out = session.run([output_name], {input_name: dummy})[0]
    print(f"  Dummy output: shape={dummy_out.shape}  dtype={dummy_out.dtype}")

    outputs_probs = dummy_out.ndim >= 2 and dummy_out.shape[-1] > 10 and "int" not in out_dtype
    if outputs_probs:
        num_classes = dummy_out.shape[-1]
        print(f"  → raw softmax probs  (num_classes={num_classes}, blank_id={num_classes-1})")
    else:
        print(f"  → argmax indices already applied in graph  (blank_id={BLANK_IDX})")

    return session, input_name, output_name, outputs_probs


# ---------------------------------------------------------------------------
# Run one plate image through the model
# ---------------------------------------------------------------------------

def run_plate(session, input_name: str, output_name: str,
              outputs_probs: bool, pil_img: Image.Image) -> tuple[str, np.ndarray]:
    inp  = preprocess(pil_img)
    outs = session.run([output_name], {input_name: inp})[0]  # [1, T] or [1, T, C]

    if outputs_probs:
        indices = outs[0].argmax(axis=-1)
    else:
        indices = outs[0].flatten()

    pred = ctc_decode(indices)
    return pred, indices


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Test NVIDIA TAO US LPRNet on synthetic plate images"
    )
    parser.add_argument("--model", required=True,
                        help="Path to us_lprnet_baseline18_deployable.onnx")
    parser.add_argument("--crop", default=None,
                        help="Path to an additional real plate crop to test")
    args = parser.parse_args()

    print(f"\n[NVIDIA TAO US LPRNet — synthetic plate test]")
    print(f"  Model      : {args.model}")
    print(f"  Alphabet   : '{ALPHABET}'  ({len(ALPHABET)} chars, 0-indexed)")
    print(f"  blank_id   : {BLANK_IDX}  (LAST — TAO convention)")
    print(f"  Input size : {INPUT_H} × {INPUT_W}")
    print(f"  Preprocess : RGB  /255.0  →  [0,1]  NCHW\n")

    session, input_name, output_name, outputs_probs = load_model(args.model)

    # ── Synthetic plates ──────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  SYNTHETIC PLATE RESULTS")
    print(f"{'='*60}")
    print(f"  {'GT':>10}   {'Pred':>12}   {'Match':>5}   raw indices")
    print(f"  {'-'*55}")

    correct = 0
    for text in TEST_PLATES:
        pil_img = render_plate(text)
        pred, indices = run_plate(session, input_name, output_name,
                                  outputs_probs, pil_img)
        match = pred == text
        correct += int(match)
        mark = "✓" if match else "✗"
        idx_str = " ".join(str(i) for i in indices[:12])
        print(f"  {mark} {text:>10}  →  {pred:>12}   [{idx_str}]")

    acc = correct / len(TEST_PLATES)
    print(f"\n  Exact match: {correct}/{len(TEST_PLATES)}  ({acc:.0%})")

    # ── Per-style variants for VRJ7774 ────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  STYLE VARIANTS — 'VRJ7774'")
    print(f"{'='*60}")
    variants = [
        ("Yellow bg, dark text (standard US)", (240, 240, 200), (10, 10, 10)),
        ("White bg, dark text",                (255, 255, 255), (20, 20, 20)),
        ("White bg, black text, larger",       (255, 255, 255), (0,  0,   0)),
        ("Dark bg, white text (inverted)",     (30,  30,  30),  (240, 240, 240)),
    ]
    for desc, bg, fg in variants:
        pil_img = render_plate("VRJ7774", bg=bg, fg=fg)
        pred, _ = run_plate(session, input_name, output_name, outputs_probs, pil_img)
        mark = "✓" if pred == "VRJ7774" else "✗"
        print(f"  {mark}  {desc}")
        print(f"     pred='{pred}'")

    # ── Preprocessing variant: BGR instead of RGB ─────────────────────────────
    print(f"\n{'='*60}")
    print(f"  BGR vs RGB TEST (model trained with OpenCV BGR)")
    print(f"{'='*60}")
    pil_rgb = render_plate("VRJ7774")
    # BGR version: swap R and B channels
    arr_rgb  = np.array(pil_rgb)
    arr_bgr  = arr_rgb[..., ::-1].copy()
    pil_bgr  = Image.fromarray(arr_bgr)

    pred_rgb, _ = run_plate(session, input_name, output_name, outputs_probs, pil_rgb)
    pred_bgr, _ = run_plate(session, input_name, output_name, outputs_probs, pil_bgr)
    print(f"  RGB input  →  '{pred_rgb}'")
    print(f"  BGR input  →  '{pred_bgr}'")
    better = "BGR" if pred_bgr == "VRJ7774" else ("RGB" if pred_rgb == "VRJ7774" else "neither")
    print(f"  → {'Both wrong; plate text just out of model distribution' if better == 'neither' else better + ' matches ground truth'}")

    # ── Real crop if provided ─────────────────────────────────────────────────
    if args.crop:
        print(f"\n{'='*60}")
        print(f"  REAL CROP: {args.crop}")
        print(f"{'='*60}")
        real = Image.open(args.crop).convert("RGB")
        print(f"  Image size: {real.size}")
        pred_real, indices_real = run_plate(session, input_name, output_name,
                                            outputs_probs, real)
        print(f"  Prediction : '{pred_real}'")
        idx_str = " ".join(str(i) for i in indices_real[:18])
        print(f"  Raw indices: [{idx_str}]")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  WHAT TO LOOK FOR")
    print(f"{'='*60}")
    print(f"  • Are any predictions correct? → model is loading and decoding")
    print(f"  • All predictions empty?       → blank_id wrong; try blank_id=0")
    print(f"  • Predictions all same char?   → softmax collapsed; check normalization")
    print(f"  • BGR better than RGB?         → model expects BGR (OpenCV convention)")
    print(f"  • No digits in predictions?    → alphabet mapping is off")


if __name__ == "__main__":
    main()
