#!/usr/bin/env python3
"""
test_lprnet_inference.py

Tests the NVIDIA TAO US LPRNet ONNX model on the OpenALPR end-to-end US
benchmark dataset — the same data NVIDIA uses in their TAO tutorial for this
exact model.

Dataset: github.com/openalpr/benchmarks  endtoend/us/
  ~220 full car images, each paired with a .txt label:
      filename  xmin  ymin  width  height  plate_text
  Script downloads via GitHub API + raw URLs, crops the plate bbox,
  and feeds each crop directly to LPRNet.

Preprocessing (NVIDIA TAO TF1 backend, img_utils.py):
  cv2.resize((96, 48)) / 255.0  →  NCHW float32  [0, 1]

Decoding:
  blank_id = len(alphabet) = 36  (LAST index — TAO convention)
  CTC greedy: collapse repeats, strip blank

Usage:
    python test_lprnet_inference.py \\
        --model weights/lprnet_deployable_onnx_v1.1/us_lprnet_baseline18_deployable.onnx
    python test_lprnet_inference.py --model <path> --cache openalpr_us/

Requires:
    pip install onnxruntime pillow numpy requests tqdm
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from collections import defaultdict

import numpy as np
import requests
from PIL import Image
from tqdm import tqdm

# ---------------------------------------------------------------------------
# NVIDIA TAO US LPRNet constants
# ---------------------------------------------------------------------------
ALPHABET  = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXY"  # 35 chars — TAO US LPRNet has no 'Z' class
BLANK_IDX = len(ALPHABET)   # 35 — blank is LAST (TAO convention)

INPUT_H = 48
INPUT_W = 96

GITHUB_API_URL  = "https://api.github.com/repos/openalpr/benchmarks/contents/endtoend/us"
GITHUB_RAW_BASE = "https://raw.githubusercontent.com/openalpr/benchmarks/master/endtoend/us"


# ---------------------------------------------------------------------------
# Dataset download
# ---------------------------------------------------------------------------

def list_dataset_files(cache_dir: Path) -> list[str]:
    """Return sorted list of base names (no extension) from the GitHub API."""
    cache_file = cache_dir / "_filelist.json"
    if cache_file.exists():
        return json.loads(cache_file.read_text())

    print("  Fetching file list from GitHub API...")
    resp = requests.get(GITHUB_API_URL, timeout=30,
                        headers={"Accept": "application/vnd.github.v3+json"})
    resp.raise_for_status()
    entries = resp.json()

    txt_names = {e["name"].rsplit(".", 1)[0]
                 for e in entries if e["name"].endswith(".txt")}
    img_names = {e["name"].rsplit(".", 1)[0]
                 for e in entries if e["name"].endswith(".jpg")}
    bases = sorted(txt_names & img_names)

    cache_file.write_text(json.dumps(bases))
    print(f"  Found {len(bases)} image+label pairs")
    return bases


def fetch_bytes(url: str, retries: int = 3) -> bytes:
    for attempt in range(retries):
        try:
            r = requests.get(url, timeout=30)
            r.raise_for_status()
            return r.content
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(1.5 ** attempt)


def load_sample(base: str, cache_dir: Path) -> tuple[Image.Image, str, tuple] | None:
    """
    Download (or load from cache) one image+label pair.
    Returns (full_car_image, plate_text, (xmin, ymin, w, h)) or None.
    Label format:  filename<TAB>xmin<TAB>ymin<TAB>width<TAB>height<TAB>plate_text
    """
    img_path = cache_dir / f"{base}.jpg"
    txt_path = cache_dir / f"{base}.txt"

    if not txt_path.exists():
        txt_path.write_bytes(fetch_bytes(f"{GITHUB_RAW_BASE}/{base}.txt"))
    if not img_path.exists():
        img_path.write_bytes(fetch_bytes(f"{GITHUB_RAW_BASE}/{base}.jpg"))

    fields = txt_path.read_text().strip().split("\t")
    if len(fields) < 6:
        return None
    try:
        xmin, ymin, w, h = int(fields[1]), int(fields[2]), int(fields[3]), int(fields[4])
    except ValueError:
        return None
    plate_text = fields[5].strip().upper()

    # Skip plates with chars outside the model alphabet (e.g. spaces, dashes)
    if not plate_text or not all(c in ALPHABET for c in plate_text):
        return None

    pil_img = Image.open(img_path).convert("RGB")
    return pil_img, plate_text, (xmin, ymin, w, h)


# ---------------------------------------------------------------------------
# Plate crop
# ---------------------------------------------------------------------------

def crop_plate(pil_img: Image.Image, bbox: tuple, padding: int = 4) -> Image.Image:
    """Crop the plate region from a full car image with a small padding."""
    xmin, ymin, w, h = bbox
    iw, ih = pil_img.size
    x1 = max(0, xmin - padding)
    y1 = max(0, ymin - padding)
    x2 = min(iw, xmin + w + padding)
    y2 = min(ih, ymin + h + padding)
    return pil_img.crop((x1, y1, x2, y2))


# ---------------------------------------------------------------------------
# Preprocessing + decode
# ---------------------------------------------------------------------------

def preprocess(pil_image: Image.Image) -> np.ndarray:
    """RGB PIL → [1, 3, 48, 96] float32 [0,1]  (NVIDIA TAO img_utils.py)."""
    rgb     = pil_image.convert("RGB")
    resized = rgb.resize((INPUT_W, INPUT_H), resample=Image.BILINEAR)
    arr     = np.array(resized, dtype=np.float32) / 255.0
    return arr.transpose(2, 0, 1)[np.newaxis]   # [1, 3, 48, 96]


def ctc_decode(indices: np.ndarray) -> str:
    """CTC greedy decode — blank_id = LAST (NVIDIA TAO convention)."""
    chars, prev = [], -1
    for idx in map(int, indices):
        if idx != prev:
            if idx != BLANK_IDX and 0 <= idx < len(ALPHABET):
                chars.append(ALPHABET[idx])
        prev = idx
    return "".join(chars)


# ---------------------------------------------------------------------------
# Model
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
    out_dtype   = session.get_outputs()[0].type

    dummy     = np.zeros((1, 3, INPUT_H, INPUT_W), dtype=np.float32)
    dummy_out = session.run([output_name], {input_name: dummy})[0]

    print(f"  Input  : {session.get_inputs()[0].shape}  {session.get_inputs()[0].type}")
    print(f"  Output : {dummy_out.shape}  {dummy_out.dtype}")

    outputs_probs = (dummy_out.ndim >= 2
                     and dummy_out.shape[-1] > 10
                     and "int" not in out_dtype)
    print(f"  Format : {'softmax probs' if outputs_probs else 'argmax indices (baked in graph)'}")
    print(f"  blank_id = {BLANK_IDX}  (last=35 — TAO US LPRNet convention, no Z class)")
    return session, input_name, output_name, outputs_probs


def run_inference(session, input_name, output_name, outputs_probs,
                  pil_img: Image.Image) -> str:
    inp = preprocess(pil_img)
    out = session.run([output_name], {input_name: inp})[0]
    if outputs_probs:
        indices = out[0].argmax(axis=-1)
    else:
        indices = out[0].flatten()
    return ctc_decode(indices)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Test NVIDIA TAO US LPRNet on OpenALPR end-to-end US benchmark"
    )
    parser.add_argument("--model", required=True,
                        help="Path to us_lprnet_baseline18_deployable.onnx")
    parser.add_argument("--cache", default="openalpr_us_cache",
                        help="Directory to cache downloaded images (default: openalpr_us_cache/)")
    parser.add_argument("--n", type=int, default=0,
                        help="Max samples to evaluate (0 = all ~220, default: all)")
    args = parser.parse_args()

    cache_dir = Path(args.cache)
    cache_dir.mkdir(exist_ok=True)

    print(f"\n[NVIDIA TAO US LPRNet — OpenALPR benchmark]")
    print(f"  Model   : {args.model}")
    print(f"  Cache   : {cache_dir}/  (images cached after first download)")
    print(f"  Dataset : openalpr/benchmarks endtoend/us  (~220 real US plate images)")

    # ── Load model ────────────────────────────────────────────────────────────
    print(f"\n[Model]")
    session, input_name, output_name, outputs_probs = load_model(args.model)

    # ── File list ─────────────────────────────────────────────────────────────
    print(f"\n[Dataset]")
    bases = list_dataset_files(cache_dir)
    if args.n > 0:
        bases = bases[:args.n]

    # ── Evaluate ──────────────────────────────────────────────────────────────
    exact_correct  = 0
    char_correct   = 0
    char_total     = 0
    skipped        = 0
    evaluated      = 0
    examples       = []
    length_buckets = defaultdict(lambda: [0, 0])

    pbar = tqdm(bases, desc="Downloading & evaluating", unit="plate")

    for base in pbar:
        try:
            result = load_sample(base, cache_dir)
        except Exception:
            skipped += 1
            continue

        if result is None:
            skipped += 1
            continue

        pil_full, gt, bbox = result
        plate_crop = crop_plate(pil_full, bbox)

        try:
            pred = run_inference(session, input_name, output_name,
                                 outputs_probs, plate_crop)
        except Exception:
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

        if len(examples) < 30:
            examples.append((gt, pred, match))

        pbar.set_postfix({"exact_acc": f"{exact_correct/evaluated:.1%}", "skip": skipped})

    pbar.close()

    # ── Results ───────────────────────────────────────────────────────────────
    exact_acc = exact_correct / evaluated if evaluated else 0.0
    char_acc  = char_correct  / char_total if char_total else 0.0

    print(f"\n{'='*60}")
    print(f"  RESULTS — OpenALPR US benchmark  [NVIDIA TAO LPRNet]")
    print(f"{'='*60}")
    print(f"  Evaluated            : {evaluated}")
    print(f"  Skipped              : {skipped}  (download errors / non-alphanumeric GT)")
    print(f"  Exact match accuracy : {exact_acc:.1%}  ({exact_correct}/{evaluated})")
    print(f"  Character accuracy   : {char_acc:.1%}  ({char_correct}/{char_total})")

    print(f"\n  Accuracy by plate length:")
    for length in sorted(length_buckets):
        ok, tot = length_buckets[length]
        print(f"    len={length:>2}  {ok/tot:.1%}  ({ok}/{tot})")

    print(f"\n  Examples  (gt → pred  [✓/✗]):")
    for gt, pred, match in examples:
        status = "✓" if match else "✗"
        print(f"    {status}  gt='{gt}'  →  pred='{pred}'")

    print(f"\n  Reference: NVIDIA NGC reports 97.49% on their internal eval set.")
    print(f"  OpenALPR benchmark is a different public US plate set — gap is expected.")
    print(f"  Note: TAO US LPRNet has no 'Z' output class (blank_id=35); plates with Z will fail.")


if __name__ == "__main__":
    main()
