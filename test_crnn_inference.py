#!/usr/bin/env python3
"""
test_crnn_inference.py

CRNN inference following the EXACT official GitYCC/crnn-pytorch configuration
sourced from the repository's dataset.py and config.py:

  Image pipeline (dataset.py):
    image = Image.open(path).convert('L')                         # PIL grayscale
    image = image.resize((img_width=100, img_height=32),          # FIXED size, BILINEAR
                          resample=Image.BILINEAR)
    image = np.array(image)                                        # uint8 [0,255]
    image = image.reshape((1, img_height, img_width))             # [1, 32, 100]
    image = (image / 127.5) - 1.0                                 # → float [-1, 1]

  Decode (evaluate_config in config.py):
    decode_method = 'beam_search'
    beam_size     = 10

  Architecture (common_config):
    img_height       = 32
    img_width        = 100
    map_to_seq_hidden = 64
    rnn_hidden        = 256
    leaky_relu        = False

Usage:
    python test_crnn_inference.py --model crnn_synth90k.pt
    python test_crnn_inference.py --model crnn_synth90k.pt --crop plate.png
    python test_crnn_inference.py --model none              # random weights baseline
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).parent))
from ocr_backends import CRNNBackend, OCRResult

# ---------------------------------------------------------------------------
# Constants matching official GitYCC config.py
# ---------------------------------------------------------------------------
OFFICIAL_IMG_WIDTH  = 100
OFFICIAL_IMG_HEIGHT = 32
OFFICIAL_BEAM_SIZE  = 10
BLANK_ID            = 0        # GitYCC: index 0 = CTC blank, 1..N = alphabet[0..N-1]
ALPHABET            = CRNNBackend.DEFAULT_ALPHABET   # '0123456789abcdefghijklmnopqrstuvwxyz'

EXPECTED_TEXT = "vrj7774"   # lowercase — model alphabet is all lowercase


# ---------------------------------------------------------------------------
# Official preprocessing  (mirrors dataset.py exactly)
# ---------------------------------------------------------------------------

def official_preprocess(image_chw_01: torch.Tensor) -> torch.Tensor:
    """
    Convert a CHW float32 [0,1] RGB tensor → [1, 1, 32, 100] float32 [-1,1]
    using the exact PIL/numpy pipeline from GitYCC dataset.py.

      1. CHW [0,1] → PIL RGB → convert('L')   (PIL greyscale, not luminance weights)
      2. PIL resize to (100, 32) with BILINEAR  (fixed size, no aspect-ratio preserve)
      3. numpy uint8 → float, reshape to [1, 32, 100]
      4. (pixel / 127.5) - 1.0  → float [-1, 1]
      5. torch FloatTensor, add batch dim → [1, 1, 32, 100]
    """
    # Step 1: tensor [0,1] → PIL RGB
    arr_uint8 = (image_chw_01.permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    pil_rgb   = Image.fromarray(arr_uint8, mode="RGB")
    pil_grey  = pil_rgb.convert("L")                          # PIL greyscale

    # Step 2: fixed resize (width, height) — PIL uses (W, H) convention
    pil_resized = pil_grey.resize(
        (OFFICIAL_IMG_WIDTH, OFFICIAL_IMG_HEIGHT),
        resample=Image.BILINEAR,
    )

    # Step 3-4: numpy normalise
    arr = np.array(pil_resized, dtype=np.float32)              # [32, 100]  uint8→float
    arr = arr.reshape((1, OFFICIAL_IMG_HEIGHT, OFFICIAL_IMG_WIDTH))  # [1, 32, 100]
    arr = (arr / 127.5) - 1.0                                  # [-1, 1]

    # Step 5: tensor, add batch dim
    tensor = torch.from_numpy(arr).unsqueeze(0)                # [1, 1, 32, 100]
    return tensor


# ---------------------------------------------------------------------------
# Prefix beam search  (official evaluate_config: beam_search, beam_size=10)
# ---------------------------------------------------------------------------

def beam_search_decode(log_probs_np: np.ndarray,
                       beam_size: int = OFFICIAL_BEAM_SIZE,
                       blank_id: int = BLANK_ID) -> tuple[list[int], float]:
    """
    Prefix beam search CTC decode.

    Follows the standard prefix beam search described in Graves (2006) and
    implemented in ctc_decoder.py of GitYCC/crnn-pytorch.

    Parameters
    ----------
    log_probs_np : np.ndarray
        Shape [T, C].  Natural-log probabilities (i.e. log_softmax output).
    beam_size : int
        Number of candidate prefixes to keep at each timestep.
    blank_id : int
        Index of the CTC blank class.

    Returns
    -------
    best_labels : list[int]
        Sequence of class indices (1-based, blank-free).
    confidence : float
        Mean probability of the best path's emitted characters.
    """
    NEG_INF = float("-inf")

    # State: prefix (tuple of int) → (log_prob_ending_with_blank,
    #                                  log_prob_ending_with_non_blank)
    beams: dict[tuple, tuple[float, float]] = {(): (0.0, NEG_INF)}

    T, C = log_probs_np.shape

    for t in range(T):
        lp = log_probs_np[t]  # [C]
        new_beams: dict[tuple, list[float]] = defaultdict(lambda: [NEG_INF, NEG_INF])

        for prefix, (p_b, p_nb) in beams.items():
            p_total = np.logaddexp(p_b, p_nb)

            # ── emit blank ────────────────────────────────────────────────────
            new_beams[prefix][0] = np.logaddexp(new_beams[prefix][0], p_total + lp[blank_id])

            # ── emit non-blank character c ────────────────────────────────────
            for c in range(C):
                if c == blank_id:
                    continue
                p_c = lp[c]
                new_prefix = prefix + (c,)

                if prefix and prefix[-1] == c:
                    # Repeated char: only blank-terminated path creates a new symbol
                    new_beams[new_prefix][1] = np.logaddexp(
                        new_beams[new_prefix][1], p_b + p_c
                    )
                    # Non-blank-terminated path extends the *existing* prefix (collapse)
                    new_beams[prefix][1] = np.logaddexp(
                        new_beams[prefix][1], p_nb + p_c
                    )
                else:
                    new_beams[new_prefix][1] = np.logaddexp(
                        new_beams[new_prefix][1], p_total + p_c
                    )

        # Prune to beam_size by total log-prob
        beams = {
            k: (v[0], v[1])
            for k, v in sorted(
                new_beams.items(),
                key=lambda x: np.logaddexp(x[1][0], x[1][1]),
                reverse=True,
            )[:beam_size]
        }

    # Best hypothesis
    best_prefix, (best_pb, best_pnb) = max(
        beams.items(), key=lambda x: np.logaddexp(x[1][0], x[1][1])
    )
    confidence = float(np.exp(np.logaddexp(best_pb, best_pnb))) if best_prefix else 0.0
    return list(best_prefix), confidence


def labels_to_text(labels: list[int], alphabet: str) -> str:
    """Convert 1-based label list to string using alphabet."""
    chars = []
    for idx in labels:
        if 1 <= idx <= len(alphabet):
            chars.append(alphabet[idx - 1])
    return "".join(chars)


# ---------------------------------------------------------------------------
# Synthetic plate helper
# ---------------------------------------------------------------------------

def make_synthetic_plate(text: str, width: int = 320, height: int = 80) -> torch.Tensor:
    """Return a CHW float32 [0,1] RGB tensor of a rendered plate."""
    img  = Image.new("RGB", (width, height), color=(220, 220, 220))
    draw = ImageDraw.Draw(img)
    for font_path in [
        "/usr/share/fonts/liberation/LiberationMono-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
    ]:
        try:
            font = ImageFont.truetype(font_path, size=48)
            break
        except Exception:
            font = ImageFont.load_default()
    draw.text((20, 10), text, fill=(10, 10, 10), font=font)
    arr = np.array(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)  # [3, H, W]


def load_crop(path: str) -> torch.Tensor:
    """Load an image file → CHW float32 [0,1] RGB."""
    img = Image.open(path).convert("RGB")
    arr = np.array(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)


# ---------------------------------------------------------------------------
# Diagnostic runner
# ---------------------------------------------------------------------------

def show_stats(name: str, t) -> None:
    if isinstance(t, torch.Tensor):
        print(f"    {name}: shape={tuple(t.shape)}  min={t.min():.4f}  max={t.max():.4f}  mean={t.mean():.4f}")
    elif isinstance(t, np.ndarray):
        print(f"    {name}: shape={t.shape}  min={t.min():.4f}  max={t.max():.4f}  mean={t.mean():.4f}")


def run_official_inference(tag: str, model: torch.nn.Module, image_chw: torch.Tensor) -> None:
    print(f"\n{'='*64}")
    print(f"  {tag}")
    print(f"{'='*64}")

    # ── Preprocessing ─────────────────────────────────────────────────────────
    print("\n  [1] Official preprocessing (GitYCC dataset.py)")
    show_stats("input (CHW [0,1])", image_chw)

    inp = official_preprocess(image_chw)      # [1, 1, 32, 100]
    show_stats("after preprocess [1,1,32,100]", inp)
    print(f"    PIL greyscale + fixed resize ({OFFICIAL_IMG_WIDTH}×{OFFICIAL_IMG_HEIGHT}) + /127.5 - 1.0")

    # ── Forward pass ──────────────────────────────────────────────────────────
    print("\n  [2] Model forward pass")
    with torch.no_grad():
        logits = model(inp)                   # [T, 1, C]
    logits = logits.squeeze(1)                # [T, C]
    show_stats("logits [T, C]", logits)

    log_probs = F.log_softmax(logits, dim=1)  # [T, C]
    probs     = log_probs.exp()

    # Entropy check
    entropy     = -(probs * (probs + 1e-9).log()).sum(dim=1)
    max_entropy = float(np.log(logits.shape[1]))
    mean_ent    = entropy.mean().item()
    print(f"    Mean softmax entropy: {mean_ent:.3f} / {max_entropy:.3f}  "
          f"({'UNIFORM — random weights?' if mean_ent > 0.9 * max_entropy else 'confident'})")

    # ── Greedy decode ─────────────────────────────────────────────────────────
    print(f"\n  [3] Greedy decode  (T={logits.shape[0]}, C={logits.shape[1]}, blank=0)")
    print(f"    {'t':>4}  {'idx':>5}  {'prob':>6}  char")
    print("    " + "-"*32)
    chars_greedy, prev = [], BLANK_ID
    for t in range(logits.shape[0]):
        idx  = probs[t].argmax().item()
        prob = probs[t, idx].item()
        if idx == BLANK_ID:
            ch = "<blank>"
        elif 1 <= idx <= len(ALPHABET):
            ch = f"'{ALPHABET[idx - 1]}'"
        else:
            ch = f"<oob:{idx}>"
        show_t = t < 6 or t >= logits.shape[0] - 3
        if show_t:
            print(f"    {t:>4}  {idx:>5}  {prob:>6.3f}  {ch}")
        elif t == 6:
            print(f"    ...  ({logits.shape[0] - 9} timesteps omitted)")
        if idx != prev and idx != BLANK_ID and 1 <= idx <= len(ALPHABET):
            chars_greedy.append(ALPHABET[idx - 1])
        prev = idx
    text_greedy = "".join(chars_greedy)
    print(f"\n    Greedy result  : '{text_greedy}'")

    # ── Beam search decode  (official evaluate_config) ────────────────────────
    print(f"\n  [4] Beam search decode  (beam_size={OFFICIAL_BEAM_SIZE})")
    lp_np  = log_probs.cpu().numpy()          # [T, C]
    labels, conf = beam_search_decode(lp_np, beam_size=OFFICIAL_BEAM_SIZE)
    text_beam    = labels_to_text(labels, ALPHABET)
    print(f"    Beam result    : '{text_beam}'  (conf={conf:.4f})")

    # ── Expected vs actual ────────────────────────────────────────────────────
    print(f"\n  [5] Expected '{EXPECTED_TEXT}' vs predictions")
    for method, pred in [("greedy", text_greedy), ("beam  ", text_beam)]:
        n_match = sum(a == b for a, b in zip(pred, EXPECTED_TEXT))
        print(f"    {method}: '{pred}'  ({n_match}/{len(EXPECTED_TEXT)} chars match)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="CRNN inference — official GitYCC config")
    parser.add_argument("--model",  default="none",
                        help="Path to crnn_synth90k.pt  (or 'none' for random-weight baseline)")
    parser.add_argument("--crop",   default=None,
                        help="Path to a real plate crop image (PNG/JPG/HEIC)")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    args = parser.parse_args()

    # ── Print configuration ───────────────────────────────────────────────────
    print("\n[Official GitYCC inference config]")
    print(f"  img_width        = {OFFICIAL_IMG_WIDTH}  (fixed, no aspect-ratio preserve)")
    print(f"  img_height       = {OFFICIAL_IMG_HEIGHT}")
    print(f"  normalisation    = pixel / 127.5 - 1.0  → [-1, 1]")
    print(f"  greyscale        = PIL convert('L')  (not weighted luminance)")
    print(f"  decode_method    = beam_search  (beam_size={OFFICIAL_BEAM_SIZE})")
    print(f"  blank_id         = 0  (labels are 1-indexed)")
    print(f"  alphabet         = '{ALPHABET}'  ({len(ALPHABET)} chars, all lowercase)")
    print(f"  map_to_seq_hidden= 64  (if official crnn_synth90k.pt)")
    print(f"  rnn_hidden       = 256")
    print(f"  checkpoint       = {args.model}")

    # ── Build and load backend ────────────────────────────────────────────────
    backend = CRNNBackend(
        model_path=args.model,
        device=args.device,
        alphabet=ALPHABET,
        img_height=OFFICIAL_IMG_HEIGHT,
        n_hidden=256,
    )
    backend.ensure_loaded()

    n_params = sum(p.numel() for p in backend.parameters())
    print(f"  model params     = {n_params:,}")

    model = backend._model
    model.eval()

    # ── Test cases ────────────────────────────────────────────────────────────

    # 1. Synthetic plate rendered at native resolution
    synth = make_synthetic_plate(EXPECTED_TEXT)
    run_official_inference(f"Synthetic plate text='{EXPECTED_TEXT}'", model, synth)

    # 2. Real crop if provided
    if args.crop:
        real = load_crop(args.crop)
        run_official_inference(f"Real crop: {args.crop}", model, real)

    # 3. All-white sanity check
    white = torch.ones(3, 32, 100)
    run_official_inference("All-white sanity check (should be low confidence)", model, white)

    # ── Summary of differences vs backend implementation ─────────────────────
    print(f"\n{'='*64}")
    print("  DIFFERENCES: official config  vs  CRNNBackend._preprocess()")
    print(f"{'='*64}")
    rows = [
        ("Greyscale method",     "PIL convert('L')",             "weighted luminance (0.299R+...)"),
        ("Resize strategy",      f"fixed {OFFICIAL_IMG_WIDTH}×{OFFICIAL_IMG_HEIGHT}",
                                                                  "aspect-ratio preserve, h=32 only"),
        ("Resize interpolation", "PIL BILINEAR",                 "torch F.interpolate bilinear"),
        ("Normalisation",        "uint8/127.5 - 1.0",           "(float-0.5)/0.5  [equivalent]"),
        ("Decode method",        f"beam_search (k={OFFICIAL_BEAM_SIZE})", "greedy"),
    ]
    col = [max(len(r[i]) for r in rows) for i in range(3)]
    hdr = f"  {'Parameter':{col[0]}}  {'Official':{col[1]}}  {'Backend':{col[2]}}"
    print(hdr)
    print("  " + "-" * (col[0] + col[1] + col[2] + 4))
    for param, official, backend_val in rows:
        print(f"  {param:{col[0]}}  {official:{col[1]}}  {backend_val}")


if __name__ == "__main__":
    main()
