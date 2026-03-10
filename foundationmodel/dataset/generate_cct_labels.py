#!/usr/bin/env python3
"""
generate_cct_labels.py

Scans all pre-cropped license-plate image directories, runs every image
through the CCT-S-V1 Global model, and writes a CSV manifest:

    image_path,label

The CCT-S prediction is used directly as the ground-truth label.

Usage:
    python foundationmodel/dataset/generate_cct_labels.py
    python foundationmodel/dataset/generate_cct_labels.py \
        --output foundationmodel/dataset/cct_labels.csv \
        --device cuda --batch-size 256

Requires:
    pip install onnx onnx2torch torch pillow numpy tqdm
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import random

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Model config (matches test_cct_own_data.py)
# ---------------------------------------------------------------------------

ALPHABET  = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ_'
BLANK     = '_'
OCR_H, OCR_W = 64, 128
OCR_PATH  = Path.home() / ".cache/fast-plate-ocr/cct-s-v1-global-model/cct_s_v1_global.onnx"

# ---------------------------------------------------------------------------
# Directories that contain pre-cropped plate images
# ---------------------------------------------------------------------------

CROP_DIRS: list[Path] = [
    Path.home() / ".cache" / "roboflow_lpr_crops",
    Path.home() / ".cache" / "kaggle_lp_crops",
    Path.home() / ".cache" / "mercosur_crops",
]


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(model_path: str, device: str) -> torch.nn.Module:
    import onnx
    import onnx2torch

    ocr_onnx = onnx.load(model_path)
    model = onnx2torch.convert(ocr_onnx).to(device).eval()

    # onnx2torch stores some ONNX initializers as plain tensor attributes
    # (not registered buffers), so .to(device) misses them — move manually.
    for module in model.modules():
        for attr, val in list(vars(module).items()):
            if isinstance(val, torch.Tensor) and not isinstance(val, torch.nn.Parameter):
                object.__setattr__(module, attr, val.to(device))

    return model


# ---------------------------------------------------------------------------
# Preprocessing / decoding
# ---------------------------------------------------------------------------

def preprocess_batch(pil_images: list[Image.Image], device: torch.device) -> torch.Tensor:
    """List of PIL images → [N, 64, 128, 3] float32 [0,255] NHWC on device."""
    tensors = []
    for img in pil_images:
        rgb = img.convert("RGB")
        arr = np.array(rgb, dtype=np.float32) / 255.0        # [H, W, 3]
        t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]
        t = F.interpolate(t, size=(OCR_H, OCR_W), mode='bilinear', align_corners=False)
        tensors.append(t.permute(0, 2, 3, 1))                 # [1, 64, 128, 3]
    return (torch.cat(tensors, dim=0) * 255).to(device)       # [N, 64, 128, 3]


def decode_batch(logits: torch.Tensor) -> list[str]:
    """[N, 9, 37] logits → list of text strings (blanks stripped)."""
    probs   = torch.softmax(logits, dim=-1)
    indices = probs.argmax(dim=-1)   # [N, 9]
    results = []
    for row in indices.tolist():
        text = "".join(ALPHABET[i] for i in row if ALPHABET[i] != BLANK)
        results.append(text)
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate CCT-S ground-truth labels for all cropped plate images"
    )
    parser.add_argument("--output", default="foundationmodel/dataset/cct_labels.csv",
                        help="Output CSV path (default: foundationmodel/dataset/cct_labels.csv)")
    parser.add_argument("--model", default=str(OCR_PATH),
                        help=f"Path to cct_s_v1_global.onnx (default: {OCR_PATH})")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=128,
                        help="Images per inference batch (default: 128)")
    parser.add_argument("--samples", type=int, default=20,
                        help="Sample images to save per dataset (default: 20)")
    parser.add_argument("--samples-dir", default="foundationmodel/dataset/samples",
                        help="Directory for sample images (default: foundationmodel/dataset/samples)")
    args = parser.parse_args()

    try:
        import onnx, onnx2torch  # noqa: F401
    except ImportError:
        print("ERROR: pip install onnx onnx2torch")
        raise SystemExit(1)

    print(f"[CCT-S Label Generator]")
    print(f"  Model  : {args.model}")
    print(f"  Device : {args.device}")
    print(f"  Output : {args.output}")
    print(f"  Batch  : {args.batch_size}")

    # -----------------------------------------------------------------------
    # Collect all image paths, grouped by dataset
    # -----------------------------------------------------------------------
    all_paths: list[Path] = []
    # Map global index → dataset name (parent dir name)
    index_to_dataset: dict[int, str] = {}

    for d in CROP_DIRS:
        if not d.exists():
            print(f"  [skip] {d} — not found")
            continue
        found = sorted(d.glob("*.png")) + sorted(d.glob("*.jpg"))
        print(f"  {d.name}: {len(found)} images")
        ds_name = d.name
        start_idx = len(all_paths)
        all_paths.extend(found)
        for i in range(len(found)):
            index_to_dataset[start_idx + i] = ds_name

    print(f"\n  Total images: {len(all_paths)}")
    if not all_paths:
        print("No images found. Exiting.")
        return

    # -----------------------------------------------------------------------
    # Load model
    # -----------------------------------------------------------------------
    print("\n[Loading model]")
    model  = load_model(args.model, args.device)
    device = torch.device(args.device)
    print("  OK")

    # -----------------------------------------------------------------------
    # Batch inference → CSV
    # -----------------------------------------------------------------------
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    written  = 0
    skipped  = 0
    filtered = 0
    bs       = args.batch_size

    # Pick evenly-spaced sample indices per dataset
    n_per_ds = args.samples
    sample_indices: set[int] = set()
    ds_to_indices: dict[str, list[int]] = {}
    for idx, ds in index_to_dataset.items():
        ds_to_indices.setdefault(ds, []).append(idx)
    for ds, idxs in ds_to_indices.items():
        n = min(n_per_ds, len(idxs))
        step = max(1, len(idxs) // n)
        sample_indices.update(idxs[::step][:n])

    samples_dir = Path(args.samples_dir)
    if samples_dir.exists():
        for f in samples_dir.iterdir():
            f.unlink()
    samples_dir.mkdir(parents=True, exist_ok=True)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
    except Exception:
        font = ImageFont.load_default()

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "label"])

        pbar = tqdm(range(0, len(all_paths), bs), unit="batch", dynamic_ncols=True)
        for start in pbar:
            batch_paths = all_paths[start : start + bs]

            pil_imgs: list[Image.Image] = []
            valid_paths: list[Path]     = []
            valid_indices: list[int]    = []

            for i, p in enumerate(batch_paths):
                try:
                    img = Image.open(p).convert("RGB")
                    if img.height < 32:
                        filtered += 1
                        img.close()
                        continue
                    pil_imgs.append(img)
                    valid_paths.append(p)
                    valid_indices.append(start + i)
                except Exception:
                    skipped += 1

            if not pil_imgs:
                continue

            try:
                inp = preprocess_batch(pil_imgs, device)
                with torch.no_grad():
                    logits = model(inp)   # [N, 9, 37]
                labels = decode_batch(logits)
            except Exception as e:
                tqdm.write(f"  Warning: batch {start}-{start+bs} failed: {e}")
                skipped += len(pil_imgs)
                for img in pil_imgs:
                    img.close()
                continue

            for img, path, label, idx in zip(pil_imgs, valid_paths, labels, valid_indices):
                writer.writerow([str(path), label])

                if idx in sample_indices:
                    vis = img.copy()
                    draw = ImageDraw.Draw(vis)
                    text = label if label else "(empty)"
                    # Black shadow + white text for legibility on any background
                    draw.text((3, 3), text, fill=(0, 0, 0),       font=font)
                    draw.text((2, 2), text, fill=(255, 255, 255), font=font)
                    out_name = f"{idx:06d}_{path.stem}.png"
                    vis.save(samples_dir / out_name)

                img.close()

            written += len(valid_paths)
            pbar.set_postfix({"written": written, "skipped": skipped})

    total = len(all_paths)
    print(f"\n{'='*50}")
    print(f"  Total images   : {total}")
    print(f"  Kept (>=32px)  : {written}")
    print(f"  Filtered (<32px): {filtered}")
    print(f"  Skipped (error): {skipped}")
    print(f"{'='*50}")
    print(f"  CSV    → {output_path}")
    print(f"  Samples → {samples_dir}/")


if __name__ == "__main__":
    main()
