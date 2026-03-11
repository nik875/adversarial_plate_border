#!/usr/bin/env python3
"""
debug_augmentations.py

Apply a random patch to one image and save N samples of what the training
augmentation pipeline actually produces, so you can visually verify that
augment_plate() in trainer.py is behaving correctly.

The augmentation is imported directly from trainer.py — there is no
separate reimplementation here, so this script always reflects the live
training code.

Usage:
    python debug_augmentations.py                      # first row in preproc_labels.csv
    python debug_augmentations.py --random-idx         # random row
    python debug_augmentations.py --idx 12             # specific row
    python debug_augmentations.py --n 20               # number of samples
    python debug_augmentations.py --out my_debug_dir
    python debug_augmentations.py --device cuda
"""

import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import kornia.geometry as KG
from PIL import Image

from trainer import PatchDecoder, PATCH_WIDTH, PATCH_HEIGHT, augment_plate


# ---------------------------------------------------------------------------
# Image I/O
# ---------------------------------------------------------------------------

def load_image(path: str) -> torch.Tensor:
    """Load any image (including HEIC) as [C, H, W] float32 RGB in [0, 1]."""
    try:
        import pillow_heif
        pillow_heif.register_heif_opener()
    except ImportError:
        pass
    img = Image.open(path).convert("RGB")
    arr = np.array(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)


def save_image(tensor: torch.Tensor, path: Path) -> None:
    """Save [1, C, H, W] or [C, H, W] tensor to PNG."""
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    arr = (tensor.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    Image.fromarray(arr).save(path)
    print(f"  saved {path.name}")


# ---------------------------------------------------------------------------
# Patch application — matches trainer.py apply_patch_to_image homography path
# ---------------------------------------------------------------------------

def apply_patch(
    image: torch.Tensor,       # [1, 3, H, W] float32 [0,1]
    corners: torch.Tensor,     # [4, 2] plate corners
    patch: torch.Tensor,       # [3, PATCH_HEIGHT, PATCH_WIDTH] float32 [0,1]
    border_scale: float = 1.4,
    device: torch.device = torch.device("cpu"),
    canonical_aug_fn=None,
) -> torch.Tensor:
    """
    Composite patch onto image in canonical space then reproject.
    canonical_aug_fn, if provided, is called on the [1,3,ph,pw] composite
    before warping back — matching trainer.py's augment=True path exactly.
    """
    B, C, H, W = image.shape
    ph, pw = PATCH_HEIGHT, PATCH_WIDTH
    image  = image.to(device)

    plate  = corners.to(device)
    cx, cy = plate[:, 0].mean(), plate[:, 1].mean()
    center = torch.tensor([cx, cy], device=device)
    border = (center.unsqueeze(0) + (plate - center.unsqueeze(0)) * border_scale).unsqueeze(0)

    src  = torch.tensor([[0, 0], [pw, 0], [pw, ph], [0, ph]],
                        dtype=torch.float32, device=device).unsqueeze(0)
    ones = torch.ones(B, 1, ph, pw, device=device)

    M_border       = KG.get_perspective_transform(src, border)
    M_to_canonical = KG.get_perspective_transform(border, src)

    canonical = KG.warp_perspective(image, M_to_canonical, (ph, pw),
                                    mode="bilinear", padding_mode="zeros",
                                    align_corners=True)

    M_c  = M_to_canonical[0]
    ph4  = torch.cat([plate, plate.new_ones(4, 1)], dim=1).T
    pc_h = M_c @ ph4
    plate_canonical = (pc_h[:2] / pc_h[2:3]).T.contiguous().unsqueeze(0)
    M_plate_in_canonical = KG.get_perspective_transform(src, plate_canonical)
    plate_mask = KG.warp_perspective(ones, M_plate_in_canonical, (ph, pw),
                                     mode="bilinear", padding_mode="zeros",
                                     align_corners=True).expand(-1, 3, -1, -1)

    # Scale patch brightness to match the plate region before compositing.
    patch_b = patch.unsqueeze(0).to(device)
    with torch.no_grad():
        plate_brightness = ((canonical * plate_mask).sum()
                            / plate_mask.sum().clamp(min=1e-6))
        patch_brightness = patch_b.mean().clamp(min=1e-6)
        brightness_scale = (plate_brightness / patch_brightness).clamp(0.2, 5.0)
    patch_b = patch_b * brightness_scale

    composite = patch_b * (1 - plate_mask) + canonical * plate_mask

    if canonical_aug_fn is not None:
        composite = canonical_aug_fn(composite)

    warped_back = KG.warp_perspective(composite, M_border, (H, W),
                                      mode="bilinear", padding_mode="zeros",
                                      align_corners=True)
    border_mask = KG.warp_perspective(ones, M_border, (H, W),
                                      mode="bilinear", padding_mode="zeros",
                                      align_corners=True).expand(-1, 3, -1, -1)

    return torch.clamp(image * (1 - border_mask) + warped_back * border_mask, 0, 1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv",          default="preproc_labels.csv")
    parser.add_argument("--idx",          type=int, default=0,
                        help="Row index into the CSV (ignored if --random-idx)")
    parser.add_argument("--random-idx",   action="store_true",
                        help="Pick a random row from the CSV")
    parser.add_argument("--n",            type=int, default=16,
                        help="Number of augmented samples to save (default 16)")
    parser.add_argument("--out",          default="debug_augmentations")
    parser.add_argument("--device",       default="cpu")
    parser.add_argument("--seed-channels", type=int, default=128)
    parser.add_argument("--border-scale", type=float, default=1.4)
    parser.add_argument("--seed",         type=int, default=None)
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    device = torch.device(args.device)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load image + corners ───────────────────────────────────────────────
    df  = pd.read_csv(args.csv)
    idx = random.randint(0, len(df) - 1) if args.random_idx else args.idx
    row = df.iloc[idx]
    print(f"Image index : {idx}")
    print(f"File        : {row['filename']}")

    image_chw = load_image(row["filename"])
    corners   = np.array([[row[f"p{i+1}_x"], row[f"p{i+1}_y"]] for i in range(4)],
                          dtype=np.float32)
    corners_t = torch.from_numpy(corners)

    _, H, W = image_chw.shape
    if max(H, W) > 2000:
        image_chw = F.interpolate(image_chw.unsqueeze(0), scale_factor=0.5,
                                  mode="bilinear", align_corners=False).squeeze(0)
        corners_t = corners_t * 0.5

    image_b = image_chw.unsqueeze(0).to(device)

    # ── Random patch ──────────────────────────────────────────────────────
    decoder = PatchDecoder(args.seed_channels).to(device)
    with torch.no_grad():
        seed  = torch.randn(1, args.seed_channels, 4, 8, device=device)
        patch = decoder(seed).squeeze(0)

    # ── Base (no augmentation) ────────────────────────────────────────────
    with torch.no_grad():
        base = apply_patch(image_b, corners_t, patch,
                           border_scale=args.border_scale, device=device)
    save_image(base, out_dir / "00_base_no_aug.png")

    # ── N samples using the live augment_plate() from trainer.py ──────────
    print(f"\nSaving {args.n} augmented samples")
    aug_fn = lambda img: augment_plate(img, str(device))
    for i in range(args.n):
        with torch.no_grad():
            result = apply_patch(image_b, corners_t, patch,
                                 border_scale=args.border_scale, device=device,
                                 canonical_aug_fn=aug_fn)
        save_image(result, out_dir / f"{i+1:02d}_augmented.png")

    print(f"\nAll samples written to: {out_dir}/")


if __name__ == "__main__":
    main()
