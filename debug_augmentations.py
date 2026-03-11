#!/usr/bin/env python3
"""
debug_augmentations.py

Apply a random patch to one image, then save one PNG per augmentation variant
to a debug directory. Useful for visually checking what each transform looks
like and tuning parameter ranges before a full training run.

Each augmentation is swept across its full parameter range in isolation.
A final section saves several random full-combination samples.

Usage:
    python debug_augmentations.py                      # first row in preproc_labels.csv
    python debug_augmentations.py --random-idx         # random row
    python debug_augmentations.py --idx 12             # specific row
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
import kornia
import kornia.geometry as KG
from PIL import Image

from trainer import PatchDecoder, PATCH_WIDTH, PATCH_HEIGHT


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
    arr = np.array(img, dtype=np.float32) / 255.0   # [H, W, 3]
    return torch.from_numpy(arr).permute(2, 0, 1)   # [3, H, W]


def save_image(tensor: torch.Tensor, path: Path) -> None:
    """Save [1, C, H, W] or [C, H, W] tensor to PNG."""
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    arr = (tensor.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    Image.fromarray(arr).save(path)
    print(f"  saved {path.name}")


# ---------------------------------------------------------------------------
# Patch application (replicates trainer's homography path)
# ---------------------------------------------------------------------------

def apply_patch(
    image: torch.Tensor,       # [1, 3, H, W] float32 [0,1]
    corners: torch.Tensor,     # [4, 2] plate corners
    patch: torch.Tensor,       # [3, PATCH_HEIGHT, PATCH_WIDTH] float32 [0,1]
    border_scale: float = 1.4,
    device: torch.device = torch.device("cpu"),
    canonical_aug_fn=None,     # optional fn([1,3,ph,pw]) → [1,3,ph,pw] applied in canonical space
) -> torch.Tensor:
    """
    Return [1, 3, H, W] image with patch composited into the border region.

    Pipeline (matches trainer.py):
      1. Warp border+plate region of the image to patch canvas (canonical space).
      2. Composite: patch fills the border ring, original plate pixels fill center.
      3. Optionally apply canonical_aug_fn in canonical space.
      4. Warp augmented canonical back to full image.
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

    # Extract canonical view
    canonical = KG.warp_perspective(image, M_to_canonical, (ph, pw),
                                    mode="bilinear", padding_mode="zeros",
                                    align_corners=True)

    # Plate mask in canonical space
    M_c  = M_to_canonical[0]
    ph4  = torch.cat([plate, plate.new_ones(4, 1)], dim=1).T
    pc_h = M_c @ ph4
    plate_canonical = (pc_h[:2] / pc_h[2:3]).T.unsqueeze(0)
    M_plate_in_canonical = KG.get_perspective_transform(src, plate_canonical)
    plate_mask = KG.warp_perspective(ones, M_plate_in_canonical, (ph, pw),
                                     mode="bilinear", padding_mode="zeros",
                                     align_corners=True).expand(-1, 3, -1, -1)

    # Composite
    patch_b   = patch.unsqueeze(0).to(device)
    composite = patch_b * (1 - plate_mask) + canonical * plate_mask

    # Optional canonical-space augmentation
    if canonical_aug_fn is not None:
        composite = canonical_aug_fn(composite)

    # Warp back
    warped_back = KG.warp_perspective(composite, M_border, (H, W),
                                      mode="bilinear", padding_mode="zeros",
                                      align_corners=True)
    border_mask = KG.warp_perspective(ones, M_border, (H, W),
                                      mode="bilinear", padding_mode="zeros",
                                      align_corners=True).expand(-1, 3, -1, -1)

    result = image * (1 - border_mask) + warped_back * border_mask
    return torch.clamp(result, 0, 1)


# ---------------------------------------------------------------------------
# Individual differentiable augmentation ops (same formulas as trainer.py)
# ---------------------------------------------------------------------------

def aug_brightness(image: torch.Tensor, factor: float) -> torch.Tensor:
    return torch.clamp(image * factor, 0, 1)


def aug_contrast(image: torch.Tensor, factor: float) -> torch.Tensor:
    mean = image.mean()
    return torch.clamp((image - mean) * factor + mean, 0, 1)


def aug_saturation(image: torch.Tensor, factor: float) -> torch.Tensor:
    return torch.clamp(kornia.enhance.adjust_saturation(image, factor), 0, 1)


def aug_color_temperature(image: torch.Tensor, shift: float) -> torch.Tensor:
    """shift > 0 = warm (boost R, reduce B); shift < 0 = cool (boost B, reduce R)."""
    scale = torch.tensor(
        [1.0 + shift * 0.3, 1.0, 1.0 - shift * 0.3],
        dtype=image.dtype, device=image.device,
    ).view(1, 3, 1, 1)
    return torch.clamp(image * scale, 0, 1)


def aug_shadow(image: torch.Tensor, angle_deg: float, intensity: float) -> torch.Tensor:
    H, W = image.shape[-2], image.shape[-1]
    device = image.device
    xs = torch.linspace(0.0, 1.0, W, device=device)
    ys = torch.linspace(0.0, 1.0, H, device=device)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    cos_a    = float(np.cos(np.radians(angle_deg)))
    sin_a    = float(np.sin(np.radians(angle_deg)))
    gradient = grid_x * cos_a + grid_y * sin_a
    gradient = (gradient - gradient.min()) / (gradient.max() - gradient.min() + 1e-6)
    shadow   = 1.0 - gradient * intensity
    return torch.clamp(image * shadow.unsqueeze(0).unsqueeze(0), 0, 1)


def aug_random_combined(image: torch.Tensor) -> torch.Tensor:
    """One random draw from the full augmentation distribution used in training."""
    image = aug_brightness(image, random.uniform(0.5, 1.5))
    image = aug_contrast(image, random.uniform(0.7, 1.3))
    image = aug_saturation(image, random.uniform(0.5, 1.5))
    image = aug_color_temperature(image, random.uniform(-0.2, 0.2))
    image = aug_shadow(image, random.uniform(0.0, 360.0), random.uniform(0.1, 0.4))
    return image


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv",         default="preproc_labels.csv")
    parser.add_argument("--idx",         type=int, default=0,
                        help="Row index into the CSV (ignored if --random-idx)")
    parser.add_argument("--random-idx",  action="store_true",
                        help="Pick a random row from the CSV")
    parser.add_argument("--out",         default="debug_augmentations",
                        help="Output directory (created if absent)")
    parser.add_argument("--device",      default="cpu")
    parser.add_argument("--seed-channels", type=int, default=128)
    parser.add_argument("--border-scale", type=float, default=1.4)
    parser.add_argument("--seed",        type=int, default=None,
                        help="Random seed for reproducible patch + combined samples")
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

    # Halve if very large (matches trainer behaviour)
    _, H, W = image_chw.shape
    if max(H, W) > 2000:
        image_chw = F.interpolate(image_chw.unsqueeze(0), scale_factor=0.5,
                                  mode="bilinear", align_corners=False).squeeze(0)
        corners_t = corners_t * 0.5

    image_b = image_chw.unsqueeze(0).to(device)   # [1, 3, H, W]

    # ── Generate random patch ──────────────────────────────────────────────
    decoder = PatchDecoder(args.seed_channels).to(device)
    with torch.no_grad():
        seed  = torch.randn(1, args.seed_channels, 4, 8, device=device)
        patch = decoder(seed).squeeze(0)   # [3, 256, 512]

    # ── Helper: apply patch with a canonical-space aug function ───────────
    def render(aug_fn=None) -> torch.Tensor:
        return apply_patch(image_b, corners_t, patch,
                           border_scale=args.border_scale, device=device,
                           canonical_aug_fn=aug_fn)

    def save(name: str, tensor: torch.Tensor) -> None:
        save_image(tensor, out_dir / name)

    # ── Base (patch only, no augmentation) ────────────────────────────────
    with torch.no_grad():
        save("00_base_patch_only.png", render())

    # Sweep values are sampled uniformly within the exact training bounds so the
    # debug images reflect what the model will actually see during training.

    # ── Brightness sweep  [0.5, 1.5] ─────────────────────────────────────
    print("\nBrightness")
    for i, factor in enumerate([0.5, 0.7, 0.9, 1.0, 1.1, 1.3, 1.5]):
        with torch.no_grad():
            save(f"01_brightness_{i+1:02d}_factor{factor:.1f}.png",
                 render(lambda img, f=factor: aug_brightness(img, f)))

    # ── Contrast sweep  [0.7, 1.3] ───────────────────────────────────────
    print("\nContrast")
    for i, factor in enumerate([0.7, 0.85, 1.0, 1.15, 1.3]):
        with torch.no_grad():
            save(f"02_contrast_{i+1:02d}_factor{factor:.2f}.png",
                 render(lambda img, f=factor: aug_contrast(img, f)))

    # ── Saturation sweep  [0.5, 1.5] ─────────────────────────────────────
    print("\nSaturation")
    for i, factor in enumerate([0.5, 0.75, 1.0, 1.25, 1.5]):
        with torch.no_grad():
            save(f"03_saturation_{i+1:02d}_factor{factor:.2f}.png",
                 render(lambda img, f=factor: aug_saturation(img, f)))

    # ── Color temperature sweep  [-0.2, 0.2] ─────────────────────────────
    print("\nColor temperature")
    for i, shift in enumerate([-0.2, -0.1, 0.0, 0.1, 0.2]):
        label = "cool" if shift < 0 else ("neutral" if shift == 0 else "warm")
        with torch.no_grad():
            save(f"04_colortemp_{i+1:02d}_{label}_shift{shift:+.1f}.png",
                 render(lambda img, s=shift: aug_color_temperature(img, s)))

    # ── Shadow angle sweep  [0°, 360°], intensity mid-point 0.25 ─────────
    print("\nDirectional shadow — angle sweep")
    for i, angle in enumerate([0, 45, 90, 135, 180, 225, 270, 315]):
        with torch.no_grad():
            save(f"05_shadow_angle_{i+1:02d}_{angle:03d}deg.png",
                 render(lambda img, a=angle: aug_shadow(img, a, intensity=0.25)))

    # ── Shadow intensity sweep  [0.1, 0.4], angle fixed 90° ──────────────
    print("\nDirectional shadow — intensity sweep")
    for i, intensity in enumerate([0.1, 0.175, 0.25, 0.325, 0.4]):
        with torch.no_grad():
            save(f"06_shadow_intensity_{i+1:02d}_int{intensity:.3f}.png",
                 render(lambda img, v=intensity: aug_shadow(img, angle_deg=90.0, intensity=v)))

    # ── Random combined samples ───────────────────────────────────────────
    print("\nRandom combined")
    for i in range(8):
        with torch.no_grad():
            save(f"07_combined_{i+1:02d}_random.png",
                 render(aug_random_combined))

    print(f"\nAll variants written to: {out_dir}/")


if __name__ == "__main__":
    main()
