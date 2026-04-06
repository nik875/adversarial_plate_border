#!/usr/bin/env python3
"""
loss_landscape.py — linear interpolation loss survey between two checkpoints.

Finds the latest .pt in each run directory, linearly interpolates the seed
and decoder weights at N evenly-spaced alpha values between 0 and 1, and
estimates the average loss over a fixed set of data batches for all four
training pipelines.  Prints a table of per-pipeline and mean losses at each
alpha, giving a picture of the loss landscape between the two solutions.

Usage:
    python loss_landscape.py runs/run_a runs/run_b \\
        --finetuned-models finetuned_models \\
        --csv finetuned_models/train_split.csv

    python loss_landscape.py runs/run_a runs/run_b --n-steps 21 --n-batches 20
"""

import argparse
import sys
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).parent))
from trainer import AdversarialPatchTrainer, PatchDecoder, PIPELINE_PAIRINGS
from detector_backends import build_backend
from ocr_backends import build_ocr_backend

FINETUNED_CHECKPOINT_MAP = {
    "fasterrcnn":   "fasterrcnn_finetuned.pt",
    "rtdetr":       "rtdetr_finetuned",
    "owlvit":       "owlvit_finetuned",
    "yolo-v9-608":  "yolo608_finetuned.pt",
    "lprnet":       "lprnet_finetuned.pt",
    "trocr":        "trocr_small_finetuned.pt",
    "doctr-vitstr": "vitstr_small_finetuned.pt",
    "cct":          "cct_s_finetuned.pt",
}


def latest_checkpoint(run_dir: Path) -> Path:
    patches_dir = run_dir / "patches"
    pts = sorted(patches_dir.glob("*.pt"), key=lambda p: p.stat().st_mtime)
    if not pts:
        raise FileNotFoundError(f"No .pt checkpoints found in {patches_dir}")
    return pts[-1]


def lerp_state_dict(sd_a: dict, sd_b: dict, alpha: float) -> dict:
    """(1 - alpha) * A + alpha * B, cast to float32 for precision."""
    return {
        k: (1.0 - alpha) * sd_a[k].float() + alpha * sd_b[k].float()
        for k in sd_a
    }


def apply_interpolated_weights(
    trainer: AdversarialPatchTrainer,
    seed_a: torch.Tensor,
    seed_b: torch.Tensor,
    sd_a: dict,
    sd_b: dict,
    alpha: float,
) -> None:
    """Interpolate A and B and write the result into trainer in-place."""
    with torch.no_grad():
        interp_seed = (1.0 - alpha) * seed_a.float() + alpha * seed_b.float()
        trainer.seed.copy_(interp_seed.to(trainer.device))
    trainer.decoder.load_state_dict(lerp_state_dict(sd_a, sd_b, alpha))
    trainer.decoder.to(trainer.device)


@torch.no_grad()
def eval_pipeline_loss(
    trainer: AdversarialPatchTrainer,
    raw_items: list,
    batch_size: int,
) -> float:
    """
    Average total loss for the currently active pipeline over raw_items.

    raw_items: list of single-item dicts (loader output with leading dim stripped).
    Returns the mean loss across all sub-batches.
    """
    patch_norm = trainer.generate_patch(training_aug=False)
    total_loss = 0.0
    n_chunks = 0

    # Prepare all items once for this pipeline.
    prepared = []
    for raw in raw_items:
        item = trainer._prepare_one(raw, patch_norm, augment=False)
        item["_patch_norm"] = patch_norm
        prepared.append(item)

    for start in range(0, len(prepared), batch_size):
        chunk = prepared[start : start + batch_size]
        loss, *_ = trainer.compute_loss_batch(chunk)
        total_loss += loss.item()
        n_chunks += 1

    return total_loss / max(n_chunks, 1)


@torch.no_grad()
def generate_patch_image(trainer: AdversarialPatchTrainer) -> Image.Image:
    """Render the current decoder weights as a PIL image."""
    patch = trainer.generate_patch(training_aug=False)  # [3, H, W] in [0, 1]
    arr = (patch.cpu().clamp(0, 1) * 255).byte().permute(1, 2, 0).numpy()
    return Image.fromarray(arr)


def make_patch_grid(
    patches: list,          # list of (label: str, pil_image: Image)
    gap: int = 8,
    label_h: int = 24,
    bg: tuple = (30, 30, 30),
) -> Image.Image:
    """Concatenate labelled patch images side by side."""
    pw, ph = patches[0][1].size
    n = len(patches)
    canvas_w = n * pw + (n - 1) * gap
    canvas_h = ph + label_h
    canvas = Image.new("RGB", (canvas_w, canvas_h), bg)
    draw = ImageDraw.Draw(canvas)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    except Exception:
        font = ImageFont.load_default()

    for i, (label, img) in enumerate(patches):
        x = i * (pw + gap)
        canvas.paste(img, (x, label_h))
        draw.text((x + 4, 4), label, fill=(220, 220, 220), font=font)

    return canvas


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("run_dir_a", help="Run directory A  (alpha = 0)")
    parser.add_argument("run_dir_b", help="Run directory B  (alpha = 1)")
    parser.add_argument("--finetuned-models", default="finetuned_models", metavar="DIR",
                        help="Directory containing finetuned model weights")
    parser.add_argument("--csv", default="finetuned_models/train_split.csv",
                        help="CCPD training CSV (same as --ccpd-train-csv in train_segmented.py)")
    parser.add_argument("--n-batches", type=int, default=10,
                        help="Number of data items to evaluate per alpha step "
                             "(same items reused for every pipeline and alpha)")
    parser.add_argument("--n-steps", type=int, default=11,
                        help="Number of interpolation points  "
                             "(default 11 → alpha = 0.0, 0.1, …, 1.0)")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Sub-batch size passed to compute_loss_batch")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--impersonation-target", default="SHX8459")
    parser.add_argument("--tv-weight", type=float, default=100.0)
    parser.add_argument("--det-loss-weight", type=float, default=3.0)
    args = parser.parse_args()

    run_a = Path(args.run_dir_a)
    run_b = Path(args.run_dir_b)
    fdir  = Path(args.finetuned_models)

    # ── Checkpoints ───────────────────────────────────────────────────────────
    ckpt_path_a = latest_checkpoint(run_a)
    ckpt_path_b = latest_checkpoint(run_b)
    print(f"Checkpoint A : {ckpt_path_a}")
    print(f"Checkpoint B : {ckpt_path_b}")

    ckpt_a = torch.load(ckpt_path_a, map_location="cpu")
    ckpt_b = torch.load(ckpt_path_b, map_location="cpu")

    seed_channels = int(ckpt_a.get("seed_channels", 128))
    seed_a = ckpt_a["seed"]   # [1, C, 4, 8]  cpu
    seed_b = ckpt_b["seed"]
    sd_a   = ckpt_a["decoder"]
    sd_b   = ckpt_b["decoder"]

    print(f"Seed channels : {seed_channels}")
    print(f"  A global_update = {ckpt_a.get('global_update', '?'):>8}   "
          f"backend = {ckpt_a.get('backend', '?')}")
    print(f"  B global_update = {ckpt_b.get('global_update', '?'):>8}   "
          f"backend = {ckpt_b.get('backend', '?')}")
    print()

    # ── Load all four pipeline backends ───────────────────────────────────────
    print("Loading pipeline backends…")
    pipeline_backends = []
    for det_name, ocr_name in PIPELINE_PAIRINGS:
        det_path = str(fdir / FINETUNED_CHECKPOINT_MAP[det_name])
        ocr_path = str(fdir / FINETUNED_CHECKPOINT_MAP[ocr_name])
        det = build_backend(det_name, det_path, device=args.device)
        det.load()
        det.eval()
        det.freeze()
        ocr = build_ocr_backend(ocr_name, ocr_path, device=args.device)
        ocr.load()
        ocr.eval()
        pipeline_backends.append((det, ocr))
        print(f"  ✓  {det_name} / {ocr_name}")
    print()

    # ── Build trainer (pipeline 0 used for data loading) ──────────────────────
    det0, ocr0 = pipeline_backends[0]
    trainer = AdversarialPatchTrainer(
        detector             = det0,
        ocr                  = ocr0,
        ccpd_csv             = args.csv,
        seed_channels        = seed_channels,
        impersonation_target = args.impersonation_target,
        tv_weight            = args.tv_weight,
        det_loss_weight      = args.det_loss_weight,
        eval_batch_size      = args.batch_size,
        skip_sanity          = True,
        training             = False,
        augment              = False,
        run_name             = "_landscape_survey",
    )

    # ── Collect a fixed set of raw items (no pipeline-specific prep yet) ───────
    print(f"Collecting {args.n_batches} data items…")
    loader = trainer._make_pipeline_loader(seed=42)
    raw_items = []
    for batch in loader:
        raw_items.append({k: v[0] for k, v in batch.items()})
        if len(raw_items) >= args.n_batches:
            break
    print(f"Collected {len(raw_items)} items.\n")

    # ── Interpolation sweep ───────────────────────────────────────────────────
    alphas = [i / (args.n_steps - 1) for i in range(args.n_steps)]
    pipeline_names = [f"{d}/{o}" for d, o in PIPELINE_PAIRINGS]

    col_w = 20
    header = (
        f"{'alpha':>6}  "
        + "  ".join(f"{n:>{col_w}}" for n in pipeline_names)
        + f"  {'mean':>{col_w}}"
    )
    sep = "-" * len(header)
    print(header)
    print(sep)

    all_results = []
    patch_images = {}   # alpha → PIL image for selected alphas
    mid_alpha = alphas[args.n_steps // 2]
    capture_alphas = {alphas[0], mid_alpha, alphas[-1]}

    for alpha in alphas:
        apply_interpolated_weights(trainer, seed_a, seed_b, sd_a, sd_b, alpha)

        # Capture patch image before switching pipelines (pipeline doesn't affect decoder)
        if alpha in capture_alphas:
            patch_images[alpha] = generate_patch_image(trainer)

        row_losses = []
        for det, ocr in pipeline_backends:
            trainer._activate_pipeline(det, ocr)
            loss = eval_pipeline_loss(trainer, raw_items, args.batch_size)
            row_losses.append(loss)

        mean_loss = sum(row_losses) / len(row_losses)
        all_results.append((alpha, row_losses, mean_loss))

        print(
            f"{alpha:>6.2f}  "
            + "  ".join(f"{l:>{col_w}.4f}" for l in row_losses)
            + f"  {mean_loss:>{col_w}.4f}"
        )

    print(sep)

    # ── Summary ───────────────────────────────────────────────────────────────
    best = min(all_results, key=lambda r: r[2])
    print(f"\nLowest mean loss: alpha={best[0]:.2f}  mean={best[2]:.4f}")

    mid_idx = args.n_steps // 2
    mid = all_results[mid_idx]
    end_a, end_b = all_results[0], all_results[-1]
    barrier = mid[2] - max(end_a[2], end_b[2])
    print(f"Midpoint (alpha={mid[0]:.2f}) mean loss: {mid[2]:.4f}")
    print(f"Loss barrier at midpoint vs worse endpoint: {barrier:+.4f}  "
          f"({'barrier present' if barrier > 0 else 'no barrier / valley'})")

    # ── Patch grid image ──────────────────────────────────────────────────────
    a0, a_mid, a1 = alphas[0], mid_alpha, alphas[-1]
    panels = [
        (f"A  (alpha={a0:.2f}  update={ckpt_a.get('global_update','?')})",  patch_images[a0]),
        (f"mid (alpha={a_mid:.2f})",                                          patch_images[a_mid]),
        (f"B  (alpha={a1:.2f}  update={ckpt_b.get('global_update','?')})",  patch_images[a1]),
    ]
    grid = make_patch_grid(panels)
    out_path = Path("loss_landscape_patches.png")
    grid.save(str(out_path))
    print(f"\nPatch grid saved → {out_path}")


if __name__ == "__main__":
    main()
