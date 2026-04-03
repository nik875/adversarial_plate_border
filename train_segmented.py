#!/usr/bin/env python3
"""
train_segmented.py — run a long holdout training in restartable 20% segments.

Splits the full 3-epoch run into 5 segments of ~20% each.  After each segment
the trainer process exits (freeing all GPU/CPU memory), then this script
relaunches it from the last checkpoint.  All segments share a single run dir.

Usage:
    python train_segmented.py --holdout owlvit
"""

import argparse
import csv
import math
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# ── Configuration (mirrors the trainer command) ────────────────────────────
FINETUNED_MODELS  = "finetuned_models"
CCPD_TRAIN_CSV    = "finetuned_models/train_split.csv"
IMPERSONATION_TGT = "SHX8459"
TOTAL_EPOCHS      = 3
LR                = "1e-5"
LR_MIN            = "1e-6"
TV_WEIGHT         = "100"
DET_LOSS_WEIGHT   = "3"
EVAL_BATCH_SIZE   = "16"
NUM_SEGMENTS      = 5        # 5 segments × 20% = 100%

# PIPELINE_PAIRINGS mirrors trainer.py — needed to predict the run_dir name.
PIPELINE_PAIRINGS = [
    ("fasterrcnn",  "lprnet"),
    ("rtdetr",      "doctr-vitstr"),
    ("owlvit",      "trocr"),
    ("yolo-v9-608", "cct"),
]


def count_csv_data_rows(csv_path: str) -> int:
    """Count non-header rows in a CSV file."""
    with open(csv_path, newline="") as f:
        reader = csv.reader(f)
        next(reader, None)  # skip header
        return sum(1 for _ in reader)


def latest_checkpoint(patches_dir: Path) -> Path | None:
    """Return the most recently modified .pt file in patches_dir, or None."""
    pts = sorted(patches_dir.glob("*.pt"), key=lambda p: p.stat().st_mtime)
    return pts[-1] if pts else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--holdout", required=True,
                        choices=[d for d, _ in PIPELINE_PAIRINGS],
                        help="Detector name to hold out (e.g. owlvit)")
    args = parser.parse_args()

    holdout = args.holdout
    holdout_ocr = next(o for d, o in PIPELINE_PAIRINGS if d == holdout)

    # Fixed run_name so all segments land in the same run dir.
    run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir  = Path("runs") / f"holdout_{holdout}_{holdout_ocr}_{run_name}"

    # Estimate total optimizer steps for 3 epochs.
    # steps_per_epoch = n_images // eval_batch_size  (update_every=1 default)
    n_images       = count_csv_data_rows(CCPD_TRAIN_CSV)
    steps_per_epoch = n_images // int(EVAL_BATCH_SIZE)
    total_steps     = TOTAL_EPOCHS * steps_per_epoch
    segment_steps   = math.ceil(total_steps / NUM_SEGMENTS)

    print(f"[segmented] holdout      : {holdout} / {holdout_ocr}")
    print(f"[segmented] run_dir      : {run_dir}")
    print(f"[segmented] n_images     : {n_images}")
    print(f"[segmented] steps/epoch  : {steps_per_epoch}")
    print(f"[segmented] total_steps  : {total_steps}")
    print(f"[segmented] segment_steps: {segment_steps}  ({NUM_SEGMENTS} segments × 20%)")
    print()

    # Base command — args shared by every segment
    base_cmd = [
        sys.executable, "trainer.py",
        "--finetuned-models", FINETUNED_MODELS,
        "--holdout",          holdout,
        "--ccpd-train-csv",   CCPD_TRAIN_CSV,
        "--impersonation-target", IMPERSONATION_TGT,
        "--epochs",           str(TOTAL_EPOCHS),
        "--lr",               LR,
        "--lr-min",           LR_MIN,
        "--tv-weight",        TV_WEIGHT,
        "--det-loss-weight",  DET_LOSS_WEIGHT,
        "--skip-sanity",
        "--eval-batch-size",  EVAL_BATCH_SIZE,
        "--augment",
        "--run-name",         run_name,
        "--max-steps",        str(segment_steps),
    ]

    continue_ckpt = None

    for seg in range(1, NUM_SEGMENTS + 1):
        cmd = list(base_cmd)
        if continue_ckpt is not None:
            cmd += ["--continue", str(continue_ckpt), "--continue-lr"]

        print(f"{'='*60}")
        print(f"[segmented] Segment {seg}/{NUM_SEGMENTS}  "
              f"(steps {(seg-1)*segment_steps + 1}–{seg*segment_steps})")
        if continue_ckpt:
            print(f"[segmented] Resuming from: {continue_ckpt}")
        print(f"[segmented] Command: {' '.join(cmd)}")
        print(f"{'='*60}")

        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"\n[segmented] Segment {seg} FAILED (exit {result.returncode}) — aborting.")
            sys.exit(result.returncode)

        # Find latest checkpoint to pass to the next segment.
        patches_dir = run_dir / "patches"
        ckpt = latest_checkpoint(patches_dir)
        if ckpt is None:
            print(f"\n[segmented] WARNING: no checkpoint found in {patches_dir} "
                  f"after segment {seg}.  Cannot continue.")
            sys.exit(1)
        continue_ckpt = ckpt
        print(f"\n[segmented] Segment {seg} complete. Checkpoint: {continue_ckpt}\n")

    print(f"{'='*60}")
    print(f"[segmented] All {NUM_SEGMENTS} segments complete.")
    print(f"[segmented] Run dir: {run_dir}")
    print(f"[segmented] Final checkpoint: {continue_ckpt}")


if __name__ == "__main__":
    main()
