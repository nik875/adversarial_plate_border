#!/usr/bin/env python3
"""
train_segmented.py — run a long holdout training in restartable segments.

Splits the full training run into N segments.  After each segment the trainer
process exits (freeing all GPU/CPU memory), then this script relaunches it
from the last checkpoint.  All segments share a single run dir.

Usage:
    python train_segmented.py --holdout owlvit
    python train_segmented.py --holdout owlvit --epochs 5 --segments 10
    python train_segmented.py --holdout owlvit --continue   # resume latest run
"""

import argparse
import csv
import math
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# PIPELINE_PAIRINGS mirrors trainer.py — needed to predict the run_dir name.
PIPELINE_PAIRINGS = [
    ("fasterrcnn",  "lprnet"),
    ("rtdetr",      "doctr-vitstr"),
    ("owlvit",      "trocr"),
    ("yolo-v9-608", "cct"),
]


def count_csv_data_rows(csv_path: str) -> int:
    with open(csv_path, newline="") as f:
        reader = csv.reader(f)
        next(reader, None)  # skip header
        return sum(1 for _ in reader)


def latest_checkpoint(patches_dir: Path):
    pts = sorted(patches_dir.glob("*.pt"), key=lambda p: p.stat().st_mtime)
    return pts[-1] if pts else None


def global_update_from_ckpt(ckpt: Path) -> int:
    """Parse global_update from checkpoint filename (avoids importing torch)."""
    m = re.search(r"update_(\d+)\.pt$", ckpt.name)
    return int(m.group(1)) if m else 0


def find_latest_run(holdout: str, holdout_ocr: str) -> Path | None:
    runs_dir = Path("runs")
    prefix = f"holdout_{holdout}_{holdout_ocr}_"
    matches = sorted(
        (p for p in runs_dir.glob(f"{prefix}*") if p.is_dir()),
        key=lambda p: p.stat().st_mtime,
    )
    return matches[-1] if matches else None


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--holdout", required=True,
                        choices=[d for d, _ in PIPELINE_PAIRINGS],
                        help="Detector name to hold out")
    parser.add_argument("--continue", dest="resume", action="store_true",
                        help="Resume the most recent matching run from its "
                             "last checkpoint, preserving LR and dataloader state")
    parser.add_argument("--segments", type=int, default=5,
                        help="Number of segments per epoch")
    # trainer.py args — all optional, override the defaults below
    parser.add_argument("--finetuned-models", default="finetuned_models", metavar="DIR")
    parser.add_argument("--ccpd-train-csv",   default="finetuned_models/train_split.csv")
    parser.add_argument("--impersonation-target", default="SHX8459")
    parser.add_argument("--epochs",           type=int,   default=3)
    parser.add_argument("--lr",               type=float, default=1e-5)
    parser.add_argument("--lr-min",           type=float, default=1e-6)
    parser.add_argument("--tv-weight",        type=float, default=100.0)
    parser.add_argument("--det-loss-weight",  type=float, default=3.0)
    parser.add_argument("--eval-batch-size",  type=int,   default=16)
    parser.add_argument("--skip-sanity",      action="store_true", default=True)
    parser.add_argument("--no-skip-sanity",   dest="skip_sanity", action="store_false")
    parser.add_argument("--augment",          action="store_true", default=True)
    parser.add_argument("--no-augment",       dest="augment", action="store_false")
    args = parser.parse_args()

    holdout     = args.holdout
    holdout_ocr = next(o for d, o in PIPELINE_PAIRINGS if d == holdout)

    n_images        = count_csv_data_rows(args.ccpd_train_csv)
    steps_per_epoch = n_images // args.eval_batch_size
    segment_steps   = math.ceil(steps_per_epoch / args.segments)
    total_steps     = args.epochs * steps_per_epoch
    total_segments  = args.epochs * args.segments

    MIN_SEGMENT_STEPS = 50
    if segment_steps < MIN_SEGMENT_STEPS:
        print(f"[segmented] ERROR: --segments {args.segments} gives only "
              f"{segment_steps} steps/segment (minimum is {MIN_SEGMENT_STEPS}).  "
              f"Max useful --segments for this dataset: "
              f"{steps_per_epoch // MIN_SEGMENT_STEPS}.")
        sys.exit(1)

    # ── Resume detection ──────────────────────────────────────────────────
    continue_ckpt = None
    start_seg     = 1

    if args.resume:
        existing_run = find_latest_run(holdout, holdout_ocr)
        if existing_run is None:
            print(f"[segmented] --continue: no existing run found for "
                  f"holdout={holdout}/{holdout_ocr} in runs/")
            sys.exit(1)

        patches_dir = existing_run / "patches"
        ckpt = latest_checkpoint(patches_dir)
        if ckpt is None:
            print(f"[segmented] --continue: no checkpoint found in {patches_dir}")
            sys.exit(1)

        prefix   = f"holdout_{holdout}_{holdout_ocr}_"
        run_name = existing_run.name[len(prefix):]
        run_dir  = existing_run
        global_update = global_update_from_ckpt(ckpt)
        remaining_steps = max(0, total_steps - global_update)
        remaining_segs  = math.ceil(remaining_steps / segment_steps)
        start_seg       = total_segments - remaining_segs + 1
        continue_ckpt   = ckpt

        print(f"[segmented] Resuming run   : {existing_run}")
        print(f"[segmented] Checkpoint     : {ckpt.name}  (global_update={global_update})")
        print(f"[segmented] Steps done     : {global_update}/{total_steps}")
        print(f"[segmented] Segments left  : {remaining_segs}/{total_segments} "
              f"(starting at segment {start_seg})")
    else:
        run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir  = Path("runs") / f"holdout_{holdout}_{holdout_ocr}_{run_name}"

    print(f"[segmented] holdout        : {holdout} / {holdout_ocr}")
    print(f"[segmented] run_dir        : {run_dir}")
    print(f"[segmented] n_images       : {n_images}")
    print(f"[segmented] steps/epoch    : {steps_per_epoch}")
    print(f"[segmented] segments/epoch : {args.segments}  ({100//args.segments}% each)")
    print(f"[segmented] segment_steps  : {segment_steps}")
    print(f"[segmented] total_segments : {total_segments}  ({args.epochs} epochs × {args.segments})")
    print()

    # Base command — passed to every segment unchanged.
    base_cmd = [
        sys.executable, "trainer.py",
        "--finetuned-models",     args.finetuned_models,
        "--holdout",              holdout,
        "--ccpd-train-csv",       args.ccpd_train_csv,
        "--impersonation-target", args.impersonation_target,
        "--epochs",               str(args.epochs),
        "--lr",                   str(args.lr),
        "--lr-min",               str(args.lr_min),
        "--tv-weight",            str(args.tv_weight),
        "--det-loss-weight",      str(args.det_loss_weight),
        "--eval-batch-size",      str(args.eval_batch_size),
        "--run-name",             run_name,
        "--max-steps",            str(segment_steps),
    ]
    if args.skip_sanity:
        base_cmd.append("--skip-sanity")
    if args.augment:
        base_cmd.append("--augment")

    for seg in range(start_seg, total_segments + 1):
        cmd = list(base_cmd)
        if continue_ckpt is not None:
            cmd += ["--continue", str(continue_ckpt), "--continue-lr"]

        lo = (seg - 1) * segment_steps + 1
        hi = seg * segment_steps
        print(f"{'='*60}")
        print(f"[segmented] Segment {seg}/{total_segments}  (steps {lo}–{hi})")
        if continue_ckpt:
            print(f"[segmented] Resuming from: {continue_ckpt}")
        print(f"[segmented] Command: {' '.join(cmd)}")
        print(f"{'='*60}")

        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"\n[segmented] Segment {seg} FAILED (exit {result.returncode}) — aborting.")
            sys.exit(result.returncode)

        patches_dir = run_dir / "patches"
        ckpt = latest_checkpoint(patches_dir)
        if ckpt is None:
            print(f"\n[segmented] WARNING: no checkpoint found in {patches_dir} "
                  f"after segment {seg}.  Cannot continue.")
            sys.exit(1)
        continue_ckpt = ckpt
        print(f"\n[segmented] Segment {seg} complete. Checkpoint: {continue_ckpt}\n")

    print(f"{'='*60}")
    print(f"[segmented] All segments complete.")
    print(f"[segmented] Run dir: {run_dir}")
    print(f"[segmented] Final checkpoint: {continue_ckpt}")


if __name__ == "__main__":
    main()
