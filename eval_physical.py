#!/usr/bin/env python3
"""
eval_physical.py

Evaluate adversarial patches over the physical-world test set
(control_plate_corners.csv) using the existing DetectorEvaluator framework.

Each patch is associated with a specific backend.  The script always also
runs a clean (no-patch) baseline on every unique backend.

Usage
-----
Single patch / backend pair:
    python eval_physical.py \
        --pairs patches/patch_yolov8__crnn_epoch_0099.pt:yolov8:weights/lp.pt \
        --output results/physical/

Multiple pairs:
    python eval_physical.py \
        --pairs patches/patch_yolov8__crnn_epoch_0099.pt:yolov8:weights/lp.pt \
                patches/patch_rtdetr__trocr_epoch_0079.pt:rtdetr:weights/detr.pt \
        --output results/physical/

Pair format:  patch_file:backend_name:backend_path[:key=val,key=val]
  e.g.        patches/p.pt:yolov8:lp.pt:conf_threshold=0.3

Options
-------
--csv       Path to corners CSV (default: control_plate_corners.csv)
--device    cpu / cuda (default: auto-detect)
--output    Output directory (default: results/)
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import torch

from detector_backends import build_backend, DetectorBackend
from evaluator import DetectorEvaluator, BackendMetrics


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------

def _parse_pair(spec: str) -> Tuple[str, str, str, dict]:
    """Parse  'patch.pt:backend_name:backend_path[:k=v,k=v]'  into parts."""
    parts = spec.split(":", 3)
    if len(parts) < 3:
        raise ValueError(
            f"Invalid pair spec '{spec}'. "
            "Expected format: patch_file:backend_name:backend_path[:k=v,...]"
        )
    patch_path   = parts[0]
    backend_name = parts[1]
    backend_path = parts[2]
    kwargs: dict = {}
    if len(parts) == 4:
        for kv in parts[3].split(","):
            k, _, v = kv.partition("=")
            try:
                kwargs[k] = float(v) if "." in v else int(v)
            except ValueError:
                kwargs[k] = v
    return patch_path, backend_name, backend_path, kwargs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate patches on physical-world test images",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--pairs", nargs="+", required=True,
        metavar="patch:backend:path[:opts]",
        help="One or more patch:backend:path specs (see module docstring)",
    )
    parser.add_argument(
        "--csv", default="control_plate_corners.csv",
        help="Corners CSV for the physical-world test set "
             "(default: control_plate_corners.csv)",
    )
    parser.add_argument("--device", default=None,
                        help="cuda / cpu (default: auto)")
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--output", default="results/",
                        help="Output directory (default: results/)")
    args = parser.parse_args()

    # Auto device
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[eval_physical] device={args.device}  csv={args.csv}")

    # Parse pairs
    pairs: List[Tuple[str, str, str, dict]] = []
    for spec in args.pairs:
        pairs.append(_parse_pair(spec))

    # Build one backend per unique (name, path, kwargs) combo
    seen: dict = {}
    backends: List[DetectorBackend] = []
    for _, bname, bpath, bkwargs in pairs:
        key = (bname, bpath, tuple(sorted(bkwargs.items())))
        if key not in seen:
            b = build_backend(bname, bpath, device=args.device, **bkwargs)
            seen[key] = b
            backends.append(b)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    evaluator = DetectorEvaluator(
        backends=backends,
        csv_path=args.csv,
        device=args.device,
        iou_threshold=args.iou_threshold,
    )

    # Build patch list: clean baseline + one entry per pair
    patch_paths = [patch for patch, _, _, _ in pairs]

    results: List[BackendMetrics] = evaluator.run(patch_paths=patch_paths)

    evaluator.save_csv(results,           str(out_dir / "metrics.csv"))
    evaluator.save_json(results,          str(out_dir / "metrics.json"))
    evaluator.save_summary_table(results, str(out_dir / "summary_table.txt"))
    evaluator.save_bar_chart(results,     str(out_dir / "bar_chart.png"))
    evaluator.save_matrix_heatmaps(results, str(out_dir))


if __name__ == "__main__":
    main()
