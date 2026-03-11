#!/usr/bin/env python3
"""
eval_physical.py

Evaluate adversarial patches over the physical-world test set
(control_plate_corners.csv) at ORIGINAL image resolution.

Unlike evaluator.py (which feeds 384×384 letterboxed images), this script
loads each HEIC/image at full resolution, applies the patch at full resolution
using the labelled corners, then hands the full-res tensor to each backend.
Backends (YOLOv8, RT-DETR, FasterRCNN) all handle their own internal resizing.

Usage
-----
python eval_physical.py \\
    --pairs patches/patch_fasterrcnn__dtrb_epoch_0059.png:fasterrcnn:weights/model.pt \\
            patches/patch_rtdetr__trocr_epoch_0079.png:rtdetr:weights/rtdetr-v2-license-plates \\
            patches/patch_yolov8__crnn_epoch_0099.png:yolov8:weights/lp_yolov8.pt \\
    --output results/physical/

Pair format:  patch_file:backend_name:backend_path[:key=val,key=val]
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
import kornia.geometry as K
from PIL import Image
from tqdm import tqdm

try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
except ImportError:
    print("Warning: pillow-heif not installed; HEIC images may fail to load.")

from detector_backends import DetectorBackend, build_backend
from evaluator import BackendMetrics


# ---------------------------------------------------------------------------
# Patch loading
# ---------------------------------------------------------------------------

def _load_patch(path: str, device: str) -> torch.Tensor:
    """Load a patch PNG/JPG or .pt checkpoint → [3, H, W] float32 in [0, 1]."""
    p = Path(path)
    if p.suffix == ".pt":
        ckpt = torch.load(path, map_location="cpu")
        raw = ckpt.get("patch", ckpt)
        patch = (torch.tanh(raw) * 0.5 + 0.5).clamp(0, 1)
    else:
        img = Image.open(path).convert("RGB")
        patch = T.ToTensor()(img)
    return patch.to(device)


# ---------------------------------------------------------------------------
# Patch application at full resolution
# ---------------------------------------------------------------------------

def _apply_patch(image: torch.Tensor,
                 corners: torch.Tensor,
                 patch: torch.Tensor,
                 border_scale: float = 1.4) -> torch.Tensor:
    """
    Warp patch as a border around the licence plate at original resolution.

    Parameters
    ----------
    image   : [3, H, W] float32 in [0, 1]
    corners : [4, 2] plate corner coordinates in original image space
    patch   : [3, Ph, Pw] float32 in [0, 1]
    """
    device = image.device
    img_h, img_w = image.shape[1], image.shape[2]
    patch_h, patch_w = patch.shape[1], patch.shape[2]

    plate = corners.to(device)                       # [4, 2]
    cx = plate[:, 0].mean()
    cy = plate[:, 1].mean()
    ctr = torch.stack([cx, cy])

    border = (ctr.unsqueeze(0) + (plate - ctr.unsqueeze(0)) * border_scale
              ).unsqueeze(0)                          # [1, 4, 2]
    plate_b = plate.unsqueeze(0)                      # [1, 4, 2]

    src = torch.tensor(
        [[0, 0], [patch_w, 0], [patch_w, patch_h], [0, patch_h]],
        dtype=torch.float32, device=device,
    ).unsqueeze(0)                                    # [1, 4, 2]

    M_border = K.get_perspective_transform(src, border)
    M_plate  = K.get_perspective_transform(src, plate_b)

    batch   = image.unsqueeze(0)
    ones    = torch.ones(1, 1, patch_h, patch_w, device=device)
    dsize   = (img_h, img_w)
    kwargs  = dict(mode="bilinear", padding_mode="zeros", align_corners=True)

    warped        = K.warp_perspective(patch.unsqueeze(0), M_border, dsize, **kwargs)
    w_border_mask = K.warp_perspective(ones,               M_border, dsize, **kwargs)
    w_plate_mask  = K.warp_perspective(ones,               M_plate,  dsize, **kwargs)

    mask = torch.clamp(w_border_mask - w_plate_mask, 0, 1).expand(-1, 3, -1, -1)
    return torch.clamp(batch * (1 - mask) + warped * mask, 0, 1).squeeze(0)


# ---------------------------------------------------------------------------
# IoU helper
# ---------------------------------------------------------------------------

def _iou(a: torch.Tensor, b: torch.Tensor) -> float:
    ix1 = max(a[0].item(), b[0].item())
    iy1 = max(a[1].item(), b[1].item())
    ix2 = min(a[2].item(), b[2].item())
    iy2 = min(a[3].item(), b[3].item())
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    ua = (a[2] - a[0]) * (a[3] - a[1])
    ub = (b[2] - b[0]) * (b[3] - b[1])
    union = ua.item() + ub.item() - inter
    return inter / (union + 1e-8) if union > 0 else 0.0


# ---------------------------------------------------------------------------
# Core evaluation loop
# ---------------------------------------------------------------------------

def preload_images(df: pd.DataFrame, device: str, scale: float = 1.0,
                   num_workers: int = 0) -> List[Tuple]:
    """Load all images and corners to GPU once. Returns list of (image, corners, gt_box)."""
    import os
    import torch.nn.functional as F
    from concurrent.futures import ThreadPoolExecutor, as_completed
    to_tensor = T.ToTensor()
    if num_workers == 0:
        num_workers = os.cpu_count() or 1

    rows = list(df.iterrows())

    def _load_one(args):
        idx, row = args
        try:
            pil_img = Image.open(row["filename"]).convert("RGB")
        except Exception as e:
            print(f"  [warn] could not load {row['filename']}: {e}")
            return idx, None
        image = to_tensor(pil_img)
        if scale != 1.0:
            image = F.interpolate(image.unsqueeze(0), scale_factor=scale,
                                  mode="bilinear", align_corners=False).squeeze(0)
        corners = torch.tensor([
            [row["p1_x"], row["p1_y"]],
            [row["p2_x"], row["p2_y"]],
            [row["p3_x"], row["p3_y"]],
            [row["p4_x"], row["p4_y"]],
        ], dtype=torch.float32) * scale
        return idx, (image, corners)

    results = [None] * len(rows)
    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        futures = {pool.submit(_load_one, r): r[0] for r in rows}
        for f in tqdm(as_completed(futures), total=len(futures),
                      desc=f"Preloading images to GPU ({num_workers} threads)"):
            idx, data = f.result()
            results[idx] = data

    samples = []
    for data in results:
        if data is None:
            continue
        image, corners = data
        image   = image.to(device)
        corners = corners.to(device)
        gt_box  = torch.stack([
            corners[:, 0].min(), corners[:, 1].min(),
            corners[:, 0].max(), corners[:, 1].max(),
        ])
        samples.append((image, corners, gt_box))
    return samples


def evaluate_one(backend: DetectorBackend,
                 samples: List[Tuple],
                 patch: Optional[torch.Tensor],
                 patch_name: str,
                 device: str,
                 iou_threshold: float = 0.5) -> BackendMetrics:
    """Evaluate a single backend × patch combo over preloaded samples."""
    m = BackendMetrics(name=backend.name, patch_name=patch_name)
    backend.eval()

    with torch.no_grad():
        for image, corners, gt_box in tqdm(samples,
                                           desc=f"  {backend.name} | {patch_name}",
                                           leave=False):
            if patch is not None:
                image = _apply_patch(image, corners, patch)

            t0   = time.perf_counter()
            dets = backend.predict(image)
            m.latency_ms.append((time.perf_counter() - t0) * 1000)

            m.num_images       += 1
            m.total_detections += len(dets)

            best_iou = best_conf = 0.0
            for det in dets:
                iou = _iou(det.box, gt_box)
                if iou > best_iou:
                    best_iou  = iou
                    best_conf = det.confidence

            m.iou_values.append(best_iou)
            m.conf_values.append(best_conf)
            if best_iou >= iou_threshold:
                m.true_positives += 1
            else:
                m.false_negatives += 1

    return m


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------

def _parse_pair(spec: str) -> Tuple[str, str, str, dict]:
    parts = spec.split(":", 3)
    if len(parts) < 3:
        raise ValueError(
            f"Invalid pair spec '{spec}'. "
            "Expected: patch_file:backend_name:backend_path[:k=v,...]"
        )
    patch_path, backend_name, backend_path = parts[0], parts[1], parts[2]
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

# Default weight paths keyed by backend name
_DEFAULT_WEIGHTS: Dict[str, str] = {
    "fasterrcnn":   "weights/model.pt",
    "rtdetr":       "weights/rtdetr-v2-license-plate",
    "yolov8":       "weights/lp_yolov8.pt",
    "yolov11":      "weights/yolov11s-license-plate.pt",
    "yolo-v9-384":  "none",
}


def _infer_backend(patch_path: str) -> Tuple[str, str]:
    """Infer backend name and default weights from a patch filename.

    Expects filenames like patch_BACKEND_*.png produced by the trainer.
    Returns (backend_name, weights_path).
    """
    stem = Path(patch_path).stem          # e.g. patch_rtdetr_best
    parts = stem.split("_")
    if len(parts) < 2 or parts[0] != "patch":
        raise ValueError(
            f"Cannot infer backend from '{patch_path}'. "
            "Use --pairs patch:backend:weights instead."
        )
    # Backend name follows the leading 'patch_' and may itself contain '_'
    # Try progressively longer candidate names against the known-weights table.
    for end in range(len(parts), 1, -1):
        candidate = "_".join(parts[1:end])
        if candidate in _DEFAULT_WEIGHTS:
            weights = _DEFAULT_WEIGHTS[candidate]
            return candidate, weights
    # Fall back to everything between 'patch_' and the last token
    backend = "_".join(parts[1:-1]) if len(parts) > 2 else parts[1]
    raise ValueError(
        f"Unknown backend '{backend}' inferred from '{patch_path}'. "
        f"Known backends: {list(_DEFAULT_WEIGHTS)}. "
        "Use --pairs patch:backend:weights to specify explicitly."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate patches on physical-world images at original resolution",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--pairs", nargs="+", default=[],
        metavar="patch:backend:path[:opts]",
        help="Explicit patch/backend/weights triplets.",
    )
    parser.add_argument(
        "--patches", nargs="+", default=[],
        metavar="patch.png",
        help="Bare patch paths; backend and weights are inferred from the filename.",
    )
    parser.add_argument(
        "--csv", default="control_plate_corners.csv",
        help="Corners CSV (default: control_plate_corners.csv)",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--gpu-preload", action="store_true",
                        help="Preload all images to GPU memory once before evaluation.")
    parser.add_argument("--num-workers", type=int, default=0,
                        help="Threads for parallel image loading (default: nproc).")
    parser.add_argument("--scale", type=float, default=1.0,
                        help="Resize images by this factor before evaluation (e.g. 0.5 to halve resolution).")
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--output", default="results/",
                        help="Output directory (default: results/)")
    args = parser.parse_args()

    if not args.pairs and not args.patches:
        parser.error("Provide at least one of --pairs or --patches.")

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[eval_physical] device={args.device}  csv={args.csv}")

    df = pd.read_csv(args.csv)
    print(f"[eval_physical] {len(df)} images loaded from {args.csv}")

    pairs: List[Tuple[str, str, str, dict]] = [_parse_pair(s) for s in args.pairs]
    for patch_path in args.patches:
        backend_name, weights_path = _infer_backend(patch_path)
        print(f"[eval_physical] Inferred backend '{backend_name}' "
              f"(weights: {weights_path}) for {patch_path}")
        pairs.append((patch_path, backend_name, weights_path, {}))

    # Build unique backends
    seen: Dict[tuple, DetectorBackend] = {}
    for _, bname, bpath, bkwargs in pairs:
        key = (bname, bpath, tuple(sorted(bkwargs.items())))
        if key not in seen:
            seen[key] = build_backend(bname, bpath, device=args.device, **bkwargs)

    # Load patches
    patch_tensors: List[Tuple[str, Optional[torch.Tensor]]] = [("clean", None)]
    for patch_path, _, _, _ in pairs:
        name   = Path(patch_path).stem
        tensor = _load_patch(patch_path, args.device)
        patch_tensors.append((name, tensor))
        print(f"[eval_physical] Loaded patch '{name}' from {patch_path}")

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    samples = preload_images(df, args.device, args.scale, args.num_workers) if args.gpu_preload else None

    # Evaluate: every backend × (clean + all patches)
    all_results: List[BackendMetrics] = []
    for (bname, bpath, bkwargs_t), backend in seen.items():
        backend.ensure_loaded()
        backend.freeze()
        print(f"\n── Backend: {bname} ──")
        eval_data = samples if samples is not None else preload_images(df, args.device, args.scale, args.num_workers)
        for patch_name, patch_tensor in patch_tensors:
            m = evaluate_one(
                backend, eval_data, patch_tensor, patch_name,
                device=args.device, iou_threshold=args.iou_threshold,
            )
            all_results.append(m)
            print(f"  {m.summary()}")

    # Save outputs via evaluator helpers
    from evaluator import DetectorEvaluator
    ev = DetectorEvaluator.__new__(DetectorEvaluator)   # no __init__ needed
    ev.save_csv(all_results,           str(out_dir / "metrics.csv"))
    ev.save_json(all_results,          str(out_dir / "metrics.json"))
    ev.save_summary_table(all_results, str(out_dir / "summary_table.txt"))
    ev.save_bar_chart(all_results,     str(out_dir / "bar_chart.png"))
    ev.save_matrix_heatmaps(all_results, str(out_dir))


if __name__ == "__main__":
    main()
